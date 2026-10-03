#include "src/models/qwen36_35b_a3b/kernels/rocm/kernels.hpp"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

#include <cmath>
#include <cstdint>
#include <hipcub/block/block_radix_sort.hpp>
#include <stdexcept>
#include <type_traits>

#include "src/models/qwen36_35b_a3b/mtp_sampling.hpp"

// HIP kernels follow the layouts and operator formulas in reference.cpp.

namespace gufo::models::qwen36_35b_a3b::rocm {
namespace {

constexpr unsigned kThreads = 256;

__device__ __forceinline__ float SigmoidF(float x) {
  return 1.0f / (1.0f + __expf(-x));
}
__device__ __forceinline__ float SoftplusF(float x) {
  return x > 20.0f ? x : log1pf(__expf(x));
}
__device__ __forceinline__ float SiluF(float x) {
  return x * SigmoidF(x);
}
__device__ __forceinline__ float Bf16ToF32(std::uint16_t h) {
  return __uint_as_float(static_cast<std::uint32_t>(h) << 16);
}

/// Reads element i of a Q8_0 / F32 / BF16 / F16 row.
__device__ __forceinline__ float RowElement(const void* row, WeightType type,
                                            std::uint32_t i) {
  switch (type) {
    case WeightType::kF32:
      return static_cast<const float*>(row)[i];
    case WeightType::kBF16:
      return Bf16ToF32(static_cast<const std::uint16_t*>(row)[i]);
    case WeightType::kF16:
      return __half2float(static_cast<const __half*>(row)[i]);
    case WeightType::kQ8_0: {
      const auto* blk = static_cast<const std::uint8_t*>(row) + (i / 32) * 34;
      const __half d = *reinterpret_cast<const __half*>(blk);
      const auto q = static_cast<const std::int8_t*>(
          static_cast<const void*>(blk + 2))[i % 32];
      return __half2float(d) * static_cast<float>(q);
    }
    default:  // Block-quantized types use the dedicated GEMV kernels.
      break;
  }
  return 0.0f;
}

__device__ __forceinline__ std::size_t RowBytes(WeightType type,
                                                std::uint32_t k) {
  switch (type) {
    case WeightType::kF32:
      return static_cast<std::size_t>(k) * 4;
    case WeightType::kBF16:
    case WeightType::kF16:
      return static_cast<std::size_t>(k) * 2;
    case WeightType::kQ8_0:
      return static_cast<std::size_t>(k / 32) * 34;
    default:
      break;
  }
  return 0;
}

/// Sum over one wave; every lane receives the total. The wave-per-row
/// kernels split rows across lanes (gfx1151 runs wave32).
__device__ __forceinline__ float WaveSum(float v) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor(v, offset);
  }
  return v;
}

/// Block-wide sum over kThreads threads; every thread receives the total.
__device__ float BlockSum(float v, float* shared) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor(v, offset);
  }
  const int lane = threadIdx.x % warpSize;
  const int warp = threadIdx.x / warpSize;
  __syncthreads();
  if (lane == 0) {
    shared[warp] = v;
  }
  __syncthreads();
  float total = 0.0f;
  for (int w = 0; w < static_cast<int>(blockDim.x / warpSize); ++w) {
    total += shared[w];
  }
  return total;
}

__device__ float BlockMax(float v, float* shared) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_xor(v, offset));
  }
  const int lane = threadIdx.x % warpSize;
  const int warp = threadIdx.x / warpSize;
  __syncthreads();
  if (lane == 0) {
    shared[warp] = v;
  }
  __syncthreads();
  float total = -INFINITY;
  for (int w = 0; w < static_cast<int>(blockDim.x / warpSize); ++w) {
    total = fmaxf(total, shared[w]);
  }
  return total;
}

__global__ void EmbedKernel(const void* table, WeightType type,
                            const std::int32_t* tokens, float* res,
                            std::uint32_t hidden, std::uint32_t streams) {
  const std::uint32_t t = blockIdx.x;
  const auto* row = static_cast<const std::uint8_t*>(table) +
                    RowBytes(type, hidden) * tokens[t];
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = RowElement(row, type, i);
    for (std::uint32_t s = 0; s < streams; ++s) {
      res[(static_cast<std::size_t>(t) * streams + s) * hidden + i] = v;
    }
  }
}

__global__ void RmsNormKernel(const float* x, const float* gamma, float* out,
                              std::uint32_t group_dim, std::uint32_t groups,
                              float eps) {
  __shared__ float shared[32];
  const std::size_t row = blockIdx.x;
  const float* src = x + row * group_dim;
  float* dst = out + row * group_dim;
  float ss = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < group_dim; i += blockDim.x) {
    ss += src[i] * src[i];
  }
  ss = BlockSum(ss, shared);
  const float scale = rsqrtf(ss / static_cast<float>(group_dim) + eps);
  // A grouped norm shares one gamma row across the groups of a token.
  const float* g =
      gamma == nullptr ? nullptr : gamma + (row % groups) * group_dim;
  for (std::uint32_t i = threadIdx.x; i < group_dim; i += blockDim.x) {
    dst[i] = src[i] * scale * (g != nullptr ? g[i] : 1.0f);
  }
}

/// Dense Q6_K projection for 1-8 decode rows: y[t][r] = W[r] . x[t].
/// One wave per weight row; a lane unpacks four bytes of each Q6_K plane
/// once (16 weights) and applies them to every row's F32 activations, so the
/// weights cost one read and one decode regardless of the row count, and
/// each row's sum keeps the same order at every batch width.
/// block_q6_K: ql[128] low nibbles, qh[64] high pairs, scales[16], d (F16).
/// One wave's dot products of kRows rows (`row_stride` bytes apart) with
/// kTokens activation rows. Each activation chunk is loaded once for all
/// rows; every (row, token) sum keeps the single-row order.
template<unsigned kTokens, unsigned kRows>
__device__ __forceinline__ void Q6KRowsDot(const std::uint8_t* __restrict__ w0,
                                           std::size_t row_stride,
                                           const float* __restrict__ x,
                                           std::uint32_t k, std::uint32_t lane,
                                           float (&acc)[kRows][kTokens]) {
  const std::uint32_t blocks = k / 256;
  // Lane -> (super-block parity, half n, l group): 16 work items cover a
  // super-block, so a wave covers two per step.
  const std::uint32_t sb_off = lane / 16;
  const std::uint32_t n = (lane / 8) % 2;
  const std::uint32_t l0 = (lane % 8) * 4;
  // Rows are repacked (RepackQ6KRows): every block's ql, then qh, then
  // scales, then d, so the 4-byte reads are aligned.
  const auto load4 = [](const std::uint8_t* p) {
    return *reinterpret_cast<const std::uint32_t*>(p);
  };
  for (std::uint32_t sb = sb_off; sb < blocks; sb += 2) {
    float wv[kRows][16];
#pragma unroll
    for (unsigned r = 0; r < kRows; ++r) {
      const std::uint8_t* wrow = w0 + r * row_stride;
      const std::uint8_t* ql = wrow + sb * 128;
      const std::uint32_t ql_lo = load4(ql + 64 * n + l0);
      const std::uint32_t ql_hi = load4(ql + 64 * n + 32 + l0);
      const std::uint32_t qh =
          load4(wrow + blocks * 128 + sb * 64 + 32 * n + l0);
      const auto* sc =
          reinterpret_cast<const std::int8_t*>(wrow + blocks * 192 + sb * 16);
      const float d = __half2float(
          reinterpret_cast<const __half*>(wrow + blocks * 208)[sb]);
      // l in [l0, l0+3] lies in one 16-element scale group per quarter.
      const std::uint32_t is = 8 * n + l0 / 16;
      const float s0 = d * sc[is + 0], s1 = d * sc[is + 2], s2 = d * sc[is + 4],
                  s3 = d * sc[is + 6];
#pragma unroll
      for (std::uint32_t j = 0; j < 4; ++j) {
        const std::uint32_t lo = (ql_lo >> (8 * j)) & 0xFFu;
        const std::uint32_t hi = (ql_hi >> (8 * j)) & 0xFFu;
        const std::uint32_t h = (qh >> (8 * j)) & 0xFFu;
        wv[r][j] =
            s0 * static_cast<float>(
                     static_cast<int>((lo & 0xFu) | ((h & 3u) << 4)) - 32);
        wv[r][4 + j] =
            s1 *
            static_cast<float>(
                static_cast<int>((hi & 0xFu) | (((h >> 2) & 3u) << 4)) - 32);
        wv[r][8 + j] =
            s2 * static_cast<float>(
                     static_cast<int>((lo >> 4) | (((h >> 4) & 3u) << 4)) - 32);
        wv[r][12 + j] =
            s3 * static_cast<float>(
                     static_cast<int>((hi >> 4) | (((h >> 6) & 3u) << 4)) - 32);
      }
    }
    const std::uint32_t base = sb * 256 + 128 * n + l0;
#pragma unroll
    for (unsigned t = 0; t < kTokens; ++t) {
      const float* xr = x + static_cast<std::size_t>(t) * k + base;
      const float4 x0 = *reinterpret_cast<const float4*>(xr);
      const float4 x1 = *reinterpret_cast<const float4*>(xr + 32);
      const float4 x2 = *reinterpret_cast<const float4*>(xr + 64);
      const float4 x3 = *reinterpret_cast<const float4*>(xr + 96);
#pragma unroll
      for (unsigned r = 0; r < kRows; ++r) {
        const float* v = wv[r];
        float a = acc[r][t];
        a = fmaf(v[0], x0.x, a);
        a = fmaf(v[1], x0.y, a);
        a = fmaf(v[2], x0.z, a);
        a = fmaf(v[3], x0.w, a);
        a = fmaf(v[4], x1.x, a);
        a = fmaf(v[5], x1.y, a);
        a = fmaf(v[6], x1.z, a);
        a = fmaf(v[7], x1.w, a);
        a = fmaf(v[8], x2.x, a);
        a = fmaf(v[9], x2.y, a);
        a = fmaf(v[10], x2.z, a);
        a = fmaf(v[11], x2.w, a);
        a = fmaf(v[12], x3.x, a);
        a = fmaf(v[13], x3.y, a);
        a = fmaf(v[14], x3.z, a);
        a = fmaf(v[15], x3.w, a);
        acc[r][t] = a;
      }
    }
  }
#pragma unroll
  for (unsigned r = 0; r < kRows; ++r)
#pragma unroll
    for (unsigned t = 0; t < kTokens; ++t)
      acc[r][t] = WaveSum(acc[r][t]);
}

/// Each wave computes kRows consecutive rows. With `up`, w is the gate and
/// each output is SiLU(gate) * up, in SwigluKernel's arithmetic; the two
/// rows are separate sums as in two ungated projections.
template<unsigned kTokens, unsigned kRows>
__launch_bounds__(256) __global__
    void DenseQ6KVecKernel(const std::uint8_t* __restrict__ w,
                           const std::uint8_t* __restrict__ up,
                           const float* __restrict__ x, float* __restrict__ y,
                           std::uint32_t rows, std::uint32_t k) {
  const std::uint32_t lane = threadIdx.x % warpSize;
  const std::uint32_t row0 =
      (blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize) * kRows;
  if (row0 >= rows)
    return;
  const std::size_t row_bytes = static_cast<std::size_t>(k / 256) * 210;
  // A ragged tail recomputes the last row; only real rows are written.
  if constexpr (kRows > 1) {
    if (row0 + kRows > rows) {
      for (std::uint32_t r = row0; r < rows; ++r) {
        float one[1][kTokens] = {};
        Q6KRowsDot<kTokens, 1>(w + r * row_bytes, row_bytes, x, k, lane, one);
        if (lane == 0)
          for (unsigned t = 0; t < kTokens; ++t)
            y[static_cast<std::size_t>(t) * rows + r] = one[0][t];
      }
      return;
    }
  }
  float acc[kRows][kTokens] = {};
  Q6KRowsDot<kTokens, kRows>(w + row0 * row_bytes, row_bytes, x, k, lane, acc);
  if (up != nullptr) {
    float u[kRows][kTokens] = {};
    Q6KRowsDot<kTokens, kRows>(up + row0 * row_bytes, row_bytes, x, k, lane, u);
#pragma unroll
    for (unsigned r = 0; r < kRows; ++r)
#pragma unroll
      for (unsigned t = 0; t < kTokens; ++t)
        acc[r][t] = SiluF(acc[r][t]) * u[r][t];
  }
  if (lane == 0) {
#pragma unroll
    for (unsigned r = 0; r < kRows; ++r)
#pragma unroll
      for (unsigned t = 0; t < kTokens; ++t)
        y[static_cast<std::size_t>(t) * rows + row0 + r] = acc[r][t];
  }
}

/// Q4_K/Q5_K sub-block scale and min (ggml get_scale_min_k4).
__device__ __forceinline__ void KScaleMin(const std::uint8_t* q, int j,
                                          std::uint32_t& sc, std::uint32_t& m) {
  if (j < 4) {
    sc = q[j] & 63u;
    m = q[j + 4] & 63u;
  } else {
    sc = (q[j + 4] & 0xFu) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

/// Dense Q4_K (kBits 4) or Q5_K (kBits 5) projection for 1-8 decode rows,
/// as DenseQ6KVecKernel: one wave per row, each lane decodes eight weights
/// of one 64-element chunk once and applies them to every row.
template<unsigned kBits, unsigned kTokens>
__launch_bounds__(256) __global__
    void DenseQ45KVecKernel(const std::uint8_t* __restrict__ w,
                            const float* __restrict__ x, float* __restrict__ y,
                            std::uint32_t rows, std::uint32_t k) {
  constexpr std::uint32_t kBlockBytes = kBits == 4 ? 144 : 176;
  constexpr std::uint32_t kQs = kBits == 4 ? 16 : 48;  // qs offset
  const std::uint32_t lane = threadIdx.x % warpSize;
  const std::uint32_t row =
      blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (row >= rows)
    return;
  const std::uint32_t blocks = k / 256;
  const std::uint8_t* wrow =
      w + static_cast<std::size_t>(row) * blocks * kBlockBytes;
  float acc[kTokens] = {};
  const std::uint32_t chunk = lane / 8;     // 64-element chunk
  const std::uint32_t l0 = (lane % 8) * 4;  // first of four positions
  for (std::uint32_t sb = 0; sb < blocks; ++sb) {
    const std::uint8_t* b = wrow + static_cast<std::size_t>(sb) * kBlockBytes;
    const float d = __half2float(*reinterpret_cast<const __half*>(b));
    const float dmin = __half2float(*reinterpret_cast<const __half*>(b + 2));
    std::uint32_t sc0, m0, sc1, m1;
    KScaleMin(b + 4, 2 * chunk, sc0, m0);
    KScaleMin(b + 4, 2 * chunk + 1, sc1, m1);
    const float d0 = d * sc0, mm0 = dmin * m0, d1 = d * sc1, mm1 = dmin * m1;
    const std::uint32_t qs =
        *reinterpret_cast<const std::uint32_t*>(b + kQs + 32 * chunk + l0);
    std::uint32_t qh = 0;
    if constexpr (kBits == 5)
      qh = *reinterpret_cast<const std::uint32_t*>(b + 16 + l0);
    float wv[8];
#pragma unroll
    for (std::uint32_t j = 0; j < 4; ++j) {
      const std::uint32_t byte = (qs >> (8 * j)) & 0xFFu;
      std::uint32_t lo = byte & 0xFu, hi = byte >> 4;
      if constexpr (kBits == 5) {
        const std::uint32_t h = (qh >> (8 * j)) & 0xFFu;
        lo |= ((h >> (2 * chunk)) & 1u) << 4;
        hi |= ((h >> (2 * chunk + 1)) & 1u) << 4;
      }
      wv[j] = d0 * static_cast<float>(lo) - mm0;
      wv[4 + j] = d1 * static_cast<float>(hi) - mm1;
    }
    const std::uint32_t base = sb * 256 + 64 * chunk + l0;
#pragma unroll
    for (unsigned t = 0; t < kTokens; ++t) {
      const float* xr = x + static_cast<std::size_t>(t) * k + base;
      const float4 x0 = *reinterpret_cast<const float4*>(xr);
      const float4 x1 = *reinterpret_cast<const float4*>(xr + 32);
      float a = acc[t];
      a = fmaf(wv[0], x0.x, a);
      a = fmaf(wv[1], x0.y, a);
      a = fmaf(wv[2], x0.z, a);
      a = fmaf(wv[3], x0.w, a);
      a = fmaf(wv[4], x1.x, a);
      a = fmaf(wv[5], x1.y, a);
      a = fmaf(wv[6], x1.z, a);
      a = fmaf(wv[7], x1.w, a);
      acc[t] = a;
    }
  }
#pragma unroll
  for (unsigned t = 0; t < kTokens; ++t) {
    const float total = WaveSum(acc[t]);
    if (lane == 0)
      y[static_cast<std::size_t>(t) * rows + row] = total;
  }
}

/// res[row] += delta[row], then out[row] = rmsnorm(res[row]) * gamma. One
/// block per row; a null gamma leaves only the residual update.
__global__ void AddRmsNormKernel(float* res, const float* delta,
                                 const float* gamma, float* out,
                                 std::uint32_t dim, float eps) {
  __shared__ float shared[32];
  const std::size_t row = blockIdx.x;
  float* r = res + row * dim;
  const float* dl = delta + row * dim;
  float ss = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    const float v = r[i] + dl[i];
    r[i] = v;
    ss += v * v;
  }
  if (gamma == nullptr)
    return;
  ss = BlockSum(ss, shared);
  const float scale = rsqrtf(ss / static_cast<float>(dim) + eps);
  float* dst = out + row * dim;
  for (std::uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    dst[i] = r[i] * scale * gamma[i];
  }
}

/// Four adjacent float lanes from F32 or F16 rows.
__device__ __forceinline__ float4 Load4(const float* p) {
  return *reinterpret_cast<const float4*>(p);
}
__device__ __forceinline__ float4 Load4(const __half* p) {
  const auto packed = *reinterpret_cast<const __half2*>(p);
  const auto packed_hi = *reinterpret_cast<const __half2*>(p + 2);
  const float2 lo = __half22float2(packed);
  const float2 hi = __half22float2(packed_hi);
  return float4{lo.x, lo.y, hi.x, hi.y};
}

/// The MoE epilogue (weighted expert rows plus the gated shared expert)
/// added into res, then the RMS norm, in one pass per row.
/// Eight adjacent float lanes from F32 or F16 rows (16- or 32-byte loads).
__device__ __forceinline__ void Load8(const float* p, float (&v)[8]) {
  const float4 a = *reinterpret_cast<const float4*>(p);
  const float4 b = *reinterpret_cast<const float4*>(p + 4);
  v[0] = a.x;
  v[1] = a.y;
  v[2] = a.z;
  v[3] = a.w;
  v[4] = b.x;
  v[5] = b.y;
  v[6] = b.z;
  v[7] = b.w;
}
__device__ __forceinline__ void Load8(const __half* p, float (&v)[8]) {
  const uint4 raw = *reinterpret_cast<const uint4*>(p);
  const auto* h = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = __half22float2(h[i]);
    v[2 * i] = f.x;
    v[2 * i + 1] = f.y;
  }
}

/// One block per row, eight lanes per thread (dim % 8 == 0): every expert
/// slot's load is issued before the weighted sum consumes them.
template<typename ExpertT>
__global__ void MoeAddRmsNormKernel(const ExpertT* expert_out,
                                    const float* weights,
                                    const float* shared_out, const float* gate,
                                    std::uint32_t gate_stride, float* res,
                                    const float* gamma, float* out,
                                    std::uint32_t k, std::uint32_t dim,
                                    float eps) {
  constexpr std::uint32_t kMaxSlots = 8;
  __shared__ float shared[32];
  const std::size_t t = blockIdx.x;
  float* r = res + t * dim;
  const float g = SigmoidF(gate[t * gate_stride]);
  const ExpertT* rows = expert_out + t * k * dim;
  float ss = 0.0f;
  for (std::uint32_t i = threadIdx.x * 8; i < dim; i += blockDim.x * 8) {
    float acc[8] = {};
    for (std::uint32_t s0 = 0; s0 < k; s0 += kMaxSlots) {
      float v[kMaxSlots][8];
      float w[kMaxSlots];
#pragma unroll
      for (std::uint32_t s = 0; s < kMaxSlots; ++s) {
        if (s0 + s < k) {
          Load8(rows + static_cast<std::size_t>(s0 + s) * dim + i, v[s]);
          w[s] = weights[t * k + s0 + s];
        }
      }
#pragma unroll
      for (std::uint32_t s = 0; s < kMaxSlots; ++s) {
        if (s0 + s < k) {
#pragma unroll
          for (int x = 0; x < 8; ++x)
            acc[x] += w[s] * v[s][x];
        }
      }
    }
    float sh[8], rv[8];
    Load8(shared_out + t * dim + i, sh);
    Load8(r + i, rv);
#pragma unroll
    for (int x = 0; x < 8; ++x) {
      rv[x] += acc[x] + g * sh[x];
      ss += rv[x] * rv[x];
    }
    *reinterpret_cast<float4*>(r + i) = make_float4(rv[0], rv[1], rv[2], rv[3]);
    *reinterpret_cast<float4*>(r + i + 4) =
        make_float4(rv[4], rv[5], rv[6], rv[7]);
  }
  if (gamma == nullptr)
    return;
  ss = BlockSum(ss, shared);
  const float scale = rsqrtf(ss / static_cast<float>(dim) + eps);
  float* dst = out + t * dim;
  for (std::uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    dst[i] = r[i] * scale * gamma[i];
  }
}

/// Four adjacent hidden lanes per thread. This keeps the stream arithmetic in
/// registers while cutting the number of blocks and inject partials by almost
/// four for the model's 2,560-wide, four-stream rows.

/// The W8A8 GEMM's activation tiles: 16 tokens x one 32-wide K block,
/// codes in fragment order (two 256-byte halves of 16 tokens x 16 codes)
/// followed by the sixteen per-token scales.
constexpr std::size_t kQ8ActTileTokens = 16;
constexpr std::size_t kQ8ActTileBytes = 576;
constexpr std::size_t kQ8ActScaleOffset = 512;
__device__ __forceinline__ std::int8_t* Q8ActTile(void* base,
                                                  std::size_t num_blocks,
                                                  std::size_t tt,
                                                  std::size_t kb) {
  return static_cast<std::int8_t*>(base) +
         (((tt * num_blocks) + kb) * kQ8ActTileBytes);
}

/// out = silu(gate) * up as F16, four elements per thread.
__global__ void SwigluHalfKernel(const float* __restrict__ gate,
                                 const float* __restrict__ up,
                                 __half* __restrict__ out, std::size_t count) {
  const std::size_t i =
      (blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x) * 4;
  if (i + 4 > count) {
    for (std::size_t j = i; j < count; ++j) {
      out[j] = __float2half(SiluF(gate[j]) * up[j]);
    }
    return;
  }
  const float4 g = *reinterpret_cast<const float4*>(gate + i);
  const float4 u = *reinterpret_cast<const float4*>(up + i);
  const __half2 lo = __floats2half2_rn(SiluF(g.x) * u.x, SiluF(g.y) * u.y);
  const __half2 hi = __floats2half2_rn(SiluF(g.z) * u.z, SiluF(g.w) * u.w);
  *reinterpret_cast<__half2*>(out + i) = lo;
  *reinterpret_cast<__half2*>(out + i + 2) = hi;
}

/// silu(gate) * up over rows of `k` elements, written only as the W8A8
/// tiled Q8 layout: four elements per lane, eight lanes per K block.
__global__ void SwigluQ8Kernel(const float* gate, const float* up, void* out_q8,
                               std::size_t n_rows, std::size_t k) {
  const std::size_t chunk =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  const std::size_t chunks_per_row = k / 4;
  const std::size_t row = chunk / chunks_per_row;
  const std::size_t i = (chunk % chunks_per_row) * 4;
  // Rows past the batch still join the wave reduction below.
  const bool live = row < n_rows;
  float4 v{0.0F, 0.0F, 0.0F, 0.0F};
  if (live) {
    const std::size_t idx = (row * k) + i;
    const float4 g = *reinterpret_cast<const float4*>(gate + idx);
    const float4 u = *reinterpret_cast<const float4*>(up + idx);
    v = float4{SiluF(g.x) * u.x, SiluF(g.y) * u.y, SiluF(g.z) * u.z,
               SiluF(g.w) * u.w};
  }
  float max_abs =
      fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
  for (int off = 4; off > 0; off >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_xor(max_abs, off));
  }
  if (!live) {
    return;
  }
  const float d = max_abs / 127.0F;
  const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
  const auto q0 = static_cast<std::uint32_t>(
      static_cast<std::uint8_t>(static_cast<std::int8_t>(roundf(v.x * id))));
  const auto q1 = static_cast<std::uint32_t>(
      static_cast<std::uint8_t>(static_cast<std::int8_t>(roundf(v.y * id))));
  const auto q2 = static_cast<std::uint32_t>(
      static_cast<std::uint8_t>(static_cast<std::int8_t>(roundf(v.z * id))));
  const auto q3 = static_cast<std::uint32_t>(
      static_cast<std::uint8_t>(static_cast<std::int8_t>(roundf(v.w * id))));
  const std::size_t tl = row % kQ8ActTileTokens;
  const std::uint32_t pos = static_cast<std::uint32_t>(i % 32);
  std::int8_t* tile = Q8ActTile(out_q8, k / 32, row / kQ8ActTileTokens, i / 32);
  *reinterpret_cast<std::uint32_t*>(tile + ((pos >> 4u) * 256) + (tl * 16) +
                                    (pos & 15u)) =
      q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
  if (pos == 0) {
    *reinterpret_cast<float*>(tile + kQ8ActScaleOffset + (tl * sizeof(float))) =
        d;
  }
}

__global__ void SwigluKernel(float* gate, const float* up, std::size_t count) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < count) {
    gate[i] = SiluF(gate[i]) * up[i];
  }
}

__global__ void SigmoidMulKernel(float* x, const float* g, std::size_t count) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < count) {
    x[i] *= SigmoidF(g[i]);
  }
}

template<typename T>
__global__ void NarrowKernel(const float* x, T* out, std::size_t count) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < count) {
    out[i] = T(x[i]);
  }
}

/// One wave per weight row, kSmallGemmRows rows per block: the row is read
/// once (four consecutive elements per lane per step) while up to eight
/// tokens accumulate in registers, then a wave reduction per token.
constexpr unsigned kSmallGemmRows = 4;
template<WeightType type, unsigned tokens, bool grouped = false>
__global__ void SmallGemmKernel(const void* w, const float* x, float* out,
                                std::uint32_t m, std::uint32_t k) {
  if constexpr (grouped) {
    x += std::size_t{blockIdx.y} * tokens * k;
    out += std::size_t{blockIdx.y} * tokens * m;
  }
  const std::uint32_t lane = threadIdx.x % warpSize;
  const std::uint32_t row =
      blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (row >= m) {
    return;
  }
  const auto* wrow =
      static_cast<const std::uint8_t*>(w) + RowBytes(type, k) * row;
  float acc[tokens] = {};
  if constexpr (type == WeightType::kF16) {
    // Eight halves per 16-byte load: the routers and alpha/beta rows are
    // short, so the loop is latency-bound and halving its trips matters.
    if (k % (warpSize * 8) == 0) {
      const auto* w8 = reinterpret_cast<const uint4*>(wrow);
      for (std::uint32_t i0 = lane * 8; i0 < k; i0 += warpSize * 8) {
        const uint4 packed = w8[i0 / 8];
        const auto* h2 = reinterpret_cast<const __half2*>(&packed);
        float wv[8];
#pragma unroll
        for (unsigned r = 0; r < 4; ++r) {
          const float2 f = __half22float2(h2[r]);
          wv[2 * r] = f.x;
          wv[2 * r + 1] = f.y;
        }
#pragma unroll
        for (unsigned j = 0; j < tokens; ++j) {
          const auto* xr = reinterpret_cast<const float4*>(
              x + static_cast<std::size_t>(j) * k + i0);
          const float4 a = xr[0], b = xr[1];
          float dot = 0.0f;
          dot += wv[0] * a.x;
          dot += wv[1] * a.y;
          dot += wv[2] * a.z;
          dot += wv[3] * a.w;
          dot += wv[4] * b.x;
          dot += wv[5] * b.y;
          dot += wv[6] * b.z;
          dot += wv[7] * b.w;
          acc[j] += dot;
        }
      }
#pragma unroll
      for (unsigned j = 0; j < tokens; ++j) {
        const float total = WaveSum(acc[j]);
        if (lane == 0) {
          out[static_cast<std::size_t>(j) * m + row] = total;
        }
      }
      return;
    }
  }
  for (std::uint32_t i0 = lane * 4; i0 < k; i0 += warpSize * 4) {
    float wv[4];
#pragma unroll
    for (unsigned r = 0; r < 4; ++r) {
      wv[r] = i0 + r < k ? RowElement(wrow, type, i0 + r) : 0.0f;
    }
#pragma unroll
    for (unsigned j = 0; j < tokens; ++j) {
      const float* xr = x + static_cast<std::size_t>(j) * k + i0;
      float dot = 0.0f;
#pragma unroll
      for (unsigned r = 0; r < 4; ++r) {
        dot += wv[r] * (i0 + r < k ? xr[r] : 0.0f);
      }
      acc[j] += dot;
    }
  }
#pragma unroll
  for (unsigned j = 0; j < tokens; ++j) {
    const float total = WaveSum(acc[j]);
    if (lane == 0) {
      out[static_cast<std::size_t>(j) * m + row] = total;
    }
  }
}

/// F16 rows of a short matrix (the routers, alpha/beta): one block of four
/// waves per row, each wave summing a quarter of k with 16-byte loads, the
/// quarters added in wave order. Fills the device where one wave per row
/// leaves it latency-bound. blockIdx.y selects a group of `tokens` rows.
template<unsigned tokens>
__launch_bounds__(128) __global__
    void SmallGemmSplitF16Kernel(const __half* w, const float* x, float* out,
                                 std::uint32_t m, std::uint32_t k) {
  __shared__ float part[4][tokens];
  const std::uint32_t lane = threadIdx.x % warpSize;
  const std::uint32_t wave = threadIdx.x / warpSize;
  const std::uint32_t row = blockIdx.x;
  x += static_cast<std::size_t>(blockIdx.y) * tokens * k;
  out += static_cast<std::size_t>(blockIdx.y) * tokens * m;
  const std::uint32_t quarter = k / 4;
  const auto* w8 = reinterpret_cast<const uint4*>(
      w + static_cast<std::size_t>(row) * k + wave * quarter);
  float acc[tokens] = {};
  for (std::uint32_t i0 = lane * 8; i0 < quarter; i0 += warpSize * 8) {
    const uint4 packed = w8[i0 / 8];
    const auto* h2 = reinterpret_cast<const __half2*>(&packed);
    float wv[8];
#pragma unroll
    for (unsigned r = 0; r < 4; ++r) {
      const float2 f = __half22float2(h2[r]);
      wv[2 * r] = f.x;
      wv[2 * r + 1] = f.y;
    }
#pragma unroll
    for (unsigned j = 0; j < tokens; ++j) {
      const auto* xr = reinterpret_cast<const float4*>(
          x + static_cast<std::size_t>(j) * k + wave * quarter + i0);
      const float4 a = xr[0], b = xr[1];
      // Rounded intrinsics: fast-math may otherwise regroup this sum
      // differently in each token-count instantiation, and a row's result
      // must not depend on how many rows share the launch.
      float dot = __fmul_rn(wv[0], a.x);
      dot = __fmaf_rn(wv[1], a.y, dot);
      dot = __fmaf_rn(wv[2], a.z, dot);
      dot = __fmaf_rn(wv[3], a.w, dot);
      dot = __fmaf_rn(wv[4], b.x, dot);
      dot = __fmaf_rn(wv[5], b.y, dot);
      dot = __fmaf_rn(wv[6], b.z, dot);
      dot = __fmaf_rn(wv[7], b.w, dot);
      acc[j] = __fadd_rn(acc[j], dot);
    }
  }
#pragma unroll
  for (unsigned j = 0; j < tokens; ++j) {
    const float total = WaveSum(acc[j]);
    if (lane == 0)
      part[wave][j] = total;
  }
  __syncthreads();
  if (threadIdx.x < tokens) {
    const unsigned j = threadIdx.x;
    out[static_cast<std::size_t>(j) * m + row] = __fadd_rn(
        __fadd_rn(__fadd_rn(part[0][j], part[1][j]), part[2][j]), part[3][j]);
  }
}

/// New history row j is row (n_tokens + j) of [history ; in].
__global__ void HistoryShiftKernel(const float* in, std::uint32_t in_stride,
                                   const float* history, float* scratch,
                                   std::uint32_t n_tokens,
                                   std::uint32_t channels, std::uint32_t hist) {
  const std::size_t idx =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<std::size_t>(hist) * channels) {
    return;
  }
  const std::uint32_t j = idx / channels;
  const std::uint32_t c = idx % channels;
  const std::uint32_t src = n_tokens + j;
  scratch[idx] = src < hist
                     ? history[static_cast<std::size_t>(src) * channels + c]
                     : in[static_cast<std::size_t>(src - hist) * in_stride + c];
}

/// L2-normalizes the convolved q and k of one (token, key head) into the
/// packed [t][kh][d] buffers the recurrence streams from, so the serial loop
/// carries no block-wide reductions. One wave per (token, head) row.
constexpr unsigned kGdnDim = 128;
template<bool kBatch>
__global__ void GdnPrepKernel(const float* conv_out, float* qn, float* kn,
                              std::uint32_t n_rows, std::uint32_t k_heads,
                              std::uint32_t channels, float eps,
                              const GdnBatchItem* batch, std::uint32_t active) {
  if constexpr (kBatch) {
    if ((active & (1U << blockIdx.z)) == 0)
      return;
    const auto& item = batch[blockIdx.z];
    conv_out = item.conv_scratch;
    qn = item.qn;
    kn = item.kn;
    n_rows = item.n_tokens * k_heads;
  }
  constexpr std::uint32_t d = kGdnDim;
  const std::uint32_t lane = threadIdx.x % warpSize;
  const std::uint32_t row =
      blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (row >= n_rows) {
    return;
  }
  const std::uint32_t t = row / k_heads;
  const std::uint32_t kh = row % k_heads;
  const float* conv = conv_out + static_cast<std::size_t>(t) * channels;
  const std::uint32_t per_lane = d / warpSize;
  float qv[d / 32];
  float kv[d / 32];
  float qs = 0.0f;
  float ks = 0.0f;
#pragma unroll
  for (std::uint32_t r = 0; r < per_lane; ++r) {
    const std::uint32_t i = r * warpSize + lane;
    qv[r] = conv[kh * d + i];
    kv[r] = conv[k_heads * d + kh * d + i];
    qs += qv[r] * qv[r];
    ks += kv[r] * kv[r];
  }
  qs = rsqrtf(WaveSum(qs) + eps);
  ks = rsqrtf(WaveSum(ks) + eps);
  float* q_out = qn + static_cast<std::size_t>(row) * d;
  float* k_out = kn + static_cast<std::size_t>(row) * d;
#pragma unroll
  for (std::uint32_t r = 0; r < per_lane; ++r) {
    const std::uint32_t i = r * warpSize + lane;
    q_out[i] = qv[r] * qs;
    k_out[i] = kv[r] * ks;
  }
}

__global__ void GdnPrepKqKernel(const float* conv_out, float* scales,
                                std::uint32_t n_tokens, std::uint32_t k_heads,
                                std::uint32_t channels, float eps) {
  constexpr std::uint32_t d = kGdnDim;
  const std::uint32_t lane = threadIdx.x;
  const std::uint32_t kh = blockIdx.x;
  const std::uint32_t t = blockIdx.y;
  if (t >= n_tokens || kh >= k_heads) {
    return;
  }
  const float* row = conv_out + static_cast<std::size_t>(t) * channels;
  const auto* q = reinterpret_cast<const float4*>(row + kh * d);
  const auto* k = reinterpret_cast<const float4*>(row + (k_heads + kh) * d);
  const float4 q4 = q[lane];
  const float4 k4 = k[lane];
  float qs = q4.x * q4.x + q4.y * q4.y + q4.z * q4.z + q4.w * q4.w;
  float ks = k4.x * k4.x + k4.y * k4.y + k4.z * k4.z + k4.w * k4.w;
  float kq = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
#pragma unroll
  for (unsigned offset = 16; offset > 0; offset >>= 1) {
    qs += __shfl_xor(qs, offset);
    ks += __shfl_xor(ks, offset);
    kq += __shfl_xor(kq, offset);
  }
  if (lane == 0) {
    float* dst = scales + (static_cast<std::size_t>(t) * k_heads + kh) * 3;
    dst[0] = rsqrtf(ks + eps);
    dst[1] = rsqrtf(qs + eps) * rsqrtf(static_cast<float>(d));
    dst[2] = kq;
  }
}

__global__ void GdnPrepAbKernel(const float* alpha_beta, const float* a,
                                const float* dt, float* ab, std::size_t count,
                                std::uint32_t v_heads) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i >= count) {
    return;
  }
  const std::uint32_t h = i % v_heads;
  const std::size_t t = i / v_heads;
  const float alpha = alpha_beta[t * 2 * v_heads + h];
  const float beta = alpha_beta[t * 2 * v_heads + v_heads + h];
  ab[i * 2] = __expf(a[h] * SoftplusF(alpha + dt[h]));
  ab[i * 2 + 1] = SigmoidF(beta);
}

template<int kMask>
__device__ __forceinline__ float GdnXorAddDpp(float x) {
  const int y = __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, x),
                                            0x160 | kMask, 0xF, 0xF, false);
  return x + __builtin_bit_cast(float, y);
}

/// Sums u and p over a row's kLanes lanes (2 to 16, within one DPP row).
template<int kLanes>
__device__ __forceinline__ void GdnRowReduce(float& u, float& p) {
  u = GdnXorAddDpp<1>(u);
  p = GdnXorAddDpp<1>(p);
  u = GdnXorAddDpp<2>(u);
  p = GdnXorAddDpp<2>(p);
  if constexpr (kLanes > 4) {
    u = GdnXorAddDpp<4>(u);
    p = GdnXorAddDpp<4>(p);
  }
  if constexpr (kLanes > 8) {
    u = GdnXorAddDpp<8>(u);
    p = GdnXorAddDpp<8>(p);
  }
}

/// Qwen row-split recurrence specialized to the model's 128x128 state.
/// d / kKeysPerLane lanes own one row and the rest of the wave holds more
/// rows; d / rows-per-block blocks cover a head.
/// The token loop is a short dependency chain that consumed its operands
/// straight from global memory (the conv output is far larger than the
/// caches), so every step paid a DRAM latency; the block now stages four
/// tokens at a time through LDS and keeps three such windows of loads in
/// flight, one window ahead of the one being committed.
constexpr int kGdnWindow = 4;
constexpr int kGdnWindowsAhead = 3;
template<int kKeysPerLane>
__launch_bounds__(256) __global__
    void GdnRowSplitKernel(const float* conv_out, const float* scales,
                           const float* ab, float* state, float* raw,
                           std::uint32_t n_tokens, std::uint32_t k_heads,
                           std::uint32_t v_heads) {
  constexpr int d = kGdnDim;
  constexpr int kVec = kKeysPerLane / 4;
  constexpr int kLanesPerRow = d / kKeysPerLane;
  constexpr int kRowsPerWave = 32 / kLanesPerRow;
  constexpr int kRowsPerBlock = kRowsPerWave * 8;
  constexpr int kW = kGdnWindow;
  static_assert(kW * (d / 4) * 2 == 256, "one q or k float4 per thread");
  static_assert(kW * kRowsPerBlock <= 256, "at most one v element per thread");
  static_assert(kW * 5 <= 256, "scales and decays per window");
  const int h = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int segment = lane % kLanesPerRow;
  const int row_group = lane / kLanesPerRow;
  const int row_local = (tid >> 5) * kRowsPerWave + row_group;
  const int row = blockIdx.x * kRowsPerBlock + row_local;
  const int kh = h % k_heads;
  const int vec0 = segment * kVec;
  const int channels = 2 * k_heads * d + v_heads * d;

  // Two window slots: q and k rows, the block's v rows, the three scales
  // and two decays per token.
  __shared__ float4 s_qk[2][2][kW][d / 4];
  __shared__ float s_v[2][kW][kRowsPerBlock];
  __shared__ float s_misc[2][kW][5];

  float* state_row = state + (static_cast<std::size_t>(h) * d + row) * d;
  auto* state4 = reinterpret_cast<float4*>(state_row);
  float4 s[kVec];
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    s[i] = state4[vec0 + i];
  }

  // Window loads: thread tid takes q (tid < 128) or k float4 (tid / 32 of
  // token (tid / 32) % kW ... laid out so a token's 128 floats are 32
  // consecutive threads), one v element and, for tid < 5 kW, one scale.
  const int qk_which = tid >> 7;      // 0: q, 1: k
  const int qk_tok = (tid >> 5) & 3;  // token in the window
  const int qk_vec = tid & 31;
  constexpr int kVLoads = kW * kRowsPerBlock;
  const int v_tok = tid / kRowsPerBlock;
  const int v_row = tid % kRowsPerBlock;
  const int m_tok = tid / 5;
  const int m_idx = tid % 5;
  const float* qk_src =
      conv_out + (qk_which == 0 ? kh : k_heads + kh) * d + (qk_vec * 4);
  const float* v_src =
      conv_out + 2 * k_heads * d + h * d + blockIdx.x * kRowsPerBlock + v_row;
  const float* m_src =
      m_idx < 3 ? scales + kh * 3 + m_idx : ab + h * 2 + (m_idx - 3);
  const std::size_t m_stride = m_idx < 3 ? k_heads * 3 : v_heads * 2;

  // Three windows of loads in flight, as named registers (a ring array
  // lands in scratch).
  struct Window {
    float4 qk;
    float v;
    float m;
  };
  Window r0;
  Window r1;
  Window r2;
  const auto load_window = [&](int w, Window& r) {
    const int last = static_cast<int>(n_tokens) - 1;
    const int t_qk = min((w * kW) + qk_tok, last);
    const int t_v = min((w * kW) + v_tok, last);
    r.qk = *reinterpret_cast<const float4*>(
        qk_src + static_cast<std::size_t>(t_qk) * channels);
    if (tid < kVLoads)
      r.v = v_src[static_cast<std::size_t>(t_v) * channels];
    if (tid < 5 * kW) {
      const int t_m = min((w * kW) + m_tok, last);
      r.m = m_src[static_cast<std::size_t>(t_m) * m_stride];
    }
  };
  // A token's 32 float4 are stored [i][segment] so the four segments of
  // a row's lanes read adjacent 16-byte chunks (row-major, they were 128
  // bytes apart: a four-way bank conflict on every fragment).
  const int qk_slot = ((qk_vec % kVec) * kLanesPerRow) + (qk_vec / kVec);
  const auto commit_window = [&](const Window& r, int lds) {
    s_qk[lds][qk_which][qk_tok][qk_slot] = r.qk;
    if (tid < kVLoads)
      s_v[lds][v_tok][v_row] = r.v;
    if (tid < 5 * kW) {
      s_misc[lds][m_tok][m_idx] = r.m;
    }
  };

  const int n_windows = (static_cast<int>(n_tokens) + kW - 1) / kW;
  load_window(0, r0);
  if (n_windows > 1) {
    load_window(1, r1);
  }
  if (n_windows > 2) {
    load_window(2, r2);
  }
  commit_window(r0, 0);
  __syncthreads();
  float* out_base = raw + h * d;
  // `cur` held window w (committed already) and takes window w + 3;
  // `next` holds window w + 1, committed after this window's tokens.
  const auto window = [&](int w, Window& cur, const Window& next) {
    if (w >= n_windows) {
      return;
    }
    const int lds = w & 1;
    if (w + kGdnWindowsAhead < n_windows) {
      load_window(w + kGdnWindowsAhead, cur);
    }
    const int t_end = min(kW, static_cast<int>(n_tokens) - (w * kW));
    for (int tl = 0; tl < t_end; ++tl) {
      const float4* q4 = s_qk[lds][0][tl];
      const float4* k4 = s_qk[lds][1][tl];
      const float decay = s_misc[lds][tl][3];
      float u = 0.0F;
      float p = 0.0F;
      float4 kc[kVec];
#pragma unroll
      for (int i = 0; i < kVec; ++i) {
        const float4 qv = q4[(i * kLanesPerRow) + segment];
        kc[i] = k4[(i * kLanesPerRow) + segment];
        s[i].x *= decay;
        s[i].y *= decay;
        s[i].z *= decay;
        s[i].w *= decay;
        u += s[i].x * kc[i].x + s[i].y * kc[i].y + s[i].z * kc[i].z +
             s[i].w * kc[i].w;
        p += s[i].x * qv.x + s[i].y * qv.y + s[i].z * qv.z + s[i].w * qv.w;
      }
      GdnRowReduce<kLanesPerRow>(u, p);
      const float inv_k = s_misc[lds][tl][0];
      const float q_scale = s_misc[lds][tl][1];
      const float delta =
          (s_v[lds][tl][row_local] - u * inv_k) * s_misc[lds][tl][4];
      if (segment == 0) {
        out_base[row] =
            p * q_scale + delta * inv_k * q_scale * s_misc[lds][tl][2];
      }
      const float update = delta * inv_k;
#pragma unroll
      for (int i = 0; i < kVec; ++i) {
        s[i].x += update * kc[i].x;
        s[i].y += update * kc[i].y;
        s[i].z += update * kc[i].z;
        s[i].w += update * kc[i].w;
      }
      out_base += v_heads * d;
    }
    // The other slot was last read one window ago (before the previous
    // barrier), so window w + 1 goes in without a second barrier.
    if (w + 1 < n_windows) {
      commit_window(next, lds ^ 1);
    }
    __syncthreads();
  };
  for (int w = 0; w < n_windows; w += kGdnWindowsAhead) {
    window(w, r0, r1);
    window(w + 1, r1, r2);
    window(w + 2, r2, r0);
  }
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    state4[vec0 + i] = s[i];
  }
}

/// grid (value heads, row groups): each block owns kGdnRowsPerBlock state
/// rows of one head, kGdnLanes threads per row, each lane a contiguous
/// slice of the key dimension. Rows of the delta rule are independent, so
/// splitting a head over blocks only re-reads its q/k. The token loop is
/// serial; every reduction stays inside a lane group, so the loop runs
/// barrier-free. The raw attention rows go out unnormalized;
/// GdnEpilogueKernel finishes them.
constexpr unsigned kGdnLanes = 4;
constexpr unsigned kGdnRowsPerBlock = 32;

// The original vector kernel rounds error*key and then beta*correction for
// the last element of each lane, fusing the state decay into that result.
// HIP's FP intrinsics alone still permit reassociation of this three-product
// expression under fast-math. Keep its actual instruction sequence explicit.
__device__ __forceinline__ float GdnLastUpdate(float state, float decay,
                                               float error, float key,
                                               float beta) {
  float value;
  asm volatile(
      "v_mul_f32 %0, %1, %2\n\t"
      "v_mul_f32 %0, %0, %3\n\t"
      "v_fma_f32 %0, %4, %5, %0"
      : "=&v"(value)
      : "v"(error), "v"(key), "v"(beta), "v"(state), "v"(decay));
  return value;
}

template<bool kBatch>
__global__ void GdnKernel(const float* conv_out, const float* qn,
                          const float* kn, const float* alpha_beta,
                          const float* a, const float* dt, float* state,
                          float* raw, RollbackRows snapshots,
                          std::uint32_t n_tokens, std::uint32_t k_heads,
                          std::uint32_t v_heads, const GdnBatchItem* batch,
                          std::uint32_t active) {
  if constexpr (kBatch) {
    if ((active & (1U << blockIdx.z)) == 0)
      return;
    const auto& item = batch[blockIdx.z];
    conv_out = item.conv_scratch;
    qn = item.qn;
    kn = item.kn;
    alpha_beta = item.alpha_beta;
    state = item.state;
    raw = item.raw;
    n_tokens = item.n_tokens;
  }
  constexpr std::uint32_t d = kGdnDim;
  constexpr std::uint32_t slice = d / kGdnLanes;
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t kh = h % k_heads;
  const std::uint32_t j =
      blockIdx.y * kGdnRowsPerBlock + threadIdx.x / kGdnLanes;
  const std::uint32_t lane = threadIdx.x % kGdnLanes;
  const std::uint32_t i0 = lane * slice;
  const std::uint32_t channels = 2 * k_heads * d + v_heads * d;
  const float a_h = a[h];
  const float dt_h = dt[h];
  float* S = state + static_cast<std::size_t>(h) * d * d + j * d + i0;
  float row[slice];
#pragma unroll
  for (std::uint32_t i = 0; i < slice; ++i) {
    row[i] = S[i];
  }
  const float q_scale = rsqrtf(static_cast<float>(d));
  for (std::uint32_t t = 0; t < n_tokens; ++t) {
    const float* q = qn + (static_cast<std::size_t>(t) * k_heads + kh) * d + i0;
    const float* k = kn + (static_cast<std::size_t>(t) * k_heads + kh) * d + i0;
    const float vv = conv_out[static_cast<std::size_t>(t) * channels +
                              2 * k_heads * d + h * d + j];
    const float alpha = alpha_beta[t * 2 * v_heads + h];
    const float beta = alpha_beta[t * 2 * v_heads + v_heads + h];
    const float decay = __expf(a_h * SoftplusF(alpha + dt_h));
    const float b = SigmoidF(beta);
    float kr[slice];
    float u = 0.0f;
    const float last = row[slice - 1];
#pragma unroll
    for (std::uint32_t i = 0; i < slice; ++i) {
      kr[i] = k[i];
      row[i] *= decay;
      u += row[i] * kr[i];
    }
#pragma unroll
    for (unsigned off = kGdnLanes / 2; off > 0; off >>= 1) {
      u += __shfl_xor(u, off, kGdnLanes);
    }
    const float error = vv - u;
    float acc = 0.0f;
#pragma unroll
    for (std::uint32_t i = 0; i < slice; ++i) {
      // Preserve the original kernel's contraction: round error*key, then
      // fuse beta*correction into the decayed state. Materializing error*beta
      // for rollback would change this order under fast-math.
      const float correction = __fmul_rn(error, kr[i]);
      row[i] = i + 1 == slice ? GdnLastUpdate(last, decay, error, kr[i], b)
                              : __fmaf_rn(b, correction, row[i]);
      acc += row[i] * q[i];
    }
#pragma unroll
    for (unsigned off = kGdnLanes / 2; off > 0; off >>= 1) {
      acc += __shfl_xor(acc, off, kGdnLanes);
    }
    if (lane == 0) {
      raw[static_cast<std::size_t>(t) * v_heads * d + h * d + j] =
          acc * q_scale;
    }
    if ((kBatch ? batch[blockIdx.z].state_snapshots.rows[0]
                : snapshots.rows[0]) != nullptr &&
        t + 1 < n_tokens) {
      float* snap = kBatch ? batch[blockIdx.z].state_snapshots.rows[t]
                           : snapshots.rows[t];
      if (t == 0) {
        snap += static_cast<std::size_t>(h) * d * d + j * d + i0;
#pragma unroll
        for (std::uint32_t i = 0; i < slice; ++i)
          snap[i] = row[i];
      } else {
        if (j == 0 && h < k_heads) {
#pragma unroll
          for (std::uint32_t i = 0; i < slice; ++i)
            snap[h * d + i0 + i] = kr[i];
        }
        if (lane == 0) {
          if (j == 0) {
            snap[k_heads * d + h] = decay;
            snap[k_heads * d + v_heads + h] = b;
          }
          snap[k_heads * d + 2 * v_heads + h * d + j] = error;
        }
      }
    }
  }
#pragma unroll
  for (std::uint32_t i = 0; i < slice; ++i) {
    S[i] = row[i];
  }
}

__global__ void RestoreGdnStateKernel(float* state, RollbackRows snapshots,
                                      std::uint32_t keep, std::uint32_t k_heads,
                                      std::uint32_t v_heads) {
  constexpr std::uint32_t d = kGdnDim;
  const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (i >= std::size_t{v_heads} * d * d)
    return;
  const auto h = i / (d * d);
  const auto j = (i / d) % d;
  const auto k = i % d;
  float value = snapshots.rows[0][i];
  for (std::uint32_t t = 1; t < keep; ++t) {
    const auto* update = snapshots.rows[t];
    const float key = update[(h % k_heads) * d + k];
    const float decay = update[k_heads * d + h];
    const float beta = update[k_heads * d + v_heads + h];
    const float error = update[k_heads * d + 2 * v_heads + h * d + j];
    const float correction = __fmul_rn(error, key);
    value = (k + 1) % (d / kGdnLanes) == 0
                ? GdnLastUpdate(value, decay, error, key, beta)
                : __fmaf_rn(beta, correction, __fmul_rn(value, decay));
  }
  state[i] = value;
}

/// Per-head RMSNorm of the raw attention rows and the SiLU output gate,
/// one wave per (token, head) row.
/// A non-null `out_q8` receives the row quantized into the tiled Q8 layout
/// of the ssm_out projection (K = v_heads * d) in place of the F32 row:
/// each wave-wide slice of 32 lanes is one K block.
template<bool kBatch>
__global__ void GdnEpilogueKernel(const float* raw, const float* z,
                                  std::uint32_t z_stride, const float* norm_w,
                                  float* out, void* out_q8, __half* out_half,
                                  std::uint32_t n_rows, std::uint32_t v_heads,
                                  float eps, const GdnBatchItem* batch,
                                  std::uint32_t active) {
  if constexpr (kBatch) {
    if ((active & (1U << blockIdx.z)) == 0)
      return;
    const auto& item = batch[blockIdx.z];
    raw = item.raw;
    z = item.z;
    out = item.out;
    n_rows = item.n_tokens * v_heads;
  }
  constexpr std::uint32_t d = kGdnDim;
  const std::uint32_t lane = threadIdx.x % warpSize;
  const std::size_t row =
      blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (row >= n_rows) {
    return;
  }
  // z rows are [t][v_heads*d] with a caller-side row stride.
  const float* zrow = z + (row / v_heads) * z_stride + (row % v_heads) * d;
  const std::uint32_t per_lane = d / warpSize;
  const float* src = raw + row * d;
  float v[d / 32];
  float ss = 0.0f;
#pragma unroll
  for (std::uint32_t r = 0; r < per_lane; ++r) {
    v[r] = src[r * warpSize + lane];
    ss += v[r] * v[r];
  }
  const float scale = rsqrtf(WaveSum(ss) / static_cast<float>(d) + eps);
  if (out_q8 == nullptr) {
#pragma unroll
    for (std::uint32_t r = 0; r < per_lane; ++r) {
      const std::uint32_t i = r * warpSize + lane;
      const float value = v[r] * scale * norm_w[i] * SiluF(zrow[i]);
      if (out_half != nullptr)
        out_half[row * d + i] = __float2half_rn(value);
      else
        out[row * d + i] = value;
    }
    return;
  }
  const std::size_t tok = row / v_heads;
  const std::size_t head = row % v_heads;
  const std::size_t num_blocks = (static_cast<std::size_t>(v_heads) * d) / 32;
  const std::size_t tl = tok % kQ8ActTileTokens;
#pragma unroll
  for (std::uint32_t r = 0; r < per_lane; ++r) {
    const std::uint32_t i = r * warpSize + lane;
    const float n = v[r] * scale * norm_w[i] * SiluF(zrow[i]);
    float max_abs = fabsf(n);
    for (int off = 16; off > 0; off >>= 1) {
      max_abs = fmaxf(max_abs, __shfl_xor(max_abs, off));
    }
    const float dq = max_abs / 127.0F;
    const float id = (dq != 0.0F) ? (1.0F / dq) : 0.0F;
    const auto q = static_cast<std::int8_t>(roundf(n * id));
    const std::size_t kb = ((head * d) + (r * warpSize)) / 32;
    std::int8_t* tile =
        Q8ActTile(out_q8, num_blocks, tok / kQ8ActTileTokens, kb);
    tile[((lane >> 4u) * 256) + (tl * 16) + (lane & 15u)] = q;
    if (lane == 0) {
      *reinterpret_cast<float*>(tile + kQ8ActScaleOffset +
                                (tl * sizeof(float))) = dq;
    }
  }
}

/// Row j of the rolling state after token t is row (t + 1 + j) of the
/// concatenation [history ; rows], for any history depth `hist`.
__global__ void RollingSnapshotKernel(
    const float* rows, std::uint32_t row_stride, const float* history,
    RollbackRows snapshots, std::uint32_t n_tokens, std::uint32_t channels,
    std::uint32_t hist) {
  const std::size_t idx =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  const std::size_t per_token = static_cast<std::size_t>(hist) * channels;
  if (idx >= per_token * n_tokens) {
    return;
  }
  const std::uint32_t t = idx / per_token;
  const std::uint32_t j = (idx % per_token) / channels;
  const std::uint32_t c = idx % channels;
  const std::uint32_t src = t + 1 + j;
  snapshots.rows[t][idx % per_token] =
      src < hist ? history[static_cast<std::size_t>(src) * channels + c]
                 : rows[static_cast<std::size_t>(src - hist) * row_stride + c];
}

/// Causal conv over the chunk with the rolling state, SiLU applied.
__global__ void SsmConvKernel(const float* qkv, std::uint32_t qkv_stride,
                              const float* w, const float* conv_state,
                              float* out, std::uint32_t n_tokens,
                              std::uint32_t channels, std::uint32_t kernel) {
  const std::size_t idx =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<std::size_t>(n_tokens) * channels) {
    return;
  }
  const std::uint32_t t = idx / channels;
  const std::uint32_t c = idx % channels;
  float acc = 0.0f;
  for (std::uint32_t k = 0; k < kernel; ++k) {
    const std::int32_t src_t = static_cast<std::int32_t>(t) -
                               static_cast<std::int32_t>(kernel - 1 - k);
    const float v =
        src_t >= 0 ? qkv[static_cast<std::size_t>(src_t) * qkv_stride + c]
                   : conv_state[static_cast<std::size_t>(kernel - 1 + src_t) *
                                    channels +
                                c];
    acc += w[static_cast<std::size_t>(c) * kernel + k] * v;
  }
  out[idx] = SiluF(acc);
}

/// SsmConvKernel with one thread per (channel, kTokensPerThread tokens) for
/// the 4-tap kernel: the taps stay in one float4 and the window slides in
/// registers, so a token costs one load and one store instead of eight
/// loads. Lanes run along channels, so every access is a contiguous row.
constexpr std::uint32_t kSsmConvTaps = 4;
constexpr std::uint32_t kSsmConvTokensPerThread = 8;
template<bool kSaveHistory, bool kBatch>
__global__ void SsmConv4Kernel(const float* qkv, std::uint32_t qkv_stride,
                               const float* w, float* conv_state, float* out,
                               std::uint32_t n_tokens, std::uint32_t channels,
                               RollbackRows snapshots,
                               const GdnBatchItem* batch,
                               std::uint32_t active) {
  if constexpr (kBatch) {
    if ((active & (1U << blockIdx.z)) == 0)
      return;
    const auto& item = batch[blockIdx.z];
    qkv = item.qkv;
    conv_state = item.conv_state;
    out = item.conv_scratch;
    n_tokens = item.n_tokens;
  }
  const std::uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels) {
    return;
  }
  const std::uint32_t t0 = blockIdx.y * kSsmConvTokensPerThread;
  const float4 taps = *reinterpret_cast<const float4*>(w + c * kSsmConvTaps);
  // window[i] holds token t0 - 3 + i.
  float window[kSsmConvTaps - 1];
#pragma unroll
  for (std::uint32_t i = 0; i < kSsmConvTaps - 1; ++i) {
    const std::int32_t src_t =
        static_cast<std::int32_t>(t0) - 3 + static_cast<std::int32_t>(i);
    window[i] =
        src_t >= 0
            ? qkv[static_cast<std::size_t>(src_t) * qkv_stride + c]
            : conv_state[static_cast<std::size_t>(kSsmConvTaps - 1 + src_t) *
                             channels +
                         c];
  }
#pragma unroll
  for (std::uint32_t i = 0; i < kSsmConvTokensPerThread; ++i) {
    const std::uint32_t t = t0 + i;
    if (t >= n_tokens) {
      break;
    }
    const float v = qkv[static_cast<std::size_t>(t) * qkv_stride + c];
    const float acc = taps.x * window[0] + taps.y * window[1] +
                      taps.z * window[2] + taps.w * v;
    out[static_cast<std::size_t>(t) * channels + c] = SiluF(acc);
    window[0] = window[1];
    window[1] = window[2];
    window[2] = v;
    if constexpr (kSaveHistory) {
      if ((kBatch ? batch[blockIdx.z].conv_snapshots.rows[0]
                  : snapshots.rows[0]) != nullptr &&
          t + 1 < n_tokens) {
        float* snapshot = kBatch ? batch[blockIdx.z].conv_snapshots.rows[t]
                                 : snapshots.rows[t];
#pragma unroll
        for (std::uint32_t j = 0; j < kSsmConvTaps - 1; ++j)
          snapshot[std::size_t{j} * channels + c] = window[j];
      }
    }
  }
  if constexpr (kSaveHistory) {
    // Only one token tile may update history: each channel then has one
    // owner, which has already consumed all three original history values.
#pragma unroll
    for (std::uint32_t j = 0; j < kSsmConvTaps - 1; ++j)
      conv_state[std::size_t{j} * channels + c] = window[j];
  }
}

__global__ void UnpackQGateKernel(const float* qg, std::uint32_t qg_stride,
                                  float* q, float* gate, float* k, float* v,
                                  std::uint32_t heads, std::uint32_t d,
                                  std::uint32_t kv_width) {
  const std::uint32_t t = blockIdx.x;
  const std::size_t width = static_cast<std::size_t>(heads) * d;
  const float* row = qg + static_cast<std::size_t>(t) * qg_stride;
  for (std::size_t i = threadIdx.x; i < width; i += blockDim.x) {
    const std::uint32_t h = i / d;
    const std::uint32_t j = i % d;
    q[t * width + i] = row[h * 2 * d + j];
    gate[t * width + i] = row[h * 2 * d + d + j];
  }
  // A stacked [q|gate ; k ; v] projection carries k and v after the heads.
  if (k != nullptr) {
    for (std::size_t i = threadIdx.x; i < kv_width; i += blockDim.x) {
      k[t * kv_width + i] = row[2 * width + i];
      v[t * kv_width + i] = row[2 * width + kv_width + i];
    }
  }
}

// Keep each head's original eight-wave reduction. Heads in a block share
// rotary angles; normalization statistics remain independent.
template<std::uint32_t kHeadsPerBlock>
__global__ void PrepareAttentionKernel(
    const float* __restrict__ packed, std::uint32_t stride,
    const float* __restrict__ q_gamma, const float* __restrict__ k_gamma,
    float* __restrict__ q, float* __restrict__ gate,
    __half* __restrict__ k_cache, __half* __restrict__ v_cache,
    std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t d,
    std::uint32_t rotary, const std::uint32_t* start_pos, float theta,
    float eps, const qwen::vision::DeviceRope* rope) {
  __shared__ float partial[kHeadsPerBlock][8];
  __shared__ float norm[kHeadsPerBlock][256];
  const std::uint32_t t = blockIdx.x, first = blockIdx.y * kHeadsPerBlock,
                      tid = threadIdx.x;
  const std::uint32_t lane = tid % 32, wave = tid / 32;
  const std::uint32_t live = min(kHeadsPerBlock, heads + kv_heads - first);
  const std::size_t width = std::size_t(heads) * d,
                    kv_width = std::size_t(kv_heads) * d;
  const float* row = packed + std::size_t(t) * stride;
  float values[kHeadsPerBlock];
#pragma unroll
  for (std::uint32_t j = 0; j < kHeadsPerBlock; ++j) {
    const std::uint32_t h = first + j;
    const bool query = h < heads;
    const std::uint32_t head = query ? h : h - heads;
    values[j] = j < live && tid < d
                    ? row[(query ? head * 2 * d : 2 * width + head * d) + tid]
                    : 0.0f;
    float ss = values[j] * values[j];
    // Match the separate RMS kernel's rounded square before its reduction.
    // Otherwise fast-math can contract it with the first shuffle addition.
    asm volatile("" : "+v"(ss));
    ss = WaveSum(ss);
    if (lane == 0)
      partial[j][wave] = ss;
  }
  __syncthreads();
#pragma unroll
  for (std::uint32_t j = 0; j < kHeadsPerBlock; ++j) {
    if (j >= live)
      continue;
    const std::uint32_t h = first + j;
    const bool query = h < heads;
    const std::uint32_t head = query ? h : h - heads;
    float total = 0.0f;
    for (int w = 0; w < static_cast<int>(blockDim.x / warpSize); ++w)
      total += partial[j][w];
    const float scale = rsqrtf(total / static_cast<float>(d) + eps);
    const float* gamma = query ? q_gamma : k_gamma;
    if (tid < d) {
      norm[j][tid] = values[j] * scale * (gamma != nullptr ? gamma[tid] : 1.0f);
      if (query)
        gate[std::size_t(t) * width + head * d + tid] =
            row[head * 2 * d + d + tid];
      else
        v_cache[std::size_t(*start_pos + t) * kv_width + head * d + tid] =
            __float2half(row[2 * width + kv_width + head * d + tid]);
    }
  }
  __syncthreads();
  const std::uint32_t half = rotary / 2;
  float sine = 0.0f, cosine = 0.0f;
  if (tid < half) {
    const float freq = powf(
        theta, -2.0f * static_cast<float>(tid) / static_cast<float>(rotary));
    sincosf(qwen::vision::RopePosition(rope, *start_pos + t, tid) * freq, &sine,
            &cosine);
  }
#pragma unroll
  for (std::uint32_t j = 0; j < kHeadsPerBlock; ++j) {
    if (j >= live)
      continue;
    const std::uint32_t h = first + j;
    const bool query = h < heads;
    const std::uint32_t head = query ? h : h - heads;
    if (tid < half) {
      const float a = norm[j][tid], b = norm[j][tid + half];
      const float lo = __fmaf_rn(a, cosine, -__fmul_rn(b, sine)),
                  hi = __fmaf_rn(a, sine, __fmul_rn(b, cosine));
      if (query) {
        q[std::size_t(t) * width + head * d + tid] = lo;
        q[std::size_t(t) * width + head * d + tid + half] = hi;
      } else {
        k_cache[std::size_t(*start_pos + t) * kv_width + head * d + tid] =
            __float2half(lo);
        k_cache[std::size_t(*start_pos + t) * kv_width + head * d + tid +
                half] = __float2half(hi);
      }
    }
    if (tid >= rotary && tid < d) {
      if (query)
        q[std::size_t(t) * width + head * d + tid] = norm[j][tid];
      else
        k_cache[std::size_t(*start_pos + t) * kv_width + head * d + tid] =
            __float2half(norm[j][tid]);
    }
  }
}

__global__ void RopeKernel(float* x, std::uint32_t heads, std::uint32_t d,
                           std::uint32_t rotary_dim,
                           const std::uint32_t* start_pos, float theta,
                           const qwen::vision::DeviceRope* rope) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t half = rotary_dim / 2;
  for (std::uint32_t idx = threadIdx.x; idx < heads * half; idx += blockDim.x) {
    const std::uint32_t h = idx / half;
    const std::uint32_t i = idx % half;
    float* v = x + (static_cast<std::size_t>(t) * heads + h) * d;
    const float freq = powf(
        theta, -2.0f * static_cast<float>(i) / static_cast<float>(rotary_dim));
    float s = 0.0f;
    float c = 0.0f;
    sincosf(qwen::vision::RopePosition(rope, *start_pos + t, i) * freq, &s, &c);
    const float a = v[i];
    const float b = v[i + half];
    v[i] = a * c - b * s;
    v[i + half] = a * s + b * c;
  }
}

__global__ void StoreKvKernel(const float* src, __half* cache,
                              std::uint32_t row_dim,
                              const std::uint32_t* start_pos) {
  const std::uint32_t t = blockIdx.x;
  for (std::uint32_t i = threadIdx.x; i < row_dim; i += blockDim.x) {
    cache[static_cast<std::size_t>(*start_pos + t) * row_dim + i] =
        __float2half(src[static_cast<std::size_t>(t) * row_dim + i]);
  }
}

// F16 WMMA fragments (wave32): sixteen halves per lane, eight F32
// accumulators per lane.
using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

// Keep indexer queries in F32: narrowing them before ranking can swap blocks
// at the selection boundary. A thread scores one key, reusing it across all
// four heads. Preserve the original wave32 F32 accumulation and reduction
// tree: even tiny changes can exchange blocks at the top-k boundary.
constexpr std::uint32_t kSelectHeads = 4;
constexpr std::uint32_t kSelectDim = 128;

// F16 fragments are used by attention and projection kernels below.
__device__ __forceinline__ v16h LoadFrag(const __half* p) {
  union {
    v16h f;
    uint4 u[2];
  } cvt;
  cvt.u[0] = *reinterpret_cast<const uint4*>(p);
  cvt.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return cvt.f;
}

/// Locate a descending histogram rank cooperatively. Each thread owns
/// consecutive bins; state receives the bin, its remaining rank and count.
template<std::uint32_t kBinsPerThread>
__device__ void FindHistogramThreshold(const std::uint32_t* histogram,
                                       std::uint32_t wanted,
                                       std::uint32_t* wave_sums,
                                       std::uint32_t* state) {
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t wave = threadIdx.x >> 5;
  const std::uint32_t first =
      kThreads * kBinsPerThread - 1 - threadIdx.x * kBinsPerThread;
  std::uint32_t count = 0;
#pragma unroll
  for (std::uint32_t j = 0; j < kBinsPerThread; ++j)
    count += histogram[first - j];
  std::uint32_t inclusive = count;
#pragma unroll
  for (std::uint32_t distance = 1; distance < 32; distance *= 2) {
    const std::uint32_t other = __shfl_up(inclusive, distance);
    if (lane >= distance)
      inclusive += other;
  }
  if (lane == 31)
    wave_sums[wave] = inclusive;
  __syncthreads();
  std::uint32_t before = inclusive - count;
  for (std::uint32_t w = 0; w < wave; ++w)
    before += wave_sums[w];
  if (before < wanted && before + count >= wanted) {
    for (std::uint32_t j = 0; j < kBinsPerThread; ++j) {
      const std::uint32_t bin = first - j;
      const std::uint32_t bin_count = histogram[bin];
      if (before + bin_count >= wanted) {
        state[0] = bin;
        state[1] = wanted - before;
        state[2] = bin_count;
        break;
      }
      before += bin_count;
    }
  }
  __syncthreads();
}

/// Eight 8-bit K/V values times their block scale, as eight F16 lanes.
__device__ __forceinline__ uint4 DequantKv8(uint2 raw, __half scale) {
  const float d = __half2float(scale);
  const auto* b = reinterpret_cast<const std::int8_t*>(&raw);
  union {
    uint4 u;
    __half h[8];
  } out;
#pragma unroll
  for (int i = 0; i < 8; ++i)
    out.h[i] = __float2half(d * static_cast<float>(b[i]));
  return out.u;
}

/// grid (tokens, kv_row / 256), 256 threads: one wave per 32-value block.
__global__ void QuantizeKvKernel(const __half* __restrict__ cache,
                                 std::int8_t* __restrict__ values,
                                 __half* __restrict__ scales,
                                 std::uint32_t kv_row,
                                 const std::uint32_t* start_pos) {
  const std::size_t i =
      static_cast<std::size_t>(*start_pos + blockIdx.x) * kv_row +
      blockIdx.y * 256 + threadIdx.x;
  const float x = __half2float(cache[i]);
  float amax = fabsf(x);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    amax = fmaxf(amax, __shfl_xor(amax, off));
  const __half scale = __float2half(amax / 127.0F);
  const float d = __half2float(scale);
  const float q = d > 0.0F ? rintf(x / d) : 0.0F;
  values[i] = static_cast<std::int8_t>(fminf(fmaxf(q, -127.0F), 127.0F));
  if (threadIdx.x % 32 == 0)
    scales[i / 32] = scale;
}

constexpr std::uint32_t kGqaTile = 32;
constexpr std::uint32_t kGqaDim = 256;

/// Decode attention grouped by KV head on the WMMA cores: one block per
/// (KV head, split) serves every query head of the group for up to eight
/// rows, so each K/V tile is read once per step instead of once per (row,
/// head). K and V come from the 8-bit cache copy (KvQ8), widened to F16 in
/// LDS.
/// Key tiles of kGqaTile positions are dealt round-robin over gridDim.y
/// splits by tile index, never by batch size. Each split writes (max, sum,
/// unnormalized row) to partials[(t * heads + h) * splits + split], which
/// AttentionMergeKernel combines.
///
/// S = Q K^T and O += P V as 16x16x16 tiles, Q staged once as F16 scaled by
/// 1/16 (the prefill route's scaling), V transposed so its fragments are
/// contiguous keys. K and the transposed V share one LDS region. A pair's
/// scores, softmax and output reduce over its own row only, and tiles are
/// dealt to splits by index, so results do not depend on the batch size.
template<std::uint32_t kRowTiles>
__launch_bounds__(256) __global__ void GqaDecodeAttentionWmmaKernel(
    const float* __restrict__ q, const std::int8_t* __restrict__ k_q8,
    const std::int8_t* __restrict__ v_q8, const __half* __restrict__ k_scale,
    const __half* __restrict__ v_scale, float* __restrict__ partials,
    const std::uint32_t* __restrict__ start_pos, std::uint32_t n_tokens,
    std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t row_base) {
  constexpr std::uint32_t d = kGqaDim;
  constexpr std::uint32_t kPairs = kRowTiles * 16;
  constexpr std::uint32_t kQStride = d + 8;          // halves
  constexpr std::uint32_t kKStride = d + 8;          // halves
  constexpr std::uint32_t kVtStride = kGqaTile + 8;  // halves
  constexpr std::uint32_t kKvHalves = (kGqaTile * kKStride > d * kVtStride)
                                          ? kGqaTile * kKStride
                                          : d * kVtStride;
  constexpr std::uint32_t kOTiles = kRowTiles * (d / 16);  // per block
  constexpr std::uint32_t kOTilesPerWave = kOTiles / 8;
  static_assert(kGqaTile == 32 && kOTiles % 8 == 0);
  __shared__ __attribute__((aligned(16))) __half q_lds[kPairs * kQStride];
  __shared__ __attribute__((aligned(16))) __half kv_lds[kKvHalves];
  // Four row tiles fit in 64 KiB of LDS only when a pair's F16
  // probabilities overwrite the front of its own score row: its wave reads
  // the scores before writing, and no other wave touches the row. Smaller
  // tiles keep separate buffers, which run faster.
  constexpr bool kShareScores = kRowTiles > 2;
  constexpr std::uint32_t kSStride =
      kShareScores ? kGqaTile + 4 : kGqaTile + 1;  // floats
  constexpr std::uint32_t kPStride =
      kShareScores ? 2 * kSStride : kVtStride;  // halves
  __shared__ __attribute__((aligned(16))) float s_lds[kPairs * kSStride];
  __shared__ __attribute__((
      aligned(16))) __half p_own[kShareScores ? 8 : kPairs * kVtStride];
  __half* const p_lds = kShareScores ? reinterpret_cast<__half*>(s_lds) : p_own;
  __shared__ float m_lds[kPairs];
  __shared__ float l_lds[kPairs];
  __shared__ float scale_lds[kPairs];

  const std::uint32_t kvh = blockIdx.x;
  const std::uint32_t split = blockIdx.y;
  const std::uint32_t splits = gridDim.y;
  const std::uint32_t group = heads / kv_heads;
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31u;
  const std::uint32_t wave = tid >> 5u;
  const std::uint32_t sub = lane & 15u;
  const std::uint32_t half_id = lane >> 4u;
  const std::uint32_t first = *start_pos;
  const std::uint32_t n_kv_max = first + n_tokens;
  const std::size_t kv_stride = static_cast<std::size_t>(kv_heads) * d;
  // Rows row_base.. of the batch; a pair's arithmetic ignores which launch
  // or position in the launch it occupies.
  const auto pair_row = [&](std::uint32_t p) { return row_base + p / group; };
  const auto pair_head = [&](std::uint32_t p) {
    return kvh * group + p % group;
  };
  const auto pair_live = [&](std::uint32_t p) {
    return p < kPairs && pair_row(p) < n_tokens;
  };

  // Q once, F16, pre-scaled so scores need no further multiply.
  for (std::uint32_t c = tid; c < kPairs * (d / 4); c += 256) {
    const std::uint32_t p = c / (d / 4);
    const std::uint32_t x = (c % (d / 4)) * 4;
    float4 v = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
    if (pair_live(p))
      v = *reinterpret_cast<const float4*>(
          q +
          (static_cast<std::size_t>(pair_row(p)) * heads + pair_head(p)) * d +
          x);
    __half* dst = q_lds + p * kQStride + x;
    dst[0] = __float2half_rn(v.x * (1.0F / 16.0F));
    dst[1] = __float2half_rn(v.y * (1.0F / 16.0F));
    dst[2] = __float2half_rn(v.z * (1.0F / 16.0F));
    dst[3] = __float2half_rn(v.w * (1.0F / 16.0F));
  }
  if (tid < kPairs) {
    m_lds[tid] = -INFINITY;
    l_lds[tid] = 0.0F;
  }
  v8f o_acc[kOTilesPerWave] = {};

  // Raw 8-bit values and their scales; converted to F16 at the LDS stores
  // so the prefetch of the next tile stays in flight under softmax and PV.
  // Sixteen values per 16-byte load.
  constexpr std::uint32_t kStage16 = kGqaTile * (d / 16) / 256;
  uint4 k_reg[kStage16];
  uint4 v_reg[kStage16];
  __half k_sc[kStage16];
  __half v_sc[kStage16];
  const auto load_tile = [&](std::uint32_t key0) {
#pragma unroll
    for (std::uint32_t j = 0; j < kStage16; ++j) {
      const std::uint32_t c = tid + 256 * j;
      const std::uint32_t pos = key0 + c / (d / 16);
      const std::uint32_t d16 = (c % (d / 16)) * 16;
      k_reg[j] = make_uint4(0u, 0u, 0u, 0u);
      v_reg[j] = make_uint4(0u, 0u, 0u, 0u);
      k_sc[j] = __float2half(0.0F);
      v_sc[j] = __float2half(0.0F);
      if (pos < n_kv_max) {
        const std::size_t off = pos * kv_stride + kvh * d + d16;
        k_reg[j] = *reinterpret_cast<const uint4*>(k_q8 + off);
        k_sc[j] = k_scale[off / 32];
      }
      // V lanes walk keys instead of dims, so the transposed LDS stores of
      // one wave land in distinct banks rather than 16-way conflicting.
      const std::uint32_t vpos = key0 + c % kGqaTile;
      const std::uint32_t vd16 = (c / kGqaTile) * 16;
      if (vpos < n_kv_max) {
        const std::size_t voff = vpos * kv_stride + kvh * d + vd16;
        v_reg[j] = *reinterpret_cast<const uint4*>(v_q8 + voff);
        v_sc[j] = v_scale[voff / 32];
      }
    }
  };
  std::uint32_t tile = split;
  if (tile * kGqaTile < n_kv_max)
    load_tile(tile * kGqaTile);

  for (; tile * kGqaTile < n_kv_max; tile += splits) {
    const std::uint32_t key0 = tile * kGqaTile;
    __syncthreads();  // previous PV has finished reading kv_lds/p_lds
#pragma unroll
    for (std::uint32_t j = 0; j < kStage16; ++j) {
      const std::uint32_t c = tid + 256 * j;
      const std::uint32_t key = c / (d / 16);
      const std::uint32_t d16 = (c % (d / 16)) * 16;
      auto* dst = reinterpret_cast<uint4*>(&kv_lds[key * kKStride + d16]);
      dst[0] = DequantKv8(make_uint2(k_reg[j].x, k_reg[j].y), k_sc[j]);
      dst[1] = DequantKv8(make_uint2(k_reg[j].z, k_reg[j].w), k_sc[j]);
    }
    __syncthreads();
    // S tiles: (row tile, key block); waves beyond the tile count idle.
    if (wave < kRowTiles * 2) {
      const std::uint32_t rt = wave / 2;
      const std::uint32_t kb = wave % 2;
      v8f s_acc = {};
#pragma unroll
      for (std::uint32_t ks = 0; ks < d / 16; ++ks) {
        const v16h a = LoadFrag(q_lds + (rt * 16 + sub) * kQStride + ks * 16);
        const v16h b = LoadFrag(kv_lds + (kb * 16 + sub) * kKStride + ks * 16);
        s_acc = Wmma(a, b, s_acc);
      }
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i)
        s_lds[(rt * 16 + 2 * i + half_id) * kSStride + kb * 16 + sub] =
            s_acc[i];
    }
    __syncthreads();
    // V transposed into the K region: vt[dim][key].
#pragma unroll
    for (std::uint32_t j = 0; j < kStage16; ++j) {
      const std::uint32_t c = tid + 256 * j;
      const std::uint32_t key = c % kGqaTile;
      const std::uint32_t d16 = (c / kGqaTile) * 16;
      const uint4 lo = DequantKv8(make_uint2(v_reg[j].x, v_reg[j].y), v_sc[j]);
      const uint4 hi = DequantKv8(make_uint2(v_reg[j].z, v_reg[j].w), v_sc[j]);
      const auto* vl = reinterpret_cast<const __half*>(&lo);
      const auto* vh = reinterpret_cast<const __half*>(&hi);
#pragma unroll
      for (std::uint32_t x = 0; x < 8; ++x) {
        kv_lds[(d16 + x) * kVtStride + key] = vl[x];
        kv_lds[(d16 + 8 + x) * kVtStride + key] = vh[x];
      }
    }
    // The next tile's loads overlap softmax and PV.
    if ((tile + splits) * kGqaTile < n_kv_max)
      load_tile((tile + splits) * kGqaTile);
    // Online softmax: one wave per pair, one lane per key.
    for (std::uint32_t pair = wave; pair < kPairs; pair += 8) {
      const bool valid =
          pair_live(pair) && key0 + lane <= first + pair_row(pair);
      const float sc = valid ? s_lds[pair * kSStride + lane] : -INFINITY;
      float tile_max = sc;
#pragma unroll
      for (std::uint32_t off = 16; off > 0; off >>= 1)
        tile_max = fmaxf(tile_max, __shfl_xor(tile_max, off));
      const float m = m_lds[pair];
      const float m_new = fmaxf(m, tile_max);
      const float w = valid ? __expf(sc - m_new) : 0.0F;
      p_lds[pair * kPStride + lane] = __float2half_rn(w);
      const float sum = WaveSum(w);
      if (lane == 0) {
        const float rescale = m == -INFINITY ? 0.0F : __expf(m - m_new);
        l_lds[pair] = l_lds[pair] * rescale + sum;
        m_lds[pair] = m_new;
        scale_lds[pair] = rescale;
      }
    }
    __syncthreads();
    // O += P V: wave w owns O tiles w, w + 8, ... (row tile, dim tile).
    // A pair takes the WMMA result for tiles it sees whole. The WMMA sum
    // also depends on V rows whose P is zero, and keys past a row hold real
    // values or zeros depending on the batch width, so the tile holding a
    // row's own position sums its visible keys alone.
#pragma unroll
    for (std::uint32_t t = 0; t < kOTilesPerWave; ++t) {
      const std::uint32_t ot = wave + 8 * t;
      const std::uint32_t rt = ot / (d / 16);
      const std::uint32_t dt = ot % (d / 16);
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i)
        o_acc[t][i] *= scale_lds[rt * 16 + 2 * i + half_id];
      v8f acc = o_acc[t];
#pragma unroll
      for (std::uint32_t kb = 0; kb < 2; ++kb) {
        const v16h a = LoadFrag(p_lds + (rt * 16 + sub) * kPStride + kb * 16);
        const v16h b = LoadFrag(kv_lds + (dt * 16 + sub) * kVtStride + kb * 16);
        acc = Wmma(a, b, acc);
      }
      if (key0 + kGqaTile - 1 > first) {
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          const std::uint32_t pair = rt * 16 + 2 * i + half_id;
          const std::uint32_t last = first + pair_row(pair);
          if (key0 + kGqaTile - 1 <= last)
            continue;
          float sum = o_acc[t][i];
          for (std::uint32_t key = 0; key < kGqaTile && key0 + key <= last;
               ++key)
            sum = fmaf(__half2float(p_lds[pair * kPStride + key]),
                       __half2float(kv_lds[(dt * 16 + sub) * kVtStride + key]),
                       sum);
          acc[i] = sum;
        }
      }
      o_acc[t] = acc;
    }
  }
  __syncthreads();
#pragma unroll
  for (std::uint32_t t = 0; t < kOTilesPerWave; ++t) {
    const std::uint32_t ot = wave + 8 * t;
    const std::uint32_t rt = ot / (d / 16);
    const std::uint32_t dt = ot % (d / 16);
#pragma unroll
    for (std::uint32_t i = 0; i < 8; ++i) {
      const std::uint32_t p = rt * 16 + 2 * i + half_id;
      if (!pair_live(p))
        continue;
      float* part =
          partials +
          ((static_cast<std::size_t>(pair_row(p)) * heads + pair_head(p)) *
               splits +
           split) *
              (d + 2);
      part[2 + dt * 16 + sub] = o_acc[t][i];
    }
  }
  if (tid < kPairs && pair_live(tid)) {
    float* part =
        partials +
        ((static_cast<std::size_t>(pair_row(tid)) * heads + pair_head(tid)) *
             splits +
         split) *
            (d + 2);
    part[0] = m_lds[tid];
    part[1] = l_lds[tid];
  }
}

/// grid (heads, queries): merges the split partials of one row with the
/// usual log-sum-exp rescaling.
__global__ void AttentionMergeKernel(const float* partials, float* out,
                                     std::uint32_t heads, std::uint32_t d,
                                     std::uint32_t splits) {
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t t = blockIdx.y;
  const std::uint32_t i = threadIdx.x;
  const float* base =
      partials + (static_cast<std::size_t>(t) * heads + h) * splits * (d + 2);
  float m = -INFINITY;
  for (std::uint32_t s = 0; s < splits; ++s) {
    m = fmaxf(m, base[s * (d + 2)]);
  }
  float l = 0.0f;
  float acc = 0.0f;
  for (std::uint32_t s = 0; s < splits; ++s) {
    const float* part = base + s * (d + 2);
    const float ms = part[0];
    const float scale = ms == -INFINITY ? 0.0f : __expf(ms - m);
    l += part[1] * scale;
    if (i < d) {
      acc += part[2 + i] * scale;
    }
  }
  if (i < d) {
    out[(static_cast<std::size_t>(t) * heads + h) * d + i] = acc / l;
  }
}

// Keep the original block-wide softmax reduction. Selection then stays in
// one wave's registers, retaining probability ordering and lowest-index ties.
template<unsigned MaxExperts>
__global__ void RouterTopKKernel(const float* logits, std::uint32_t stride,
                                 std::int32_t* ids, float* weights,
                                 std::uint32_t n_experts, std::uint32_t k) {
  __shared__ float probs[1024];
  __shared__ float shared[32];
  __shared__ std::uint32_t chosen[32];
  __shared__ float chosen_p[32];
  const std::uint32_t t = blockIdx.x;
  const float* src = logits + static_cast<std::size_t>(t) * stride;
  float local_max = -INFINITY;
  for (std::uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
    local_max = fmaxf(local_max, src[e]);
  }
  const float max_logit = BlockMax(local_max, shared);
  float local_sum = 0.0f;
  for (std::uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
    probs[e] = __expf(src[e] - max_logit);
    local_sum += probs[e];
  }
  const float denom = BlockSum(local_sum, shared);
  for (std::uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
    probs[e] /= denom;
  }
  __syncthreads();

  if (threadIdx.x >= 32)
    return;
  constexpr unsigned Items = MaxExperts / 32;
  float values[Items];
#pragma unroll
  for (unsigned j = 0; j < Items; ++j) {
    unsigned e = threadIdx.x + j * 32;
    values[j] = e < n_experts ? probs[e] : -1.0f;
  }
  for (unsigned slot = 0; slot < k; ++slot) {
    float best = -1.0f;
    unsigned index = 0xffffffffu;
#pragma unroll
    for (unsigned j = 0; j < Items; ++j) {
      unsigned e = threadIdx.x + j * 32;
      if (values[j] > best) {
        best = values[j];
        index = e;
      }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      float other = __shfl_xor(best, offset);
      unsigned oi = __shfl_xor(index, offset);
      if (other > best || (other == best && oi < index)) {
        best = other;
        index = oi;
      }
    }
    if (threadIdx.x == 0) {
      chosen[slot] = index;
      chosen_p[slot] = best;
    }
#pragma unroll
    for (unsigned j = 0; j < Items; ++j) {
      if (threadIdx.x + j * 32 == index)
        values[j] = -1.0f;
    }
  }
  if (threadIdx.x == 0) {
    float sum = 0.0f;
    for (std::uint32_t slot = 0; slot < k; ++slot) {
      sum += chosen_p[slot];
    }
    sum = fmaxf(sum, 6.103515625e-5f);
    for (std::uint32_t slot = 0; slot < k; ++slot) {
      ids[t * k + slot] = static_cast<std::int32_t>(chosen[slot]);
      weights[t * k + slot] = chosen_p[slot] / sum;
    }
  }
}

/// grid (tokens, dim chunks of kThreads).
__global__ void MoeEpilogueKernel(const float* expert_out, const float* weights,
                                  const float* shared_out, const float* gate,
                                  std::uint32_t gate_stride, float* out,
                                  std::uint32_t k, std::uint32_t dim) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t i = blockIdx.y * blockDim.x + threadIdx.x;
  if (i >= dim) {
    return;
  }
  float acc = 0.0f;
  for (std::uint32_t s = 0; s < k; ++s) {
    acc += weights[t * k + s] *
           expert_out[(static_cast<std::size_t>(t) * k + s) * dim + i];
  }
  const std::size_t idx = static_cast<std::size_t>(t) * dim + i;
  out[idx] = acc + SigmoidF(gate[static_cast<std::size_t>(t) * gate_stride]) *
                       shared_out[idx];
}

/// Four adjacent output lanes per thread: the same reduction over the top-k
/// slots with a quarter of the waves, so the per-wave issue overhead no
/// longer hides the streaming reads.
/// `ExpertT` is float, or __half when the routed down projection wrote its
/// rows as F16.
template<typename ExpertT>
__global__ void MoeEpilogueVec4Kernel(const ExpertT* expert_out,
                                      const float* weights,
                                      const float* shared_out,
                                      const float* gate,
                                      std::uint32_t gate_stride, float* out,
                                      std::uint32_t k, std::uint32_t dim) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t i = (blockIdx.y * blockDim.x + threadIdx.x) * 4;
  if (i >= dim) {
    return;
  }
  float4 acc{0.0F, 0.0F, 0.0F, 0.0F};
  const ExpertT* rows = expert_out + static_cast<std::size_t>(t) * k * dim + i;
  for (std::uint32_t s = 0; s < k; ++s) {
    const float w = weights[t * k + s];
    const float4 v = Load4(rows + static_cast<std::size_t>(s) * dim);
    acc.x += w * v.x;
    acc.y += w * v.y;
    acc.z += w * v.z;
    acc.w += w * v.w;
  }
  const std::size_t idx = static_cast<std::size_t>(t) * dim + i;
  const float g = SigmoidF(gate[static_cast<std::size_t>(t) * gate_stride]);
  const float4 sh = *reinterpret_cast<const float4*>(shared_out + idx);
  acc.x += g * sh.x;
  acc.y += g * sh.y;
  acc.z += g * sh.z;
  acc.w += g * sh.w;
  *reinterpret_cast<float4*>(out + idx) = acc;
}

/// dst[t] = row < 0 ? alt[t] : base[(row + t)]: the draft block's hidden
/// input, a kept trunk row or its own carried residual.
/// The draft block's hidden input, normalized as the model was trained:
/// kept trunk rows through the trunk's final norm (`base_gamma`), the
/// block's own carried residual through its shared head norm (`alt_gamma`).
__global__ void MtpHiddenKernel(const float* base, const float* alt,
                                const std::int32_t* row, float* dst,
                                std::uint32_t width, const float* base_gamma,
                                const float* alt_gamma, float eps) {
  __shared__ float shared[32];
  const std::uint32_t t = blockIdx.x;
  const bool carried = *row < 0;
  const float* src = carried
                         ? alt + static_cast<std::size_t>(t) * width
                         : base + (static_cast<std::size_t>(*row) + t) * width;
  const float* gamma = carried ? alt_gamma : base_gamma;
  float ss = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < width; i += blockDim.x)
    ss += src[i] * src[i];
  ss = BlockSum(ss, shared);
  const float scale = rsqrtf(ss / static_cast<float>(width) + eps);
  for (std::uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
    dst[static_cast<std::size_t>(t) * width + i] = src[i] * scale * gamma[i];
  }
}

__global__ void MtpAddEmbeddingKernel(const float* embedding, float* residual,
                                      std::uint32_t hidden,
                                      std::uint32_t streams) {
  const std::uint32_t t = blockIdx.x;
  for (std::uint32_t i = threadIdx.x; i < streams * hidden; i += blockDim.x) {
    residual[static_cast<std::size_t>(t) * streams * hidden + i] +=
        embedding[static_cast<std::size_t>(t) * hidden + i % hidden];
  }
}

__device__ __forceinline__ ArgmaxCandidate BetterCandidate(ArgmaxCandidate a,
                                                           ArgmaxCandidate b) {
  return b.value > a.value || (b.value == a.value && b.index < a.index) ? b : a;
}

__device__ ArgmaxCandidate ArgmaxBlock(ArgmaxCandidate best,
                                       ArgmaxCandidate* shared) {
  const unsigned lane = threadIdx.x & 31u;
  const unsigned wave = threadIdx.x >> 5u;
  for (int offset = 16; offset > 0; offset >>= 1) {
    best = BetterCandidate(
        best, {__shfl_xor(best.value, offset), __shfl_xor(best.index, offset)});
  }
  if (lane == 0) {
    shared[wave] = best;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (unsigned w = 1; w < kThreads / 32; ++w) {
      best = BetterCandidate(best, shared[w]);
    }
  }
  return best;
}

__global__ void ArgmaxPartialKernel(const float* logits,
                                    ArgmaxCandidate* partial,
                                    std::uint32_t vocab) {
  __shared__ ArgmaxCandidate shared[kThreads / 32];
  const std::uint32_t t = blockIdx.y;
  const float* src = logits + static_cast<std::size_t>(t) * vocab;
  ArgmaxCandidate best{-INFINITY, INT32_MAX};
  for (std::uint32_t i = blockIdx.x * kThreads + threadIdx.x; i < vocab;
       i += kArgmaxParts * kThreads) {
    best = BetterCandidate(best, {src[i], static_cast<std::int32_t>(i)});
  }
  best = ArgmaxBlock(best, shared);
  if (threadIdx.x == 0) {
    partial[t * kArgmaxParts + blockIdx.x] = best;
  }
}

__global__ void ArgmaxFinishKernel(const float* logits,
                                   const ArgmaxCandidate* partial,
                                   std::int32_t* out, std::uint32_t vocab) {
  __shared__ ArgmaxCandidate shared[kThreads / 32];
  const std::uint32_t t = blockIdx.x;
  ArgmaxCandidate best{-INFINITY, INT32_MAX};
  if (threadIdx.x < kArgmaxParts) {
    best = partial[t * kArgmaxParts + threadIdx.x];
  }
  best = ArgmaxBlock(best, shared);
  if (threadIdx.x == 0) {
    // std::max_element keeps element zero when it is NaN; later NaNs
    // never replace a finite candidate.
    out[t] =
        isnan(logits[static_cast<std::size_t>(t) * vocab]) ? 0 : best.index;
  }
}

/// Argmax plus the softmax probability of the winner, for draft gating:
/// each part keeps its best and its sum of exp(logit - best).
__global__ void ArgmaxProbPartialKernel(const float* logits,
                                        ArgmaxCandidate* partial,
                                        float* partial_sum,
                                        std::uint32_t vocab) {
  __shared__ ArgmaxCandidate shared[kThreads / 32];
  __shared__ float reduce[32];
  ArgmaxCandidate best{-INFINITY, INT32_MAX};
  for (std::uint32_t i = blockIdx.x * kThreads + threadIdx.x; i < vocab;
       i += kArgmaxParts * kThreads) {
    best = BetterCandidate(best, {logits[i], static_cast<std::int32_t>(i)});
  }
  best = ArgmaxBlock(best, shared);
  __syncthreads();
  if (threadIdx.x == 0)
    shared[0] = best;
  __syncthreads();
  const float m = shared[0].value;
  float sum = 0.0F;
  for (std::uint32_t i = blockIdx.x * kThreads + threadIdx.x; i < vocab;
       i += kArgmaxParts * kThreads) {
    sum += __expf(logits[i] - m);
  }
  sum = BlockSum(sum, reduce);
  if (threadIdx.x == 0) {
    partial[blockIdx.x] = shared[0];
    partial_sum[blockIdx.x] = m == -INFINITY ? 0.0F : sum;
  }
}

__global__ void ArgmaxProbFinishKernel(const float* logits,
                                       const ArgmaxCandidate* partial,
                                       const float* partial_sum,
                                       std::int32_t* out, float* prob) {
  __shared__ ArgmaxCandidate shared[kThreads / 32];
  __shared__ float reduce[32];
  ArgmaxCandidate best{-INFINITY, INT32_MAX};
  if (threadIdx.x < kArgmaxParts)
    best = partial[threadIdx.x];
  best = ArgmaxBlock(best, shared);
  __syncthreads();
  if (threadIdx.x == 0)
    shared[0] = best;
  __syncthreads();
  const float m = shared[0].value;
  float sum = 0.0F;
  if (threadIdx.x < kArgmaxParts && partial[threadIdx.x].value != -INFINITY)
    sum = partial_sum[threadIdx.x] * __expf(partial[threadIdx.x].value - m);
  sum = BlockSum(sum, reduce);
  if (threadIdx.x == 0) {
    *out = isnan(logits[0]) ? 0 : shared[0].index;
    *prob = sum > 0.0F ? 1.0F / sum : 0.0F;
  }
}

__device__ __forceinline__ PenaltyArgmaxCandidate
BetterPenaltyCandidate(PenaltyArgmaxCandidate a, PenaltyArgmaxCandidate b) {
  return b.value > a.value || (b.value == a.value && b.index < a.index) ? b : a;
}

__device__ PenaltyArgmaxCandidate PenaltyArgmaxBlock(
    PenaltyArgmaxCandidate best, PenaltyArgmaxCandidate* shared) {
  const unsigned lane = threadIdx.x & 31u;
  const unsigned wave = threadIdx.x >> 5u;
  for (int offset = 16; offset > 0; offset >>= 1)
    best = BetterPenaltyCandidate(
        best, {__shfl_xor(best.value, offset), __shfl_xor(best.index, offset)});
  if (lane == 0)
    shared[wave] = best;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (unsigned w = 1; w < kThreads / 32; ++w)
      best = BetterPenaltyCandidate(best, shared[w]);
  }
  return best;
}

__global__ void PenaltyArgmaxPartialKernel(const float* logits,
                                           GreedyPenaltyRows penalties,
                                           float repeat, float frequency,
                                           float presence,
                                           PenaltyArgmaxCandidate* partial,
                                           std::uint32_t vocab) {
  __shared__ PenaltyArgmaxCandidate shared[kThreads / 32];
  const unsigned row = blockIdx.y;
  const auto* src = logits + std::size_t(row) * vocab;
  PenaltyArgmaxCandidate best{-INFINITY, INT32_MAX};
  for (unsigned token = blockIdx.x * kThreads + threadIdx.x; token < vocab;
       token += kArgmaxParts * kThreads) {
    if (!isfinite(src[token]))
      continue;
    auto begin = penalties.offsets[row];
    auto end = penalties.offsets[row + 1];
    while (begin < end) {
      const auto mid = begin + (end - begin) / 2;
      if (penalties.penalties[mid].token < token)
        begin = mid + 1;
      else
        end = mid;
    }
    double value = src[token];
    if (begin < penalties.offsets[row + 1] &&
        penalties.penalties[begin].token == token) {
      const auto penalty = penalties.penalties[begin];
      // Preserve the CPU's FP64 operations. In particular, do not narrow
      // the adjusted logit before comparing close candidates.
      if (penalty.repeated && repeat != 1.0F)
        value = value <= 0 ? __dmul_rn(value, double(repeat))
                           : __ddiv_rn(value, double(repeat));
      value = __dadd_rn(value, -__dmul_rn(double(frequency),
                                          double(penalty.generated_count)));
      if (penalty.generated_count != 0)
        value = __dadd_rn(value, -double(presence));
    }
    best =
        BetterPenaltyCandidate(best, {value, static_cast<std::int32_t>(token)});
  }
  best = PenaltyArgmaxBlock(best, shared);
  if (threadIdx.x == 0)
    partial[row * kArgmaxParts + blockIdx.x] = best;
}

__global__ void PenaltyArgmaxFinishKernel(const float* logits,
                                          const PenaltyArgmaxCandidate* partial,
                                          ArgmaxCandidate* out,
                                          std::uint32_t vocab) {
  __shared__ PenaltyArgmaxCandidate shared[kThreads / 32];
  const auto row = blockIdx.x;
  PenaltyArgmaxCandidate best{-INFINITY, INT32_MAX};
  if (threadIdx.x < kArgmaxParts)
    best = partial[row * kArgmaxParts + threadIdx.x];
  best = PenaltyArgmaxBlock(best, shared);
  if (threadIdx.x == 0)
    out[row] =
        best.index < vocab
            ? ArgmaxCandidate{logits[std::size_t(row) * vocab + best.index],
                              best.index}
            : ArgmaxCandidate{NAN, -1};
}

constexpr unsigned kMtpCandidateTile = 1024;

__global__ void GatherArgmaxCandidatesKernel(const float* logits,
                                             const std::uint32_t* ids,
                                             ArgmaxCandidate* out,
                                             std::uint32_t rows,
                                             std::uint32_t vocab) {
  const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows) {
    const auto id = ids[row];
    out[row] = id < vocab
                   ? ArgmaxCandidate{logits[std::size_t(row) * vocab + id],
                                     static_cast<std::int32_t>(id)}
                   : ArgmaxCandidate{NAN, -1};
  }
}

// A token excluded from its tile's top Keep cannot enter the global top Keep.
// Reduce tiles repeatedly, retaining the original order of equal scores.
// Only token IDs need to survive between passes; the original logit buffer
// supplies scores and preserves the sign of zero in the final output.
template<unsigned Keep>
__global__ void MtpCandidateTileKernel(const float* logits,
                                       const std::uint32_t* input,
                                       std::uint32_t* output, float* scores,
                                       std::uint32_t size,
                                       std::uint32_t vocab) {
  using Sort = hipcub::BlockRadixSort<float, kThreads, 4, std::uint32_t>;
  __shared__ Sort::TempStorage scratch;
  float keys[4];
  std::uint32_t ids[4];
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    const std::size_t i =
        std::size_t(blockIdx.x) * kMtpCandidateTile + threadIdx.x * 4 + j;
    const auto id = i < size ? (input != nullptr ? input[i] : i) : UINT32_MAX;
    ids[j] = static_cast<std::uint32_t>(id);
    const float value = id < vocab ? logits[id] : -INFINITY;
    keys[j] = isfinite(value) ? (value == 0.0F ? 0.0F : value) : -INFINITY;
  }
  Sort(scratch).SortDescending(keys, ids);
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    const unsigned rank = threadIdx.x * 4 + j;
    if (rank < min(Keep, size)) {
      output[blockIdx.x * Keep + rank] = ids[j];
      if (scores != nullptr) {
        const float value = ids[j] < vocab ? logits[ids[j]] : -INFINITY;
        scores[rank] = isfinite(value) ? value : -INFINITY;
      }
    }
  }
}

__global__ void CopyKernel(const float* src, float* dst, std::size_t count) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < count) {
    dst[i] = src[i];
  }
}

inline unsigned Blocks(std::size_t count) {
  return static_cast<unsigned>((count + kThreads - 1) / kThreads);
}

// Prefill attention geometry: 16 x 256 query heads over two KV heads.
constexpr std::uint32_t kWmmaHeadDim = 256;
constexpr std::uint32_t kWmmaQueryHeads = 16;
constexpr std::uint32_t kWmmaKvHeads = 2;
constexpr std::uint32_t kWmmaGqa = kWmmaQueryHeads / kWmmaKvHeads;
constexpr std::uint32_t kWmmaAttnWidth = kWmmaQueryHeads * kWmmaHeadDim;
constexpr std::uint32_t kWmmaKvWidth = kWmmaKvHeads * kWmmaHeadDim;
// A prefill block owns 16 queries of four heads of one KV group (21.9
// TFLOPS at 18K depth against 18.2 with two heads; 32-key rounds of four
// heads spill registers).
constexpr std::uint32_t kWmmaHeads = 4;
constexpr std::uint32_t kWmmaQueryRows = 16;

// W8A8 int8 WMMA GEMM over the GGUF Q8_0 weights, ported from the Qwen 27B
// route (src/models/qwen/hip/kernels/prefill_quant_gemm.hip,
// opt-c163-blocked-w8a8 with the opt-c179 addressing). Weights stay in their
// row-major 34-byte block_q8_0 layout; activations are quantized per 32-wide
// block into WMMA B-fragment order:
//
//   tile(tt, kb) = [b0: 16 tokens x 16 bytes]   offset   0
//                  [b1: 16 tokens x 16 bytes]   offset 256
//                  [16 fp32 token scales    ]   offset 512
//
// with tt = token / 16 and kb the 32-element K block, 576 bytes per tile.
struct Q8_0Block {
  __half d;
  std::int8_t qs[32];
};
static_assert(sizeof(Q8_0Block) == 34, "block_q8_0 must be 34 bytes");

using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;

__device__ __forceinline__ int32x8_t WmmaI8(int32x4_t a, int32x4_t b,
                                            int32x8_t c) {
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, true);
}

/// Four elements per lane, eight lanes per 32-wide K block, four blocks per
/// wave: per-block absmax scale, four codes stored as one word straight
/// into the fragment order the GEMM stages from.
__global__ void QuantizeQ8TiledVec4Kernel(const float* __restrict__ x,
                                          void* __restrict__ y,
                                          std::size_t batch, std::size_t k) {
  const std::size_t num_blocks = k / 32;
  const std::size_t b_idx =
      (blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x) >> 3u;
  const std::uint32_t lane8 = threadIdx.x & 7u;
  if (b_idx >= batch * num_blocks) {
    return;
  }
  const std::size_t tok = b_idx / num_blocks;
  const std::size_t blk = b_idx % num_blocks;
  const float4 v = *reinterpret_cast<const float4*>(x + (tok * k) + (blk * 32) +
                                                    (lane8 * 4));
  float max_abs =
      fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
  for (int off = 4; off > 0; off >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_xor(max_abs, off));
  }
  const float d = max_abs / 127.0F;
  const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
  const auto q0 = static_cast<std::uint32_t>(
      static_cast<std::uint8_t>(static_cast<std::int8_t>(roundf(v.x * id))));
  const auto q1 = static_cast<std::uint32_t>(
      static_cast<std::uint8_t>(static_cast<std::int8_t>(roundf(v.y * id))));
  const auto q2 = static_cast<std::uint32_t>(
      static_cast<std::uint8_t>(static_cast<std::int8_t>(roundf(v.z * id))));
  const auto q3 = static_cast<std::uint32_t>(
      static_cast<std::uint8_t>(static_cast<std::int8_t>(roundf(v.w * id))));
  const std::size_t tt = tok / kQ8ActTileTokens;
  const std::size_t tl = tok % kQ8ActTileTokens;
  std::int8_t* tile = Q8ActTile(y, num_blocks, tt, blk);
  const std::uint32_t pos = lane8 * 4;  // 0, 4, ..., 28
  *reinterpret_cast<std::uint32_t*>(tile + ((pos >> 4u) * 256) + (tl * 16) +
                                    (pos & 15u)) =
      q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
  if (lane8 == 0) {
    *reinterpret_cast<float*>(tile + kQ8ActScaleOffset + (tl * sizeof(float))) =
        d;
  }
}

/// One wave per (token, 32-wide K block): per-block absmax scale, codes
/// written straight into the fragment order the GEMM stages from.
__global__ void QuantizeQ8TiledKernel(const float* __restrict__ x,
                                      void* __restrict__ y, std::size_t batch,
                                      std::size_t k) {
  const std::size_t num_blocks = k / 32;
  const std::size_t b_idx =
      (blockIdx.x * (blockDim.x >> 5u)) + (threadIdx.x >> 5u);
  const std::size_t lane_id = threadIdx.x & 31u;
  if (b_idx >= batch * num_blocks) {
    return;
  }
  const std::size_t tok = b_idx / num_blocks;
  const std::size_t blk = b_idx % num_blocks;
  const float val = x[(tok * k) + (blk * 32) + lane_id];
  float max_abs = fabsf(val);
  for (int off = 16; off > 0; off >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_xor(max_abs, off));
  }
  const float d = max_abs / 127.0F;
  const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
  const auto q = static_cast<std::int8_t>(roundf(val * id));
  const std::size_t tt = tok / kQ8ActTileTokens;
  const std::size_t tl = tok % kQ8ActTileTokens;
  std::int8_t* tile = Q8ActTile(y, num_blocks, tt, blk);
  const std::size_t half = lane_id >> 4u;
  const std::size_t pos = lane_id & 15u;
  tile[(half * 256) + (tl * 16) + pos] = q;
  if (lane_id == 0) {
    *reinterpret_cast<float*>(tile + kQ8ActScaleOffset + (tl * sizeof(float))) =
        d;
  }
}

/// Two-dimensionally blocked W8A8 WMMA GEMM: block = BM rows x BN tokens, BK
/// 32-element K blocks per LDS stage, waves = WM row groups x WN token groups.
/// Out-of-range rows and K blocks are clamped and their scale zeroed, so
/// they contribute exactly zero without divergence. y is [batch][m].
template<int BM, int BN, int BK, int WM, int WN>
__launch_bounds__(256) __global__
    void W8A8BlockedWmmaGEMMKernel(const void* __restrict__ w,
                                   const void* __restrict__ x_blocks,
                                   float* __restrict__ y, std::size_t batch,
                                   std::size_t m, std::size_t k) {
  static_assert(WM * WN == 8, "256 threads is 8 waves");
  static_assert(BM % (16 * WM) == 0 && BN % (16 * WN) == 0);
  static_assert(BN / 16 <= 8,
                "the activation stage assigns one wave per token subtile");
  constexpr int kRowTiles = BM / 16;
  constexpr int kTokTiles = BN / 16;
  constexpr int kWaveRowTiles = kRowTiles / WM;
  constexpr int kWaveTokTiles = kTokTiles / WN;

  // One LDS plane: codes and scales of both operands; the epilogue reuses
  // the first 16 KB to transpose 16 x 32 result tiles per wave.
  constexpr int kABytes = BK * kRowTiles * 32 * 16;
  constexpr int kDwBytes = BK * kRowTiles * 16 * 4;
  constexpr int kBBytes = BK * kTokTiles * 32 * 16;
  constexpr int kDxBytes = BK * kTokTiles * 16 * 4;
  constexpr int kLdsBytes = kABytes + kDwBytes + kBBytes + kDxBytes;
  static_assert(kLdsBytes >= 8 * 512 * 4, "epilogue transposes 16 KB");
  static_assert(kWaveRowTiles % 2 == 0, "the epilogue pairs row tiles");
  __shared__ __attribute__((aligned(16))) std::uint8_t lds[kLdsBytes];
  auto* s_a = reinterpret_cast<int32x4_t(*)[kRowTiles][32]>(lds);
  auto* s_dw = reinterpret_cast<float (*)[kRowTiles][16]>(lds + kABytes);
  auto* s_b =
      reinterpret_cast<int32x4_t(*)[kTokTiles][32]>(lds + kABytes + kDwBytes);
  auto* s_dx = reinterpret_cast<float (*)[kTokTiles][16]>(lds + kABytes +
                                                          kDwBytes + kBBytes);

  const auto* w_blocks = static_cast<const Q8_0Block*>(w);
  const std::size_t num_blocks = k / 32;

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  const int wave_row = wave_id / WN;
  const int wave_tok = wave_id % WN;

  const std::size_t r_block = static_cast<std::size_t>(blockIdx.y) * BM;
  const std::size_t t_block = static_cast<std::size_t>(blockIdx.x) * BN;
  const std::size_t tt_block = t_block / kQ8ActTileTokens;

  float acc[kWaveRowTiles][kWaveTokTiles][8];
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        acc[i][j][l] = 0.0F;
      }
    }
  }

  // Staging registers for the next K stage, fetched one stage ahead so the
  // scattered 34-byte block reads overlap the WMMA work.
  constexpr int kPrefetch = (BM * BK) / 256;
  int32x4_t r_q0[kPrefetch];
  int32x4_t r_q1[kPrefetch];
  float r_dw[kPrefetch];
  int32x4_t r_b[BK];
  float r_dx[BK];
  const int b_tile = wave_id;  // one wave per token subtile

  const int num_kb = static_cast<int>(num_blocks);
  const int m_i = static_cast<int>(m);
  const Q8_0Block* w_row[kPrefetch];
  float row_live[kPrefetch];
#pragma unroll
  for (int p = 0; p < kPrefetch; ++p) {
    const int r = static_cast<int>(r_block) + (((p * 256) + tid) / BK);
    const int r_clamped = (r < m_i) ? r : (m_i - 1);
    w_row[p] = w_blocks + (static_cast<std::size_t>(r_clamped) * num_blocks);
    row_live[p] = (r < m_i) ? 1.0F : 0.0F;
  }
  const auto* b_base = static_cast<const std::int8_t*>(x_blocks) +
                       ((tt_block + static_cast<std::size_t>(b_tile)) *
                        num_blocks * kQ8ActTileBytes);
  const bool b_live = b_tile < kTokTiles;

  const auto fetch_stage = [&](int kb0) {
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
      const int kb = kb0 + (((p * 256) + tid) % BK);
      const int kb_clamped = (kb < num_kb) ? kb : (num_kb - 1);
      const Q8_0Block& blk = w_row[p][kb_clamped];
      __builtin_memcpy(&r_q0[p], blk.qs + 0, 16);
      __builtin_memcpy(&r_q1[p], blk.qs + 16, 16);
      r_dw[p] = __half2float(blk.d) * ((kb < num_kb) ? row_live[p] : 0.0F);
    }
    if (b_live) {
#pragma unroll
      for (int i = 0; i < BK; ++i) {
        const int kb = kb0 + i;
        const int kb_clamped = (kb < num_kb) ? kb : (num_kb - 1);
        const auto* tile =
            b_base + (static_cast<std::size_t>(kb_clamped) * kQ8ActTileBytes);
        r_b[i] = reinterpret_cast<const int32x4_t*>(tile)[lane_id];
        r_dx[i] = ((kb < num_kb) ? 1.0F : 0.0F) *
                  reinterpret_cast<const float*>(
                      tile + kQ8ActScaleOffset)[lane_id & 15];
      }
    }
  };

  const auto commit_stage = [&]() {
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
      const int idx = (p * 256) + tid;
      const int rr = idx / BK;
      const int kk = idx % BK;
      const int rs = rr / 16;
      const int rl = rr % 16;
      s_a[kk][rs][rl] = r_q0[p];
      s_a[kk][rs][16 + rl] = r_q1[p];
      s_dw[kk][rs][(rl % 2 == 0) ? (rl / 2) : (8 + (rl / 2))] = r_dw[p];
    }
    if (b_live) {
#pragma unroll
      for (int i = 0; i < BK; ++i) {
        s_b[i][b_tile][lane_id] = r_b[i];
        if (lane_id < 16) {
          s_dx[i][b_tile][lane_id] = r_dx[i];
        }
      }
    }
  };

  fetch_stage(0);
  for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
    commit_stage();
    __syncthreads();
    if (kb0 + BK < num_kb) {
      fetch_stage(kb0 + BK);
    }

#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      int32x4_t a0[kWaveRowTiles];
      int32x4_t a1[kWaveRowTiles];
      float dw[kWaveRowTiles][8];
#pragma unroll
      for (int i = 0; i < kWaveRowTiles; ++i) {
        const int rs = (wave_row * kWaveRowTiles) + i;
        a0[i] = s_a[kb][rs][sub_lane];
        a1[i] = s_a[kb][rs][16 + sub_lane];
        const float4 lo =
            *reinterpret_cast<const float4*>(&s_dw[kb][rs][half_id * 8]);
        const float4 up =
            *reinterpret_cast<const float4*>(&s_dw[kb][rs][(half_id * 8) + 4]);
        dw[i][0] = lo.x;
        dw[i][1] = lo.y;
        dw[i][2] = lo.z;
        dw[i][3] = lo.w;
        dw[i][4] = up.x;
        dw[i][5] = up.y;
        dw[i][6] = up.z;
        dw[i][7] = up.w;
      }
      // Token-tile operands are read one tile at a time to keep the live
      // register set small enough for the BK=2 stage.
#pragma unroll
      for (int j = 0; j < kWaveTokTiles; ++j) {
        const int ts = (wave_tok * kWaveTokTiles) + j;
        const int32x4_t b0 = s_b[kb][ts][sub_lane];
        const int32x4_t b1 = s_b[kb][ts][16 + sub_lane];
        const float dx = s_dx[kb][ts][sub_lane];
#pragma unroll
        for (int i = 0; i < kWaveRowTiles; ++i) {
          int32x8_t c = {0, 0, 0, 0, 0, 0, 0, 0};
          c = WmmaI8(a0[i], b0, c);
          c = WmmaI8(a1[i], b1, c);
#pragma unroll
          for (int l = 0; l < 8; ++l) {
            acc[i][j][l] += (dw[i][l] * dx) * static_cast<float>(c[l]);
          }
        }
      }
#if __clang_major__ >= 23
      // LLVM 23 hoists the next K block's operands above this one's WMMAs,
      // which spills registers on gfx1151. ROCm 7.2.3 (LLVM 22) does not, and
      // the fence costs it about 3%.
      __builtin_amdgcn_sched_barrier(0);
#endif
    }
    __syncthreads();
  }

  // Transpose the result through LDS, two row tiles at a time, so every
  // global store covers 32 consecutive rows of one token: a full 128-byte
  // line (half lines cost a read-modify-write on the fabric).
  __syncthreads();
  constexpr unsigned kOutputStride = 32;
  static_assert(8 * 16 * kOutputStride * sizeof(float) <= sizeof(lds));
  float* tile_scratch =
      reinterpret_cast<float*>(lds) + wave_id * 16 * kOutputStride;
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; i += 2) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
      // scratch[token][row] over 16 tokens x 32 rows.
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        tile_scratch[(sub_lane * kOutputStride) + (2 * l) + half_id] =
            acc[i][j][l];
        tile_scratch[(sub_lane * kOutputStride) + 16 + (2 * l) + half_id] =
            acc[i + 1][j][l];
      }
      __builtin_amdgcn_wave_barrier();
      const std::size_t r0 =
          r_block +
          static_cast<std::size_t>((((wave_row * kWaveRowTiles) + i) * 16));
      const std::size_t t0 =
          t_block +
          static_cast<std::size_t>((((wave_tok * kWaveTokTiles) + j) * 16));
      // Lane pair (2p, 2p + 1) stores token p's 32 rows as eight float4.
      const int tok_l = lane_id >> 1;
      const int row_l = (lane_id & 1) * 16;
      const std::size_t tok = t0 + static_cast<std::size_t>(tok_l);
      const auto* src = reinterpret_cast<const float4*>(
          tile_scratch + (tok_l * kOutputStride) + row_l);
      if (tok < batch && r0 + 32 <= m) {
        auto* dst = reinterpret_cast<float4*>(y + (tok * m) + r0 +
                                              static_cast<std::size_t>(row_l));
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          dst[q] = src[q];
        }
      } else if (tok < batch) {
#pragma unroll
        for (int q = 0; q < 16; ++q) {
          const std::size_t r = r0 + static_cast<std::size_t>(row_l + q);
          if (r < m) {
            const float value =
                tile_scratch[(tok_l * kOutputStride) + row_l + q];
            y[(tok * m) + r] = value;
          }
        }
      }
      __builtin_amdgcn_wave_barrier();
    }
  }
}

// Routed expert GEMMs: assignment rows are compacted by expert with every
// bucket padded to a 16-row tile (`pad_bounds`), so a token tile never
// straddles experts; `rows_out` maps a compact row to its (token, slot)
// output row or -1 for padding.
constexpr std::size_t kRoutedTileTokens = 16;

struct Q4KBlock {
  __half d;
  __half dmin;
  std::uint8_t scales[12];
  std::uint8_t qs[128];
};
static_assert(sizeof(Q4KBlock) == 144, "block_q4_K must be 144 bytes");

struct Q5_1Block {
  __half d;
  __half m;
  std::uint32_t qh;
  std::uint8_t qs[16];
};
static_assert(sizeof(Q5_1Block) == 24, "block_q5_1 must be 24 bytes");

/// Byte `i` of the 16-byte block header (d, dmin, scales[12]).
__device__ __forceinline__ std::uint32_t HeaderByte(const uint4& h,
                                                    std::uint32_t i) {
  const std::uint32_t word = i < 4 ? h.x : i < 8 ? h.y : i < 12 ? h.z : h.w;
  return (word >> (8U * (i & 3U))) & 0xFFU;
}

// Routed F16 WMMA expert GEMM. The int8 kernel above pays a float epilogue
// and an activation-sum correction every K block because the per-32 scales
// of both operands sit outside the integer dot product; here the weights are
// dequantized to F16 right after the LDS read (each wave decodes only its own
// sixteen rows) and the activations are F16 rows, so the matrix core
// accumulates the whole K extent in F32 with no per-block work. The codes
// stay packed in LDS (4 bits for Q4_K, 4 + 1 for Q5_1), which keeps a
// two-K-block stage at 11-12 KB and five blocks resident per WGP.
//
// A code becomes a half through a byte permute into the mantissa of 1024.0
// (0x6400 | q = 1024 + q exactly for q < 32, the half's unit being 1 there),
// a packed subtract of 1024 (exact), then one packed FMA:
//
//     w = q * scale + bias
//
// with (scale, bias) = (d * sc, -dmin * mn) for Q4_K and (d, m) for Q5_1,
// staged per (row, K block) as a half2.
//
// grid (m / BM, tiles): `tiles[y]` packs the expert in the low 16 bits and
// the token macro tile index in the high 16, so no block is launched for an
// empty tile; the row blocks of one tile are consecutive in dispatch order so
// they share the tile's gathered activations through L2. Block (x, y)
// computes rows x*BM.. of the expert against its
// compact rows [pad_bounds[e] + j*BN, +BN) and scatters them to
// out[rows_out[c]][row] (F32, or F16 with the SwiGLU applied when `out_half`
// is given: the up projection then writes the down projection's input).
constexpr std::uint32_t kHalfMagic = 0x64646464U;  // 1024.0 high bytes

/// block_q5_K: the Q4_K header, 32 high-bit bytes (bit s of byte j is the
/// fifth bit of element j of K block s), then the Q4_K nibble layout.
constexpr std::size_t kQ5KBlockBytes = 176;

template<WeightType kType>
__device__ __forceinline__ std::size_t RoutedF16RowBytes(std::size_t k) {
  return kType == WeightType::kQ4_K   ? (k / 256) * sizeof(Q4KBlock)
         : kType == WeightType::kQ5_K ? (k / 256) * kQ5KBlockBytes
         : kType == WeightType::kQ5_1 ? (k / 32) * sizeof(Q5_1Block)
                                      : (k / 32) * sizeof(Q8_0Block);
}

/// Bit `s` of each of the four bytes of `w`, packed into bits 0-3.
__device__ __forceinline__ std::uint32_t GatherBit(std::uint32_t w, int s) {
  // 0x01020408 moves byte b's bit to bit 24 + b.
  return (((w >> s) & 0x01010101U) * 0x01020408U) >> 24U;
}

/// Four packed 5-bit codes: `nib` holds the 4-bit parts one per byte, `bits`
/// bits j..j+3 of the Q5_1 high-bit word, spread to bit 4 of each byte.
__device__ __forceinline__ std::uint32_t SpreadHighBits(std::uint32_t bits) {
  // 0x00204081 = 1 + 2^7 + 2^14 + 2^21: bit b of `bits` lands at 8b, every
  // cross term falls off the 0x01010101 mask.
  return (__umul24(bits, 0x00204081U) & 0x01010101U) << 4U;
}

/// Four halves from four code bytes: 1024 + q as F16, minus `magic` (1024,
/// or 1152 for a signed byte carried as q + 128), then the affine.
__device__ __forceinline__ void CodesToHalves(std::uint32_t codes,
                                              __half2 magic, __half2 scale2,
                                              __half2 bias2, __half2& lo,
                                              __half2& hi) {
  const std::uint32_t p0 =
      __builtin_amdgcn_perm(codes, kHalfMagic, 0x01050004U);
  const std::uint32_t p1 =
      __builtin_amdgcn_perm(codes, kHalfMagic, 0x03070206U);
  lo = __hfma2(__hadd2(__builtin_bit_cast(__half2, p0), magic), scale2, bias2);
  hi = __hfma2(__hadd2(__builtin_bit_cast(__half2, p1), magic), scale2, bias2);
}

template<WeightType kType, int BM, int BN, int BK, bool kPair = false>
__launch_bounds__(256) __global__
    void RoutedF16GEMMKernel(const void* __restrict__ w,
                             const __half* __restrict__ x,
                             const std::int32_t* __restrict__ tiles,
                             const std::int32_t* __restrict__ pad_bounds,
                             const std::int32_t* __restrict__ rows_in,
                             const std::int32_t* __restrict__ rows_out,
                             const float* __restrict__ swiglu_gate,
                             float* __restrict__ out,
                             __half* __restrict__ out_half, std::size_t m,
                             std::size_t k, const void* __restrict__ w_up) {
  static_assert(BM == 128 || BM == 256, "eight waves, 16-row tiles");
  static_assert(!kPair || BM == 128);
  static_assert(BN % 16 == 0 && BN / 16 <= 8);
  static_assert(BK == 2, "one stage is one 32-byte Q4_K nibble group");
  constexpr int kTokTiles = BN / 16;
  constexpr int kWaveRowTiles = BM / 128;  // 16-row tiles per wave
  constexpr bool kQ5 = kType == WeightType::kQ5_1;
  constexpr bool kQ5K = kType == WeightType::kQ5_K;
  constexpr bool kQ8 = kType == WeightType::kQ8_0;
  constexpr bool kKQuant = kType == WeightType::kQ4_K || kQ5K;
  // 16-byte code chunks per row and stage: Q4_K's nibble pair and Q5_1's
  // two nibble blocks are two, Q8_0's two byte blocks are four.
  constexpr int kChunks = kQ8 ? 2 * BK : BK;

  // LDS plan (bytes): the code plane holds BM rows x kChunks 16-byte chunks
  // with the chunks of nearby rows permuted so a fragment read (one row per
  // lane) covers all bank groups; the activation plane is
  // [kb][16-element quarter][token][16 B] so a fragment read is 256
  // contiguous bytes; the epilogue reuses it all.
  constexpr int kCodeBytes = BM * kChunks * 16;
  constexpr int kHighBytes = (kQ5 || kQ5K) ? BK * BM * 4 : 0;
  constexpr int kScaleBytes = BK * BM * 4;
  // One slot of padding per activation quarter plane: the eight chunks of
  // a token then land on eight bank groups when they are written.
  constexpr int kActStride = BN + 1;
  constexpr int kActBytes = BK * 4 * kActStride * 16;
  // The epilogue transposes one 16x16 tile per wave through the same
  // bytes (8 KB), which the narrow tile's stages do not reach.
  constexpr int kStageBytes = kCodeBytes + kHighBytes + kScaleBytes + kActBytes;
  constexpr int kLdsBytes = kStageBytes > 8 * 1024 ? kStageBytes : 8 * 1024;
  __shared__ __attribute__((aligned(16))) std::uint8_t lds[kLdsBytes];
  auto* s_codes = reinterpret_cast<uint4*>(lds);
  auto* s_high = reinterpret_cast<std::uint32_t*>(lds + kCodeBytes);
  auto* s_scale =
      reinterpret_cast<std::uint32_t*>(lds + kCodeBytes + kHighBytes);
  auto* s_act =
      reinterpret_cast<uint4*>(lds + kCodeBytes + kHighBytes + kScaleBytes);

  const std::int32_t tile = tiles[blockIdx.y];
  const int expert = tile & 0xFFFF;
  const int t_local = (tile >> 16) * BN;
  const int bucket_begin = pad_bounds[expert];
  const int bucket_rows = pad_bounds[expert + 1] - bucket_begin;
  const int live_tok_tiles =
      std::min(kTokTiles, (bucket_rows - t_local + 15) / 16);
  const int num_kb = static_cast<int>(k / 32);
  const int m_i = static_cast<int>(m);
  const std::size_t row_bytes = RoutedF16RowBytes<kType>(k);
  const auto* w_expert = static_cast<const std::uint8_t*>(w) +
                         static_cast<std::size_t>(expert) * m * row_bytes;

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  constexpr int kRows = kPair ? BM / 2 : BM;
  const int r_block = static_cast<int>(blockIdx.x) * kRows;

  // Weight fetch: unit u of a thread is (row = tid / 2 + 128 u, chunk c =
  // tid % 2). Q4_K: the two 16-byte halves of one 32-byte nibble group (two
  // K blocks, low and high nibbles); Q5_1: one 24-byte K block each.
  const int f_c = tid & 1;
  const std::uint8_t* f_ptr[kWaveRowTiles];
  bool f_live[kWaveRowTiles];
  uint4 f_header[kWaveRowTiles];
#pragma unroll
  for (int u = 0; u < kWaveRowTiles; ++u) {
    const int r =
        r_block + (kPair ? (tid >> 1) % kRows : (tid >> 1) + (u * 128));
    f_live[u] = r < m_i;
    const std::uint8_t* weights = w_expert;
    if constexpr (kPair) {
      if ((tid >> 1) >= kRows) {
        weights = static_cast<const std::uint8_t*>(w_up) +
                  static_cast<std::size_t>(expert) * m * row_bytes;
      }
    }
    f_ptr[u] = weights +
               static_cast<std::size_t>(f_live[u] ? r : (m_i - 1)) * row_bytes;
    f_header[u] = make_uint4(0u, 0u, 0u, 0u);
  }
  // The next stage's weights and activations, fetched one stage ahead. The
  // (scale, bias) pair is derived from the raw header word only when the
  // stage is committed, so nothing waits on the loads before the compute.
  uint4 f_codes[kWaveRowTiles];
  // Paired gate/up tiles reuse the full quantized block over four stages.
  // Keep its remaining codes in registers alongside the cached header.
  uint4 code_cache[kWaveRowTiles][4];
  uint4 f_codes_hi[kWaveRowTiles];  ///< Q8_0: the block's second 16 codes
  uint4 f_qh[kWaveRowTiles][2];     ///< Q5_K: the superblock's high bits
  std::uint32_t f_high[kWaveRowTiles];
  std::uint32_t f_dm[kWaveRowTiles];  ///< Q5_1: d | m; Q8_0: d
  int f_sb32[kWaveRowTiles];          ///< Q4_K: the K block in its superblock
  constexpr int kActFetch = BN <= 64 ? 2 : 4;
  uint4 a_data[kActFetch];

  // Activation fetch: BN tokens x (BK * 64) bytes per stage in 16-byte
  // chunks, eight per token; each thread fetches consecutive 256-chunk
  // strides, up to four for a 128-token tile.
  constexpr int kActChunks = BN * BK * 4;
  static_assert(kActChunks <= kActFetch * 256);
  const __half* a_src[kActFetch];
  int a_slot[kActFetch];
#pragma unroll
  for (int i = 0; i < kActFetch; ++i) {
    const int chunk = tid + (i * 256);
    const int t = chunk / (BK * 4);
    const int sub = chunk % (BK * 4);
    const int c_row = t_local + t;
    const std::int32_t src = (chunk < kActChunks && c_row < bucket_rows)
                                 ? rows_in[bucket_begin + c_row]
                                 : -1;
    a_src[i] = src >= 0 ? x + (static_cast<std::size_t>(src) * k) + (sub * 8)
                        : nullptr;
    // s_act[(kb * 4 + quarter) * kActStride + t]
    a_slot[i] = chunk < kActChunks ? (sub * kActStride) + t : -1;
  }

  const auto swizzle = [](int row, int c) {
    return (row * kChunks) +
           (c ^ (kChunks == 4 ? ((row >> 1) & 3) : ((row >> 2) & 1)));
  };

  const auto fetch_stage = [&](int kb0) {
#pragma unroll
    for (int u = 0; u < kWaveRowTiles; ++u) {
      if constexpr (kQ5) {
        const int kb = kb0 + f_c;
        const auto* words = reinterpret_cast<const uint2*>(f_ptr[u]) + (kb * 3);
        const uint2 w0 = words[0];
        const uint2 w1 = words[1];
        const uint2 w2 = words[2];
        f_codes[u] = make_uint4(w1.x, w1.y, w2.x, w2.y);
        f_high[u] = w0.y;
        f_dm[u] = w0.x;
      } else if constexpr (kQ8) {
        // block_q8_0 is 34 bytes, so the code loads are 2-byte aligned.
        const auto* blk = f_ptr[u] + ((kb0 + f_c) * 34);
        f_dm[u] = *reinterpret_cast<const std::uint16_t*>(blk);
        __builtin_memcpy(&f_codes[u], blk + 2, 16);
        __builtin_memcpy(&f_codes_hi[u], blk + 18, 16);
      } else {
        constexpr int kBlockChunks = kQ5K ? 11 : 9;
        constexpr int kCodeChunk = kQ5K ? 3 : 1;
        const int block = kb0 / 8;
        const auto* blk =
            reinterpret_cast<const uint4*>(f_ptr[u]) + (block * kBlockChunks);
        // The K sweep enters a new superblock every eight Q8-sized blocks.
        if (kb0 % 8 == 0) {
          f_header[u] = blk[0];
          if constexpr (kPair) {
#pragma unroll
            for (int group = 0; group < 4; ++group)
              code_cache[u][group] = blk[kCodeChunk + group * 2 + f_c];
          }
          if constexpr (kQ5K) {
            f_qh[u][0] = blk[1];
            f_qh[u][1] = blk[2];
          }
        }
        const int sb32 = (kb0 % 8) + f_c;
        if constexpr (kPair) {
          const int group = sb32 / 2;
          f_codes[u] = group == 0   ? code_cache[u][0]
                       : group == 1 ? code_cache[u][1]
                       : group == 2 ? code_cache[u][2]
                                    : code_cache[u][3];
        } else {
          f_codes[u] = blk[kCodeChunk + (sb32 / 2) * 2 + f_c];
        }
        f_sb32[u] = sb32;
        if constexpr (kQ5K) {
          // Bit sb32 of the 32 high-bit bytes, packed as the Q5_1 word.
          const std::uint32_t qh[8] = {f_qh[u][0].x, f_qh[u][0].y, f_qh[u][0].z,
                                       f_qh[u][0].w, f_qh[u][1].x, f_qh[u][1].y,
                                       f_qh[u][1].z, f_qh[u][1].w};
          std::uint32_t high = 0;
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            high |= GatherBit(qh[i], sb32) << (4 * i);
          }
          f_high[u] = high;
        }
      }
    }
#pragma unroll
    for (int i = 0; i < kActFetch; ++i) {
      a_data[i] = a_src[i] != nullptr
                      ? *reinterpret_cast<const uint4*>(a_src[i] + (kb0 * 32))
                      : make_uint4(0u, 0u, 0u, 0u);
    }
  };

  const auto commit_stage = [&]() {
#pragma unroll
    for (int u = 0; u < kWaveRowTiles; ++u) {
      const int row = (tid >> 1) + (u * 128);
      std::uint32_t scale_bias = 0;
      if constexpr (kQ8) {
        s_codes[swizzle(row, 2 * f_c)] = f_codes[u];
        s_codes[swizzle(row, (2 * f_c) + 1)] = f_codes_hi[u];
        scale_bias = f_live[u] ? f_dm[u] : 0U;  // half2 (d, 0)
      } else {
        s_codes[swizzle(row, f_c)] = f_codes[u];
      }
      if constexpr (kQ5K) {
        s_high[(f_c * BM) + row] = f_high[u];
      }
      if constexpr (kQ8) {
      } else if constexpr (kQ5) {
        s_high[(f_c * BM) + row] = f_high[u];
        const __half2 dm = __builtin_bit_cast(__half2, f_dm[u]);
        const float d = f_live[u] ? __low2float(dm) : 0.0F;
        const float mn = f_live[u] ? __high2float(dm) : 0.0F;
        scale_bias =
            __builtin_bit_cast(std::uint32_t, __floats2half2_rn(d, mn));
      } else {
        const int sb32 = f_sb32[u];
        std::uint32_t sc = 0;
        std::uint32_t mn = 0;
        if (sb32 < 4) {
          sc = HeaderByte(f_header[u], 4 + sb32) & 0x3FU;
          mn = HeaderByte(f_header[u], 8 + sb32) & 0x3FU;
        } else {
          sc = (HeaderByte(f_header[u], 8 + sb32) & 0x0FU) |
               ((HeaderByte(f_header[u], sb32) >> 6U) << 4U);
          mn = (HeaderByte(f_header[u], 8 + sb32) >> 4U) |
               ((HeaderByte(f_header[u], 4 + sb32) >> 6U) << 4U);
        }
        const __half2 dm = __builtin_bit_cast(__half2, f_header[u].x);
        const float scale =
            f_live[u] ? __low2float(dm) * static_cast<float>(sc) : 0.0F;
        const float offset =
            f_live[u] ? __high2float(dm) * static_cast<float>(mn) : 0.0F;
        scale_bias = __builtin_bit_cast(std::uint32_t,
                                        __floats2half2_rn(scale, -offset));
      }
      s_scale[(f_c * BM) + row] = scale_bias;
    }
#pragma unroll
    for (int i = 0; i < kActFetch; ++i) {
      if (a_slot[i] >= 0) {
        s_act[a_slot[i]] = a_data[i];
      }
    }
  };

  v8f acc[kWaveRowTiles][kTokTiles];
#pragma unroll
  for (int u = 0; u < kWaveRowTiles; ++u) {
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) {
      acc[u][j] = v8f{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    }
  }

  const __half2 magic =
      __floats2half2_rn(kQ8 ? -1152.0F : -1024.0F, kQ8 ? -1152.0F : -1024.0F);
  const auto compute_stage = [&]() {
    uint4 raw[kWaveRowTiles][BK];
    if constexpr (!kQ8) {
#pragma unroll
      for (int u = 0; u < kWaveRowTiles; ++u) {
        const int row = (wave_id * 16) + (u * 128) + sub_lane;
#pragma unroll
        for (int c = 0; c < BK; ++c) {
          raw[u][c] = s_codes[swizzle(row, c)];
        }
      }
    }
#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      v16h a_lo[kWaveRowTiles];
      v16h a_hi[kWaveRowTiles];
#pragma unroll
      for (int u = 0; u < kWaveRowTiles; ++u) {
        const int row = (wave_id * 16) + (u * 128) + sub_lane;
        const __half2 sb =
            __builtin_bit_cast(__half2, s_scale[(kb * BM) + row]);
        const __half2 scale2 = __low2half2(sb);
        const __half2 bias2 = __high2half2(sb);
        std::uint32_t nib[8];
        if constexpr (kQ8) {
          // Q8_0: the block's 32 signed bytes are chunks 2 kb and 2 kb + 1;
          // flipping the sign bit carries q + 128, which the 1152 magic
          // takes back out.
          const uint4 c0 = s_codes[swizzle(row, 2 * kb)];
          const uint4 c1 = s_codes[swizzle(row, (2 * kb) + 1)];
          const std::uint32_t words[8] = {c0.x, c0.y, c0.z, c0.w,
                                          c1.x, c1.y, c1.z, c1.w};
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            nib[i] = words[i] ^ 0x80808080U;
          }
        } else if constexpr (kQ5) {
          // Q5_1: K block kb's 16 bytes are chunk kb; elements 0-15 take
          // the low nibbles, 16-31 the high, plus bit j of the high-bit
          // word.
          const uint4 r = raw[u][kb];
          const std::uint32_t high = s_high[(kb * BM) + row];
          const std::uint32_t words[4] = {r.x, r.y, r.z, r.w};
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            nib[i] = (words[i] & 0x0F0F0F0FU) |
                     SpreadHighBits((high >> (4 * i)) & 0xFU);
            nib[4 + i] = ((words[i] >> 4U) & 0x0F0F0F0FU) |
                         SpreadHighBits((high >> (16 + 4 * i)) & 0xFU);
          }
        } else {
          // Q4_K / Q5_K: elements 0-15 of K block kb0 + kb are the low
          // (kb = 0) or high (kb = 1) nibbles of chunk 0, elements 16-31
          // of chunk 1; Q5_K adds bit j of the staged high-bit word.
          const unsigned shift = 4U * static_cast<unsigned>(kb);
          const std::uint32_t words[8] = {raw[u][0].x, raw[u][0].y, raw[u][0].z,
                                          raw[u][0].w, raw[u][1].x, raw[u][1].y,
                                          raw[u][1].z, raw[u][1].w};
          const std::uint32_t high = kQ5K ? s_high[(kb * BM) + row] : 0U;
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            nib[i] = (words[i] >> shift) & 0x0F0F0F0FU;
            if constexpr (kQ5K) {
              nib[i] |= SpreadHighBits((high >> (4 * i)) & 0xFU);
            }
          }
        }
        __half2 h[16];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          CodesToHalves(nib[i], magic, scale2, bias2, h[2 * i], h[2 * i + 1]);
        }
        __builtin_memcpy(&a_lo[u], &h[0], 32);
        __builtin_memcpy(&a_hi[u], &h[8], 32);
      }
#pragma unroll
      for (int j = 0; j < kTokTiles; ++j) {
        if constexpr (kPair || ((kQ5 || kQ8) && BN >= 48)) {
          // Keep one token tile's LDS fragments live at a time. Hoisting
          // all eight tiles spills registers and defeats the wider tile's
          // reuse of each weight decode. This is a compiler barrier only.
          asm volatile("" ::: "memory");
        }
        // A short expert bucket has no output in the remaining token
        // tiles, so omit their WMMA work.
        if constexpr ((kPair || ((kQ5 || kQ8) && BN >= 48)) && kTokTiles > 1) {
          if (j >= live_tok_tiles)
            continue;
        }
        const uint4* frag =
            s_act + ((kb * 4) * kActStride) + (j * 16) + sub_lane;
        uint4 b[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          b[q] = frag[q * kActStride];
        }
        v16h b_lo;
        v16h b_hi;
        __builtin_memcpy(&b_lo, &b[0], 32);
        __builtin_memcpy(&b_hi, &b[2], 32);
#pragma unroll
        for (int u = 0; u < kWaveRowTiles; ++u) {
          acc[u][j] = Wmma(a_lo[u], b_lo, acc[u][j]);
          acc[u][j] = Wmma(a_hi[u], b_hi, acc[u][j]);
        }
      }
    }
  };

  fetch_stage(0);
  for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
    commit_stage();
    __syncthreads();
    if (kb0 + BK < num_kb) {
      fetch_stage(kb0 + BK);
    }
    compute_stage();
    __syncthreads();
  }

  if constexpr (kPair) {
    // Four waves compute gate rows and four compute the matching up rows.
    // Pair them in the existing LDS allocation, keeping the K accumulation
    // order and avoiding the gate's F32 write/read between projections.
    // Two padding floats keep the accumulator scatter off repeated banks.
    constexpr unsigned stride = 18;
    constexpr unsigned plane = 16 * stride;
    static_assert(8 * plane * sizeof(float) <= kLdsBytes);
    float* base = reinterpret_cast<float*>(lds);
    float* scratch = base + wave_id * plane;
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        scratch[sub_lane * stride + 2 * l + half_id] = acc[0][j][l];
      }
      __syncthreads();
#pragma unroll
      for (int unit = 0; unit < 2; ++unit) {
        const int flat = (unit * 256 + tid) * 2;
        const int t = t_local + j * 16 + flat / kRows;
        const int r = flat % kRows;
        if (t < bucket_rows && r_block + r < m_i) {
          const std::int32_t dst = rows_out[bucket_begin + t];
          if (dst >= 0) {
            const int idx = (r / 16) * plane + (flat / kRows) * stride + r % 16;
            // Preserve the separate projection epilogue's F32 evaluation
            // order before narrowing. Fast-math can otherwise regroup the
            // products and change an F16 rounding tie.
            __half values[2];
#pragma unroll
            for (int v = 0; v < 2; ++v) {
              float product = base[idx + v + 4 * plane] * base[idx + v];
              asm volatile("" : "+v"(product));
              float value = product * SigmoidF(base[idx + v]);
              asm volatile("" : "+v"(value));
              values[v] = __float2half(value);
            }
            const auto offset = static_cast<std::size_t>(dst) * m + r_block + r;
            if (m % 2 == 0 && r_block + r + 1 < m_i) {
              *reinterpret_cast<__half2*>(out_half + offset) =
                  __halves2half2(values[0], values[1]);
            } else {
              out_half[offset] = values[0];
              if (r_block + r + 1 < m_i)
                out_half[offset + 1] = values[1];
            }
          }
        }
      }
      __syncthreads();
    }
    return;
  }

  // One wave writes a complete 128-byte line of F16 output. Padding the
  // shared row by two floats also makes the accumulator scatter conflict-free.
  // Narrow buckets keep the lighter wave-local epilogue below.
  if constexpr (BN >= 48) {
    static_assert(kLdsBytes >= 16 * (BM + 2) * sizeof(float));
    if (out_half != nullptr) {
      constexpr unsigned stride = BM + 2;
      float* scratch = reinterpret_cast<float*>(lds);
#pragma unroll
      for (int j = 0; j < kTokTiles; ++j) {
#pragma unroll
        for (int u = 0; u < kWaveRowTiles; ++u) {
#pragma unroll
          for (int l = 0; l < 8; ++l)
            scratch[sub_lane * stride + wave_id * 16 + u * 128 + 2 * l +
                    half_id] = acc[u][j][l];
        }
        __syncthreads();
#pragma unroll
        for (int round = 0; round < 16 * BM / (256 * 2); ++round) {
          const unsigned flat = (round * 256 + tid) * 2;
          const unsigned tr = flat / BM, row = flat % BM;
          const unsigned t = t_local + j * 16 + tr, r = r_block + row;
          if (t < unsigned(bucket_rows) && r < m) {
            const int dst = rows_out[bucket_begin + t];
            if (dst >= 0) {
              float2 v =
                  *reinterpret_cast<const float2*>(scratch + tr * stride + row);
              const size_t o = size_t(dst) * m + r;
              if (swiglu_gate != nullptr) {
                v.x *= SiluF(swiglu_gate[o]);
                if (r + 1 < m)
                  v.y *= SiluF(swiglu_gate[o + 1]);
              }
              if (m % 2 == 0 && r + 1 < m)
                *reinterpret_cast<__half2*>(out_half + o) =
                    __floats2half2_rn(v.x, v.y);
              else {
                out_half[o] = __float2half(v.x);
                if (r + 1 < m)
                  out_half[o + 1] = __float2half(v.y);
              }
            }
          }
        }
        __syncthreads();
      }
      return;
    }
  }
  // Transpose each 16x16 tile through LDS, then scatter the 16 rows of each
  // token to its output row.
  float* tile_scratch = reinterpret_cast<float*>(lds) + (wave_id * 256);
#pragma unroll
  for (int u = 0; u < kWaveRowTiles; ++u) {
    const int r0 = r_block + (wave_id * 16) + (u * 128);
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        tile_scratch[(sub_lane * 16) + (2 * l) + half_id] = acc[u][j][l];
      }
      __builtin_amdgcn_wave_barrier();
      const int t0 = t_local + (j * 16);
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const int flat = (s * 32) + lane_id;
        const int t = t0 + (flat >> 4);
        const int r = r0 + (flat & 15);
        if (t < bucket_rows && r < m_i) {
          const std::int32_t dst = rows_out[bucket_begin + t];
          if (dst >= 0) {
            const std::size_t o = (static_cast<std::size_t>(dst) * m) +
                                  static_cast<std::size_t>(r);
            float v = tile_scratch[flat];
            if (out_half != nullptr) {
              out_half[o] = __float2half(
                  swiglu_gate != nullptr ? v * SiluF(swiglu_gate[o]) : v);
            } else {
              out[o] = v;
            }
          }
        }
      }
      __builtin_amdgcn_wave_barrier();
    }
  }
}

/// Compacts the routed assignments by expert with 16-row padded buckets.
/// One block: exclusive scan of the padded counts into pad_bounds[0..E].
__global__ void RoutedPadBoundsKernel(const std::uint32_t* __restrict__ counts,
                                      std::int32_t* __restrict__ pad_bounds,
                                      std::int32_t* __restrict__ cursors,
                                      std::uint32_t n_experts) {
  __shared__ std::int32_t padded[1024];
  for (std::uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
    padded[e] = static_cast<std::int32_t>((counts[e] + 15u) / 16u * 16u);
    cursors[e] = 0;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    std::int32_t running = 0;
    for (std::uint32_t e = 0; e < n_experts; ++e) {
      pad_bounds[e] = running;
      running += padded[e];
    }
    pad_bounds[n_experts] = running;
  }
}

/// rows_token[c] / rows_slot[c] for every routed (token, slot); the order
/// inside a bucket is whatever the atomics produce, which changes nothing:
/// every output row is computed from its own inputs only.
__global__ void RoutedScatterKernel(const std::int32_t* __restrict__ ids,
                                    const std::int32_t* __restrict__ pad_bounds,
                                    std::int32_t* __restrict__ cursors,
                                    std::int32_t* __restrict__ rows_token,
                                    std::int32_t* __restrict__ rows_slot,
                                    std::uint32_t slots, std::uint32_t k) {
  const std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= slots) {
    return;
  }
  const std::int32_t e = ids[slot];
  if (e < 0) {
    return;
  }
  const std::int32_t c = pad_bounds[e] + atomicAdd(&cursors[e], 1);
  rows_token[c] = static_cast<std::int32_t>(slot / k);
  rows_slot[c] = static_cast<std::int32_t>(slot);
}

}  // namespace

void EmbedTokens(const void* table, WeightType type, const std::int32_t* tokens,
                 float* res, std::uint32_t n_tokens, std::uint32_t hidden,
                 std::uint32_t streams, hipStream_t stream) {
  hipLaunchKernelGGL(EmbedKernel, dim3(n_tokens), dim3(kThreads), 0, stream,
                     table, type, tokens, res, hidden, streams);
}

void RmsNormRows(const float* x, const float* gamma, float* out,
                 std::uint32_t n_rows, std::uint32_t dim, std::uint32_t groups,
                 float eps, hipStream_t stream) {
  // Every (row, group) pair is one block; the kernel recovers the group
  // from the block index to pick its gamma slice.
  hipLaunchKernelGGL(RmsNormKernel, dim3(n_rows * groups), dim3(kThreads), 0,
                     stream, x, gamma, out, dim / groups, groups, eps);
}

/// One thread per (row, block): block_q6_K fields of a row regrouped by
/// field (see Q6KRowDot).
__global__ void RepackQ6KRowsKernel(const std::uint8_t* __restrict__ src,
                                    std::uint8_t* __restrict__ dst,
                                    std::uint32_t rows, std::uint32_t blocks) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i >= static_cast<std::size_t>(rows) * blocks)
    return;
  const std::size_t row = i / blocks;
  const std::uint32_t sb = i % blocks;
  const std::uint8_t* b = src + i * 210;
  std::uint8_t* r = dst + row * blocks * 210;
  for (std::uint32_t j = 0; j < 128; ++j)
    r[sb * 128 + j] = b[j];
  for (std::uint32_t j = 0; j < 64; ++j)
    r[blocks * 128 + sb * 64 + j] = b[128 + j];
  for (std::uint32_t j = 0; j < 16; ++j)
    r[blocks * 192 + sb * 16 + j] = b[192 + j];
  r[blocks * 208 + sb * 2] = b[208];
  r[blocks * 208 + sb * 2 + 1] = b[209];
}

bool RepackQ6KRows(void* w, std::uint32_t rows, std::uint32_t k) {
  if (k % 512 != 0)
    return false;
  const std::uint32_t blocks = k / 256;
  const std::size_t bytes = static_cast<std::size_t>(rows) * blocks * 210;
  void* tmp = nullptr;
  if (hipMalloc(&tmp, bytes) != hipSuccess)
    return false;
  const std::size_t n = static_cast<std::size_t>(rows) * blocks;
  hipLaunchKernelGGL(RepackQ6KRowsKernel, dim3((n + 255) / 256), dim3(256), 0,
                     nullptr, static_cast<const std::uint8_t*>(w),
                     static_cast<std::uint8_t*>(tmp), rows, blocks);
  const bool ok =
      hipMemcpy(w, tmp, bytes, hipMemcpyDeviceToDevice) == hipSuccess;
  (void)hipFree(tmp);
  return ok && hipGetLastError() == hipSuccess;
}

bool DenseQ6KVec(const void* w, const float* x, float* y, std::uint32_t tokens,
                 std::uint32_t rows, std::uint32_t k, hipStream_t stream,
                 const void* up) {
  if (tokens == 0 || tokens > 8 || k % 512 != 0)
    return false;
  const auto* wb = static_cast<const std::uint8_t*>(w);
  const auto* ub = static_cast<const std::uint8_t*>(up);
  // Two rows per wave share each activation load of a multi-token batch
  // (measured with the weights out of cache: faster than one or four);
  // one token keeps one row per wave.
  const std::uint32_t per_wave =
      up == nullptr && tokens >= 2 && rows >= 2048 ? 2 : 1;
  const dim3 grid((rows + 8 * per_wave - 1) / (8 * per_wave));
  switch (tokens) {
#define Q6K_CASE(T)                                                     \
  case T:                                                               \
    if (per_wave == 2)                                                  \
      hipLaunchKernelGGL((DenseQ6KVecKernel<T, 2>), grid, dim3(256), 0, \
                         stream, wb, ub, x, y, rows, k);                \
    else                                                                \
      hipLaunchKernelGGL((DenseQ6KVecKernel<T, 1>), grid, dim3(256), 0, \
                         stream, wb, ub, x, y, rows, k);                \
    break;
    Q6K_CASE(1)
    Q6K_CASE(2)
    Q6K_CASE(3)
    Q6K_CASE(4)
    Q6K_CASE(5)
    Q6K_CASE(6)
    Q6K_CASE(7)
    Q6K_CASE(8)
#undef Q6K_CASE
  }
  return true;
}

__global__ void QuantizeRowsQ8_0Kernel(const __half* __restrict__ src,
                                       std::uint8_t* __restrict__ dst,
                                       std::size_t blocks) {
  const std::size_t b =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (b >= blocks)
    return;
  const __half* v = src + b * 32;
  float amax = 0.0F;
  for (int i = 0; i < 32; ++i)
    amax = fmaxf(amax, fabsf(__half2float(v[i])));
  const float d = amax / 127.0F;
  const float id = d != 0.0F ? 1.0F / d : 0.0F;
  std::uint8_t* out = dst + b * 34;
  *reinterpret_cast<__half*>(out) = __float2half_rn(d);
  for (int i = 0; i < 32; ++i)
    reinterpret_cast<std::int8_t*>(out + 2)[i] =
        static_cast<std::int8_t>(roundf(__half2float(v[i]) * id));
}

void QuantizeRowsQ8_0(const __half* src, void* dst, std::size_t count,
                      hipStream_t stream) {
  const std::size_t blocks = count / 32;
  hipLaunchKernelGGL(QuantizeRowsQ8_0Kernel, dim3(Blocks(blocks)),
                     dim3(kThreads), 0, stream, src,
                     static_cast<std::uint8_t*>(dst), blocks);
}

/// Launches DenseQ45KVecKernel<kBits, tokens> for tokens in [kTokens, 8].
template<unsigned kBits, std::uint32_t kTokens = 1>
void LaunchDenseQ45KVec(const std::uint8_t* w, const float* x, float* y,
                        std::uint32_t tokens, std::uint32_t rows,
                        std::uint32_t k, hipStream_t stream) {
  if (tokens == kTokens) {
    hipLaunchKernelGGL((DenseQ45KVecKernel<kBits, kTokens>),
                       dim3((rows + 7) / 8), dim3(256), 0, stream, w, x, y,
                       rows, k);
    return;
  }
  if constexpr (kTokens < 8)
    LaunchDenseQ45KVec<kBits, kTokens + 1>(w, x, y, tokens, rows, k, stream);
}

bool DenseQ45KVec(unsigned bits, const void* w, const float* x, float* y,
                  std::uint32_t tokens, std::uint32_t rows, std::uint32_t k,
                  hipStream_t stream) {
  if (tokens == 0 || tokens > 8 || k % 256 != 0 || (bits != 4 && bits != 5))
    return false;
  const auto* wb = static_cast<const std::uint8_t*>(w);
  if (bits == 4)
    LaunchDenseQ45KVec<4>(wb, x, y, tokens, rows, k, stream);
  else
    LaunchDenseQ45KVec<5>(wb, x, y, tokens, rows, k, stream);
  return true;
}

void AddRmsNormRows(float* res, const float* delta, const float* gamma,
                    float* out, std::uint32_t n_rows, std::uint32_t dim,
                    float eps, hipStream_t stream) {
  hipLaunchKernelGGL(AddRmsNormKernel, dim3(n_rows), dim3(kThreads), 0, stream,
                     res, delta, gamma, out, dim, eps);
}

void Swiglu(float* gate, const float* up, std::size_t count,
            hipStream_t stream) {
  hipLaunchKernelGGL(SwigluKernel, dim3(Blocks(count)), dim3(kThreads), 0,
                     stream, gate, up, count);
}

void SwigluHalf(const float* gate, const float* up, __half* out,
                std::size_t count, hipStream_t stream) {
  hipLaunchKernelGGL(SwigluHalfKernel, dim3(Blocks((count + 3) / 4)),
                     dim3(kThreads), 0, stream, gate, up, out, count);
}

bool SwigluQ8Tiled(const float* gate, const float* up, void* out_q8,
                   std::size_t n_rows, std::size_t k, hipStream_t stream) {
  if (k % 32 != 0) {
    return false;
  }
  const std::size_t chunks = n_rows * (k / 4);
  hipLaunchKernelGGL(SwigluQ8Kernel, dim3(Blocks(chunks)), dim3(kThreads), 0,
                     stream, gate, up, out_q8, n_rows, k);
  return true;
}

void SigmoidMul(float* x, const float* g, std::size_t count,
                hipStream_t stream) {
  hipLaunchKernelGGL(SigmoidMulKernel, dim3(Blocks(count)), dim3(kThreads), 0,
                     stream, x, g, count);
}

void NarrowActivations(const float* x, void* out, bool bf16, std::size_t count,
                       hipStream_t stream) {
  if (bf16) {
    hipLaunchKernelGGL(NarrowKernel<hip_bfloat16>, dim3(Blocks(count)),
                       dim3(kThreads), 0, stream, x,
                       static_cast<hip_bfloat16*>(out), count);
  } else {
    hipLaunchKernelGGL(NarrowKernel<__half>, dim3(Blocks(count)),
                       dim3(kThreads), 0, stream, x, static_cast<__half*>(out),
                       count);
  }
}

std::size_t Q8TiledBytes(std::size_t batch, std::size_t k) {
  // The GEMM stages whole 128-token macro tiles, so the buffer is sized to
  // the batch rounded up to one (the padding tiles are never read as
  // results, only staged).
  constexpr std::size_t kMacroTokens = 128;
  const std::size_t padded =
      (batch + kMacroTokens - 1) / kMacroTokens * kMacroTokens;
  return (padded / kQ8ActTileTokens) * (k / 32) * kQ8ActTileBytes;
}

void QuantizeQ8Tiled(const float* x, void* out, std::size_t batch,
                     std::size_t k, hipStream_t stream) {
  const std::size_t blocks = batch * (k / 32);
  if (batch * k >= 4096) {
    const std::size_t per_block = kThreads / 8;
    hipLaunchKernelGGL(QuantizeQ8TiledVec4Kernel,
                       dim3((blocks + per_block - 1) / per_block),
                       dim3(kThreads), 0, stream, x, out, batch, k);
    return;
  }
  const std::size_t waves = kThreads / 32;
  hipLaunchKernelGGL(QuantizeQ8TiledKernel, dim3((blocks + waves - 1) / waves),
                     dim3(kThreads), 0, stream, x, out, batch, k);
}

bool W8A8Gemm(const void* w, const void* x_tiled, float* out, std::size_t batch,
              std::size_t m, std::size_t k, hipStream_t stream) {
  if (m == 0 || k == 0 || batch == 0 || k % 32 != 0) {
    return false;
  }
  // Qwen's wave64 matrix kernel with four row groups improves the model's
  // large output projections while preserving every K32 accumulator update.
  if (batch >= 1024 && m == 2048 && k == 4096) {
    W8A8GemmWave64(w, x_tiled, out, batch, m, k, stream);
    return true;
  }
  // A 128-token macro tile is the throughput configuration; short chunks
  // would leave most of it idle and take the 64-token variant. A narrow
  // projection (at most 512 rows) gets 64-row tiles so it still fills the
  // device.
  constexpr int kBM = 128;
  if (m <= 512 && batch >= 96) {
    constexpr int kBN = 128;
    constexpr int kNarrowBM = 64;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kNarrowBM - 1) / kNarrowBM));
    hipLaunchKernelGGL((W8A8BlockedWmmaGEMMKernel<kNarrowBM, kBN, 4, 2, 4>),
                       grid, dim3(kThreads), 0, stream, w, x_tiled, out, batch,
                       m, k);
  } else if (batch >= 96) {
    constexpr int kBN = 128;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kBM - 1) / kBM));
    hipLaunchKernelGGL((W8A8BlockedWmmaGEMMKernel<kBM, kBN, 2, 4, 2>), grid,
                       dim3(kThreads), 0, stream, w, x_tiled, out, batch, m, k);
  } else {
    constexpr int kBN = 64;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kBM - 1) / kBM));
    hipLaunchKernelGGL((W8A8BlockedWmmaGEMMKernel<kBM, kBN, 4, 4, 2>), grid,
                       dim3(kThreads), 0, stream, w, x_tiled, out, batch, m, k);
  }
  return true;
}

std::size_t RoutedCompactRows(std::size_t slots, std::size_t n_experts) {
  return slots + (n_experts * (kRoutedTileTokens - 1));
}

void RoutedCompact(const std::int32_t* ids, const std::uint32_t* counts,
                   std::int32_t* pad_bounds, std::int32_t* cursors,
                   std::int32_t* rows_token, std::int32_t* rows_slot,
                   std::uint32_t n_tokens, std::uint32_t k,
                   std::uint32_t n_experts, hipStream_t stream) {
  const std::size_t slots = static_cast<std::size_t>(n_tokens) * k;
  const std::size_t rows = RoutedCompactRows(slots, n_experts);
  (void)hipMemsetAsync(rows_token, 0xFF, rows * sizeof(std::int32_t), stream);
  (void)hipMemsetAsync(rows_slot, 0xFF, rows * sizeof(std::int32_t), stream);
  hipLaunchKernelGGL(RoutedPadBoundsKernel, dim3(1), dim3(1024), 0, stream,
                     counts, pad_bounds, cursors, n_experts);
  hipLaunchKernelGGL(RoutedScatterKernel, dim3(Blocks(slots)), dim3(kThreads),
                     0, stream, ids, pad_bounds, cursors, rows_token, rows_slot,
                     static_cast<std::uint32_t>(slots), k);
}

template<int BN>
bool LaunchRoutedF16(const void* w, WeightType type, const __half* x,
                     const std::int32_t* tiles, std::uint32_t n_tiles,
                     const std::int32_t* pad_bounds,
                     const std::int32_t* rows_in, const std::int32_t* rows_out,
                     const float* swiglu_gate, float* out, __half* out_half,
                     std::size_t m, std::size_t k, hipStream_t stream) {
  constexpr int kBM = 128;
  constexpr int kBK = 2;
  const dim3 grid(static_cast<unsigned int>((m + kBM - 1) / kBM), n_tiles);
  switch (type) {
    case WeightType::kQ4_K:
      if constexpr (BN > 48) {
        return false;
      } else {
        hipLaunchKernelGGL(
            (RoutedF16GEMMKernel<WeightType::kQ4_K, kBM, BN, kBK>), grid,
            dim3(kThreads), 0, stream, w, x, tiles, pad_bounds, rows_in,
            rows_out, swiglu_gate, out, out_half, m, k, nullptr);
        return true;
      }
    case WeightType::kQ5_1:
      hipLaunchKernelGGL((RoutedF16GEMMKernel<WeightType::kQ5_1, kBM, BN, kBK>),
                         grid, dim3(kThreads), 0, stream, w, x, tiles,
                         pad_bounds, rows_in, rows_out, swiglu_gate, out,
                         out_half, m, k, nullptr);
      return true;
    case WeightType::kQ8_0:
      hipLaunchKernelGGL((RoutedF16GEMMKernel<WeightType::kQ8_0, kBM, BN, kBK>),
                         grid, dim3(kThreads), 0, stream, w, x, tiles,
                         pad_bounds, rows_in, rows_out, swiglu_gate, out,
                         out_half, m, k, nullptr);
      return true;
    case WeightType::kQ5_K:
      if constexpr (BN > 48) {
        return false;
      } else {
        hipLaunchKernelGGL(
            (RoutedF16GEMMKernel<WeightType::kQ5_K, kBM, BN, kBK>), grid,
            dim3(kThreads), 0, stream, w, x, tiles, pad_bounds, rows_in,
            rows_out, swiglu_gate, out, out_half, m, k, nullptr);
        return true;
      }
    default:
      return false;
  }
}

bool RoutedF16Gemm(const void* w, WeightType type, const __half* x,
                   const std::int32_t* tiles, std::uint32_t n_tiles,
                   std::uint32_t tile_rows, const std::int32_t* pad_bounds,
                   const std::int32_t* rows_in, const std::int32_t* rows_out,
                   const float* swiglu_gate, float* out, __half* out_half,
                   std::size_t m, std::size_t k, hipStream_t stream) {
  const std::size_t block_elems =
      (type == WeightType::kQ4_K || type == WeightType::kQ5_K) ? 256 : 64;
  if (m == 0 || k == 0 || k % block_elems != 0 || n_tiles == 0 ||
      (out_half == nullptr) == (out == nullptr)) {
    return false;
  }
  switch (tile_rows) {
    case 16:
      return LaunchRoutedF16<16>(w, type, x, tiles, n_tiles, pad_bounds,
                                 rows_in, rows_out, swiglu_gate, out, out_half,
                                 m, k, stream);
    case 48:
      return LaunchRoutedF16<48>(w, type, x, tiles, n_tiles, pad_bounds,
                                 rows_in, rows_out, swiglu_gate, out, out_half,
                                 m, k, stream);
    case 64:
      return LaunchRoutedF16<64>(w, type, x, tiles, n_tiles, pad_bounds,
                                 rows_in, rows_out, swiglu_gate, out, out_half,
                                 m, k, stream);
    default:
      return false;
  }
}

template<int BN>
bool LaunchRoutedGatedF16(const void* gate, const void* up, WeightType type,
                          const __half* x, const std::int32_t* tiles,
                          std::uint32_t n_tiles, const std::int32_t* pad_bounds,
                          const std::int32_t* rows_in,
                          const std::int32_t* rows_out, __half* out,
                          std::size_t m, std::size_t k, hipStream_t stream) {
  if (m == 0 || k == 0 || k % 256 != 0 || n_tiles == 0 || out == nullptr) {
    return false;
  }
  const dim3 grid(static_cast<unsigned int>((m + 63) / 64), n_tiles);
  switch (type) {
    case WeightType::kQ4_K:
      hipLaunchKernelGGL(
          (RoutedF16GEMMKernel<WeightType::kQ4_K, 128, BN, 2, true>), grid,
          dim3(kThreads), 0, stream, gate, x, tiles, pad_bounds, rows_in,
          rows_out, nullptr, nullptr, out, m, k, up);
      return true;
    case WeightType::kQ5_K:
      hipLaunchKernelGGL(
          (RoutedF16GEMMKernel<WeightType::kQ5_K, 128, BN, 2, true>), grid,
          dim3(kThreads), 0, stream, gate, x, tiles, pad_bounds, rows_in,
          rows_out, nullptr, nullptr, out, m, k, up);
      return true;
    default:
      return false;
  }
}

bool RoutedGatedF16Gemm(const void* gate, const void* up, WeightType type,
                        const __half* x, const std::int32_t* tiles,
                        std::uint32_t n_tiles, std::uint32_t tile_rows,
                        const std::int32_t* pad_bounds,
                        const std::int32_t* rows_in,
                        const std::int32_t* rows_out, __half* out,
                        std::size_t m, std::size_t k, hipStream_t stream) {
  if (tile_rows == 128) {
    return LaunchRoutedGatedF16<128>(gate, up, type, x, tiles, n_tiles,
                                     pad_bounds, rows_in, rows_out, out, m, k,
                                     stream);
  }
  return tile_rows == 64 &&
         LaunchRoutedGatedF16<64>(gate, up, type, x, tiles, n_tiles, pad_bounds,
                                  rows_in, rows_out, out, m, k, stream);
}

// Keep the separate four-tap convolution's F32 rounding order when its
// inputs come from the projection's LDS tile.
__device__ __forceinline__ float SsmConv4Value(float4 w, float x0, float x1,
                                               float x2, float x3) {
  float acc = __fmaf_rn(w.x, x0, __fmul_rn(w.y, x1));
  acc = __fmaf_rn(w.z, x2, acc);
  acc = __fmaf_rn(w.w, x3, acc);
  return SiluF(acc);
}

constexpr unsigned kSsmProjectionTileTokens = 32;

// The fused projection leaves each 32-token tile's first and last three
// raw rows in qkv. Only the first three convolutions need another tile or
// the previous chunk's history; all other rows are produced in LDS.
__global__ void SsmConvBoundaryKernel(const float* qkv, const float* w,
                                      const float* history, float* out,
                                      std::uint32_t n_tokens,
                                      std::uint32_t channels,
                                      std::uint32_t stride) {
  const std::uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t t = blockIdx.y * kSsmProjectionTileTokens + blockIdx.z;
  if (c >= channels || t >= n_tokens) {
    return;
  }
  const float4 taps = *reinterpret_cast<const float4*>(w + c * 4);
  float v[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int src = static_cast<int>(t) - 3 + j;
    v[j] = src < 0 ? history[static_cast<std::size_t>(src + 3) * channels + c]
                   : qkv[static_cast<std::size_t>(src) * stride + c];
  }
  out[static_cast<std::size_t>(t) * channels + c] =
      SsmConv4Value(taps, v[0], v[1], v[2], v[3]);
}

// Optional output layout for the stacked QKV projection. The projection's
// 256-row tile covers one query, gate, key or value head.
struct AttentionProjectionOutput {
  const float* q_gamma;
  const float* k_gamma;
  float* query;
  float* gate;
  __half* keys;
  __half* values;
  const std::uint32_t* position;
  float theta;
  float eps;
  const qwen::vision::DeviceRope* rope;
};

/// Dense F16 WMMA GEMM over Q8_0 or F16 weights: block = BM rows x BN tokens,
/// BK 32-element K blocks per LDS stage, waves = WM row groups x WN token
/// groups. The codes are dequantized to F16 once per stage as they are
/// committed to LDS (magic-number F16 construction, exact for a Q8_0 code),
/// and the activations are F16 rows [batch][k], so the matrix cores
/// accumulate in F32 with no per-block scaling. F16 weights skip decoding
/// and use the same ordered K16 products. y is [batch][m].
template<int BM, int BN, int BK, int WM, int WN, bool kSsmConv = false,
         bool kAttention = false, bool kHalfWeights = false>
__launch_bounds__(256) __global__
    void DenseF16GEMMKernel(const void* __restrict__ w,
                            const __half* __restrict__ x, float* __restrict__ y,
                            std::size_t batch, std::size_t m, std::size_t k,
                            const float* conv_w = nullptr,
                            float* conv_out = nullptr,
                            AttentionProjectionOutput attention = {}) {
  static_assert(WM * WN == 8, "256 threads is 8 waves");
  static_assert(BM % (16 * WM) == 0 && BN % (16 * WN) == 0);
  constexpr int kRowTiles = BM / 16;
  constexpr int kTokTiles = BN / 16;
  constexpr int kWaveRowTiles = kRowTiles / WM;
  constexpr int kWaveTokTiles = kTokTiles / WN;
  // Weight and activation K blocks staged per thread; a unit past the
  // stage's block count is idle.
  constexpr int kAUnits = BM * BK;
  constexpr int kBUnits = BN * BK;
  constexpr int kAPer = (kAUnits + 255) / 256;
  constexpr int kBPer = (kBUnits + 255) / 256;

  // One K block of one row or token is four 16-byte chunks; the chunks of
  // nearby rows are permuted so a fragment read (one row per lane, 64-byte
  // stride) covers all bank groups.
  constexpr int kLdsChunks = BK * (BM + BN) * 4;
  static_assert(kLdsChunks * 16 >= 8 * 512 * 4, "epilogue transposes 16 KB");
  constexpr int kTransposeChunks = 8 * 16 * 36 * sizeof(float) / sizeof(uint4);
  constexpr int kLdsStorage =
      kLdsChunks < kTransposeChunks ? kTransposeChunks : kLdsChunks;
  __shared__ __attribute__((aligned(16))) uint4 s_lds[kLdsStorage];
  auto* s_a = reinterpret_cast<uint4(*)[BM][4]>(s_lds);
  auto* s_b = reinterpret_cast<uint4(*)[BN][4]>(s_lds + (BK * BM * 4));
  const auto swizzle = [](int row, int c) { return c ^ ((row >> 1) & 3); };

  const int num_kb = static_cast<int>(k / 32);
  const int m_i = static_cast<int>(m);
  const auto* w_bytes = static_cast<const std::uint8_t*>(w);

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  const int wave_row = wave_id / WN;
  const int wave_tok = wave_id % WN;
  const int r_block = static_cast<int>(blockIdx.y) * BM;
  const int t_block = static_cast<int>(blockIdx.x) * BN;

  // Weight fetch unit p of a thread: row (p * 256 + tid) / BK, K block
  // (p * 256 + tid) % BK of the stage; rows past m read the last row with
  // a zero scale.
  const std::uint8_t* a_ptr[kAPer];
  bool a_live[kAPer];
#pragma unroll
  for (int p = 0; p < kAPer; ++p) {
    const int idx = (p * 256) + tid;
    const int r = r_block + (idx / BK);
    a_live[p] = idx < kAUnits && r < m_i;
    a_ptr[p] = w_bytes + static_cast<std::size_t>(a_live[p] ? r : (m_i - 1)) *
                             static_cast<std::size_t>(num_kb) *
                             (kHalfWeights ? 64 : 34);
  }
  const __half* b_ptr[kBPer];
#pragma unroll
  for (int p = 0; p < kBPer; ++p) {
    const int idx = (p * 256) + tid;
    const int t = t_block + (idx / BK);
    b_ptr[p] = idx < kBUnits && t < static_cast<int>(batch)
                   ? x + (static_cast<std::size_t>(t) * k)
                   : nullptr;
  }
  uint4 a_codes[kAPer][2];
  uint4 a_half[kHalfWeights ? kAPer : 1][4];
  std::uint32_t a_d[kAPer];
  uint4 b_data[kBPer][4];

  const auto fetch_stage = [&](int kb0) {
#pragma unroll
    for (int p = 0; p < kAPer; ++p) {
      const int kb = kb0 + (((p * 256) + tid) % BK);
      const bool live = a_live[p] && kb < num_kb;
      if constexpr (kHalfWeights) {
        const auto* src = reinterpret_cast<const uint4*>(
            a_ptr[p] + static_cast<std::size_t>(kb) * 64);
#pragma unroll
        for (int c = 0; c < 4; ++c)
          a_half[p][c] = live ? src[c] : make_uint4(0u, 0u, 0u, 0u);
      } else {
        const std::uint8_t* blk =
            a_ptr[p] +
            (static_cast<std::size_t>(live ? kb : (num_kb - 1)) * 34);
        a_d[p] = live ? *reinterpret_cast<const std::uint16_t*>(blk) : 0U;
        __builtin_memcpy(&a_codes[p][0], blk + 2, 16);
        __builtin_memcpy(&a_codes[p][1], blk + 18, 16);
      }
    }
#pragma unroll
    for (int p = 0; p < kBPer; ++p) {
      const int kb = kb0 + (((p * 256) + tid) % BK);
      if (b_ptr[p] != nullptr && kb < num_kb) {
        const auto* src = reinterpret_cast<const uint4*>(b_ptr[p] + (kb * 32));
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          b_data[p][c] = src[c];
        }
      } else {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          b_data[p][c] = make_uint4(0u, 0u, 0u, 0u);
        }
      }
    }
  };

  const __half2 magic = __floats2half2_rn(-1152.0F, -1152.0F);
  const __half2 zero2 = __floats2half2_rn(0.0F, 0.0F);
  const auto commit_stage = [&]() {
#pragma unroll
    for (int p = 0; p < kAPer; ++p) {
      const int idx = (p * 256) + tid;
      if (idx >= kAUnits) {
        break;
      }
      const int row = idx / BK;
      const int kk = idx % BK;
      if constexpr (kHalfWeights) {
#pragma unroll
        for (int c = 0; c < 4; ++c)
          s_a[kk][row][swizzle(row, c)] = a_half[p][c];
      } else {
        // Q8_0 codes are signed; flipping the sign bit carries q + 128, which
        // the 1152 magic takes back out.
        const std::uint32_t words[8] = {
            a_codes[p][0].x, a_codes[p][0].y, a_codes[p][0].z, a_codes[p][0].w,
            a_codes[p][1].x, a_codes[p][1].y, a_codes[p][1].z, a_codes[p][1].w};
        const __half2 scale2 = __half2half2(
            __builtin_bit_cast(__half, static_cast<std::uint16_t>(a_d[p])));
        __half2 h[16];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          CodesToHalves(words[i] ^ 0x80808080U, magic, scale2, zero2, h[2 * i],
                        h[2 * i + 1]);
        }
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          uint4 v;
          __builtin_memcpy(&v, &h[4 * c], 16);
          s_a[kk][row][swizzle(row, c)] = v;
        }
      }
    }
#pragma unroll
    for (int p = 0; p < kBPer; ++p) {
      const int idx = (p * 256) + tid;
      if (idx >= kBUnits) {
        break;
      }
      const int t = idx / BK;
      const int kk = idx % BK;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        s_b[kk][t][swizzle(t, c)] = b_data[p][c];
      }
    }
  };

  v8f acc[kWaveRowTiles][kWaveTokTiles];
  // Separate K16 chains reduce FP32 accumulation error for the sensitive
  // unquantized router/gate projections, with the same order in every chunk.
  v8f acc_high[kHalfWeights ? kWaveRowTiles : 1]
              [kHalfWeights ? kWaveTokTiles : 1]{};
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
      acc[i][j] = v8f{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    }
  }

  fetch_stage(0);
  for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
    commit_stage();
    __syncthreads();
    if (kb0 + BK < num_kb) {
      fetch_stage(kb0 + BK);
    }
    // Keeps the next stage's loads ahead of the matrix work: scheduled
    // freely, the compiler sinks them below it and the commit waits on
    // the full memory latency.
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      v16h a_lo[kWaveRowTiles];
      v16h a_hi[kWaveRowTiles];
#pragma unroll
      for (int i = 0; i < kWaveRowTiles; ++i) {
        const int row = (((wave_row * kWaveRowTiles) + i) * 16) + sub_lane;
        uint4 c[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          c[q] = s_a[kb][row][swizzle(row, q)];
        }
        __builtin_memcpy(&a_lo[i], &c[0], 32);
        __builtin_memcpy(&a_hi[i], &c[2], 32);
      }
#pragma unroll
      for (int j = 0; j < kWaveTokTiles; ++j) {
        const int t = (((wave_tok * kWaveTokTiles) + j) * 16) + sub_lane;
        uint4 c[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          c[q] = s_b[kb][t][swizzle(t, q)];
        }
        v16h b_lo;
        v16h b_hi;
        __builtin_memcpy(&b_lo, &c[0], 32);
        __builtin_memcpy(&b_hi, &c[2], 32);
#pragma unroll
        for (int i = 0; i < kWaveRowTiles; ++i) {
          acc[i][j] = Wmma(a_lo[i], b_lo, acc[i][j]);
          if constexpr (kHalfWeights)
            acc_high[i][j] = Wmma(a_hi[i], b_hi, acc_high[i][j]);
          else
            acc[i][j] = Wmma(a_hi[i], b_hi, acc[i][j]);
        }
      }
    }
    __syncthreads();
  }

  if constexpr (kHalfWeights) {
#pragma unroll
    for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
      for (int j = 0; j < kWaveTokTiles; ++j)
        acc[i][j] += acc_high[i][j];
    }
  }

  if constexpr (kAttention) {
    static_assert(BM == 256 && BN == 128 && BK == 2 && WM == 8 && WN == 1);
    static_assert(!kSsmConv);
    constexpr unsigned stride = 36, dim = 256, width = 4096, kvwidth = 512;
    float* scratch = reinterpret_cast<float*>(s_lds);
    float* tile = scratch + wave_id * 16 * stride;
    const unsigned projection_head = r_block / 256;
    const bool query = projection_head < 32 && (projection_head % 2) == 0;
    const bool key = projection_head >= 32 && projection_head < 34;
    const bool gate = projection_head < 32 && (projection_head % 2) == 1;
    const unsigned head =
        projection_head < 32 ? projection_head / 2 : projection_head % 2;
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        tile[sub_lane * stride + 2 * l + half_id] = acc[0][j][l];
        tile[sub_lane * stride + 16 + 2 * l + half_id] = acc[1][j][l];
      }
      __syncthreads();
#pragma unroll
      for (unsigned phase = 0; phase < 2; ++phase) {
        // One wave handles a token, emulating its original eight-wave norm.
        const unsigned tok_local = wave_id * 2 + phase;
        const std::size_t tok = t_block + j * 16 + tok_local;
        if (tok < batch) {
          float v[8];
#pragma unroll
          for (unsigned c = 0; c < 8; ++c)
            v[c] = scratch[(c * 16 + tok_local) * stride + lane_id];
          if (query || key) {
            float total = 0.0F;
#pragma unroll
            for (unsigned c = 0; c < 8; ++c) {
              // Match the separate norm: round each square and each wave sum
              // before accumulating the eight partials in their original order.
              float sq = v[c] * v[c];
              asm volatile("" : "+v"(sq));
              float ss = WaveSum(sq);
              asm volatile("" : "+v"(ss));
              total += ss;
              asm volatile("" : "+v"(total));
            }
            const float scale = rsqrtf(total / 256.0F + attention.eps);
            const float* gamma = query ? attention.q_gamma : attention.k_gamma;
#pragma unroll
            for (unsigned c = 0; c < 8; ++c) {
              v[c] = v[c] * scale;
              asm volatile("" : "+v"(v[c]));
              v[c] = v[c] * gamma[c * 32 + lane_id];
              asm volatile("" : "+v"(v[c]));
            }
            const float freq = powf(
                attention.theta, -2.0F * static_cast<float>(lane_id) / 64.0F);
            float sn = 0.0F, cs = 0.0F;
            sincosf(qwen::vision::RopePosition(
                        attention.rope, *attention.position + tok, lane_id) *
                        freq,
                    &sn, &cs);
            const float lo = __fmaf_rn(v[0], cs, -__fmul_rn(v[1], sn));
            const float hi = __fmaf_rn(v[0], sn, __fmul_rn(v[1], cs));
            v[0] = lo;
            v[1] = hi;
          }
#pragma unroll
          for (unsigned c = 0; c < 8; ++c) {
            const unsigned col = c * 32 + lane_id;
            if (query)
              attention.query[tok * width + head * dim + col] = v[c];
            else if (gate)
              attention.gate[tok * width + head * dim + col] = v[c];
            else if (key)
              attention.keys[std::size_t(*attention.position + tok) * kvwidth +
                             head * dim + col] = __float2half_rn(v[c]);
            else
              attention
                  .values[std::size_t(*attention.position + tok) * kvwidth +
                          head * dim + col] = __float2half_rn(v[c]);
          }
        }
      }
      __syncthreads();
    }
    return;
  }

  // Transpose the result through LDS, two row tiles at a time, so every
  // global store covers 32 consecutive rows of one token: a full 128-byte
  // line (half lines cost a read-modify-write on the fabric).
  // Four padding floats reduce scatter bank conflicts and retain float4
  // alignment for both output stores and the fused convolution's reads.
  constexpr unsigned kOutputStride = 36;
  // Pair token tiles for SSM: half as many convolutions cross a tile edge.
  // The larger transpose still fits the projection's existing LDS allocation.
  constexpr unsigned kOutputTokens = kSsmConv ? kSsmProjectionTileTokens : 16;
  constexpr unsigned kOutputGroups = kOutputTokens / 16;
  static_assert(8 * kOutputTokens * kOutputStride * sizeof(float) <=
                sizeof(s_lds));
  static_assert(kWaveRowTiles % 2 == 0, "the epilogue pairs row tiles");
  float* tile_scratch =
      reinterpret_cast<float*>(s_lds) + wave_id * kOutputTokens * kOutputStride;
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; i += 2) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; j += kOutputGroups) {
#pragma unroll
      for (unsigned group = 0; group < kOutputGroups; ++group) {
        // scratch[token][row], with 32 output rows per token.
#pragma unroll
        for (int l = 0; l < 8; ++l) {
          tile_scratch[((sub_lane + group * 16) * kOutputStride) + (2 * l) +
                       half_id] = acc[i][j + group][l];
          tile_scratch[((sub_lane + group * 16) * kOutputStride) + 16 +
                       (2 * l) + half_id] = acc[i + 1][j + group][l];
        }
      }
      __builtin_amdgcn_wave_barrier();
      const std::size_t r0 =
          static_cast<std::size_t>(r_block) +
          static_cast<std::size_t>((((wave_row * kWaveRowTiles) + i) * 16));
      const std::size_t t0 =
          static_cast<std::size_t>(t_block) +
          static_cast<std::size_t>((((wave_tok * kWaveTokTiles) + j) * 16));
#pragma unroll
      for (unsigned group = 0; group < kOutputGroups; ++group) {
        // Lane pair (2p, 2p + 1) stores token p's 32 rows as eight float4.
        const int tok_l = (lane_id >> 1) + group * 16;
        const int row_l = (lane_id & 1) * 16;
        const std::size_t tok = t0 + static_cast<std::size_t>(tok_l);
        const auto* src = reinterpret_cast<const float4*>(
            tile_scratch + (tok_l * kOutputStride) + row_l);
        if constexpr (kSsmConv) {
          static_assert(BM == 256 && BN == 128 && BK == 2 && WM == 8 &&
                        WN == 1);
          constexpr std::uint32_t channels = 8192;
          if (tok < batch) {
#pragma unroll
            for (int v = 0; v < 4; ++v) {
              const unsigned row = r0 + row_l + v * 4;
              if (row >= m)
                continue;
              const float4 current = src[v];
              if (row >= channels || tok_l < 3 || tok_l >= kOutputTokens - 3 ||
                  tok + 3 >= batch) {
                *reinterpret_cast<float4*>(y + tok * m + row) = current;
              }
              if (row < channels && tok_l >= 3) {
                const float4 x0 = *reinterpret_cast<const float4*>(
                    tile_scratch + (tok_l - 3) * kOutputStride + row_l + v * 4);
                const float4 x1 = *reinterpret_cast<const float4*>(
                    tile_scratch + (tok_l - 2) * kOutputStride + row_l + v * 4);
                const float4 x2 = *reinterpret_cast<const float4*>(
                    tile_scratch + (tok_l - 1) * kOutputStride + row_l + v * 4);
                const float4 w0 =
                    *reinterpret_cast<const float4*>(conv_w + (row + 0) * 4);
                const float4 w1 =
                    *reinterpret_cast<const float4*>(conv_w + (row + 1) * 4);
                const float4 w2 =
                    *reinterpret_cast<const float4*>(conv_w + (row + 2) * 4);
                const float4 w3 =
                    *reinterpret_cast<const float4*>(conv_w + (row + 3) * 4);
                const float4 value{
                    SsmConv4Value(w0, x0.x, x1.x, x2.x, current.x),
                    SsmConv4Value(w1, x0.y, x1.y, x2.y, current.y),
                    SsmConv4Value(w2, x0.z, x1.z, x2.z, current.z),
                    SsmConv4Value(w3, x0.w, x1.w, x2.w, current.w)};
                *reinterpret_cast<float4*>(conv_out + tok * channels + row) =
                    value;
              }
            }
          }
        } else {
          if (tok < batch && r0 + 32 <= m && ((tok * m) % 4 == 0)) {
            auto* dst = reinterpret_cast<float4*>(
                y + (tok * m) + r0 + static_cast<std::size_t>(row_l));
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              dst[q] = src[q];
            }
          } else if (tok < batch) {
#pragma unroll
            for (int q = 0; q < 16; ++q) {
              const std::size_t r = r0 + static_cast<std::size_t>(row_l + q);
              if (r < m) {
                y[(tok * m) + r] =
                    tile_scratch[(tok_l * kOutputStride) + row_l + q];
              }
            }
          }
        }
      }
      __builtin_amdgcn_wave_barrier();
    }
  }
}

bool AttentionF16Gemm(const void* weights, const __half* input,
                      const float* q_gamma, const float* k_gamma, float* query,
                      float* gate, __half* keys, __half* values,
                      std::uint32_t n_tokens, const std::uint32_t* position,
                      float theta, float eps, hipStream_t stream,
                      const qwen::vision::DeviceRope* rope) {
  if (n_tokens < 1024 || weights == nullptr || input == nullptr ||
      q_gamma == nullptr || k_gamma == nullptr || query == nullptr ||
      gate == nullptr || keys == nullptr || values == nullptr ||
      position == nullptr)
    return false;
  const AttentionProjectionOutput output{q_gamma, k_gamma,  query, gate, keys,
                                         values,  position, theta, eps,  rope};
  hipLaunchKernelGGL((DenseF16GEMMKernel<256, 128, 2, 8, 1, false, true>),
                     dim3((n_tokens + 127) / 128, 36), dim3(kThreads), 0,
                     stream, weights, input, nullptr, n_tokens, 9216, 2048,
                     nullptr, nullptr, output);
  return true;
}

bool UnquantizedF16Gemm(const void* w, const __half* x, float* out,
                        std::size_t batch, std::size_t m, std::size_t k,
                        hipStream_t stream) {
  if (m == 0 || batch == 0 || k == 0 || k % 32 != 0)
    return false;
  hipLaunchKernelGGL((DenseF16GEMMKernel<64, 64, 2, 2, 4, false, false, true>),
                     dim3((batch + 63) / 64, (m + 63) / 64), dim3(kThreads), 0,
                     stream, w, x, out, batch, m, k);
  return true;
}

bool DenseF16Gemm(const void* w, const __half* x, float* out, std::size_t batch,
                  std::size_t m, std::size_t k, hipStream_t stream) {
  if (m == 0 || k == 0 || batch == 0 || k % 32 != 0) {
    return false;
  }
  // The same tile plan as the W8A8 route: a 128-token macro tile for wide
  // batches, 64-row tiles for narrow projections, 64 tokens below 96.
  constexpr int kBM = 128;
  if (m <= 512 && batch >= 96) {
    constexpr int kBN = 128;
    constexpr int kNarrowBM = 64;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kNarrowBM - 1) / kNarrowBM));
    hipLaunchKernelGGL((DenseF16GEMMKernel<kNarrowBM, kBN, 2, 2, 4>), grid,
                       dim3(kThreads), 0, stream, w, x, out, batch, m, k);
  } else if (batch >= 96) {
    // 64 x 64 wave tiles: half the LDS fragment bytes per matrix product
    // of the 32 x 64 tile (the F16 fragments are twice the int8 ones).
    constexpr int kWideBM = 256;
    constexpr int kBN = 128;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kWideBM - 1) / kWideBM));
    if (batch >= 1024 && (((m == 12288 || m == 9216) && k == 2048) ||
                          (m == 2048 && k == 4096))) {
      // Eight row groups reuse each weight fragment across all token tiles
      // and keep fewer weight fragments live. K accumulation is unchanged.
      hipLaunchKernelGGL((DenseF16GEMMKernel<kWideBM, kBN, 2, 8, 1>), grid,
                         dim3(kThreads), 0, stream, w, x, out, batch, m, k);
    } else {
      hipLaunchKernelGGL((DenseF16GEMMKernel<kWideBM, kBN, 1, 4, 2>), grid,
                         dim3(kThreads), 0, stream, w, x, out, batch, m, k);
    }
  } else {
    constexpr int kBN = 64;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kBM - 1) / kBM));
    hipLaunchKernelGGL((DenseF16GEMMKernel<kBM, kBN, 4, 4, 2>), grid,
                       dim3(kThreads), 0, stream, w, x, out, batch, m, k);
  }
  return true;
}

bool DenseF16SsmGemm(const void* w, const __half* x, const float* conv_w,
                     const float* history, float* qkvz, float* convolved,
                     std::uint32_t n_tokens, std::uint32_t m, std::uint32_t k,
                     std::uint32_t channels, std::uint32_t kernel,
                     hipStream_t stream) {
  if (n_tokens < 1024 || m != 12288 || k != 2048 || channels != 8192 ||
      kernel != kSsmConvTaps) {
    return false;
  }
  hipLaunchKernelGGL((DenseF16GEMMKernel<256, 128, 2, 8, 1, true>),
                     dim3((n_tokens + 127) / 128, m / 256), dim3(kThreads), 0,
                     stream, w, x, qkvz, n_tokens, m, k, conv_w, convolved);
  hipLaunchKernelGGL(
      SsmConvBoundaryKernel,
      dim3(Blocks(channels),
           (n_tokens + kSsmProjectionTileTokens - 1) / kSsmProjectionTileTokens,
           3),
      dim3(kThreads), 0, stream, qkvz, conv_w, history, convolved, n_tokens,
      channels, m);
  return true;
}

/// Shapes the split-K F16 kernel serves; every row group of such a
/// projection takes it, so a row's arithmetic is independent of the width.
constexpr bool SmallGemmSplitF16(std::uint32_t m, std::uint32_t k) {
  return k % (4 * 32 * 8) == 0 && m <= 1024;
}

template<WeightType type, unsigned tokens>
void LaunchSmallGemm(const void* w, const float* x, float* out, std::uint32_t m,
                     std::uint32_t k, hipStream_t stream,
                     std::uint32_t groups = 1) {
  if constexpr (type == WeightType::kF16) {
    if (SmallGemmSplitF16(m, k)) {
      hipLaunchKernelGGL((SmallGemmSplitF16Kernel<tokens>), dim3(m, groups),
                         dim3(128), 0, stream, static_cast<const __half*>(w), x,
                         out, m, k);
      return;
    }
  }
  hipLaunchKernelGGL((SmallGemmKernel<type, tokens>),
                     dim3((m + kSmallGemmRows - 1) / kSmallGemmRows),
                     dim3(kSmallGemmRows * 32), 0, stream, w, x, out, m, k);
}

template<WeightType type>
void SmallGemmForType(const void* w, const float* x, float* out,
                      std::uint32_t tokens, std::uint32_t m, std::uint32_t k,
                      hipStream_t stream) {
  if (tokens > 8) {
    const auto groups = tokens / 8;
    if (type == WeightType::kF16 && SmallGemmSplitF16(m, k)) {
      LaunchSmallGemm<type, 8>(w, x, out, m, k, stream, groups);
    } else {
      hipLaunchKernelGGL(
          (SmallGemmKernel<type, 8, true>),
          dim3((m + kSmallGemmRows - 1) / kSmallGemmRows, groups),
          dim3(kSmallGemmRows * 32), 0, stream, w, x, out, m, k);
    }
    const auto consumed = groups * 8;
    x += std::size_t{consumed} * k;
    out += std::size_t{consumed} * m;
    tokens -= consumed;
    if (tokens == 0)
      return;
  }
  switch (tokens) {
    case 1:
      return LaunchSmallGemm<type, 1>(w, x, out, m, k, stream);
    case 2:
      return LaunchSmallGemm<type, 2>(w, x, out, m, k, stream);
    case 3:
      return LaunchSmallGemm<type, 3>(w, x, out, m, k, stream);
    case 4:
      return LaunchSmallGemm<type, 4>(w, x, out, m, k, stream);
    case 5:
      return LaunchSmallGemm<type, 5>(w, x, out, m, k, stream);
    case 6:
      return LaunchSmallGemm<type, 6>(w, x, out, m, k, stream);
    case 7:
      return LaunchSmallGemm<type, 7>(w, x, out, m, k, stream);
    case 8:
      return LaunchSmallGemm<type, 8>(w, x, out, m, k, stream);
    default:
      throw std::logic_error("small projection requires 1-8 token rows");
  }
}

void SmallGemm(const void* w, WeightType type, const float* x, float* out,
               std::uint32_t n_tokens, std::uint32_t m, std::uint32_t k,
               hipStream_t stream) {
  switch (type) {
    case WeightType::kF32:
      return SmallGemmForType<WeightType::kF32>(w, x, out, n_tokens, m, k,
                                                stream);
    case WeightType::kBF16:
      return SmallGemmForType<WeightType::kBF16>(w, x, out, n_tokens, m, k,
                                                 stream);
    case WeightType::kF16:
      return SmallGemmForType<WeightType::kF16>(w, x, out, n_tokens, m, k,
                                                stream);
    default:
      throw std::logic_error("unsupported small projection format");
  }
}

void RestoreGdnState(float* state, RollbackRows snapshots, std::uint32_t keep,
                     std::uint32_t k_heads, std::uint32_t v_heads,
                     hipStream_t stream) {
  const auto count = std::size_t{v_heads} * kGdnDim * kGdnDim;
  RestoreGdnStateKernel<<<(count + 255) / 256, 256, 0, stream>>>(
      state, snapshots, keep, k_heads, v_heads);
}

void GatedDeltaNet(const float* qkv, std::uint32_t qkv_stride, const float* z,
                   std::uint32_t z_stride, const float* alpha_beta,
                   const float* conv_w, const float* a, const float* dt,
                   const float* norm_w, float* conv_state, float* conv_scratch,
                   float* qn, float* kn, float* raw, float* state, float* out,
                   void* out_q8, RollbackRows state_snapshots,
                   RollbackRows conv_snapshots, std::uint32_t n_tokens,
                   std::uint32_t k_heads, std::uint32_t v_heads,
                   std::uint32_t d, std::uint32_t kernel, bool row_split,
                   bool convolved, float eps, hipStream_t stream,
                   __half* out_half) {
  const std::uint32_t channels = 2 * k_heads * d + v_heads * d;
  const std::size_t count = static_cast<std::size_t>(n_tokens) * channels;
  const bool saved_history = !convolved && kernel == kSsmConvTaps &&
                             n_tokens <= kSsmConvTokensPerThread;
  if (!convolved) {
    if (saved_history) {
      hipLaunchKernelGGL((SsmConv4Kernel<true, false>), dim3(Blocks(channels)),
                         dim3(kThreads), 0, stream, qkv, qkv_stride, conv_w,
                         conv_state, conv_scratch, n_tokens, channels,
                         conv_snapshots, nullptr, 0);
    } else if (kernel == kSsmConvTaps) {
      hipLaunchKernelGGL(
          (SsmConv4Kernel<false, false>),
          dim3(Blocks(channels), (n_tokens + kSsmConvTokensPerThread - 1) /
                                     kSsmConvTokensPerThread),
          dim3(kThreads), 0, stream, qkv, qkv_stride, conv_w, conv_state,
          conv_scratch, n_tokens, channels, RollbackRows{}, nullptr, 0);
    } else {
      hipLaunchKernelGGL(SsmConvKernel, dim3(Blocks(count)), dim3(kThreads), 0,
                         stream, qkv, qkv_stride, conv_w, conv_state,
                         conv_scratch, n_tokens, channels, kernel);
    }
  }
  if (!saved_history) {
    if (conv_snapshots.rows[0] != nullptr && n_tokens > 1) {
      const std::size_t saved =
          static_cast<std::size_t>(n_tokens - 1) * channels;
      hipLaunchKernelGGL(RollingSnapshotKernel,
                         dim3(Blocks(saved * (kernel - 1))), dim3(kThreads), 0,
                         stream, qkv, qkv_stride, conv_state, conv_snapshots,
                         n_tokens - 1, channels, kernel - 1);
    }
    // The rolling state is the last kernel-1 projections: [history ; qkv].
    const std::uint32_t hist = kernel - 1;
    const std::size_t hist_count = static_cast<std::size_t>(hist) * channels;
    hipLaunchKernelGGL(HistoryShiftKernel, dim3(Blocks(hist_count)),
                       dim3(kThreads), 0, stream, qkv, qkv_stride, conv_state,
                       conv_scratch + count, n_tokens, channels, hist);
    hipLaunchKernelGGL(CopyKernel, dim3(Blocks(hist_count)), dim3(kThreads), 0,
                       stream, conv_scratch + count, conv_state, hist_count);
  }
  const unsigned waves = kThreads / 32;
  if (row_split && d == kGdnDim && state_snapshots.rows[0] == nullptr) {
    hipLaunchKernelGGL(GdnPrepKqKernel, dim3(k_heads, n_tokens), dim3(32), 0,
                       stream, conv_scratch, qn, n_tokens, k_heads, channels,
                       eps);
    hipLaunchKernelGGL(
        GdnPrepAbKernel,
        dim3(Blocks(static_cast<std::size_t>(n_tokens) * v_heads)),
        dim3(kThreads), 0, stream, alpha_beta, a, dt, kn,
        static_cast<std::size_t>(n_tokens) * v_heads, v_heads);
    // Sixteen keys per lane (eight lanes per row) measured fastest.
    hipLaunchKernelGGL(GdnRowSplitKernel<16>, dim3(kGdnDim / 32, v_heads),
                       dim3(kThreads), 0, stream, conv_scratch, qn, kn, state,
                       raw, n_tokens, k_heads, v_heads);
  } else {
    hipLaunchKernelGGL(GdnPrepKernel<false>,
                       dim3((n_tokens * k_heads + waves - 1) / waves),
                       dim3(kThreads), 0, stream, conv_scratch, qn, kn,
                       n_tokens * k_heads, k_heads, channels, eps, nullptr, 0);
    hipLaunchKernelGGL(GdnKernel<false>,
                       dim3(v_heads, kGdnDim / kGdnRowsPerBlock),
                       dim3(kGdnRowsPerBlock * kGdnLanes), 0, stream,
                       conv_scratch, qn, kn, alpha_beta, a, dt, state, raw,
                       state_snapshots, n_tokens, k_heads, v_heads, nullptr, 0);
  }
  hipLaunchKernelGGL(
      GdnEpilogueKernel<false>, dim3((n_tokens * v_heads + waves - 1) / waves),
      dim3(kThreads), 0, stream, raw, z, z_stride, norm_w, out, out_q8,
      out_half, n_tokens * v_heads, v_heads, eps, nullptr, 0);
}

bool GatedDeltaNetBatch(const GdnBatchItem* items, std::uint32_t count,
                        std::uint32_t max_tokens, std::uint32_t active,
                        std::uint32_t qkv_stride, std::uint32_t z_stride,
                        const float* conv_w, const float* a, const float* dt,
                        const float* norm_w, std::uint32_t k_heads,
                        std::uint32_t v_heads, float eps, hipStream_t stream) {
  if (items == nullptr || count == 0 || count > 8 || max_tokens == 0 ||
      max_tokens > kSsmConvTokensPerThread)
    return false;
  const auto channels = (2 * k_heads + v_heads) * kGdnDim;
  constexpr auto waves = kThreads / 32;
  hipLaunchKernelGGL((SsmConv4Kernel<true, true>),
                     dim3(Blocks(channels), 1, count), dim3(kThreads), 0,
                     stream, nullptr, qkv_stride, conv_w, nullptr, nullptr,
                     max_tokens, channels, RollbackRows{}, items, active);
  hipLaunchKernelGGL(GdnPrepKernel<true>,
                     dim3((max_tokens * k_heads + waves - 1) / waves, 1, count),
                     dim3(kThreads), 0, stream, nullptr, nullptr, nullptr,
                     max_tokens * k_heads, k_heads, channels, eps, items,
                     active);
  hipLaunchKernelGGL(
      GdnKernel<true>, dim3(v_heads, kGdnDim / kGdnRowsPerBlock, count),
      dim3(kGdnRowsPerBlock * kGdnLanes), 0, stream, nullptr, nullptr, nullptr,
      nullptr, a, dt, nullptr, nullptr, RollbackRows{}, max_tokens, k_heads,
      v_heads, items, active);
  hipLaunchKernelGGL(GdnEpilogueKernel<true>,
                     dim3((max_tokens * v_heads + waves - 1) / waves, 1, count),
                     dim3(kThreads), 0, stream, nullptr, nullptr, z_stride,
                     norm_w, nullptr, nullptr, nullptr, max_tokens * v_heads,
                     v_heads, eps, items, active);
  return hipGetLastError() == hipSuccess;
}

void UnpackQGate(const float* qg, std::uint32_t qg_stride, float* q,
                 float* gate, float* k, float* v, std::uint32_t n_tokens,
                 std::uint32_t heads, std::uint32_t d, std::uint32_t kv_width,
                 hipStream_t stream) {
  hipLaunchKernelGGL(UnpackQGateKernel, dim3(n_tokens), dim3(kThreads), 0,
                     stream, qg, qg_stride, q, gate, k, v, heads, d, kv_width);
}

bool PrepareAttention(const float* packed, std::uint32_t stride,
                      const float* q_gamma, const float* k_gamma, float* q,
                      float* gate, __half* k_cache, __half* v_cache,
                      std::uint32_t n_tokens, std::uint32_t heads,
                      std::uint32_t kv_heads, std::uint32_t d,
                      std::uint32_t rotary_dim, const std::uint32_t* start_pos,
                      float theta, float eps, hipStream_t stream,
                      const qwen::vision::DeviceRope* rope, bool prefill) {
  if (d == 0 || d > 256 || rotary_dim == 0 || rotary_dim > d ||
      rotary_dim % 2 != 0 || kv_heads == 0 ||
      stride < static_cast<std::size_t>(2) * (heads + kv_heads) * d) {
    return false;
  }
  if (n_tokens == 0)
    return true;
  if (!prefill && n_tokens < 32) {
    hipLaunchKernelGGL((PrepareAttentionKernel<1>),
                       dim3(n_tokens, heads + kv_heads), dim3(kThreads), 0,
                       stream, packed, stride, q_gamma, k_gamma, q, gate,
                       k_cache, v_cache, heads, kv_heads, d, rotary_dim,
                       start_pos, theta, eps, rope);
  } else {
    hipLaunchKernelGGL((PrepareAttentionKernel<4>),
                       dim3(n_tokens, (heads + kv_heads + 3) / 4),
                       dim3(kThreads), 0, stream, packed, stride, q_gamma,
                       k_gamma, q, gate, k_cache, v_cache, heads, kv_heads, d,
                       rotary_dim, start_pos, theta, eps, rope);
  }
  return true;
}

void Rope(float* x, std::uint32_t n_tokens, std::uint32_t heads,
          std::uint32_t d, std::uint32_t rotary_dim,
          const std::uint32_t* start_pos, float theta, hipStream_t stream,
          const qwen::vision::DeviceRope* rope) {
  hipLaunchKernelGGL(RopeKernel, dim3(n_tokens), dim3(kThreads), 0, stream, x,
                     heads, d, rotary_dim, start_pos, theta, rope);
}

void StoreKv(const float* src, __half* cache, std::uint32_t n_tokens,
             std::uint32_t row_dim, const std::uint32_t* start_pos,
             hipStream_t stream) {
  hipLaunchKernelGGL(StoreKvKernel, dim3(n_tokens), dim3(kThreads), 0, stream,
                     src, cache, row_dim, start_pos);
}

void QuantizeKv(const __half* cache, std::int8_t* values, __half* scales,
                std::uint32_t n_tokens, std::uint32_t kv_row,
                const std::uint32_t* start_pos, hipStream_t stream) {
  hipLaunchKernelGGL(QuantizeKvKernel, dim3(n_tokens, kv_row / 256), dim3(256),
                     0, stream, cache, values, scales, kv_row, start_pos);
}

bool GqaDecodeAttention(const float* q, const KvQ8& kv, float* out,
                        float* partials, std::uint32_t splits,
                        std::uint32_t n_tokens, const std::uint32_t* start_pos,
                        std::uint32_t heads, std::uint32_t kv_heads,
                        std::uint32_t d, hipStream_t stream) {
  if (d != kGqaDim || kv_heads == 0 || heads % kv_heads != 0 ||
      heads / kv_heads != 8 || n_tokens == 0 || n_tokens > 8 || splits == 0) {
    return false;
  }
  const dim3 grid(kv_heads, splits);
  // Pair capacity rounds the batch up; every row keeps its own arithmetic
  // in the one WMMA kernel family, so verification matches decoding.
  // Up to eight rows read each K/V tile once.
  if (n_tokens <= 2) {
    hipLaunchKernelGGL(GqaDecodeAttentionWmmaKernel<1>, grid, dim3(256), 0,
                       stream, q, kv.k, kv.v, kv.k_scale, kv.v_scale, partials,
                       start_pos, n_tokens, heads, kv_heads, 0u);
  } else if (n_tokens <= 4) {
    hipLaunchKernelGGL(GqaDecodeAttentionWmmaKernel<2>, grid, dim3(256), 0,
                       stream, q, kv.k, kv.v, kv.k_scale, kv.v_scale, partials,
                       start_pos, n_tokens, heads, kv_heads, 0u);
  } else {
    hipLaunchKernelGGL(GqaDecodeAttentionWmmaKernel<4>, grid, dim3(256), 0,
                       stream, q, kv.k, kv.v, kv.k_scale, kv.v_scale, partials,
                       start_pos, n_tokens, heads, kv_heads, 0u);
  }
  hipLaunchKernelGGL(AttentionMergeKernel, dim3(heads, n_tokens),
                     dim3(kThreads), 0, stream, partials, out, heads, d,
                     splits);
  return true;
}

/// Dense causal prefill attention for 16 query heads over two KV heads,
/// head dim 256, on the WMMA cores. A block owns 16 queries x kRowBlocks
/// heads (one 16-row block each) and sweeps kKeys keys per round: the S
/// tiles (row block x 16-key block) are split over the eight waves along the
/// head dimension, a lane group runs each row's online softmax, and each
/// wave accumulates O for dims `wave` and `wave + 8` of every row block. The
/// next round's K/V are prefetched into registers behind S, softmax and PV;
/// V is written to LDS transposed with one key per lane. Key rounds are
/// aligned to absolute positions and each (row, round) pair picks its PV
/// arithmetic from those positions alone, so results do not depend on the
/// prefill chunking. Measured fastest with 16-key rounds.
template<std::uint32_t kKeys, std::uint32_t kRowBlocks>
__launch_bounds__(256, 2) __global__ void DenseCausalAttentionKernel(
    const float* __restrict__ q, const float* __restrict__ gate,
    const __half* __restrict__ k_cache, const __half* __restrict__ v_cache,
    float* __restrict__ out, std::uint32_t start_pos, std::uint32_t n_tokens,
    std::uint32_t first_query_group) {
  constexpr std::uint32_t kD = kWmmaHeadDim;
  constexpr std::uint32_t kRows = 16 * kRowBlocks;
  constexpr std::uint32_t kKeyBlocks = kKeys / 16;
  constexpr std::uint32_t kSTiles = kRowBlocks * kKeyBlocks;
  constexpr std::uint32_t kSplit = 8 / kSTiles;  // waves per S tile
  constexpr std::uint32_t kKStepsPerWave = (kD / 16) / kSplit;
  constexpr std::uint32_t kKStride = kD + 8;
  constexpr std::uint32_t kVtStride = kKeys + 8;
  constexpr std::uint32_t kLanes = 256 / kRows;  // softmax lanes per row
  static_assert(kLanes * kRows == 256 && kKeys % kLanes == 0);
  constexpr std::uint32_t kPerLane = kKeys / kLanes;
  constexpr std::uint32_t kKRegs = kKeys * (kD / 8) / 256;
  constexpr std::uint32_t kVRegs = kKeys * kD / (256 * 8);
  static_assert(kKeys == 16 || kKeys == 32);
  static_assert(kSplit * kSTiles == 8 && kKStepsPerWave * kSplit == kD / 16);

  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31u;
  const std::uint32_t wave = tid >> 5u;
  const std::uint32_t sub = lane & 15u;
  const std::uint32_t half_id = lane >> 4u;

  const std::uint32_t query_start = (first_query_group + blockIdx.x) * 16;
  const std::uint32_t kv_head = blockIdx.y / (kWmmaGqa / kRowBlocks);
  const std::uint32_t first_head =
      kv_head * kWmmaGqa + (blockIdx.y % (kWmmaGqa / kRowBlocks)) * kRowBlocks;
  constexpr float kScale = 1.0F / 16.0F;

  constexpr std::uint32_t kKvHalves =
      kKeys * kKStride > kD * kVtStride ? kKeys * kKStride : kD * kVtStride;
  __shared__ __attribute__((aligned(16))) __half kv_lds[kKvHalves];
  __shared__ float s_lds[kSplit][kSTiles][16][17];
  __shared__ __attribute__((aligned(16))) __half p_lds[kRows][kKeys + 8];
  __shared__ float row_sum[kRows];
  __shared__ float row_scale[kRows];

  const std::uint32_t s_tile = wave % kSTiles;
  const std::uint32_t s_kh = wave / kSTiles;
  const std::uint32_t s_rb = s_tile % kRowBlocks;
  const std::uint32_t s_kb = s_tile / kRowBlocks;

  v16h q_frag[kKStepsPerWave];
  {
    const std::uint32_t query = query_start + sub;
    const bool live = query < n_tokens;
    const float* q_row =
        q + static_cast<std::size_t>(live ? query : 0) * kWmmaAttnWidth +
        static_cast<std::size_t>(first_head + s_rb) * kD;
#pragma unroll
    for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
      const std::uint32_t d0 = (s_kh * kKStepsPerWave + ks) * 16;
      const auto* qp = reinterpret_cast<const float4*>(q_row + d0);
#pragma unroll
      for (std::uint32_t v = 0; v < 4; ++v) {
        const float4 f = live ? qp[v] : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        q_frag[ks][v * 4 + 0] = static_cast<_Float16>(f.x * kScale);
        q_frag[ks][v * 4 + 1] = static_cast<_Float16>(f.y * kScale);
        q_frag[ks][v * 4 + 2] = static_cast<_Float16>(f.z * kScale);
        q_frag[ks][v * 4 + 3] = static_cast<_Float16>(f.w * kScale);
      }
    }
  }

  v8f o_acc[kRowBlocks][2] = {};
  float running_max = -INFINITY;
  float running_sum = 0.0F;
  const std::uint32_t context_end = start_pos + n_tokens;
  const std::uint32_t max_visible =
      min(context_end, start_pos + query_start + 16);
  const std::uint32_t n_tiles = (max_visible + kKeys - 1) / kKeys;

  // V: one key per lane and a slice of dims per thread, so the transposed
  // LDS writes stay conflict-free.
  const std::uint32_t v_key = lane % kKeys;
  const std::uint32_t v_slice = (tid / kKeys) * (kVRegs * 8);
  const auto* v_base =
      v_cache + static_cast<std::size_t>(kv_head) * kD + v_slice;
  const auto* k_base = k_cache + static_cast<std::size_t>(kv_head) * kD;
  const auto load_k = [&](std::uint32_t key0, uint4* dst) {
#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n) {
      const std::uint32_t idx = tid + n * 256;
      const std::uint32_t key = key0 + idx / (kD / 8);
      const std::uint32_t d8 = (idx % (kD / 8)) * 8;
      dst[n] =
          key < context_end
              ? *reinterpret_cast<const uint4*>(
                    k_base + static_cast<std::size_t>(key) * kWmmaKvWidth + d8)
              : make_uint4(0u, 0u, 0u, 0u);
    }
  };
  const auto load_v = [&](std::uint32_t key0, uint4* dst) {
    const std::uint32_t key = key0 + v_key;
    const bool live = key < context_end;
    const auto* src =
        v_base + static_cast<std::size_t>(live ? key : 0) * kWmmaKvWidth;
#pragma unroll
    for (std::uint32_t j = 0; j < kVRegs; ++j)
      dst[j] = live ? *reinterpret_cast<const uint4*>(src + j * 8)
                    : make_uint4(0u, 0u, 0u, 0u);
  };

  uint4 k_cur[kKRegs], v_cur[kVRegs], k_pre[kKRegs], v_pre[kVRegs];
  if (n_tiles != 0) {
    load_k(0, k_cur);
    load_v(0, v_cur);
  }
  for (std::uint32_t kt = 0; kt < n_tiles; ++kt) {
    const std::uint32_t key0 = kt * kKeys;
    __syncthreads();
#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n) {
      const std::uint32_t idx = tid + n * 256;
      *reinterpret_cast<uint4*>(
          &kv_lds[(idx / (kD / 8)) * kKStride + (idx % (kD / 8)) * 8]) =
          k_cur[n];
    }
    __syncthreads();
    if (kt + 1 < n_tiles) {
      load_k(key0 + kKeys, k_pre);
      load_v(key0 + kKeys, v_pre);
    }
    // --- S = Q K^T
    {
      v8f s_acc = {};
#pragma unroll
      for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
        const std::uint32_t d0 = (s_kh * kKStepsPerWave + ks) * 16;
        s_acc =
            Wmma(q_frag[ks],
                 LoadFrag(&kv_lds[(s_kb * 16 + sub) * kKStride + d0]), s_acc);
      }
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i)
        s_lds[s_kh][s_tile][2 * i + half_id][sub] = s_acc[i];
    }
    __syncthreads();
    // --- V transposed into the K region
#pragma unroll
    for (std::uint32_t j = 0; j < kVRegs; ++j) {
      const auto* packed = reinterpret_cast<const __half*>(&v_cur[j]);
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i)
        kv_lds[(v_slice + j * 8 + i) * kVtStride + v_key] = packed[i];
    }
    // --- online softmax, kLanes threads per row
    {
      const std::uint32_t rg = tid / kLanes;
      const std::uint32_t seg = tid % kLanes;
      const std::uint32_t rb = rg / 16;
      const std::uint32_t row = rg % 16;
      const std::uint32_t query = query_start + row;
      const bool live_row = query < n_tokens;
      const std::uint32_t absolute_query = start_pos + query;
      float vals[kPerLane];
      float part_max = -INFINITY;
#pragma unroll
      for (std::uint32_t m = 0; m < kPerLane; ++m) {
        const std::uint32_t col = seg * kPerLane + m;
        const std::uint32_t key = key0 + col;
        const bool valid =
            live_row && key <= absolute_query && key < context_end;
        const std::uint32_t tile = (col / 16) * kRowBlocks + rb;
        float sum = 0.0F;
#pragma unroll
        for (std::uint32_t h = 0; h < kSplit; ++h)
          sum += s_lds[h][tile][row][col % 16];
        vals[m] = valid ? sum : -INFINITY;
        part_max = fmaxf(part_max, vals[m]);
      }
#pragma unroll
      for (std::uint32_t off = 1; off < kLanes; off <<= 1)
        part_max = fmaxf(part_max, __shfl_xor(part_max, off));
      const float prev_max = running_max;
      const float next_max = fmaxf(prev_max, part_max);
      const float prior_scale =
          isfinite(prev_max) ? __expf(prev_max - next_max) : 0.0F;
      float part_sum = 0.0F;
#pragma unroll
      for (std::uint32_t m = 0; m < kPerLane; ++m) {
        const float w = isfinite(vals[m]) ? __expf(vals[m] - next_max) : 0.0F;
        part_sum += w;
        p_lds[rg][seg * kPerLane + m] = static_cast<__half>(w);
      }
#pragma unroll
      for (std::uint32_t off = 1; off < kLanes; off <<= 1)
        part_sum += __shfl_xor(part_sum, off);
      running_max = next_max;
      running_sum = running_sum * prior_scale + part_sum;
      if (seg == 0)
        row_scale[rg] = prior_scale;
    }
    __syncthreads();
#pragma unroll
    for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        const float s = row_scale[rb * 16 + 2 * i + half_id];
        o_acc[rb][0][i] *= s;
        o_acc[rb][1][i] *= s;
      }
    }
    // --- O += P V. A row takes the WMMA result for rounds it sees whole.
    // The WMMA sum also depends on V rows whose P is zero, and those hold
    // zeros or real keys depending on where the prefill chunk ends, so a
    // row's diagonal round instead sums its visible keys alone. The choice
    // depends only on absolute positions, never on the block's first query.
    const bool block_diagonal = key0 + kKeys - 1 > start_pos + query_start;
#pragma unroll
    for (std::uint32_t t = 0; t < 2; ++t) {
      const std::uint32_t dim_tile = wave + t * 8;
      v16h v_frag[kKeyBlocks];
#pragma unroll
      for (std::uint32_t kb = 0; kb < kKeyBlocks; ++kb)
        v_frag[kb] =
            LoadFrag(&kv_lds[(dim_tile * 16 + sub) * kVtStride + kb * 16]);
#pragma unroll
      for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
        v8f acc = o_acc[rb][t];
#pragma unroll
        for (std::uint32_t kb = 0; kb < kKeyBlocks; ++kb)
          acc = Wmma(LoadFrag(&p_lds[rb * 16 + sub][kb * 16]), v_frag[kb], acc);
        if (block_diagonal) {
#pragma unroll
          for (std::uint32_t i = 0; i < 8; ++i) {
            const std::uint32_t row = 2 * i + half_id;
            const std::uint32_t absolute_query = start_pos + query_start + row;
            if (key0 + kKeys - 1 <= absolute_query)
              continue;
            float sum = o_acc[rb][t][i];
            for (std::uint32_t key = 0;
                 key < kKeys && key0 + key <= absolute_query; ++key)
              sum = fmaf(
                  __half2float(p_lds[rb * 16 + row][key]),
                  __half2float(kv_lds[(dim_tile * 16 + sub) * kVtStride + key]),
                  sum);
            acc[i] = sum;
          }
        }
        o_acc[rb][t] = acc;
      }
    }
#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n)
      k_cur[n] = k_pre[n];
#pragma unroll
    for (std::uint32_t n = 0; n < kVRegs; ++n)
      v_cur[n] = v_pre[n];
  }
  if (tid % kLanes == 0)
    row_sum[tid / kLanes] = running_sum;
  __syncthreads();
#pragma unroll
  for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
#pragma unroll
    for (std::uint32_t t = 0; t < 2; ++t) {
      const std::uint32_t dim_tile = wave + t * 8;
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        const std::uint32_t row = 2 * i + half_id;
        const std::uint32_t query = query_start + row;
        if (query >= n_tokens)
          continue;
        const float denominator = row_sum[rb * 16 + row];
        const std::size_t offset =
            static_cast<std::size_t>(query) * kWmmaAttnWidth +
            static_cast<std::size_t>(first_head + rb) * kD + dim_tile * 16 +
            sub;
        float value = denominator > 0.0F ? o_acc[rb][t][i] / denominator : 0.0F;
        if (gate != nullptr)
          value *= SigmoidF(gate[offset]);
        out[offset] = value;
      }
    }
  }
}

bool WmmaCausalAttention(const float* q, const float* gate,
                         const __half* k_cache, const __half* v_cache,
                         float* out, std::uint32_t n_tokens,
                         std::uint32_t start_pos, std::uint32_t heads,
                         std::uint32_t kv_heads, std::uint32_t d,
                         hipStream_t stream, bool last_only) {
  if (heads != kWmmaQueryHeads || kv_heads != kWmmaKvHeads ||
      d != kWmmaHeadDim || n_tokens == 0) {
    return false;
  }
  const std::uint32_t first_group =
      last_only ? (n_tokens - 1) / kWmmaQueryRows : 0;
  const dim3 grid(
      (n_tokens + kWmmaQueryRows - 1) / kWmmaQueryRows - first_group,
      kWmmaKvHeads * (kWmmaGqa / kWmmaHeads));
  hipLaunchKernelGGL((DenseCausalAttentionKernel<16, kWmmaHeads>), grid,
                     dim3(kThreads), 0, stream, q, gate, k_cache, v_cache, out,
                     start_pos, n_tokens, first_group);
  return true;
}

__global__ void ExpertCountsKernel(const std::int32_t* ids,
                                   std::uint32_t* counts, std::size_t slots) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < slots && ids[i] >= 0) {
    atomicAdd(counts + ids[i], 1u);
  }
}

void RouterTopK(const float* logits, std::uint32_t stride, std::int32_t* ids,
                float* weights, std::uint32_t n_tokens, std::uint32_t n_experts,
                std::uint32_t k, hipStream_t stream) {
  if (n_experts <= 512) {
    hipLaunchKernelGGL((RouterTopKKernel<512>), dim3(n_tokens), dim3(kThreads),
                       0, stream, logits, stride, ids, weights, n_experts, k);
  } else {
    hipLaunchKernelGGL((RouterTopKKernel<1024>), dim3(n_tokens), dim3(kThreads),
                       0, stream, logits, stride, ids, weights, n_experts, k);
  }
}

void ExpertCounts(const std::int32_t* ids, std::uint32_t* counts,
                  std::uint32_t n_tokens, std::uint32_t n_experts,
                  std::uint32_t k, hipStream_t stream) {
  (void)hipMemsetAsync(counts, 0, n_experts * sizeof(std::uint32_t), stream);
  const std::size_t slots = static_cast<std::size_t>(n_tokens) * k;
  hipLaunchKernelGGL(ExpertCountsKernel, dim3(Blocks(slots)), dim3(kThreads), 0,
                     stream, ids, counts, slots);
}

void MoeEpilogue(const float* expert_out, const float* weights,
                 const float* shared, const float* gate,
                 std::uint32_t gate_stride, float* out, std::uint32_t n_tokens,
                 std::uint32_t k, std::uint32_t dim, hipStream_t stream) {
  hipLaunchKernelGGL(MoeEpilogueKernel, dim3(n_tokens, Blocks(dim)),
                     dim3(kThreads), 0, stream, expert_out, weights, shared,
                     gate, gate_stride, out, k, dim);
}

void MoeAddRmsNormRows(const float* expert_out, const float* weights,
                       const float* shared, const float* gate,
                       std::uint32_t gate_stride, float* res,
                       const float* gamma, float* out, std::uint32_t n_tokens,
                       std::uint32_t k, std::uint32_t dim, float eps,
                       hipStream_t stream) {
  hipLaunchKernelGGL(MoeAddRmsNormKernel<float>, dim3(n_tokens), dim3(kThreads),
                     0, stream, expert_out, weights, shared, gate, gate_stride,
                     res, gamma, out, k, dim, eps);
}

void MoeAddRmsNormRows(const __half* expert_out, const float* weights,
                       const float* shared, const float* gate,
                       std::uint32_t gate_stride, float* res,
                       const float* gamma, float* out, std::uint32_t n_tokens,
                       std::uint32_t k, std::uint32_t dim, float eps,
                       hipStream_t stream) {
  hipLaunchKernelGGL(MoeAddRmsNormKernel<__half>, dim3(n_tokens),
                     dim3(kThreads), 0, stream, expert_out, weights, shared,
                     gate, gate_stride, res, gamma, out, k, dim, eps);
}

void MoeEpilogueVec4(const float* expert_out, const float* weights,
                     const float* shared, const float* gate,
                     std::uint32_t gate_stride, float* out,
                     std::uint32_t n_tokens, std::uint32_t k, std::uint32_t dim,
                     hipStream_t stream) {
  if (dim % 4 != 0) {
    MoeEpilogue(expert_out, weights, shared, gate, gate_stride, out, n_tokens,
                k, dim, stream);
    return;
  }
  hipLaunchKernelGGL(MoeEpilogueVec4Kernel<float>,
                     dim3(n_tokens, Blocks(dim / 4)), dim3(kThreads), 0, stream,
                     expert_out, weights, shared, gate, gate_stride, out, k,
                     dim);
}

void MoeEpilogueVec4F16(const __half* expert_out, const float* weights,
                        const float* shared, const float* gate,
                        std::uint32_t gate_stride, float* out,
                        std::uint32_t n_tokens, std::uint32_t k,
                        std::uint32_t dim, hipStream_t stream) {
  hipLaunchKernelGGL(MoeEpilogueVec4Kernel<__half>,
                     dim3(n_tokens, Blocks(dim / 4)), dim3(kThreads), 0, stream,
                     expert_out, weights, shared, gate, gate_stride, out, k,
                     dim);
}

void MtpHidden(const float* base, const float* alt, const std::int32_t* row,
               float* dst, std::uint32_t n_tokens, std::uint32_t width,
               const float* base_gamma, const float* alt_gamma, float eps,
               hipStream_t stream) {
  hipLaunchKernelGGL(MtpHiddenKernel, dim3(n_tokens), dim3(kThreads), 0, stream,
                     base, alt, row, dst, width, base_gamma, alt_gamma, eps);
}

void MtpAddEmbedding(const float* embedding, float* residual,
                     std::uint32_t n_tokens, std::uint32_t hidden,
                     std::uint32_t streams, hipStream_t stream) {
  hipLaunchKernelGGL(MtpAddEmbeddingKernel, dim3(n_tokens), dim3(kThreads), 0,
                     stream, embedding, residual, hidden, streams);
}

void Argmax(const float* logits, ArgmaxCandidate* scratch, std::int32_t* out,
            std::uint32_t n_tokens, std::uint32_t vocab, hipStream_t stream) {
  hipLaunchKernelGGL(ArgmaxPartialKernel, dim3(kArgmaxParts, n_tokens),
                     dim3(kThreads), 0, stream, logits, scratch, vocab);
  hipLaunchKernelGGL(ArgmaxFinishKernel, dim3(n_tokens), dim3(kThreads), 0,
                     stream, logits, scratch, out, vocab);
}

void ArgmaxProb(const float* logits, ArgmaxCandidate* scratch,
                std::int32_t* out, float* prob, std::uint32_t vocab,
                hipStream_t stream) {
  auto* sums = reinterpret_cast<float*>(scratch + kArgmaxParts);
  hipLaunchKernelGGL(ArgmaxProbPartialKernel, dim3(kArgmaxParts),
                     dim3(kThreads), 0, stream, logits, scratch, sums, vocab);
  hipLaunchKernelGGL(ArgmaxProbFinishKernel, dim3(1), dim3(kThreads), 0, stream,
                     logits, scratch, sums, out, prob);
}

void PenalizedArgmax(const float* logits, GreedyPenaltyRows penalties,
                     float repeat, float frequency, float presence,
                     PenaltyArgmaxCandidate* partial, ArgmaxCandidate* out,
                     std::uint32_t rows, std::uint32_t vocab,
                     hipStream_t stream) {
  if (rows == 0 || rows > 7 || vocab == 0)
    throw std::invalid_argument("invalid penalty argmax shape");
  PenaltyArgmaxPartialKernel<<<dim3(kArgmaxParts, rows), kThreads, 0, stream>>>(
      logits, penalties, repeat, frequency, presence, partial, vocab);
  PenaltyArgmaxFinishKernel<<<rows, kThreads, 0, stream>>>(logits, partial, out,
                                                           vocab);
}

void GatherArgmaxCandidates(const float* logits, const std::uint32_t* ids,
                            ArgmaxCandidate* out, std::uint32_t rows,
                            std::uint32_t vocab, hipStream_t stream) {
  hipLaunchKernelGGL(GatherArgmaxCandidatesKernel, dim3(Blocks(rows)),
                     dim3(kThreads), 0, stream, logits, ids, out, rows, vocab);
}

template<unsigned Keep>
void SelectMtpCandidates(const float* logits, std::uint32_t* ids,
                         std::uint32_t* scratch_ids, float* scores,
                         std::uint32_t vocab, hipStream_t stream) {
  if (logits == nullptr || ids == nullptr || scratch_ids == nullptr ||
      vocab == 0) {
    throw std::invalid_argument("invalid MTP candidate selection");
  }
  const auto tiles = [](std::uint32_t n) {
    return 1U + (n - 1U) / kMtpCandidateTile;
  };
  unsigned passes = 1;
  for (auto size = vocab; size > kMtpCandidateTile; size = tiles(size) * Keep) {
    ++passes;
  }
  // Choose the first buffer so the final pass always lands in `ids`.
  auto* destination = passes % 2 != 0 ? ids : scratch_ids;
  const std::uint32_t* source = nullptr;
  auto size = vocab;
  for (;;) {
    const auto blocks = tiles(size);
    hipLaunchKernelGGL((MtpCandidateTileKernel<Keep>), dim3(blocks),
                       dim3(kThreads), 0, stream, logits, source, destination,
                       blocks == 1 ? scores : nullptr, size, vocab);
    if (blocks == 1) {
      break;
    }
    size = blocks * Keep;
    source = destination;
    destination = destination == ids ? scratch_ids : ids;
  }
}

std::uint32_t MtpCandidateWorkspaceSize(std::uint32_t vocab) {
  const auto tiles = (vocab + 1023) / 1024;
  return (tiles > 2 ? tiles : 2) * kMtpCandidates;
}

void MtpTopCandidates(const float* logits, std::uint32_t* ids,
                      std::uint32_t* scratch_ids, float* scores,
                      std::uint32_t vocab, hipStream_t stream) {
  if (scores == nullptr)
    throw std::invalid_argument("invalid MTP candidate scores");
  SelectMtpCandidates<kMtpCandidates>(logits, ids, scratch_ids, scores, vocab,
                                      stream);
}

}  // namespace gufo::models::qwen36_35b_a3b::rocm
