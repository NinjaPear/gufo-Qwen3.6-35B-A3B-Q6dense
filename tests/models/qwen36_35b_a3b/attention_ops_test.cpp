#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "src/models/qwen36_35b_a3b/kernels/rocm/kernels.hpp"

namespace q = gufo::models::qwen36_35b_a3b::rocm;
namespace {

// The model's full-attention geometry: 16 query heads over 2 KV heads,
// head_dim 256.
constexpr std::uint32_t kHeads = 16;
constexpr std::uint32_t kKvHeads = 2;
constexpr std::uint32_t kDim = 256;
constexpr std::uint32_t kQWidth = kHeads * kDim;
constexpr std::uint32_t kKvWidth = kKvHeads * kDim;
// The WMMA routes round Q and P to FP16; their outputs (order one) stay
// within this absolute envelope of an FP64 evaluation.
constexpr double kWmmaLimit = 2e-2;

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

template<typename T>
class HipBuffer {
public:
  explicit HipBuffer(std::size_t count) : count_(count) {
    void* allocation = nullptr;
    CheckHip(hipMalloc(&allocation, bytes()), "hipMalloc");
    CheckHip(hipMemset(allocation, 0, bytes()), "hipMemset");
    data_ = static_cast<T*>(allocation);
  }
  ~HipBuffer() {
    if (data_ != nullptr) {
      (void)hipFree(data_);
    }
  }

  HipBuffer(const HipBuffer&) = delete;
  HipBuffer& operator=(const HipBuffer&) = delete;
  HipBuffer(HipBuffer&&) = delete;
  HipBuffer& operator=(HipBuffer&&) = delete;

  [[nodiscard]] T* get() noexcept { return data_; }
  [[nodiscard]] std::size_t bytes() const noexcept {
    return count_ * sizeof(T);
  }

private:
  T* data_{nullptr};
  std::size_t count_{0};
};

std::uint32_t NextRandom(std::uint32_t* state) noexcept {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

std::vector<float> MakeValues(std::size_t count, std::uint32_t seed,
                              float scale) {
  std::vector<float> values(count);
  for (float& value : values) {
    value = scale *
            static_cast<float>(static_cast<int>(NextRandom(&seed) & 0xFFFFU) -
                               32768) /
            32768.0F;
  }
  return values;
}

template<typename T>
void Upload(HipBuffer<T>* destination, const std::vector<T>& source) {
  CheckHip(hipMemcpy(destination->get(), source.data(), destination->bytes(),
                     hipMemcpyHostToDevice),
           "upload");
}

std::vector<float> Download(HipBuffer<float>* source, std::size_t count) {
  std::vector<float> values(count);
  CheckHip(hipMemcpy(values.data(), source->get(), source->bytes(),
                     hipMemcpyDeviceToHost),
           "download");
  return values;
}

void CheckPreparation(std::uint32_t n, std::uint32_t start,
                      std::uint32_t rotary) {
  constexpr std::uint32_t stride = 2 * (kQWidth + kKvWidth);
  const std::size_t count = static_cast<std::size_t>(n) * kQWidth;
  const std::size_t cache_rows = start + n + 2;
  HipBuffer<float> packed(static_cast<std::size_t>(n) * stride);
  HipBuffer<float> q_gamma(kDim), k_gamma(kDim);
  HipBuffer<float> q_ref(count), gate_ref(count), q_out(count), gate_out(count);
  HipBuffer<float> k(static_cast<std::size_t>(n) * kKvWidth);
  HipBuffer<float> v(static_cast<std::size_t>(n) * kKvWidth);
  HipBuffer<__half> k_ref(cache_rows * kKvWidth), v_ref(cache_rows * kKvWidth);
  HipBuffer<__half> k_out(cache_rows * kKvWidth), v_out(cache_rows * kKvWidth);
  HipBuffer<std::uint32_t> pos(1);
  Upload(&packed, MakeValues(static_cast<std::size_t>(n) * stride, 17, 4.0F));
  auto qg = MakeValues(kDim, 37, 0.5F);
  auto kg = MakeValues(kDim, 91, 0.5F);
  for (auto& x : qg)
    x += 1.0F;
  for (auto& x : kg)
    x += 1.0F;
  Upload(&q_gamma, qg);
  Upload(&k_gamma, kg);

  struct Capture {
    hipStream_t stream{nullptr};
    hipGraph_t graph{nullptr};
    hipGraphExec_t exec{nullptr};
    Capture() {
      CheckHip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
               "preparation stream");
    }
    ~Capture() {
      if (exec)
        (void)hipGraphExecDestroy(exec);
      if (graph)
        (void)hipGraphDestroy(graph);
      (void)hipStreamDestroy(stream);
    }
  } capture;
  const auto same = [&](auto* expected, auto* actual, std::size_t size,
                        const char* name, std::uint32_t replay) {
    using T = std::remove_pointer_t<decltype(expected)>;
    std::vector<T> a(size), b(size);
    CheckHip(
        hipMemcpy(a.data(), expected, size * sizeof(T), hipMemcpyDeviceToHost),
        "preparation reference");
    CheckHip(
        hipMemcpy(b.data(), actual, size * sizeof(T), hipMemcpyDeviceToHost),
        "preparation result");
    for (std::size_t i = 0; i < size; ++i) {
      if (std::memcmp(&a[i], &b[i], sizeof(T)) == 0)
        continue;
      std::cerr << "preparation n=" << n << " start=" << start
                << " rotary=" << rotary << " replay=" << replay
                << " index=" << i << " expected=" << std::hexfloat
                << static_cast<float>(a[i])
                << " actual=" << static_cast<float>(b[i]) << std::defaultfloat
                << '\n';
      throw std::runtime_error(std::string("attention preparation changed ") +
                               name);
    }
  };
  for (std::uint32_t replay = 0; replay < 2; ++replay) {
    const std::vector<std::uint32_t> position{start + replay};
    Upload(&pos, position);
    q::UnpackQGate(packed.get(), stride, q_ref.get(), gate_ref.get(), k.get(),
                   v.get(), n, kHeads, kDim, kKvWidth, nullptr);
    q::RmsNormRows(q_ref.get(), q_gamma.get(), q_ref.get(), n * kHeads, kDim, 1,
                   1e-6F, nullptr);
    q::RmsNormRows(k.get(), k_gamma.get(), k.get(), n * kKvHeads, kDim, 1,
                   1e-6F, nullptr);
    q::Rope(q_ref.get(), n, kHeads, kDim, rotary, pos.get(), 1e7F, nullptr);
    q::Rope(k.get(), n, kKvHeads, kDim, rotary, pos.get(), 1e7F, nullptr);
    q::StoreKv(k.get(), k_ref.get(), n, kKvWidth, pos.get(), nullptr);
    q::StoreKv(v.get(), v_ref.get(), n, kKvWidth, pos.get(), nullptr);
    CheckHip(hipDeviceSynchronize(), "preparation reference ready");
    if (replay == 0) {
      CheckHip(
          hipStreamBeginCapture(capture.stream, hipStreamCaptureModeGlobal),
          "preparation capture");
      const bool supported = q::PrepareAttention(
          packed.get(), stride, q_gamma.get(), k_gamma.get(), q_out.get(),
          gate_out.get(), k_out.get(), v_out.get(), n, kHeads, kKvHeads, kDim,
          rotary, pos.get(), 1e7F, 1e-6F, capture.stream);
      CheckHip(hipStreamEndCapture(capture.stream, &capture.graph),
               "preparation capture end");
      if (!supported)
        throw std::runtime_error("attention preparation rejected geometry");
      CheckHip(hipGraphInstantiate(&capture.exec, capture.graph, nullptr,
                                   nullptr, 0),
               "preparation instantiate");
    }
    CheckHip(hipGraphLaunch(capture.exec, capture.stream),
             "preparation replay");
    CheckHip(hipStreamSynchronize(capture.stream), "preparation ready");
    same(q_ref.get(), q_out.get(), count, "queries", replay);
    same(gate_ref.get(), gate_out.get(), count, "gates", replay);
    // Include the neighboring cache rows and replay at a new device position.
    const std::size_t first = start == 0 ? 0 : start - 1;
    const std::size_t window = (cache_rows - first) * kKvWidth;
    same(k_ref.get() + first * kKvWidth, k_out.get() + first * kKvWidth, window,
         "key cache", replay);
    same(v_ref.get() + first * kKvWidth, v_out.get() + first * kKvWidth, window,
         "value cache", replay);
  }
  if (n >= 32) {
    // Each small prefill piece must match the full preparation, including
    // normalization rounding and absolute rotary positions.
    for (const std::uint32_t chunk : {1U, 8U, 9U, 16U, 32U}) {
      for (std::uint32_t off = 0; off < n; off += chunk) {
        Upload(&pos, std::vector<std::uint32_t>{start + 1 + off});
        if (!q::PrepareAttention(
                packed.get() + std::size_t(off) * stride, stride, q_gamma.get(),
                k_gamma.get(), q_out.get() + std::size_t(off) * kQWidth,
                gate_out.get() + std::size_t(off) * kQWidth, k_out.get(),
                v_out.get(), std::min(chunk, n - off), kHeads, kKvHeads, kDim,
                rotary, pos.get(), 1e7F, 1e-6F, nullptr, nullptr, true))
          throw std::runtime_error("prefill preparation rejected a short tail");
      }
      CheckHip(hipDeviceSynchronize(), "short prefill preparation");
      same(q_ref.get(), q_out.get(), count, "prefill queries", chunk);
      same(gate_ref.get(), gate_out.get(), count, "prefill gates", chunk);
      same(k_ref.get(), k_out.get(), cache_rows * kKvWidth, "prefill keys",
           chunk);
      same(v_ref.get(), v_out.get(), cache_rows * kKvWidth, "prefill values",
           chunk);
    }
  }
  std::cout << "attention preparation n=" << n << " start=" << start
            << " rotary=" << rotary << ": exact, including graph positions\n";
}

/// FP64 attention of each row over every key below pos + 1, gated by
/// sigmoid(gate) when `gate` is non-empty. Keys/values come from `kv(row)`.
template<typename Kv>
std::vector<double> Fp64Attention(const std::vector<float>& qv,
                                  const std::vector<float>& gate, const Kv& kv,
                                  std::uint32_t n_tokens,
                                  std::uint32_t start_pos) {
  std::vector<double> out(static_cast<std::size_t>(n_tokens) * kQWidth);
  const double scale = 1.0 / std::sqrt(static_cast<double>(kDim));
  std::vector<double> scores;
  for (std::uint32_t t = 0; t < n_tokens; ++t) {
    const std::uint32_t n_kv = start_pos + t + 1;
    for (std::uint32_t h = 0; h < kHeads; ++h) {
      const std::size_t row = (static_cast<std::size_t>(t) * kHeads + h) * kDim;
      const std::size_t kv_head = (h / (kHeads / kKvHeads)) * kDim;
      scores.assign(n_kv, 0.0);
      double maximum = -INFINITY;
      for (std::uint32_t j = 0; j < n_kv; ++j) {
        const std::size_t base = j * std::size_t{kKvWidth} + kv_head;
        double dot = 0.0;
        for (std::uint32_t d = 0; d < kDim; ++d)
          dot += static_cast<double>(qv[row + d]) * kv.Key(base + d);
        scores[j] = dot * scale;
        maximum = std::max(maximum, scores[j]);
      }
      double sum = 0.0;
      for (double& score : scores) {
        score = std::exp(score - maximum);
        sum += score;
      }
      for (std::uint32_t d = 0; d < kDim; ++d) {
        double acc = 0.0;
        for (std::uint32_t j = 0; j < n_kv; ++j)
          acc += scores[j] * kv.Value(j * std::size_t{kKvWidth} + kv_head + d);
        const double g =
            gate.empty() ? INFINITY : static_cast<double>(gate[row + d]);
        out[row + d] = acc / sum / (1.0 + std::exp(-g));
      }
    }
  }
  return out;
}

struct HalfKv {
  const std::vector<__half>& k;
  const std::vector<__half>& v;
  double Key(std::size_t i) const { return __half2float(k[i]); }
  double Value(std::size_t i) const { return __half2float(v[i]); }
};

std::vector<__half> ToHalf(const std::vector<float>& values) {
  std::vector<__half> half(values.size());
  std::transform(values.begin(), values.end(), half.begin(),
                 [](float x) { return __float2half(x); });
  return half;
}

/// Prefill attention (gated, causal, WMMA) against FP64, its replay and the
/// last-tile mode that only writes the final query tile.
double CheckPrefill(std::uint32_t n_tokens, std::uint32_t start_pos,
                    std::uint32_t seed) {
  const std::uint32_t n_kv = start_pos + n_tokens;
  const std::size_t q_count = static_cast<std::size_t>(n_tokens) * kQWidth;
  const std::size_t kv_count = static_cast<std::size_t>(n_kv) * kKvWidth;
  const auto qv = MakeValues(q_count, seed, 4.0F);
  const auto gate = MakeValues(q_count, seed ^ 0x5555U, 3.0F);
  const auto kh = ToHalf(MakeValues(kv_count, seed ^ 0xAAAAU, 1.0F));
  const auto vh = ToHalf(MakeValues(kv_count, seed ^ 0x3333U, 1.0F));
  HipBuffer<float> d_q(q_count), d_gate(q_count), d_out(q_count);
  HipBuffer<__half> d_k(kv_count), d_v(kv_count);
  Upload(&d_q, qv);
  Upload(&d_gate, gate);
  Upload(&d_k, kh);
  Upload(&d_v, vh);
  const auto run = [&](bool last_only) {
    if (!q::WmmaCausalAttention(d_q.get(), d_gate.get(), d_k.get(), d_v.get(),
                                d_out.get(), n_tokens, start_pos, kHeads,
                                kKvHeads, kDim, nullptr, last_only))
      throw std::runtime_error("prefill attention rejected the model geometry");
    CheckHip(hipDeviceSynchronize(), "prefill attention");
    return Download(&d_out, q_count);
  };
  const auto out = run(false);
  if (out != run(false))
    throw std::runtime_error("prefill attention replay changed the output");
  constexpr float poison = -12345.0F;
  Upload(&d_out, std::vector<float>(q_count, poison));
  const auto tail = run(true);
  constexpr std::uint32_t kTileRows = 16;
  const std::size_t begin =
      static_cast<std::size_t>((n_tokens - 1) / kTileRows * kTileRows) *
      kQWidth;
  for (std::size_t i = 0; i < q_count; ++i) {
    if (i < begin ? tail[i] != poison
                  : std::memcmp(&out[i], &tail[i], sizeof(float)) != 0)
      throw std::runtime_error("last-tile attention changed a row or guard");
  }
  const auto exact =
      Fp64Attention(qv, gate, HalfKv{kh, vh}, n_tokens, start_pos);
  double worst = 0.0;
  for (std::size_t i = 0; i < q_count; ++i) {
    if (!std::isfinite(out[i]))
      throw std::runtime_error("prefill attention output is not finite");
    worst = std::max(worst, std::abs(out[i] - exact[i]));
  }
  if (q::WmmaCausalAttention(d_q.get(), d_gate.get(), d_k.get(), d_v.get(),
                             d_out.get(), n_tokens, start_pos, 24, kKvHeads,
                             kDim, nullptr))
    throw std::runtime_error("prefill attention accepted another geometry");
  return worst;
}

/// Decode attention reads the 8-bit cache copy. QuantizeKv must stay within
/// half a step of the F16 cache and leave other rows alone; the attention
/// must match FP64 over the dequantized copy for every split count, and a
/// row's result must not depend on how many rows share the launch (MTP
/// verification widths change every cycle).
void CheckDecode(std::uint32_t start_pos, std::uint32_t seed) {
  constexpr std::uint32_t kMaxRows = 8;
  const std::uint32_t n_kv = start_pos + kMaxRows;
  const std::size_t kv_count = static_cast<std::size_t>(n_kv) * kKvWidth;
  const std::size_t q_count = std::size_t{kMaxRows} * kQWidth;
  const auto qv = MakeValues(q_count, seed, 4.0F);
  const auto kh = ToHalf(MakeValues(kv_count, seed ^ 0xAAAAU, 1.0F));
  const auto vh = ToHalf(MakeValues(kv_count, seed ^ 0x3333U, 1.0F));
  HipBuffer<float> d_q(q_count), d_out(q_count);
  HipBuffer<__half> d_k(kv_count), d_v(kv_count);
  HipBuffer<std::int8_t> d_kq(kv_count + 64), d_vq(kv_count + 64);
  HipBuffer<__half> d_ks(kv_count / 32 + 8), d_vs(kv_count / 32 + 8);
  HipBuffer<std::uint32_t> d_pos(1);
  Upload(&d_q, qv);
  Upload(&d_k, kh);
  Upload(&d_v, vh);
  constexpr std::int8_t kGuard = 0x5A;
  for (auto* b : {&d_kq, &d_vq})
    CheckHip(hipMemset(b->get(), kGuard, b->bytes()), "KV8 guards");
  // Quantize in two pieces: the prefix, then the decode rows.
  Upload(&d_pos, std::vector<std::uint32_t>{0});
  if (start_pos > 0) {
    q::QuantizeKv(d_k.get(), d_kq.get(), d_ks.get(), start_pos, kKvWidth,
                  d_pos.get(), nullptr);
    q::QuantizeKv(d_v.get(), d_vq.get(), d_vs.get(), start_pos, kKvWidth,
                  d_pos.get(), nullptr);
  }
  Upload(&d_pos, std::vector<std::uint32_t>{start_pos});
  q::QuantizeKv(d_k.get(), d_kq.get(), d_ks.get(), kMaxRows, kKvWidth,
                d_pos.get(), nullptr);
  q::QuantizeKv(d_v.get(), d_vq.get(), d_vs.get(), kMaxRows, kKvWidth,
                d_pos.get(), nullptr);
  CheckHip(hipDeviceSynchronize(), "KV8 quantization");
  std::vector<std::int8_t> kq(kv_count + 64), vq(kv_count + 64);
  std::vector<__half> ks(kv_count / 32), vs(kv_count / 32);
  CheckHip(hipMemcpy(kq.data(), d_kq.get(), kq.size(), hipMemcpyDeviceToHost),
           "KV8 keys");
  CheckHip(hipMemcpy(vq.data(), d_vq.get(), vq.size(), hipMemcpyDeviceToHost),
           "KV8 values");
  CheckHip(hipMemcpy(ks.data(), d_ks.get(), ks.size() * sizeof(__half),
                     hipMemcpyDeviceToHost),
           "KV8 key scales");
  CheckHip(hipMemcpy(vs.data(), d_vs.get(), vs.size() * sizeof(__half),
                     hipMemcpyDeviceToHost),
           "KV8 value scales");
  for (std::size_t i = kv_count; i < kq.size(); ++i)
    if (kq[i] != kGuard || vq[i] != kGuard)
      throw std::runtime_error("KV8 quantization wrote past the cache");
  for (std::size_t i = 0; i < kv_count; ++i) {
    const auto check = [&](const std::vector<__half>& f16,
                           const std::vector<std::int8_t>& q8,
                           const std::vector<__half>& scales) {
      const double step = __half2float(scales[i / 32]);
      const double x = __half2float(f16[i]);
      if (std::abs(q8[i] * step - x) > 0.5 * step + 1e-3 * std::abs(x) + 1e-7)
        throw std::runtime_error("KV8 value outside half a quantization step");
    };
    check(kh, kq, ks);
    check(vh, vq, vs);
  }
  struct Q8Kv {
    const std::vector<std::int8_t>& kq;
    const std::vector<std::int8_t>& vq;
    const std::vector<__half>& ks;
    const std::vector<__half>& vs;
    double Key(std::size_t i) const {
      return kq[i] * double(__half2float(ks[i / 32]));
    }
    double Value(std::size_t i) const {
      return vq[i] * double(__half2float(vs[i / 32]));
    }
  };
  const q::KvQ8 kv{d_kq.get(), d_vq.get(), d_ks.get(), d_vs.get()};
  const auto exact =
      Fp64Attention(qv, {}, Q8Kv{kq, vq, ks, vs}, kMaxRows, start_pos);
  for (const std::uint32_t splits : {32U, 128U}) {
    HipBuffer<float> partials(std::size_t{kMaxRows} * kHeads * splits *
                              (kDim + 2));
    // Row t alone at position start_pos + t.
    std::vector<float> single(q_count);
    for (std::uint32_t t = 0; t < kMaxRows; ++t) {
      Upload(&d_pos, std::vector<std::uint32_t>{start_pos + t});
      if (!q::GqaDecodeAttention(d_q.get() + std::size_t(t) * kQWidth, kv,
                                 d_out.get(), partials.get(), splits, 1,
                                 d_pos.get(), kHeads, kKvHeads, kDim, nullptr))
        throw std::runtime_error("decode attention rejected one row");
      CheckHip(hipMemcpy(single.data() + std::size_t(t) * kQWidth, d_out.get(),
                         kQWidth * sizeof(float), hipMemcpyDeviceToHost),
               "decode row");
    }
    double worst = 0.0;
    for (std::size_t i = 0; i < q_count; ++i) {
      if (!std::isfinite(single[i]))
        throw std::runtime_error("decode attention output is not finite");
      worst = std::max(worst, std::abs(single[i] - exact[i]));
    }
    std::cout << "KV8 decode attention start=" << start_pos
              << " splits=" << splits << ": worst |GPU - F64| " << worst
              << '\n';
    if (worst > kWmmaLimit)
      throw std::runtime_error("decode attention exceeds its FP64 envelope");
    Upload(&d_pos, std::vector<std::uint32_t>{start_pos});
    for (std::uint32_t n = 2; n <= kMaxRows; ++n) {
      Upload(&d_out, std::vector<float>(q_count, -12345.0F));
      if (!q::GqaDecodeAttention(d_q.get(), kv, d_out.get(), partials.get(),
                                 splits, n, d_pos.get(), kHeads, kKvHeads, kDim,
                                 nullptr))
        throw std::runtime_error("decode attention rejected a batch");
      const auto batch = Download(&d_out, q_count);
      if (std::memcmp(batch.data(), single.data(),
                      std::size_t(n) * kQWidth * sizeof(float)) != 0)
        throw std::runtime_error("decode attention changes with batch width " +
                                 std::to_string(n));
      for (std::size_t i = std::size_t(n) * kQWidth; i < q_count; ++i)
        if (batch[i] != -12345.0F)
          throw std::runtime_error("decode attention wrote an inactive row");
    }
  }
  HipBuffer<float> partials(std::size_t{9} * kHeads * 32 * (kDim + 2));
  if (q::GqaDecodeAttention(d_q.get(), kv, d_out.get(), partials.get(), 32, 9,
                            d_pos.get(), kHeads, kKvHeads, kDim, nullptr) ||
      q::GqaDecodeAttention(d_q.get(), kv, d_out.get(), partials.get(), 32, 1,
                            d_pos.get(), 24, kKvHeads, kDim, nullptr))
    throw std::runtime_error("decode attention accepted unsupported geometry");
}

void CheckChunks(std::uint32_t n, std::uint32_t split) {
  const std::size_t count = std::size_t(n) * kQWidth;
  HipBuffer<float> queries(count), gates(count), full(count), chunked(count);
  HipBuffer<__half> keys(std::size_t(n) * kKvWidth),
      values(std::size_t(n) * kKvWidth);
  Upload(&queries, MakeValues(count, 412, 4.0F));
  Upload(&gates, MakeValues(count, 721, 3.0F));
  for (auto* destination : {&keys, &values}) {
    const auto f = MakeValues(std::size_t(n) * kKvWidth,
                              destination == &keys ? 891 : 347, 1.0F);
    std::vector<__half> half(f.size());
    std::transform(f.begin(), f.end(), half.begin(),
                   [](float x) { return __float2half(x); });
    Upload(destination, half);
  }
  const auto run = [&](std::uint32_t start, std::uint32_t rows, float* out) {
    const auto offset = std::size_t(start) * kQWidth;
    if (!q::WmmaCausalAttention(queries.get() + offset, gates.get() + offset,
                                keys.get(), values.get(), out + offset, rows,
                                start, kHeads, kKvHeads, kDim, nullptr)) {
      throw std::runtime_error("chunk attention rejected geometry");
    }
  };
  run(0, n, full.get());
  run(0, split, chunked.get());
  run(split, n - split, chunked.get());
  const auto a = Download(&full, count);
  const auto b = Download(&chunked, count);
  std::size_t differences = 0, first = count;
  float max_error = 0;
  for (std::size_t i = 0; i < count; ++i) {
    if (a[i] != b[i]) {
      ++differences;
      first = std::min(first, i);
      max_error = std::max(max_error, std::abs(a[i] - b[i]));
    }
  }
  std::cout << "chunk attention n=" << n << " split=" << split
            << " differing=" << differences << " max=" << max_error
            << " first_row=" << first / kQWidth << '\n';
  for (std::size_t i = 0, shown = 0; i < count && shown < 12; ++i)
    if (a[i] != b[i] && (i == 0 || a[i - 1] == b[i - 1]))
      std::cout << "  row " << i / kQWidth << " head " << i % kQWidth / kDim
                << " dim " << i % kDim << '\n',
          ++shown;
  if (differences != 0)
    throw std::runtime_error("attention depends on prefill chunk boundary");
}

}  // namespace

int main() {
  try {
    // Chunks ending on, just after and inside a 16-key round: the last row
    // of the first chunk may be alone in its query block.
    for (const std::uint32_t split : {17U, 18U, 33U, 94U, 97U, 129U})
      CheckChunks(136, split);
    for (const std::uint32_t split : {1025U, 1039U})
      CheckChunks(2048, split);
    CheckPreparation(1, 0, 64);
    CheckPreparation(8, 4096, 64);
    CheckPreparation(65, 131069, 64);
    CheckPreparation(7, 131069, 256);
    for (const std::uint32_t start : {0U, 1U, 37U, 4093U, 16381U})
      CheckDecode(start, 0xD0C0DE00U + start);
    struct Case {
      std::uint32_t n_tokens;
      std::uint32_t start_pos;
    };
    const Case cases[] = {{1, 0},     {4, 4096},  {100, 0},   {64, 37},
                          {77, 51},   {17, 2047}, {96, 4096}, {130, 8000},
                          {5, 32765}, {3, 65533}};
    bool ok = true;
    std::uint32_t seed = 0x1234ABCDU;
    for (const Case& c : cases) {
      const double worst = CheckPrefill(c.n_tokens, c.start_pos, seed++);
      std::cout << "prefill attention n=" << c.n_tokens
                << " start=" << c.start_pos << ": worst |GPU - F64| " << worst
                << '\n';
      ok = ok && worst < kWmmaLimit;
    }
    return ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
