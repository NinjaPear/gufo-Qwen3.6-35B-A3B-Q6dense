#ifndef GUFO_TESTS_MODELS_QWEN36_35B_A3B_QUANT_FIXTURES_HPP_
#define GUFO_TESTS_MODELS_QWEN36_35B_A3B_QUANT_FIXTURES_HPP_

#include <hip/hip_fp16.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

// Random GGUF weight blocks with their dequantized values, shared by the
// Qwen3.6 operator tests.
namespace gufo::models::qwen36_35b_a3b::test {

inline std::uint32_t NextRandom(std::uint32_t* state) noexcept {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

inline float Uniform(std::uint32_t* state, float scale) {
  return scale *
         static_cast<float>(static_cast<int>(NextRandom(state) & 0xFFFFU) -
                            32768) /
         32768.0F;
}

/// Packed expert weights plus their dequantized values [e][m][k].
struct Experts {
  std::vector<std::uint8_t> packed;
  std::vector<float> values;
};

/// Random Q4_K blocks with the real 6-bit scale/min packing.
inline Experts MakeQ4K(std::size_t e, std::size_t m, std::size_t k,
                       std::uint32_t seed) {
  Experts w;
  const std::size_t blocks = e * m * (k / 256);
  w.packed.resize(blocks * 144);
  w.values.resize(e * m * k);
  for (std::size_t b = 0; b < blocks; ++b) {
    std::uint8_t* blk = w.packed.data() + b * 144;
    const __half d = __float2half(Uniform(&seed, 0.02F) + 0.03F);
    const __half dmin = __float2half(Uniform(&seed, 0.01F) + 0.015F);
    std::memcpy(blk, &d, 2);
    std::memcpy(blk + 2, &dmin, 2);
    std::uint8_t sc[8];
    std::uint8_t mn[8];
    for (int i = 0; i < 8; ++i) {
      sc[i] = static_cast<std::uint8_t>(NextRandom(&seed) % 64);
      mn[i] = static_cast<std::uint8_t>(NextRandom(&seed) % 64);
    }
    std::uint8_t* scales = blk + 4;
    for (int i = 0; i < 4; ++i) {
      scales[i] = static_cast<std::uint8_t>(sc[i] | ((sc[i + 4] >> 4) << 6));
      scales[i + 4] =
          static_cast<std::uint8_t>(mn[i] | ((mn[i + 4] >> 4) << 6));
      scales[i + 8] = static_cast<std::uint8_t>((sc[i + 4] & 0xF) |
                                                ((mn[i + 4] & 0xF) << 4));
    }
    std::uint8_t* qs = blk + 16;
    for (int i = 0; i < 128; ++i) {
      qs[i] = static_cast<std::uint8_t>(NextRandom(&seed) & 0xFF);
    }
    const float df = __half2float(d);
    const float mf = __half2float(dmin);
    for (int j = 0; j < 256; ++j) {
      const int sb32 = j / 32;
      const int base = (sb32 / 2) * 32 + (j % 32);
      const int shift = 4 * (sb32 & 1);
      const int code = (qs[base] >> shift) & 0xF;
      w.values[b * 256 + j] =
          df * static_cast<float>(sc[sb32]) * static_cast<float>(code) -
          mf * static_cast<float>(mn[sb32]);
    }
  }
  return w;
}

/// Random Q5_K blocks: the Q4_K header, 32 high-bit bytes, 128 nibble bytes.
inline Experts MakeQ5K(std::size_t e, std::size_t m, std::size_t k,
                       std::uint32_t seed) {
  Experts w;
  const std::size_t blocks = e * m * (k / 256);
  w.packed.resize(blocks * 176);
  w.values.resize(e * m * k);
  for (std::size_t b = 0; b < blocks; ++b) {
    std::uint8_t* blk = w.packed.data() + b * 176;
    const __half d = __float2half(Uniform(&seed, 0.02F) + 0.03F);
    const __half dmin = __float2half(Uniform(&seed, 0.01F) + 0.015F);
    std::memcpy(blk, &d, 2);
    std::memcpy(blk + 2, &dmin, 2);
    std::uint8_t sc[8];
    std::uint8_t mn[8];
    for (int i = 0; i < 8; ++i) {
      sc[i] = static_cast<std::uint8_t>(NextRandom(&seed) % 64);
      mn[i] = static_cast<std::uint8_t>(NextRandom(&seed) % 64);
    }
    std::uint8_t* scales = blk + 4;
    for (int i = 0; i < 4; ++i) {
      scales[i] = static_cast<std::uint8_t>(sc[i] | ((sc[i + 4] >> 4) << 6));
      scales[i + 4] =
          static_cast<std::uint8_t>(mn[i] | ((mn[i + 4] >> 4) << 6));
      scales[i + 8] = static_cast<std::uint8_t>((sc[i + 4] & 0xF) |
                                                ((mn[i + 4] & 0xF) << 4));
    }
    std::uint8_t* qh = blk + 16;
    for (int i = 0; i < 32; ++i) {
      qh[i] = static_cast<std::uint8_t>(NextRandom(&seed) & 0xFF);
    }
    std::uint8_t* qs = blk + 48;
    for (int i = 0; i < 128; ++i) {
      qs[i] = static_cast<std::uint8_t>(NextRandom(&seed) & 0xFF);
    }
    const float df = __half2float(d);
    const float mf = __half2float(dmin);
    for (int j = 0; j < 256; ++j) {
      const int sb32 = j / 32;
      const int base = (sb32 / 2) * 32 + (j % 32);
      const int shift = 4 * (sb32 & 1);
      const int code =
          ((qs[base] >> shift) & 0xF) | (((qh[j % 32] >> sb32) & 1) << 4);
      w.values[b * 256 + j] =
          df * static_cast<float>(sc[sb32]) * static_cast<float>(code) -
          mf * static_cast<float>(mn[sb32]);
    }
  }
  return w;
}

/// Random Q8_0 blocks: d and 32 signed codes.
inline Experts MakeQ8_0(std::size_t e, std::size_t m, std::size_t k,
                        std::uint32_t seed) {
  Experts w;
  const std::size_t blocks = e * m * (k / 32);
  w.packed.resize(blocks * 34);
  w.values.resize(e * m * k);
  for (std::size_t b = 0; b < blocks; ++b) {
    std::uint8_t* blk = w.packed.data() + b * 34;
    const __half d = __float2half(Uniform(&seed, 0.02F) + 0.03F);
    std::memcpy(blk, &d, 2);
    const float df = __half2float(d);
    for (int j = 0; j < 32; ++j) {
      const auto q = static_cast<std::int8_t>(NextRandom(&seed) & 0xFF);
      blk[2 + j] = static_cast<std::uint8_t>(q);
      w.values[b * 32 + j] = df * static_cast<float>(q);
    }
  }
  return w;
}

/// Random Q6_K blocks: 128 low-nibble bytes, 64 high-bit bytes, 16 signed
/// scales and a trailing F16 d.
inline Experts MakeQ6K(std::size_t e, std::size_t m, std::size_t k,
                       std::uint32_t seed) {
  Experts w;
  const std::size_t blocks = e * m * (k / 256);
  w.packed.resize(blocks * 210);
  w.values.resize(e * m * k);
  for (std::size_t b = 0; b < blocks; ++b) {
    std::uint8_t* blk = w.packed.data() + b * 210;
    for (int i = 0; i < 192; ++i)
      blk[i] = static_cast<std::uint8_t>(NextRandom(&seed) & 0xFF);
    auto* sc = reinterpret_cast<std::int8_t*>(blk + 192);
    for (int i = 0; i < 16; ++i)
      sc[i] = static_cast<std::int8_t>(int(NextRandom(&seed) % 64) - 32);
    const __half d = __float2half(Uniform(&seed, 0.001F) + 0.002F);
    std::memcpy(blk + 208, &d, 2);
    const float df = __half2float(d);
    float* y = w.values.data() + b * 256;
    for (int half = 0; half < 2; ++half) {
      const std::uint8_t* ql = blk + half * 64;
      const std::uint8_t* qh = blk + 128 + half * 32;
      const std::int8_t* s8 = sc + half * 8;
      for (int l = 0; l < 32; ++l) {
        const int is = l / 16;
        const int q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
        const int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
        const int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        const int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
        y[half * 128 + l] = df * s8[is] * q1;
        y[half * 128 + l + 32] = df * s8[is + 2] * q2;
        y[half * 128 + l + 64] = df * s8[is + 4] * q3;
        y[half * 128 + l + 96] = df * s8[is + 6] * q4;
      }
    }
  }
  return w;
}

}  // namespace gufo::models::qwen36_35b_a3b::test

#endif  // GUFO_TESTS_MODELS_QWEN36_35B_A3B_QUANT_FIXTURES_HPP_
