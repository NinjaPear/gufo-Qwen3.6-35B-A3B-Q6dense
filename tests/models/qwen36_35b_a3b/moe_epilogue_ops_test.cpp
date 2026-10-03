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

// The model's MoE combine geometry: top-8 slots over a 2,048-wide hidden
// state, router row stride num_experts + 1.
constexpr std::uint32_t kTokens = 37;
constexpr std::uint32_t kSlots = 8;
constexpr std::uint32_t kHidden = 2048;
constexpr std::uint32_t kGateStride = 257;

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
                              float scale, float offset = 0.0F) {
  std::vector<float> values(count);
  for (float& value : values) {
    value = offset +
            scale *
                static_cast<float>(
                    static_cast<int>(NextRandom(&seed) & 0xFFFFU) - 32768) /
                32768.0F;
  }
  return values;
}

void Upload(HipBuffer<float>* destination, const std::vector<float>& source) {
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

/// The combine that decode and prefill run: the routed rows (F32, or F16
/// from the WMMA route) weighted, plus the gated shared expert, added into
/// the residual, then the next block's RMSNorm. Both outputs must match an
/// FP64 evaluation, the normalized rows must stay untouched without gamma,
/// and each row must not depend on how many rows share the launch.
template<typename ExpertT>
void CheckAddRmsNorm(const std::vector<float>& expert_values,
                     const std::vector<float>& weights,
                     const std::vector<float>& shared,
                     const std::vector<float>& gate,
                     HipBuffer<float>* d_weights, HipBuffer<float>* d_shared,
                     HipBuffer<float>* d_gate) {
  constexpr std::size_t kOut = static_cast<std::size_t>(kTokens) * kHidden;
  std::vector<ExpertT> experts(expert_values.size());
  std::vector<double> rounded(expert_values.size());
  for (std::size_t i = 0; i < experts.size(); ++i) {
    if constexpr (std::is_same_v<ExpertT, float>) {
      experts[i] = expert_values[i];
      rounded[i] = expert_values[i];
    } else {
      experts[i] = __float2half(expert_values[i]);
      rounded[i] = __half2float(experts[i]);
    }
  }
  HipBuffer<ExpertT> d_experts(experts.size());
  CheckHip(hipMemcpy(d_experts.get(), experts.data(), d_experts.bytes(),
                     hipMemcpyHostToDevice),
           "expert upload");
  const auto residual = MakeValues(kOut, 0x5EEDU, 2.0F);
  auto gamma = MakeValues(kHidden, 0x6A33U, 0.4F, 1.0F);
  HipBuffer<float> d_res(kOut), d_gamma(kHidden), d_out(kOut + 8);
  Upload(&d_gamma, gamma);
  const auto run = [&](std::uint32_t first, std::uint32_t rows,
                       const float* g) {
    q::MoeAddRmsNormRows(
        d_experts.get() + std::size_t{first} * kSlots * kHidden,
        d_weights->get() + std::size_t{first} * kSlots,
        d_shared->get() + std::size_t{first} * kHidden,
        d_gate->get() + std::size_t{first} * kGateStride, kGateStride,
        d_res.get() + std::size_t{first} * kHidden, g,
        d_out.get() + std::size_t{first} * kHidden, rows, kSlots, kHidden,
        1e-6F, nullptr);
  };
  Upload(&d_res, residual);
  CheckHip(hipMemset(d_out.get(), 0xA5, d_out.bytes()), "poison output");
  run(0, kTokens, nullptr);
  const auto ungated = Download(&d_res, kOut);
  {
    std::vector<float> out(kOut + 8);
    CheckHip(hipMemcpy(out.data(), d_out.get(), d_out.bytes(),
                       hipMemcpyDeviceToHost),
             "ungated output");
    for (const float v : out) {
      std::uint32_t bits;
      std::memcpy(&bits, &v, sizeof(bits));
      if (bits != 0xA5A5A5A5U)
        throw std::runtime_error("combine without gamma wrote normalized rows");
    }
  }
  Upload(&d_res, residual);
  run(0, kTokens, d_gamma.get());
  const auto res = Download(&d_res, kOut);
  std::vector<float> out(kOut);
  CheckHip(hipMemcpy(out.data(), d_out.get(), kOut * sizeof(float),
                     hipMemcpyDeviceToHost),
           "normalized output");
  if (std::memcmp(res.data(), ungated.data(), kOut * sizeof(float)) != 0)
    throw std::runtime_error("gamma changed the combined residual");
  double worst_res = 0.0, worst_out = 0.0;
  for (std::uint32_t t = 0; t < kTokens; ++t) {
    const double g = 1.0 / (1.0 + std::exp(-double(gate[t * kGateStride])));
    std::vector<double> row(kHidden);
    double ss = 0.0;
    for (std::uint32_t i = 0; i < kHidden; ++i) {
      double acc = 0.0;
      for (std::uint32_t s = 0; s < kSlots; ++s)
        acc += double(weights[t * kSlots + s]) *
               rounded[(std::size_t{t} * kSlots + s) * kHidden + i];
      row[i] = residual[std::size_t{t} * kHidden + i] + acc +
               g * shared[std::size_t{t} * kHidden + i];
      ss += row[i] * row[i];
    }
    const double scale = 1.0 / std::sqrt(ss / kHidden + 1e-6);
    for (std::uint32_t i = 0; i < kHidden; ++i) {
      const std::size_t idx = std::size_t{t} * kHidden + i;
      worst_res = std::max(worst_res, std::abs(res[idx] - row[i]));
      worst_out =
          std::max(worst_out, std::abs(out[idx] - row[i] * scale * gamma[i]));
    }
  }
  if (worst_res > 1e-5 || worst_out > 1e-5)
    throw std::runtime_error("fused combine differs from FP64");
  // Row by row: no row depends on the launch width.
  Upload(&d_res, residual);
  for (std::uint32_t t = 0; t < kTokens; ++t)
    run(t, 1, d_gamma.get());
  if (Download(&d_res, kOut) != res)
    throw std::runtime_error("fused combine residual depends on launch width");
  std::vector<float> single(kOut);
  CheckHip(hipMemcpy(single.data(), d_out.get(), kOut * sizeof(float),
                     hipMemcpyDeviceToHost),
           "single-row output");
  if (single != out)
    throw std::runtime_error("fused combine norm depends on launch width");
  std::cout << "fused MoE combine + RMSNorm ("
            << (std::is_same_v<ExpertT, float> ? "F32" : "F16")
            << " experts): worst residual " << worst_res << ", norm "
            << worst_out << ", width independent\n";
}

}  // namespace

int main() {
  try {
    constexpr std::size_t kExpert =
        static_cast<std::size_t>(kTokens) * kSlots * kHidden;
    constexpr std::size_t kOut = static_cast<std::size_t>(kTokens) * kHidden;
    const auto expert_out = MakeValues(kExpert, 0x1234ABCDU, 1.0F);
    const auto weights = MakeValues(static_cast<std::size_t>(kTokens) * kSlots,
                                    0xBADC0FFEU, 0.5F, 0.5F);
    const auto shared = MakeValues(kOut, 0xDEADBEEFU, 1.0F);
    const auto gate = MakeValues(
        static_cast<std::size_t>(kTokens) * kGateStride, 0xC0FFEE11U, 3.0F);
    HipBuffer<float> d_expert(kExpert);
    HipBuffer<float> d_weights(weights.size());
    HipBuffer<float> d_shared(kOut);
    HipBuffer<float> d_gate(gate.size());
    HipBuffer<float> d_ref(kOut);
    HipBuffer<float> d_vec(kOut);
    Upload(&d_expert, expert_out);
    Upload(&d_weights, weights);
    Upload(&d_shared, shared);
    Upload(&d_gate, gate);
    q::MoeEpilogue(d_expert.get(), d_weights.get(), d_shared.get(),
                   d_gate.get(), kGateStride, d_ref.get(), kTokens, kSlots,
                   kHidden, nullptr);
    q::MoeEpilogueVec4(d_expert.get(), d_weights.get(), d_shared.get(),
                       d_gate.get(), kGateStride, d_vec.get(), kTokens, kSlots,
                       kHidden, nullptr);
    CheckHip(hipDeviceSynchronize(), "MoE epilogue synchronization");
    const auto ref = Download(&d_ref, kOut);
    const auto vec = Download(&d_vec, kOut);
    double worst = 0.0;
    for (std::size_t i = 0; i < kOut; ++i) {
      worst = std::max(worst, std::abs(static_cast<double>(ref[i] - vec[i])));
    }
    std::cout << "MoE epilogue vec4 worst absolute error " << worst << '\n';
    if (worst != 0.0)
      return 1;
    CheckAddRmsNorm<float>(expert_out, weights, shared, gate, &d_weights,
                           &d_shared, &d_gate);
    CheckAddRmsNorm<__half>(expert_out, weights, shared, gate, &d_weights,
                            &d_shared, &d_gate);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
