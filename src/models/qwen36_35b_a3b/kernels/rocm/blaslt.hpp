#ifndef GUFO_MODELS_QWEN36_35B_A3B_KERNELS_ROCM_BLASLT_HPP_
#define GUFO_MODELS_QWEN36_35B_A3B_KERNELS_ROCM_BLASLT_HPP_

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>

namespace gufo::models::qwen36_35b_a3b::rocm {

/// Dense 16-bit projections with F32 accumulation and output. Plans depend
/// only on tensor geometry and the pinned HIP library, never runtime timing.
class BlasLt {
public:
  ~BlasLt();
  BlasLt(const BlasLt&) = delete;
  BlasLt& operator=(const BlasLt&) = delete;

  [[nodiscard]] static std::unique_ptr<BlasLt> Create(hipStream_t stream,
                                                      std::string* error_msg);

  /// Row-major out[n][m] = input[n][k] * weights[m][k]^T.
  [[nodiscard]] bool Gemm(const void* weights, const void* input, float* out,
                          hipDataType type, int m, int n, int k,
                          std::string* error_msg);

private:
  BlasLt() = default;

  struct Plan;
  std::unique_ptr<Plan> MakePlan(hipDataType type, int m, int n, int k,
                                 std::string* error_msg) const;

  hipblasLtHandle_t handle_{nullptr};
  hipStream_t stream_{nullptr};
  std::map<std::array<int, 4>, std::unique_ptr<Plan>> plans_;
};

}  // namespace gufo::models::qwen36_35b_a3b::rocm

#endif  // GUFO_MODELS_QWEN36_35B_A3B_KERNELS_ROCM_BLASLT_HPP_
