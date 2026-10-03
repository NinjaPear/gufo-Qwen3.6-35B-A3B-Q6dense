#include "src/models/qwen/hip/kernels/prefill_quant_gemm.hpp"
#include "src/models/qwen36_35b_a3b/kernels/rocm/kernels.hpp"

namespace gufo::models::qwen36_35b_a3b::rocm {

void W8A8GemmWave64(const void* w, const void* x_tiled, float* out,
                    std::size_t batch, std::size_t m, std::size_t k,
                    hipStream_t stream) {
  const dim3 grid(static_cast<unsigned>((batch + 127) / 128),
                  static_cast<unsigned>((m + 127) / 128));
  // The 2048 x 4096 output projections. This geometry is distinct from
  // Qwen27B's Q8 instantiation, so each retains its qualified schedule.
  gufo::hip::WKQuantA8BlockedWmmaGEMMKernel<128, 128, 2, 4, 1,
                                            core::GgmlType::kQ8_0, 64>
      <<<grid, 256, 0, stream>>>(w, x_tiled, out, batch, m, k);
}

}  // namespace gufo::models::qwen36_35b_a3b::rocm
