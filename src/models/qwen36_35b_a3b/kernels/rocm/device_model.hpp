#ifndef GUFO_MODELS_QWEN36_35B_A3B_KERNELS_ROCM_DEVICE_MODEL_HPP_
#define GUFO_MODELS_QWEN36_35B_A3B_KERNELS_ROCM_DEVICE_MODEL_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen36_35b_a3b/weights.hpp"

namespace gufo::models::qwen36_35b_a3b::rocm {

/// One weight resident in device memory, still in its GGUF encoding.
struct DeviceTensor {
  void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint32_t cols{0};
  std::uint32_t rows{0};
  std::uint32_t experts{1};
  /// Q8_0 copy of a K-quant dense matrix for the prefill kernels; decode
  /// reads the compact `data`.
  void* prefill{nullptr};

  [[nodiscard]] bool empty() const noexcept { return data == nullptr; }
  [[nodiscard]] bool kquant() const noexcept {
    return type == core::GgmlType::kQ4_K || type == core::GgmlType::kQ5_K ||
           type == core::GgmlType::kQ6_K;
  }
  [[nodiscard]] const float* f32() const noexcept {
    return static_cast<const float*>(data);
  }
};

struct DeviceLayer {
  bool linear{false};
  DeviceTensor attn_norm, post_norm;

  DeviceTensor ssm_qkv, ssm_gate, ssm_conv1d, ssm_dt, ssm_a, ssm_norm, ssm_out;
  /// qkv and gate rows stacked ([hidden -> conv channels + value dim]) when
  /// both are Q8_0; then ssm_qkv/ssm_gate are empty.
  DeviceTensor ssm_in;
  /// alpha and beta rows stacked: [hidden -> 2 * v_heads] F16.
  DeviceTensor ssm_alpha_beta;
  DeviceTensor attn_q, attn_k, attn_v, attn_out, attn_q_norm, attn_k_norm;
  /// [q|gate ; k ; v] rows stacked when all are Q8_0; then attn_q/k/v are
  /// empty.
  DeviceTensor attn_qkv;
  /// Router rows followed by the shared-expert gate row:
  /// [hidden -> num_experts + 1] F16.
  DeviceTensor router;
  DeviceTensor ffn_gate_exps, ffn_up_exps, ffn_down_exps, shexp_gate, shexp_up,
      shexp_down;
  DeviceTensor nextn_enorm, nextn_hnorm, nextn_fc_embedding, nextn_fc_hidden,
      nextn_shared_norm;
};

/// The trunk and, when the artifact carries it, the MTP draft block uploaded
/// to the GPU.
class DeviceModel {
public:
  ~DeviceModel();
  DeviceModel(const DeviceModel&) = delete;
  DeviceModel& operator=(const DeviceModel&) = delete;

  /// Streams tensors from the same open files used to bind their metadata.
  [[nodiscard]] static std::unique_ptr<DeviceModel> Upload(
      const ModelWeights& weights, const core::GgufReader& reader,
      const MtpWeights* mtp, std::string* error_msg = nullptr);

  const Config& config() const noexcept { return config_; }
  const DeviceTensor& token_embd() const noexcept { return token_embd_; }
  const DeviceTensor& output() const noexcept { return output_; }
  const DeviceTensor& output_norm() const noexcept { return output_norm_; }
  const std::vector<DeviceLayer>& layers() const noexcept { return layers_; }
  /// The layers prefill runs: K-quant dense matrices replaced by their Q8_0
  /// copies (the decode layers when there are none).
  const std::vector<DeviceLayer>& prefill_layers() const noexcept {
    return prefill_layers_;
  }
  [[nodiscard]] bool has_mtp() const noexcept { return has_mtp_; }
  const DeviceLayer& mtp() const noexcept { return mtp_; }
  [[nodiscard]] std::size_t resident_bytes() const noexcept { return bytes_; }
  /// Widest K among the BF16/F16 matrices (activation staging for hipBLAS).
  [[nodiscard]] std::size_t max_half_cols() const noexcept {
    return max_half_cols_;
  }
  /// Widest K among the Q8_0 matrices (decode activation quantization).
  [[nodiscard]] std::size_t max_q8_cols() const noexcept {
    return max_q8_cols_;
  }

private:
  DeviceModel() = default;

  Config config_;
  DeviceTensor token_embd_;
  DeviceTensor output_;
  DeviceTensor output_norm_;
  std::vector<DeviceLayer> layers_;
  std::vector<DeviceLayer> prefill_layers_;
  DeviceLayer mtp_;
  bool has_mtp_{false};
  std::vector<void*> allocations_;
  std::size_t bytes_{0};
  std::size_t max_half_cols_{1};
  std::size_t max_q8_cols_{32};
};

}  // namespace gufo::models::qwen36_35b_a3b::rocm

#endif  // GUFO_MODELS_QWEN36_35B_A3B_KERNELS_ROCM_DEVICE_MODEL_HPP_
