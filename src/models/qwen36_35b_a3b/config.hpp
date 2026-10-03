#ifndef GUFO_MODELS_QWEN36_35B_A3B_CONFIG_HPP_
#define GUFO_MODELS_QWEN36_35B_A3B_CONFIG_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::qwen36_35b_a3b {

/// Architecture parameters of a `qwen35moe` GGUF artifact. Every value is
/// read from the file; the shape gate in FromGguf bounds what this runtime
/// was written and validated for (Qwen3.6-35B-A3B, 40 trunk layers).
struct Config {
  std::uint32_t num_layers{0};      ///< Trunk layers (MTP block excluded).
  std::uint32_t num_layers_all{0};  ///< block_count, includes the MTP block.
  std::uint32_t hidden_size{0};     ///< n_embd, 2048.
  std::uint32_t vocab_size{0};      ///< 248320.
  std::uint32_t context_length{0};  ///< 262144.
  float rms_eps{1e-6F};

  // Full (gated GQA) attention on every full_attention_interval-th layer.
  std::uint32_t full_attention_interval{0};  ///< 4
  std::uint32_t num_heads{0};                ///< 16
  std::uint32_t num_kv_heads{0};             ///< 2
  std::uint32_t head_dim{0};                 ///< 256
  std::uint32_t rotary_dim{0};               ///< 64
  float rope_theta{0.0F};                    ///< 1e7
  std::array<std::uint32_t, 4> rope_sections{};

  // Gated DeltaNet linear attention.
  std::uint32_t ssm_conv_kernel{0};  ///< 4
  std::uint32_t ssm_head_dim{0};     ///< 128 (state_size; key and value dim)
  std::uint32_t ssm_num_k_heads{0};  ///< 16 (group_count)
  std::uint32_t ssm_num_v_heads{0};  ///< 32 (time_step_rank)
  std::uint32_t ssm_inner_size{0};   ///< 4096

  // Mixture of experts with one always-on, sigmoid-gated shared expert.
  std::uint32_t num_experts{0};       ///< 256
  std::uint32_t num_experts_used{0};  ///< 8
  std::uint32_t expert_ff{0};         ///< 512
  std::uint32_t shared_expert_ff{0};  ///< 512

  // Native MTP block (`nextn`), stored as the last block of the same file.
  std::uint32_t nextn_layers{0};

  [[nodiscard]] std::uint32_t SsmKeyDim() const noexcept {
    return ssm_num_k_heads * ssm_head_dim;
  }
  [[nodiscard]] std::uint32_t SsmValueDim() const noexcept {
    return ssm_num_v_heads * ssm_head_dim;
  }
  /// Channels of the fused q|k|v projection and its causal convolution.
  [[nodiscard]] std::uint32_t SsmConvChannels() const noexcept {
    return 2 * SsmKeyDim() + SsmValueDim();
  }
  [[nodiscard]] std::uint32_t AttentionQDim() const noexcept {
    return num_heads * head_dim;
  }
  [[nodiscard]] std::uint32_t AttentionKvDim() const noexcept {
    return num_kv_heads * head_dim;
  }
  [[nodiscard]] bool IsLinearLayer(std::uint32_t layer) const noexcept {
    return layer < num_layers && ((layer + 1) % full_attention_interval) != 0;
  }
  [[nodiscard]] bool HasMtp() const noexcept { return nextn_layers == 1; }

  /// Reads and validates the `qwen35moe.*` metadata.
  [[nodiscard]] static std::optional<Config> FromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

}  // namespace gufo::models::qwen36_35b_a3b

#endif  // GUFO_MODELS_QWEN36_35B_A3B_CONFIG_HPP_
