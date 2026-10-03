#ifndef GUFO_MODELS_QWEN36_35B_A3B_WEIGHTS_HPP_
#define GUFO_MODELS_QWEN36_35B_A3B_WEIGHTS_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen36_35b_a3b/config.hpp"

namespace gufo::models::qwen36_35b_a3b {

/// Non-owning view of one GGUF tensor. `rows` x `cols` follows the GGUF
/// convention: cols (ne[0]) is the contiguous reduction dimension, rows
/// (ne[1]) the output dimension, and `experts` (ne[2]) stacks whole matrices.
struct TensorRef {
  const void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint64_t cols{0};
  std::uint64_t rows{1};
  std::uint64_t experts{1};
  std::uint64_t file_offset{0};  ///< Byte offset inside the owning shard.
  std::uint32_t shard{0};        ///< Mapped region index in the reader.
  std::string_view name;

  [[nodiscard]] bool empty() const noexcept { return data == nullptr; }
  [[nodiscard]] std::uint64_t ElementCount() const noexcept {
    return cols * rows * experts;
  }
  /// Encoded bytes of one row (`cols` elements) in this format.
  [[nodiscard]] std::size_t RowBytes() const noexcept;
  [[nodiscard]] std::size_t SizeBytes() const noexcept {
    return RowBytes() * rows * experts;
  }
  /// Base address of one stacked expert matrix.
  [[nodiscard]] const std::uint8_t* Expert(std::uint64_t e) const noexcept {
    return static_cast<const std::uint8_t*>(data) + RowBytes() * rows * e;
  }
};

/// One pre-norm residual block: `attn_norm` -> token mixer -> add ->
/// `post_attention_norm` -> experts -> add.
struct LayerWeights {
  bool linear{false};
  TensorRef attn_norm;  ///< [hidden]
  TensorRef post_norm;  ///< [hidden]

  // Gated DeltaNet (linear layers).
  TensorRef ssm_qkv;     ///< [hidden -> 2*key_dim + value_dim]
  TensorRef ssm_gate;    ///< [hidden -> value_dim], the z output gate.
  TensorRef ssm_conv1d;  ///< [conv_kernel, channels]
  TensorRef ssm_alpha;   ///< [hidden -> v_heads]
  TensorRef ssm_beta;    ///< [hidden -> v_heads]
  TensorRef ssm_dt;      ///< [v_heads] softplus bias
  TensorRef ssm_a;       ///< [v_heads] = -exp(A_log)
  TensorRef ssm_norm;    ///< [ssm_head_dim]
  TensorRef ssm_out;     ///< [value_dim -> hidden]

  // Gated GQA (attention layers).
  TensorRef attn_q;       ///< [hidden -> heads * 2 * head_dim], q|gate per head
  TensorRef attn_k;       ///< [hidden -> kv_heads * head_dim]
  TensorRef attn_v;       ///< [hidden -> kv_heads * head_dim]
  TensorRef attn_out;     ///< [heads * head_dim -> hidden]
  TensorRef attn_q_norm;  ///< [head_dim]
  TensorRef attn_k_norm;  ///< [head_dim]

  // Mixture of experts.
  TensorRef router;          ///< [hidden -> num_experts] F32 or BF16
  TensorRef ffn_gate_exps;   ///< [hidden -> expert_ff] x experts
  TensorRef ffn_up_exps;     ///< [hidden -> expert_ff] x experts
  TensorRef ffn_down_exps;   ///< [expert_ff -> hidden] x experts
  TensorRef shexp_gate_inp;  ///< [hidden] -> scalar sigmoid gate
  TensorRef shexp_gate;      ///< [hidden -> shared_ff]
  TensorRef shexp_up;        ///< [hidden -> shared_ff]
  TensorRef shexp_down;      ///< [shared_ff -> hidden]

  // Native MTP (`nextn`) block only.
  TensorRef nextn_enorm;        ///< [hidden], normalizes the token embedding.
  TensorRef nextn_hnorm;        ///< [hidden], normalizes the trunk hidden.
  TensorRef nextn_eh_proj;      ///< GGUF rows: [fc_embedding | fc_hidden]
  TensorRef nextn_shared_norm;  ///< [hidden], norm before the shared head.
};

struct ModelWeights {
  Config config;
  TensorRef token_embd;   ///< [hidden -> vocab]
  TensorRef output;       ///< [hidden -> vocab]
  TensorRef output_norm;  ///< [hidden]
  std::vector<LayerWeights> layers;

  /// Binds and validates the trunk artifact. Tensor payloads stay mapped and
  /// untouched; only headers are read.
  [[nodiscard]] static std::optional<ModelWeights> Bind(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

/// The MTP draft block stored as block `num_layers` of the trunk file: one
/// gated-attention layer with its own experts; token embedding and LM head
/// are shared with the trunk.
struct MtpWeights {
  LayerWeights block;

  [[nodiscard]] static std::optional<MtpWeights> Bind(
      const core::GgufReader& reader, const Config& trunk,
      std::string* error_msg = nullptr);
};

}  // namespace gufo::models::qwen36_35b_a3b

#endif  // GUFO_MODELS_QWEN36_35B_A3B_WEIGHTS_HPP_
