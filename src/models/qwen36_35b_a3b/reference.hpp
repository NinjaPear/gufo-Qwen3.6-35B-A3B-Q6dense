#ifndef GUFO_MODELS_QWEN36_35B_A3B_REFERENCE_HPP_
#define GUFO_MODELS_QWEN36_35B_A3B_REFERENCE_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "src/models/qwen36_35b_a3b/weights.hpp"

namespace gufo::models::qwen36_35b_a3b {

/// Single-token, float32, scalar reference of the Qwen3.6-35B-A3B text graph.
/// It pins the semantics of every operator the GPU runtime must reproduce:
/// pre-norm residual blocks, Gated DeltaNet with a SiLU output gate, gated
/// GQA, softmax top-k MoE with the sigmoid-gated shared expert, and the native
/// MTP block. Speed is irrelevant here.
class ReferenceModel {
public:
  /// Float32 formulas versus an independent emulation of decode storage:
  /// Q8 activation blocks, F16 KV and F16 uploaded router weights.
  enum class Storage { kFloat32, kDecode };
  ReferenceModel(const ModelWeights& weights, std::uint32_t max_context,
                 Storage storage = Storage::kFloat32);

  /// Runs one token at the next position; `logits` (vocab floats) receives
  /// the output distribution when non-empty. `hidden_out` (hidden floats)
  /// receives the final residual before `output_norm`, the MTP hidden input.
  [[nodiscard]] bool Step(std::int32_t token, std::span<float> logits,
                          std::span<float> hidden_out = {},
                          std::string* error_msg = nullptr);

  /// Independent scalar predictor. An empty hidden input uses the previous
  /// predictor residual; otherwise the supplied pre-norm trunk residual.
  [[nodiscard]] bool MtpStep(const MtpWeights& mtp, std::int32_t token,
                             std::span<const float> hidden,
                             std::span<float> logits,
                             std::string* error_msg = nullptr);

  [[nodiscard]] std::uint32_t Position() const noexcept { return position_; }
  void Reset();

private:
  void MatVec(const TensorRef& weight, std::uint64_t expert,
              std::span<const float> x, std::span<float> out,
              bool narrow_weights = false);
  struct LinearState {
    std::vector<float> conv;   ///< [kernel-1][channels], oldest first
    std::vector<float> state;  ///< [v_heads][head_dim(v)][head_dim(k)]
  };
  struct AttentionState {
    std::vector<float> k;  ///< [pos][kv_heads][head_dim], rotated
    std::vector<float> v;  ///< [pos][kv_heads][head_dim]
  };

  void LinearAttention(const LayerWeights& l, LinearState& s,
                       std::span<const float> x, std::span<float> out);
  void Attention(const LayerWeights& l, AttentionState& s,
                 std::span<const float> x, std::uint32_t pos,
                 std::span<float> out);
  void Moe(const LayerWeights& l, std::span<const float> x,
           std::span<float> out);
  /// One residual block: x += mixer(norm(x)); x += moe(post_norm(x)).
  void Block(const LayerWeights& l, std::uint32_t il, std::span<float> x,
             std::uint32_t pos, AttentionState* attention_override = nullptr);

  const ModelWeights& w_;
  const Config& c_;
  Storage storage_;
  std::uint32_t max_context_;
  std::uint32_t position_{0};
  std::vector<LinearState> linear_;
  std::vector<AttentionState> attention_;
  AttentionState mtp_attention_;
  std::vector<float> mtp_hidden_;
  std::uint32_t mtp_position_{0};
};

}  // namespace gufo::models::qwen36_35b_a3b

#endif  // GUFO_MODELS_QWEN36_35B_A3B_REFERENCE_HPP_
