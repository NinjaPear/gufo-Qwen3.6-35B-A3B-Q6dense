#include "src/models/qwen36_35b_a3b/reference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>

#include "src/models/qwen36_35b_a3b/cpu_ops.hpp"

namespace gufo::models::qwen36_35b_a3b {

using cpu::Sigmoid;
using cpu::Silu;

ReferenceModel::ReferenceModel(const ModelWeights& weights,
                               std::uint32_t max_context, Storage storage)
    : w_(weights),
      c_(weights.config),
      storage_(storage),
      max_context_(max_context) {
  linear_.resize(c_.num_layers);
  attention_.resize(c_.num_layers);
  Reset();
}

namespace {
float Half(float value) {
  // The supported Linux x86-64 compiler implements IEEE binary16 conversion.
  return static_cast<float>(static_cast<_Float16>(value));
}

std::vector<float> QuantizedInput(std::span<const float> input) {
  std::vector<float> result(input.size());
  for (std::size_t base = 0; base < input.size(); base += 32) {
    float peak = 0;
    for (std::size_t j = 0; j < 32; ++j)
      peak = std::max(peak, std::abs(input[base + j]));
    const float scale = peak / 127.0F, stored = Half(scale);
    for (std::size_t j = 0; j < 32; ++j)
      result[base + j] =
          peak == 0 ? 0 : stored * std::round(input[base + j] / scale);
  }
  return result;
}
}  // namespace

void ReferenceModel::MatVec(const TensorRef& weight, std::uint64_t expert,
                            std::span<const float> x, std::span<float> out,
                            bool narrow_weights) {
  if (storage_ == Storage::kFloat32) {
    cpu::MatVec(weight, expert, x, out);
    return;
  }
  const auto quantized = weight.type == core::GgmlType::kQ8_0
                             ? QuantizedInput(x)
                             : std::vector<float>{};
  if (!quantized.empty())
    x = quantized;
  std::vector<float> row(weight.cols);
  for (std::size_t r = 0; r < weight.rows; ++r) {
    cpu::DequantizeRow(weight, expert, r, row.data());
    double sum = 0;
    for (std::size_t j = 0; j < x.size(); ++j)
      sum += double(narrow_weights ? Half(row[j]) : row[j]) * x[j];
    out[r] = static_cast<float>(sum);
  }
}

void ReferenceModel::Reset() {
  position_ = 0;
  mtp_position_ = 0;
  mtp_attention_ = {};
  mtp_hidden_.clear();
  for (std::uint32_t il = 0; il < c_.num_layers; ++il) {
    if (c_.IsLinearLayer(il)) {
      auto& s = linear_[il];
      s.conv.assign(static_cast<std::size_t>(c_.ssm_conv_kernel - 1) *
                        c_.SsmConvChannels(),
                    0.0F);
      s.state.assign(static_cast<std::size_t>(c_.ssm_num_v_heads) *
                         c_.ssm_head_dim * c_.ssm_head_dim,
                     0.0F);
    } else {
      attention_[il] = {};
    }
  }
}

void ReferenceModel::LinearAttention(const LayerWeights& l, LinearState& s,
                                     std::span<const float> x,
                                     std::span<float> out) {
  const std::uint32_t channels = c_.SsmConvChannels();
  const std::uint32_t key_dim = c_.SsmKeyDim();
  const std::uint32_t value_dim = c_.SsmValueDim();
  const std::uint32_t d = c_.ssm_head_dim;
  const std::uint32_t kern = c_.ssm_conv_kernel;

  std::vector<float> qkv(channels);
  std::vector<float> z(value_dim);
  std::vector<float> alpha(c_.ssm_num_v_heads);
  std::vector<float> beta(c_.ssm_num_v_heads);
  MatVec(l.ssm_qkv, 0, x, qkv);
  MatVec(l.ssm_gate, 0, x, z);
  MatVec(l.ssm_alpha, 0, x, alpha);
  MatVec(l.ssm_beta, 0, x, beta);

  // Causal depthwise conv over the last `kern` projections, then SiLU.
  const auto* conv_w = static_cast<const float*>(l.ssm_conv1d.data);
  std::vector<float> conv(channels);
  for (std::uint32_t ch = 0; ch < channels; ++ch) {
    float acc =
        conv_w[static_cast<std::size_t>(ch) * kern + kern - 1] * qkv[ch];
    for (std::uint32_t k = 0; k + 1 < kern; ++k) {
      acc += conv_w[static_cast<std::size_t>(ch) * kern + k] *
             s.conv[static_cast<std::size_t>(k) * channels + ch];
    }
    conv[ch] = Silu(acc);
  }
  if (kern > 1) {
    std::copy(s.conv.begin() + channels, s.conv.end(), s.conv.begin());
    std::copy(qkv.begin(), qkv.end(), s.conv.end() - channels);
  }

  float* q = conv.data();
  float* k = conv.data() + key_dim;
  float* v = conv.data() + 2 * key_dim;
  for (std::uint32_t h = 0; h < c_.ssm_num_k_heads; ++h) {
    cpu::L2Norm(std::span<float>(q + static_cast<std::size_t>(h) * d, d),
                c_.rms_eps);
    cpu::L2Norm(std::span<float>(k + static_cast<std::size_t>(h) * d, d),
                c_.rms_eps);
  }

  const auto* a = static_cast<const float*>(l.ssm_a.data);
  const auto* dt = static_cast<const float*>(l.ssm_dt.data);
  const auto* norm_w = static_cast<const float*>(l.ssm_norm.data);
  const float q_scale = 1.0F / std::sqrt(static_cast<float>(d));
  std::vector<float> attn(value_dim);
  std::vector<float> u(d);
  for (std::uint32_t h = 0; h < c_.ssm_num_v_heads; ++h) {
    // Value head h reads key head h % n_k_heads, the tiled layout the GGUF
    // converter permuted the value-side tensors into.
    const float* qh = q + static_cast<std::size_t>(h % c_.ssm_num_k_heads) * d;
    const float* kh = k + static_cast<std::size_t>(h % c_.ssm_num_k_heads) * d;
    const float* vh = v + static_cast<std::size_t>(h) * d;
    float* S = s.state.data() + static_cast<std::size_t>(h) * d * d;
    const float decay = std::exp(a[h] * cpu::Softplus(alpha[h] + dt[h]));
    const float b = Sigmoid(beta[h]);
    // S[j][i]: j over value dim, i over key dim.
    for (std::uint32_t j = 0; j < d; ++j) {
      float* row = S + static_cast<std::size_t>(j) * d;
      double acc = 0.0;
      for (std::uint32_t i = 0; i < d; ++i) {
        row[i] *= decay;
        acc += static_cast<double>(row[i]) * kh[i];
      }
      u[j] = static_cast<float>(acc);
    }
    float* o = attn.data() + static_cast<std::size_t>(h) * d;
    for (std::uint32_t j = 0; j < d; ++j) {
      const float delta = (vh[j] - u[j]) * b;
      float* row = S + static_cast<std::size_t>(j) * d;
      double acc = 0.0;
      for (std::uint32_t i = 0; i < d; ++i) {
        row[i] += delta * kh[i];
        acc += static_cast<double>(row[i]) * qh[i];
      }
      o[j] = static_cast<float>(acc) * q_scale;
    }
    // Per-head RMSNorm, then the SiLU output gate.
    cpu::RmsNorm(std::span<float>(o, d), norm_w, c_.rms_eps);
    for (std::uint32_t j = 0; j < d; ++j) {
      o[j] *= Silu(z[static_cast<std::size_t>(h) * d + j]);
    }
  }
  MatVec(l.ssm_out, 0, attn, out);
}

void ReferenceModel::Attention(const LayerWeights& l, AttentionState& s,
                               std::span<const float> x, std::uint32_t pos,
                               std::span<float> out) {
  const std::uint32_t hd = c_.head_dim;
  const std::uint32_t nh = c_.num_heads;
  const std::uint32_t nkv = c_.num_kv_heads;
  const std::uint32_t group = nh / nkv;

  std::vector<float> qg(2 * c_.AttentionQDim());
  std::vector<float> k(c_.AttentionKvDim());
  std::vector<float> v(c_.AttentionKvDim());
  MatVec(l.attn_q, 0, x, qg);
  MatVec(l.attn_k, 0, x, k);
  MatVec(l.attn_v, 0, x, v);

  // wq interleaves [q | gate] per head.
  std::vector<float> q(c_.AttentionQDim());
  std::vector<float> gate(c_.AttentionQDim());
  for (std::uint32_t h = 0; h < nh; ++h) {
    std::copy_n(qg.begin() + static_cast<std::size_t>(h) * 2 * hd, hd,
                q.begin() + static_cast<std::size_t>(h) * hd);
    std::copy_n(qg.begin() + static_cast<std::size_t>(h) * 2 * hd + hd, hd,
                gate.begin() + static_cast<std::size_t>(h) * hd);
    cpu::RmsNorm(
        std::span<float>(q.data() + static_cast<std::size_t>(h) * hd, hd),
        static_cast<const float*>(l.attn_q_norm.data), c_.rms_eps);
  }
  for (std::uint32_t h = 0; h < nkv; ++h) {
    cpu::RmsNorm(
        std::span<float>(k.data() + static_cast<std::size_t>(h) * hd, hd),
        static_cast<const float*>(l.attn_k_norm.data), c_.rms_eps);
  }
  // Text-only mRoPE has equal positions on every section.
  cpu::Rope(q.data(), nh, hd, c_.rotary_dim, pos, c_.rope_theta);
  cpu::Rope(k.data(), nkv, hd, c_.rotary_dim, pos, c_.rope_theta);
  if (storage_ == Storage::kDecode) {
    for (auto& value : k)
      value = Half(value);
    for (auto& value : v)
      value = Half(value);
  }
  s.k.insert(s.k.end(), k.begin(), k.end());
  s.v.insert(s.v.end(), v.begin(), v.end());

  const std::uint32_t n_kv = pos + 1;
  const float scale = 1.0F / std::sqrt(static_cast<float>(hd));
  std::vector<float> ctx(c_.AttentionQDim(), 0.0F);
  std::vector<float> scores(n_kv);
  for (std::uint32_t h = 0; h < nh; ++h) {
    const std::uint32_t kvh = h / group;
    const float* qh = q.data() + static_cast<std::size_t>(h) * hd;
    float max_score = -INFINITY;
    for (std::uint32_t j = 0; j < n_kv; ++j) {
      const float* kj =
          s.k.data() + (static_cast<std::size_t>(j) * nkv + kvh) * hd;
      double dot = 0.0;
      for (std::uint32_t i = 0; i < hd; ++i) {
        dot += static_cast<double>(qh[i]) * kj[i];
      }
      scores[j] = static_cast<float>(dot) * scale;
      max_score = std::max(max_score, scores[j]);
    }
    double denom = 0.0;
    for (float& sc : scores) {
      sc = std::exp(sc - max_score);
      denom += sc;
    }
    float* oh = ctx.data() + static_cast<std::size_t>(h) * hd;
    for (std::uint32_t j = 0; j < n_kv; ++j) {
      const float p = static_cast<float>(scores[j] / denom);
      const float* vj =
          s.v.data() + (static_cast<std::size_t>(j) * nkv + kvh) * hd;
      for (std::uint32_t i = 0; i < hd; ++i) {
        oh[i] += p * vj[i];
      }
    }
    for (std::uint32_t i = 0; i < hd; ++i) {
      oh[i] *= Sigmoid(gate[static_cast<std::size_t>(h) * hd + i]);
    }
  }
  MatVec(l.attn_out, 0, ctx, out);
}

void ReferenceModel::Moe(const LayerWeights& l, std::span<const float> x,
                         std::span<float> out) {
  std::vector<float> logits(c_.num_experts);
  MatVec(l.router, 0, x, logits, true);
  // Softmax over every expert, top-k of the probabilities, renormalized.
  const float max_logit = *std::max_element(logits.begin(), logits.end());
  double denom = 0.0;
  for (float& v : logits) {
    v = std::exp(v - max_logit);
    denom += v;
  }
  for (float& v : logits) {
    v = static_cast<float>(v / denom);
  }
  std::vector<std::uint32_t> order(c_.num_experts);
  std::iota(order.begin(), order.end(), 0U);
  std::partial_sort(
      order.begin(), order.begin() + c_.num_experts_used, order.end(),
      [&](std::uint32_t a, std::uint32_t b) { return logits[a] > logits[b]; });
  float sum = 0.0F;
  for (std::uint32_t i = 0; i < c_.num_experts_used; ++i) {
    sum += logits[order[i]];
  }
  sum = std::max(sum, 6.103515625e-5F);

  std::fill(out.begin(), out.end(), 0.0F);
  std::vector<float> gate(c_.expert_ff);
  std::vector<float> up(c_.expert_ff);
  std::vector<float> down(c_.hidden_size);
  for (std::uint32_t i = 0; i < c_.num_experts_used; ++i) {
    const std::uint32_t e = order[i];
    const float weight = logits[e] / sum;
    MatVec(l.ffn_gate_exps, e, x, gate);
    MatVec(l.ffn_up_exps, e, x, up);
    for (std::uint32_t j = 0; j < c_.expert_ff; ++j) {
      gate[j] = Silu(gate[j]) * up[j];
    }
    MatVec(l.ffn_down_exps, e, gate, down);
    for (std::uint32_t j = 0; j < c_.hidden_size; ++j) {
      out[j] += weight * down[j];
    }
  }

  // Shared expert with its own scalar sigmoid gate.
  std::vector<float> sgate(c_.shared_expert_ff);
  std::vector<float> sup(c_.shared_expert_ff);
  MatVec(l.shexp_gate, 0, x, sgate);
  MatVec(l.shexp_up, 0, x, sup);
  for (std::uint32_t j = 0; j < c_.shared_expert_ff; ++j) {
    sgate[j] = Silu(sgate[j]) * sup[j];
  }
  MatVec(l.shexp_down, 0, sgate, down);
  std::array<float, 1> shared_gate{};
  MatVec(l.shexp_gate_inp, 0, x, shared_gate, true);
  const float sg = Sigmoid(shared_gate[0]);
  for (std::uint32_t j = 0; j < c_.hidden_size; ++j) {
    out[j] += sg * down[j];
  }
}

void ReferenceModel::Block(const LayerWeights& l, std::uint32_t il,
                           std::span<float> x, std::uint32_t pos,
                           AttentionState* attention_override) {
  std::vector<float> normed(x.begin(), x.end());
  std::vector<float> block_out(c_.hidden_size);
  cpu::RmsNorm(normed, static_cast<const float*>(l.attn_norm.data), c_.rms_eps);
  if (l.linear) {
    LinearAttention(l, linear_[il], normed, block_out);
  } else {
    Attention(l, attention_override ? *attention_override : attention_[il],
              normed, pos, block_out);
  }
  for (std::uint32_t i = 0; i < c_.hidden_size; ++i) {
    x[i] += block_out[i];
  }
  normed.assign(x.begin(), x.end());
  cpu::RmsNorm(normed, static_cast<const float*>(l.post_norm.data), c_.rms_eps);
  Moe(l, normed, block_out);
  for (std::uint32_t i = 0; i < c_.hidden_size; ++i) {
    x[i] += block_out[i];
  }
}

bool ReferenceModel::Step(std::int32_t token, std::span<float> logits,
                          std::span<float> hidden_out, std::string* error_msg) {
  if (position_ >= max_context_) {
    if (error_msg != nullptr) {
      *error_msg = "reference context is full";
    }
    return false;
  }
  if (token < 0 || static_cast<std::uint32_t>(token) >= c_.vocab_size) {
    if (error_msg != nullptr) {
      *error_msg = "token out of range";
    }
    return false;
  }
  std::vector<float> x(c_.hidden_size);
  cpu::DequantizeRow(w_.token_embd, 0, static_cast<std::uint64_t>(token),
                     x.data());
  for (std::uint32_t il = 0; il < c_.num_layers; ++il) {
    Block(w_.layers[il], il, x, position_);
  }
  if (!hidden_out.empty()) {
    std::copy(x.begin(), x.end(), hidden_out.begin());
  }
  if (!logits.empty()) {
    cpu::RmsNorm(x, static_cast<const float*>(w_.output_norm.data), c_.rms_eps);
    MatVec(w_.output, 0, x, logits);
  }
  ++position_;
  return true;
}

bool ReferenceModel::MtpStep(const MtpWeights& mtp, std::int32_t token,
                             std::span<const float> hidden,
                             std::span<float> logits, std::string* error_msg) {
  const auto H = c_.hidden_size;
  if (hidden.empty())
    hidden = mtp_hidden_;
  if (token < 0 || static_cast<std::uint32_t>(token) >= c_.vocab_size ||
      hidden.size() != H || mtp_position_ >= max_context_ ||
      (!logits.empty() && logits.size() != c_.vocab_size)) {
    if (error_msg)
      *error_msg = "invalid reference MTP input";
    return false;
  }
  const auto& l = mtp.block;
  std::vector<float> embedding(H), norm(hidden.begin(), hidden.end());
  cpu::DequantizeRow(w_.token_embd, 0, token, embedding.data());
  cpu::RmsNorm(embedding, static_cast<const float*>(l.nextn_enorm.data),
               c_.rms_eps);
  cpu::RmsNorm(norm, static_cast<const float*>(l.nextn_hnorm.data), c_.rms_eps);
  if (storage_ == Storage::kDecode &&
      l.nextn_eh_proj.type == core::GgmlType::kQ8_0) {
    embedding = QuantizedInput(embedding);
    norm = QuantizedInput(norm);
  }
  // Read the original combined GGUF rows, independently of GPU repacking.
  // Separate double accumulators implement fc_embedding + fc_hidden.
  std::vector<float> x(H), row(2 * H);
  for (std::uint32_t r = 0; r < H; ++r) {
    cpu::DequantizeRow(l.nextn_eh_proj, 0, r, row.data());
    double ep = 0, hp = 0;
    for (std::uint32_t j = 0; j < H; ++j) {
      ep += static_cast<double>(row[j]) * embedding[j];
      hp += static_cast<double>(row[H + j]) * norm[j];
    }
    x[r] = static_cast<float>(ep) + static_cast<float>(hp);
  }
  Block(l, c_.num_layers, x, mtp_position_, &mtp_attention_);
  mtp_hidden_ = x;
  if (!logits.empty()) {
    cpu::RmsNorm(x, static_cast<const float*>(l.nextn_shared_norm.data),
                 c_.rms_eps);
    MatVec(w_.output, 0, x, logits);
  }
  ++mtp_position_;
  return true;
}

}  // namespace gufo::models::qwen36_35b_a3b
