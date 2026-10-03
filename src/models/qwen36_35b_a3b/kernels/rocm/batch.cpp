#include <algorithm>
#include <array>
#include <bit>
#include <stdexcept>
#include <utility>

#include "q36_mmq.h"
#include "src/models/qwen36_35b_a3b/kernels/rocm/executor.hpp"
#include "src/models/qwen36_35b_a3b/kernels/rocm/kernels.hpp"
#include "src/models/qwen36_35b_a3b/mtp_sampling.hpp"

namespace gufo::models::qwen36_35b_a3b::rocm {
namespace {

constexpr std::uint32_t kBatchSessions = 8;
constexpr std::uint32_t kDecodeRows = 8;

bool Fail(std::string* error, const std::string& message) {
  if (error != nullptr) {
    *error = message;
  }
  return false;
}

bool Check(hipError_t status, std::string* error) {
  return status == hipSuccess || Fail(error, std::string("batched forward: ") +
                                                 hipGetErrorString(status));
}

// A cancelled row stops touching its private state. Shared projections retain
// their row layout so cancellation cannot change a peer's arithmetic/order.
template<class Item>
bool AnyActive(std::span<const Item> items) {
  bool active = false;
  for (const auto& item : items)
    active |= item.session->CheckCancellation(nullptr);
  return active;
}

WeightType EmbeddingType(core::GgmlType type) {
  switch (type) {
    case core::GgmlType::kBF16:
      return WeightType::kBF16;
    case core::GgmlType::kF16:
      return WeightType::kF16;
    case core::GgmlType::kQ8_0:
      return WeightType::kQ8_0;
    case core::GgmlType::kF32:
      return WeightType::kF32;
    default:
      throw std::logic_error("unsupported Qwen3.6 embedding format");
  }
}

}  // namespace

Executor::Scratch Executor::RowScratch(const Scratch& b,
                                       std::uint32_t offset) const {
  const Config& c = config();
  const std::size_t r = offset;
  auto s = b;
  s.tokens += r;
  s.res += r * c.hidden_size;
  s.xn += r * c.hidden_size;
  s.block_out += r * c.hidden_size;
  s.qkv += r * c.SsmConvChannels();
  s.z += r * c.SsmValueDim();
  s.qkvz += r * (c.SsmConvChannels() + c.SsmValueDim());
  s.alpha_beta += r * 2 * c.ssm_num_v_heads;
  s.qn += r * c.SsmKeyDim();
  s.kn += r * c.SsmKeyDim();
  s.gdn_raw += r * c.SsmValueDim();
  s.gdn_out += r * c.SsmValueDim();
  s.qg += r * (2 * c.AttentionQDim() + 2 * c.AttentionKvDim());
  s.q += r * c.AttentionQDim();
  s.attn_gate += r * c.AttentionQDim();
  s.k += r * c.AttentionKvDim();
  s.v += r * c.AttentionKvDim();
  s.ctx += r * c.AttentionQDim();
  s.router += r * (c.num_experts + 1);
  s.ids += r * c.num_experts_used;
  s.weights += r * c.num_experts_used;
  s.gate_e += r * c.num_experts_used * c.expert_ff;
  s.up_e += r * c.num_experts_used * c.expert_ff;
  s.down_e += r * c.num_experts_used * c.hidden_size;
  s.shexp_gate += r * c.shared_expert_ff;
  s.shexp_up += r * c.shared_expert_ff;
  s.shexp_out += r * c.hidden_size;
  s.shexp_half += r * c.shared_expert_ff;
  if (has_mtp()) {
    s.mtp_h += r * c.hidden_size;
    s.mtp_embd += r * c.hidden_size;
    // This allocation aliases qkvz, but has a different row stride.
    s.mtp_eproj += r * c.hidden_size;
    s.mtp_res += r * c.hidden_size;
  }
  // Activation staging, attention partials and convolution staging have
  // one consumer at a time on stream_. They remain at the allocation base.
  return s;
}

void Executor::UseScratch(const Scratch& scratch) const {
  FlushMoeEpilogue();
  s_ = scratch;
  q8t_src_ = nullptr;
  half_src_ = nullptr;
  moe_pending_ = false;
}

bool Executor::AllocateBatch(std::string* error) const {
  return batch_logits_ != nullptr ||
         Check(hipMalloc(&batch_logits_,
                         static_cast<std::size_t>(kBatchSessions) *
                             std::min(kDecodeRows, options_.max_logit_rows) *
                             config().vocab_size * sizeof(float)),
               error);
}

bool Executor::MtpForwardBatch(std::span<const MtpBatchItem> items,
                               std::string* error) const {
  const Config& c = config();
  if (!has_mtp() || items.empty() || items.size() > kBatchSessions) {
    return Fail(error, "invalid MTP body batch");
  }
  std::array<std::uint32_t, kBatchSessions> offsets{};
  std::array<std::uint32_t, kBatchSessions> final_rows{};
  std::uint32_t rows = 0;
  for (std::size_t i = 0; i < items.size(); ++i) {
    const auto& item = items[i];
    const auto n = item.tokens.size();
    if (item.session == nullptr || item.session->owner_ != this ||
        !item.session->mtp_enabled_ || n == 0 || n > kDecodeRows ||
        (item.hidden_row < 0 &&
         (n != 1 || !item.session->mtp_.residual_valid)) ||
        (item.hidden_row >= 0 &&
         static_cast<std::uint32_t>(item.hidden_row) + n >
             options_.max_speculative) ||
        item.session->mtp_.position + n > item.session->max_context_) {
      return Fail(error, "invalid MTP body request");
    }
    for (std::size_t j = 0; j < i; ++j) {
      if (items[j].session == item.session)
        return Fail(error, "MTP body requests must be independent");
    }
    for (const auto token : item.tokens) {
      if (token < 0 || static_cast<std::uint32_t>(token) >= c.vocab_size)
        return Fail(error, "MTP token out of range");
    }
    offsets[i] = rows;
    rows += n;
    final_rows[i] = rows - 1;
  }
  if (rows > options_.max_batch)
    return Fail(error, "MTP body batch exceeds executor capacity");
  if (!AnyActive(items))
    return true;
  if (items.size() == 1) {
    const auto& item = items.front();
    return MtpForward(*item.session, item.tokens, item.hidden_row, {}, error);
  }
  if (!AllocateBatch(error) ||
      (batch_controls_ == nullptr &&
       !Check(hipHostMalloc(&batch_controls_,
                            kBatchSessions * sizeof(Session::Control)),
              error))) {
    return false;
  }
  for (std::size_t i = 0; i < items.size(); ++i) {
    const auto& item = items[i];
    const auto& session = *item.session;
    batch_controls_[i] = {session.position_, session.mtp_.position,
                          item.hidden_row};
    std::copy(item.tokens.begin(), item.tokens.end(),
              tokens_host_ + offsets[i]);
  }
  const Scratch base = s_;
  const auto& l = model_->mtp();
  const auto body = [&]() {
    if (!AnyActive(items))
      return true;
    if (!Check(hipMemcpyAsync(base.tokens, tokens_host_,
                              rows * sizeof(std::int32_t),
                              hipMemcpyHostToDevice, stream_),
               error))
      return false;
    EmbedTokens(model_->token_embd().data,
                EmbeddingType(model_->token_embd().type), base.tokens,
                base.mtp_embd, rows, c.hidden_size, 1, stream_);
    RmsNormRows(base.mtp_embd, l.nextn_enorm.f32(), base.mtp_embd, rows,
                c.hidden_size, 1, c.rms_eps, stream_);
    for (std::size_t i = 0; i < items.size(); ++i) {
      auto& session = *items[i].session;
      const auto n = static_cast<std::uint32_t>(items[i].tokens.size());
      UseScratch(RowScratch(base, offsets[i]));
      if (!Check(hipMemcpyAsync(session.control_, batch_controls_ + i,
                                sizeof(Session::Control), hipMemcpyHostToDevice,
                                stream_),
                 error))
        return false;
      MtpHidden(session.mtp_.target_hidden, session.mtp_.h,
                &session.control_->hidden_row, s_.mtp_h, n, c.hidden_size,
                model_->output_norm().f32(), l.nextn_shared_norm.f32(),
                c.rms_eps, stream_);
    }
    RmsNormRows(base.mtp_h, l.nextn_hnorm.f32(), base.mtp_h, rows,
                c.hidden_size, 1, c.rms_eps, stream_);
    UseScratch(base);
    // Persistent draft KV uses the same projections for every prompt split,
    // catch-up length and number of independent sessions.
    if (!DenseBatch(l.nextn_fc_embedding, base.mtp_embd, base.mtp_eproj, rows,
                    error) ||
        !DenseBatch(l.nextn_fc_hidden, base.mtp_h, base.mtp_res, rows, error))
      return false;
    MtpAddEmbedding(base.mtp_eproj, base.mtp_res, rows, c.hidden_size, 1,
                    stream_);
    RmsNormRows(base.mtp_res, l.attn_norm.f32(), base.xn, rows, c.hidden_size,
                1, c.rms_eps, stream_);
    UseScratch(base);
    if (!l.attn_qkv.empty()) {
      if (!DenseBatch(l.attn_qkv, base.xn, base.qg, rows, error))
        return false;
    } else if (!DenseBatch(l.attn_q, base.xn, base.qg, rows, error) ||
               !DenseBatch(l.attn_k, base.xn, base.k, rows, error) ||
               !DenseBatch(l.attn_v, base.xn, base.v, rows, error)) {
      return false;
    }
    for (std::size_t i = 0; i < items.size(); ++i) {
      auto& session = *items[i].session;
      const auto n = static_cast<std::uint32_t>(items[i].tokens.size());
      auto view = RowScratch(base, offsets[i]);
      if (l.attn_qkv.empty())
        view.qg = base.qg + std::size_t{offsets[i]} * 2 * c.AttentionQDim();
      UseScratch(view);
      if (!session.CheckCancellation(nullptr)) {
        if (!Check(hipMemsetAsync(
                       s_.ctx, 0,
                       std::size_t{n} * c.AttentionQDim() * sizeof(float),
                       stream_),
                   error))
          return false;
        continue;
      }
      ++session.mutation_epoch_;
      session.mtp_.residual_valid = false;
      Session::AttentionState attention;
      attention.rope = session.vision_input_.rope();
      attention.k_cache = session.mtp_.k_cache;
      attention.v_cache = session.mtp_.v_cache;
      attention.k_q8 = session.mtp_.k_q8;
      attention.v_q8 = session.mtp_.v_q8;
      attention.k_scale = session.mtp_.k_scale;
      attention.v_scale = session.mtp_.v_scale;
      attention.draft = true;
      if (!Attention(l, attention, s_.xn, s_.block_out, n,
                     &session.control_->mtp_position, session.mtp_.position,
                     error, false, true, false))
        return false;
    }
    // Attention retains every catch-up KV row. Only each final residual
    // contributes to the next proposal, so compact those rows into idle
    // trunk scratch before the output projection and FFN. A one-row tail
    // in the original eight-row projection has different scalar rounding.
    const bool compact_tail = rows > kDecodeRows && rows % kDecodeRows != 1;
    Scratch tail = base;
    if (compact_tail) {
      // mtp_res aliases res; the compact rows go to block_out's tail space
      // in qkvz, which no later stage of this body reads.
      tail.mtp_res = base.qkvz + std::size_t{rows} * c.hidden_size;
      tail.ctx = base.qg;
      for (std::size_t i = 0; i < items.size(); ++i) {
        const auto r = final_rows[i];
        const auto copy = [&](float* dst, const float* src, std::size_t width) {
          return Check(hipMemcpyAsync(dst + i * width, src + r * width,
                                      width * sizeof(float),
                                      hipMemcpyDeviceToDevice, stream_),
                       error);
        };
        if (!copy(tail.mtp_res, base.mtp_res, c.hidden_size) ||
            !copy(tail.ctx, base.ctx, c.AttentionQDim()))
          return false;
      }
    }
    const auto tail_rows =
        compact_tail ? static_cast<std::uint32_t>(items.size()) : rows;
    UseScratch(tail);
    if (!DenseBatch(l.attn_out, tail.ctx, tail.block_out, tail_rows, error))
      return false;
    Combine(tail.mtp_res, l.post_norm.f32(), tail_rows);
    if (!AnyActive(items))
      return true;
    if (!MoeBatch(l, tail.xn, tail.block_out, tail_rows, error))
      return false;
    UseScratch(tail);
    Combine(tail.mtp_res, nullptr, tail_rows);
    for (std::size_t i = 0; i < items.size(); ++i) {
      if (!items[i].session->CheckCancellation(nullptr))
        continue;
      if (!Check(
              hipMemcpyAsync(items[i].session->mtp_.h,
                             tail.mtp_res + (compact_tail ? i : final_rows[i]) *
                                                c.hidden_size,
                             c.hidden_size * sizeof(float),
                             hipMemcpyDeviceToDevice, stream_),
              error))
        return false;
    }
    return true;
  };
  bool ok = false;
  try {
    ok = body();
  } catch (...) {
    (void)hipStreamSynchronize(stream_);
    UseScratch(base);
    throw;
  }
  const auto status = hipStreamSynchronize(stream_);
  UseScratch(base);
  if (!ok || !Check(status, error))
    return false;
  for (const auto& item : items) {
    if (item.session->Cancelled())
      continue;
    item.session->mtp_.position += item.tokens.size();
    item.session->mtp_.residual_valid = true;
  }
  return true;
}

bool Executor::MtpHeads(std::span<const MtpHeadItem> items,
                        std::string* error) const {
  selected_logits_ = nullptr;
  if (!has_mtp() || items.empty() || items.size() > kBatchSessions ||
      items.size() > options_.max_batch) {
    return Fail(error, "invalid MTP head batch");
  }
  for (std::size_t i = 0; i < items.size(); ++i) {
    const auto& item = items[i];
    if (item.session == nullptr || item.session->owner_ != this ||
        !item.session->mtp_enabled_ || item.session->mtp_.position == 0 ||
        !item.session->mtp_.residual_valid || item.output.kv_only ||
        (item.output.token == nullptr && item.output.candidates == nullptr)) {
      return Fail(error, "invalid MTP head request");
    }
    for (std::size_t j = 0; j < i; ++j) {
      if (items[j].session == item.session) {
        return Fail(error, "MTP head requests must be independent");
      }
    }
  }
  const auto& output = model_->output();
  const auto& head_norm = model_->mtp().nextn_shared_norm;
  if (!AnyActive(items))
    return true;
  if (items.size() == 1) {
    for (const auto& item : items) {
      if (!item.session->CheckCancellation(nullptr))
        continue;
      if (!MtpHead(item.session->mtp_.h, item.output.token != nullptr,
                   item.output.candidates != nullptr, error) ||
          !Check(hipStreamSynchronize(stream_), error)) {
        return false;
      }
      if (item.output.token != nullptr) {
        *item.output.token = mtp_token_host_[0];
        if (item.output.prob != nullptr)
          *item.output.prob = std::bit_cast<float>(mtp_token_host_[1]);
      }
      if (item.output.candidates != nullptr)
        *item.output.candidates = *mtp_candidates_host_;
    }
    return true;
  }
  if (!AllocateBatch(error)) {
    return false;
  }
  if (batch_candidates_host_ == nullptr) {
    if (!Check(hipHostMalloc(&batch_candidates_host_,
                             kBatchSessions * sizeof(MtpCandidateLogits)),
               error))
      return false;
    for (std::uint32_t i = 0; i < kBatchSessions; ++i)
      std::construct_at(batch_candidates_host_ + i);
  }
  const auto n = static_cast<std::uint32_t>(items.size());
  const auto count = std::min<std::size_t>(output.rows, kMtpCandidates);
  const Scratch base = s_;
  const auto body = [&]() {
    if (!AnyActive(items))
      return true;
    for (std::uint32_t i = 0; i < n; ++i) {
      RmsNormRows(items[i].session->mtp_.h, head_norm.f32(),
                  base.xn + std::size_t{i} * config().hidden_size, 1,
                  config().hidden_size, 1, config().rms_eps, stream_);
    }
    UseScratch(base);
    if (!AnyActive(items))
      return true;
    // Greedy proposals read the draft vocabulary, as a single request does;
    // the full head is needed only when some request samples candidates.
    DeviceTensor head = output;
    if (std::ranges::none_of(items, [](const auto& item) {
          return item.output.candidates != nullptr;
        }))
      head.rows = std::min(kDraftVocabulary, output.rows);
    if (!DenseBatch(head, base.xn, batch_logits_, n, error))
      return false;
    for (std::uint32_t i = 0; i < n; ++i) {
      if (!items[i].session->CheckCancellation(nullptr))
        continue;
      if (items[i].output.candidates == nullptr) {
        ArgmaxProb(batch_logits_ + std::size_t(i) * head.rows, s_.mtp_argmax,
                   s_.mtp_token, reinterpret_cast<float*>(s_.mtp_token + 1),
                   std::min(kDraftVocabulary, output.rows), stream_);
        if (!Check(hipMemcpyAsync(batch_candidates_host_[i].ids.data(),
                                  s_.mtp_token, 2 * sizeof(std::int32_t),
                                  hipMemcpyDeviceToHost, stream_),
                   error))
          return false;
        continue;
      }
      MtpTopCandidates(batch_logits_ + std::size_t(i) * head.rows, s_.mtp_ids,
                       s_.mtp_scratch_ids, s_.mtp_scores, output.rows, stream_);
      if (!Check(hipMemcpyAsync(batch_candidates_host_ + i, s_.mtp_ids,
                                offsetof(MtpCandidateLogits, logits) +
                                    count * sizeof(float),
                                hipMemcpyDeviceToHost, stream_),
                 error)) {
        return false;
      }
      batch_candidates_host_[i].size = count;
    }
    return true;
  };
  bool ok = false;
  try {
    ok = body();
  } catch (...) {
    (void)hipStreamSynchronize(stream_);
    UseScratch(base);
    throw;
  }
  const auto status = hipStreamSynchronize(stream_);
  UseScratch(base);
  if (!ok || !Check(status, error))
    return false;
  for (std::size_t i = 0; i < items.size(); ++i) {
    if (items[i].session->Cancelled())
      continue;
    if (items[i].output.token != nullptr) {
      *items[i].output.token = batch_candidates_host_[i].ids[0];
      if (items[i].output.prob != nullptr)
        *items[i].output.prob =
            std::bit_cast<float>(batch_candidates_host_[i].ids[1]);
    }
    if (items[i].output.candidates != nullptr)
      *items[i].output.candidates = batch_candidates_host_[i];
  }
  return true;
}

bool Executor::QuantizeBatch(const float* x, std::uint32_t rows,
                             std::uint32_t cols, std::string* error) const {
  // Quantize once for all independent rows. The prefill staging allocation
  // is idle during batch decoding; borrow it without adding another buffer.
  const auto padded = (rows + kDecodeRows - 1) / kDecodeRows * kDecodeRows;
  const auto bytes = q36_mmq_q8_1_bytes(padded, cols);
  if (bytes > Q8TiledBytes(options_.max_batch, model_->max_q8_cols()))
    return Fail(error, "batched projection exceeds activation scratch");
  void* quantized = s_.x_q8t;
  q8t_src_ = nullptr;
  if (padded != rows &&
      !Check(
          hipMemsetAsync(static_cast<std::uint8_t*>(quantized) +
                             q36_mmq_q8_1_bytes(rows, cols),
                         0, q36_mmq_q8_1_bytes(padded - rows, cols), stream_),
          error))
    return false;
  return q36_mmq_quantize_q8_1(x, quantized, rows, cols, stream_) == 0 ||
         Fail(error, "batched activation quantization failed");
}

bool Executor::DenseBatch(const DeviceTensor& w, const float* x, float* out,
                          std::uint32_t rows, std::string* error) const {
  if (rows <= kDecodeRows)
    return Dense(w, x, out, rows, error);
  if (w.kquant()) {
    // Decode arithmetic in eight-row groups, as for a single request.
    for (std::uint32_t r = 0; r < rows; r += kDecodeRows) {
      const auto n = std::min(kDecodeRows, rows - r);
      if (!Dense(w, x + std::size_t{r} * w.cols, out + std::size_t{r} * w.rows,
                 n, error))
        return false;
    }
    return true;
  }
  if (w.type != core::GgmlType::kQ8_0) {
    SmallGemm(w.data, EmbeddingType(w.type), x, out, rows, w.rows, w.cols,
              stream_);
    return true;
  }
  if (!QuantizeBatch(x, rows, w.cols, error))
    return false;
  const auto project = [&](std::uint32_t first, std::uint32_t count) {
    const auto* input = static_cast<const std::uint8_t*>(s_.x_q8t) +
                        q36_mmq_q8_1_bytes(first, w.cols);
    return q36_mmq_q8_0_dense_vec_preq(
               w.data, nullptr, input,
               out + static_cast<std::size_t>(first) * w.rows, w.rows, count,
               w.cols, stream_) == 0 ||
           Fail(error, "batched Q8 projection failed");
  };
  const auto chunk = kDecodeRows;
  for (std::uint32_t r = 0; r < rows; r += chunk) {
    if (!project(r, std::min(chunk, rows - r)))
      return false;
  }
  return true;
}

bool Executor::GatedDenseBatch(const DeviceTensor& up, const DeviceTensor& gate,
                               const float* x, float* out, std::uint32_t rows,
                               std::string* error) const {
  const Scratch base = s_;
  const bool q8 = up.type == core::GgmlType::kQ8_0 &&
                  gate.type == core::GgmlType::kQ8_0 && up.rows == gate.rows &&
                  up.cols == gate.cols;
  if (q8 && !QuantizeBatch(x, rows, up.cols, error))
    return false;
  for (std::uint32_t r = 0; r < rows; r += kDecodeRows) {
    const auto n = std::min(kDecodeRows, rows - r);
    if (q8) {
      const auto* input = static_cast<const std::uint8_t*>(base.x_q8t) +
                          q36_mmq_q8_1_bytes(r, up.cols);
      if (q36_mmq_q8_0_dense_vec_preq(up.data, gate.data, input,
                                      out + std::size_t{r} * up.rows, up.rows,
                                      n, up.cols, stream_) != 0)
        return Fail(error, "batched gated Q8 projection failed");
    } else {
      UseScratch(RowScratch(base, r));
      const bool ok =
          GatedDense(up, gate, x + std::size_t{r} * up.cols,
                     out + std::size_t{r} * up.rows, n, nullptr, error);
      UseScratch(base);
      if (!ok)
        return false;
    }
  }
  return true;
}

bool Executor::MoeBatch(const DeviceLayer& l, const float* x, float* out,
                        std::uint32_t rows, std::string* error) const {
  if (rows <= kDecodeRows)
    return Moe(l, x, out, rows, error);
  const auto& c = config();
  const Scratch base = s_;
  if (!DenseBatch(l.router, x, base.router, rows, error))
    return false;
  RouterTopK(base.router, c.num_experts + 1, base.ids, base.weights, rows,
             c.num_experts, c.num_experts_used, stream_);
  if (!GatedDenseBatch(l.shexp_up, l.shexp_gate, x, base.shexp_up, rows,
                       error) ||
      !DenseBatch(l.shexp_down, base.shexp_up, base.shexp_out, rows, error))
    return false;
  if (l.ffn_gate_exps.type == core::GgmlType::kQ4_K &&
      l.ffn_up_exps.type == core::GgmlType::kQ4_K) {
    // Share expert weights across request boundaries while retaining the
    // scalar Q8 activation quantization and dot-product reduction.
    if (q36_mmq_moe_gated_vec(static_cast<int>(l.ffn_gate_exps.type),
                              l.ffn_gate_exps.data, l.ffn_up_exps.data, x,
                              base.ids, base.gate_e, c.expert_ff, c.hidden_size,
                              rows, c.num_experts, c.num_experts_used,
                              stream_) != 0 ||
        q36_mmq_moe_vec(
            static_cast<int>(l.ffn_down_exps.type), l.ffn_down_exps.data,
            base.gate_e, base.ids, base.down_e, c.hidden_size, c.expert_ff,
            rows * c.num_experts_used, c.num_experts, 1, stream_) != 0)
      return Fail(error, "batched routed vector projection failed");
    DeferMoeEpilogue(out, rows);
    return true;
  }
  // Each row group leaves its expert rows deferred in its own view of the
  // shared scratch; the views tile the base layout, so one deferred
  // epilogue covers every row.
  std::uint32_t deferred = 0;
  for (std::uint32_t r = 0; r < rows; r += kDecodeRows) {
    UseScratch(RowScratch(base, r));
    const bool ok = MoeExperts(l, x + std::size_t{r} * c.hidden_size,
                               out + std::size_t{r} * c.hidden_size,
                               std::min(kDecodeRows, rows - r), error);
    deferred += std::exchange(moe_vec_pending_, 0);
    UseScratch(base);
    if (!ok)
      return false;
  }
  if (deferred != rows)
    return Fail(error, "batched MoE did not defer its epilogue");
  DeferMoeEpilogue(out, rows);
  return true;
}

void Executor::DeferMoeEpilogue(float* out, std::uint32_t rows) const {
  // Combine folds the epilogue into the residual norm pass, the same
  // arithmetic a single request uses; any other destination runs it now.
  if (out == s_.block_out) {
    moe_vec_pending_ = rows;
    return;
  }
  const Config& c = config();
  MoeEpilogue(s_.down_e, s_.weights, s_.shexp_out, s_.router + c.num_experts,
              c.num_experts + 1, out, rows, c.num_experts_used, c.hidden_size,
              stream_);
}

bool Executor::ForwardBatch(std::span<const BatchItem> items,
                            std::string* error) const {
  selected_logits_ = nullptr;
  const Config& c = config();
  if (items.empty() || items.size() > kBatchSessions) {
    return Fail(error, "decode batch must contain 1..8 sessions");
  }
  std::array<std::uint32_t, kBatchSessions> offsets{};
  std::uint32_t rows = 0;
  std::uint32_t max_tokens = 0;
  for (std::size_t i = 0; i < items.size(); ++i) {
    const auto& item = items[i];
    if (item.session == nullptr || item.session->owner_ != this ||
        item.tokens.empty() || item.tokens.size() > kDecodeRows ||
        item.tokens.size() > options_.max_logit_rows ||
        (item.speculative && item.tokens.size() > options_.max_speculative) ||
        item.session->position_ + item.tokens.size() >
            item.session->max_context_) {
      return Fail(error, "invalid session or chain in decode batch");
    }
    for (std::size_t j = 0; j < i; ++j) {
      if (items[j].session == item.session) {
        return Fail(error, "decode batch contains a duplicate session");
      }
    }
    for (const auto token : item.tokens) {
      if (token < 0 || static_cast<std::uint32_t>(token) >= c.vocab_size) {
        return Fail(error, "token out of range");
      }
    }
    offsets[i] = rows;
    rows += static_cast<std::uint32_t>(item.tokens.size());
    max_tokens =
        std::max(max_tokens, static_cast<std::uint32_t>(item.tokens.size()));
  }
  if (rows > options_.max_batch) {
    return Fail(error, "decode batch exceeds executor capacity");
  }
  if (!AnyActive(items))
    return true;
  for (const auto& item : items) {
    if (item.speculative &&
        !EnsureRollback(*item.session, item.tokens.size() - 1, error))
      return false;
  }
  if (!AllocateBatch(error)) {
    return false;
  }
  if (batch_controls_ == nullptr &&
      !Check(hipHostMalloc(&batch_controls_,
                           kBatchSessions * sizeof(Session::Control)),
             error)) {
    return false;
  }
  const bool batch_gdn =
      items.size() > 1 && c.ssm_head_dim == 128 && c.ssm_conv_kernel == 4;
  if (batch_gdn) {
    const std::size_t bytes =
        std::size_t{c.num_layers} * kBatchSessions * sizeof(GdnBatchItem);
    if ((batch_gdn_host_ == nullptr &&
         !Check(hipHostMalloc(&batch_gdn_host_, bytes, hipHostMallocMapped),
                error)) ||
        (batch_gdn_ == nullptr &&
         !Check(hipHostGetDevicePointer(reinterpret_cast<void**>(&batch_gdn_),
                                        batch_gdn_host_, 0),
                error)))
      return false;
  }
  batch_rows_ = 0;
  for (std::size_t i = 0; i < items.size(); ++i) {
    const auto& item = items[i];
    auto& session = *item.session;
    ++session.mutation_epoch_;
    session.spec_base_ = session.position_;
    session.spec_tokens_ = item.speculative ? item.tokens.size() : 0;
    batch_controls_[i] = {session.position_, session.mtp_.position, -1};
    std::copy(item.tokens.begin(), item.tokens.end(),
              tokens_host_ + offsets[i]);
  }

  const Scratch base = s_;
  // Every exit restores the ordinary single-session scratch view. Enqueued
  // work is drained before a caller can reuse its pinned staging buffers.
  const auto body = [&]() -> bool {
    if (!AnyActive(items))
      return true;
    for (std::size_t i = 0; i < items.size(); ++i) {
      if (!Check(hipMemcpyAsync(items[i].session->control_, batch_controls_ + i,
                                sizeof(Session::Control), hipMemcpyHostToDevice,
                                stream_),
                 error)) {
        return false;
      }
    }
    if (!Check(hipMemcpyAsync(base.tokens, tokens_host_,
                              rows * sizeof(std::int32_t),
                              hipMemcpyHostToDevice, stream_),
               error)) {
      return false;
    }
    EmbedTokens(model_->token_embd().data,
                EmbeddingType(model_->token_embd().type), base.tokens, base.res,
                rows, c.hidden_size, 1, stream_);
    for (std::size_t i = 0; i < items.size(); ++i) {
      items[i].session->vision_input_.Inject(
          base.res + std::size_t{offsets[i]} * c.hidden_size,
          items[i].session->position_, items[i].tokens.size(), c.hidden_size, 1,
          stream_);
    }
    const auto& layers = model_->layers();
    RmsNormRows(base.res, layers[0].attn_norm.f32(), base.xn, rows,
                c.hidden_size, 1, c.rms_eps, stream_);
    UseScratch(base);
    for (std::uint32_t il = 0; il < c.num_layers; ++il) {
      if (!AnyActive(items))
        return true;
      const auto& l = layers[il];
      if (l.linear) {
        if ((!l.ssm_in.empty()
                 ? !DenseBatch(l.ssm_in, base.xn, base.qkvz, rows, error)
                 : (!DenseBatch(l.ssm_qkv, base.xn, base.qkv, rows, error) ||
                    !DenseBatch(l.ssm_gate, base.xn, base.z, rows, error))) ||
            !DenseBatch(l.ssm_alpha_beta, base.xn, base.alpha_beta, rows,
                        error)) {
          return false;
        }
      } else if (!l.attn_qkv.empty()) {
        if (!DenseBatch(l.attn_qkv, base.xn, base.qg, rows, error)) {
          return false;
        }
      } else {
        // Separate Q/gate has a shorter row stride than the stacked buffer.
        // This model normally stacks QKV; keep the separate shape explicit.
        if (!DenseBatch(l.attn_q, base.xn, base.qg, rows, error) ||
            !DenseBatch(l.attn_k, base.xn, base.k, rows, error) ||
            !DenseBatch(l.attn_v, base.xn, base.v, rows, error)) {
          return false;
        }
      }
      std::uint32_t gdn_active = 0;
      for (std::size_t i = 0; i < items.size(); ++i) {
        const auto& item = items[i];
        auto& session = *item.session;
        const auto n = static_cast<std::uint32_t>(item.tokens.size());
        auto view = RowScratch(base, offsets[i]);
        if (!l.linear && l.attn_qkv.empty()) {
          view.qg = base.qg + static_cast<std::size_t>(offsets[i]) * 2 *
                                  c.AttentionQDim();
        }
        UseScratch(view);
        if (!session.CheckCancellation(nullptr)) {
          auto* output = l.linear ? s_.gdn_out : s_.ctx;
          const auto width = l.linear ? c.SsmValueDim() : c.AttentionQDim();
          if (!Check(hipMemsetAsync(output, 0,
                                    std::size_t{n} * width * sizeof(float),
                                    stream_),
                     error))
            return false;
          continue;
        }
        if (l.linear) {
          if (batch_gdn) {
            const auto& state = session.linear_[il];
            batch_gdn_host_[il * kBatchSessions + i] = {
                l.ssm_in.empty() ? view.qkv : view.qkvz,
                l.ssm_in.empty() ? view.z : view.qkvz + c.SsmConvChannels(),
                view.alpha_beta, state.conv_state,
                // Short convolution saves history in registers, so each
                // request needs only its own token rows of staging.
                base.conv_scratch +
                    std::size_t{offsets[i]} * c.SsmConvChannels(),
                view.qn, view.kn, view.gdn_raw, state.state, view.gdn_out,
                item.speculative ? state.state_snapshots : RollbackRows{},
                item.speculative ? state.conv_snapshots : RollbackRows{}, n};
            gdn_active |= 1U << i;
          } else if (!LinearAttention(l, session.linear_[il], s_.xn,
                                      s_.block_out, n, item.speculative, error,
                                      true, false)) {
            return false;
          }
        } else if (!Attention(l, session.attention_[il], s_.xn, s_.block_out, n,
                              &session.control_->position, session.position_,
                              error, false, true, false)) {
          return false;
        }
      }
      if (l.linear && batch_gdn && gdn_active != 0) {
        const auto offset = il * kBatchSessions;
        if (!GatedDeltaNetBatch(
                batch_gdn_ + offset, items.size(), max_tokens, gdn_active,
                l.ssm_in.empty() ? c.SsmConvChannels() : l.ssm_in.rows,
                l.ssm_in.empty() ? c.SsmValueDim() : l.ssm_in.rows,
                l.ssm_conv1d.f32(), l.ssm_a.f32(), l.ssm_dt.f32(),
                l.ssm_norm.f32(), c.ssm_num_k_heads, c.ssm_num_v_heads,
                c.rms_eps, stream_))
          return Fail(error, "batched GatedDeltaNet launch failed");
      }
      UseScratch(base);
      if (!DenseBatch(l.linear ? l.ssm_out : l.attn_out,
                      l.linear ? base.gdn_out : base.ctx, base.block_out, rows,
                      error)) {
        return false;
      }
      Combine(base.res, l.post_norm.f32(), rows);
      if (!MoeBatch(l, base.xn, base.block_out, rows, error))
        return false;
      Combine(base.res,
              il + 1 < c.num_layers ? layers[il + 1].attn_norm.f32()
                                    : model_->output_norm().f32(),
              rows);
    }
    for (std::size_t i = 0; i < items.size(); ++i) {
      const auto n = static_cast<std::uint32_t>(items[i].tokens.size());
      UseScratch(RowScratch(base, offsets[i]));
      if (!items[i].session->Cancelled() && items[i].session->mtp_enabled_ &&
          !Check(hipMemcpyAsync(items[i].session->mtp_.target_hidden, s_.res,
                                static_cast<std::size_t>(n) * c.hidden_size *
                                    sizeof(float),
                                hipMemcpyDeviceToDevice, stream_),
                 error)) {
        return false;
      }
    }
    UseScratch(base);
    // The last combine already normed every row with output_norm.
    return DenseBatch(model_->output(), base.xn, batch_logits_, rows, error);
  };
  bool ok = false;
  try {
    ok = body();
  } catch (...) {
    (void)hipStreamSynchronize(stream_);
    UseScratch(base);
    throw;
  }
  const auto status = hipStreamSynchronize(stream_);
  UseScratch(base);
  if (!ok || !Check(status, error)) {
    return false;
  }
  for (std::size_t i = 0; i < items.size(); ++i) {
    if (items[i].session->Cancelled())
      continue;
    items[i].session->position_ += items[i].tokens.size();
  }
  batch_rows_ = rows;
  return true;
}

bool Executor::SelectBatchLogits(std::uint32_t offset, std::uint32_t rows,
                                 float* logits, std::string* error) const {
  if (rows == 0 || rows > options_.max_logit_rows || offset > batch_rows_ ||
      rows > batch_rows_ - offset) {
    return Fail(error, "logit rows outside the completed decode batch");
  }
  const std::size_t count =
      static_cast<std::size_t>(rows) * config().vocab_size;
  selected_logits_ =
      batch_logits_ + static_cast<std::size_t>(offset) * config().vocab_size;
  if (logits != nullptr) {
    if (!Check(hipMemcpyAsync(logits_host_, selected_logits_,
                              count * sizeof(float), hipMemcpyDeviceToHost,
                              stream_),
               error) ||
        !Check(hipStreamSynchronize(stream_), error)) {
      return false;
    }
    std::copy_n(logits_host_, count, logits);
  }
  return true;
}

}  // namespace gufo::models::qwen36_35b_a3b::rocm
