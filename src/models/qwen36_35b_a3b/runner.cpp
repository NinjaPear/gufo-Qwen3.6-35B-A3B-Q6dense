#include "src/models/qwen36_35b_a3b/runner.hpp"

#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gufo/ext/gguf_identity.hpp"
#include "gufo/ext/host_budget.hpp"
#include "gufo/ext/text.hpp"
#include "src/families/qwen/text/chat_template.hpp"
#include "src/families/qwen/text/tokenizer.hpp"
#include "src/families/qwen/vision/prompt.hpp"
#include "src/models/qwen36_35b_a3b/continuation_identity.hpp"
#include "src/models/qwen36_35b_a3b/engine.hpp"

// Each Qwen3.6 request owns its recurrent/KV state and snapshots (the
// snapshot-based runner contract, not the component cache). Target
// projections share batches; sampling and rollback stay request-local. The
// behavior is the fork's pre-extension runner, moved here unchanged.

namespace gufo::models::qwen36_35b_a3b {
namespace {

using namespace gufo::ext;

constexpr std::string_view kName = "Qwen3.6-35B-A3B";
constexpr std::string_view kShortName = "Qwen3.6";
constexpr std::string_view kStateAbi = "qwen36-35b-a3b-rocm-session-v1";
constexpr auto kTemplateFamily = tokenization::QwenTemplateFamily::kQwen36;

std::string Message(std::string_view prefix, std::string_view text) {
  std::string message(prefix);
  message += text;
  return message;
}

std::vector<std::int32_t> EngineTokens(std::span<const TextRunnerToken> tokens) {
  std::vector<std::int32_t> converted;
  converted.reserve(tokens.size());
  for (const TextRunnerToken token : tokens) {
    if (token > static_cast<TextRunnerToken>(
                    std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument(
          Message(kName, " token ID exceeds engine range"));
    }
    converted.push_back(static_cast<std::int32_t>(token));
  }
  return converted;
}

tokenization::ChatTemplateOptions QwenChatOptions(const ChatRequest& request,
                                                  std::uint32_t max_context) {
  auto options = tokenization::ResolveQwenChatOptions(
      request.reasoning, request.add_vision_id, kTemplateFamily);
  // gufo #285: the rendered prompt may be as long as the context can hold.
  options.max_output_bytes =
      tokenization::RenderedPromptBoundBytes(max_context);
  return options;
}

std::span<const tokenization::ChatTool> RequestTools(
    const ChatRequest& request) {
  return request.tool_choice == ChatRequest::ToolChoice::kNone
             ? std::span<const tokenization::ChatTool>{}
             : std::span<const tokenization::ChatTool>{request.tools};
}

class Qwen36TextRunnerState final : public TextRunnerState {
public:
  Qwen36TextRunnerState(const std::shared_ptr<Model>& model,
                        std::uint32_t max_context, bool use_mtp) {
    std::string error;
    session_ = model->CreateSession(
        use_mtp ? core::SessionMode::kSpeculative
                : core::SessionMode::kAutoregressive,
        max_context, &error);
    if (session_ == nullptr) {
      throw std::runtime_error(
          Message("Failed to create Qwen3.6-35B-A3B session: ", error));
    }
  }

  void Invalidate() noexcept override {
    session_->SetCancellationCheck({});
    session_->Reset();
    position_ = 0;
  }
  void SetCancellationCheck(const CancellationCheck& check) override {
    session_->SetCancellationCheck(check);
  }
  [[nodiscard]] TextRunnerMeasuredResources MeasuredResources()
      const noexcept override {
    return {.per_request_state_bytes = session_->AllocatedBytes(),
            .temporary_scratch_bytes = 0};
  }

  [[nodiscard]] Session& session() const { return *session_; }
  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  void set_position(std::size_t position) noexcept { position_ = position; }

private:
  std::unique_ptr<Session> session_;
  std::size_t position_{0};
};

class Qwen36TextRunnerSnapshot final : public TextRunnerSnapshot {
public:
  Qwen36TextRunnerSnapshot(std::shared_ptr<Model> model,
                           std::unique_ptr<SessionSnapshot> snapshot,
                           std::size_t position)
      : model(std::move(model)),
        snapshot(std::move(snapshot)),
        position(position) {}

  [[nodiscard]] std::size_t PayloadBytes() const noexcept override {
    if (snapshot == nullptr ||
        snapshot->SizeBytes() > static_cast<std::uint64_t>(
                                    std::numeric_limits<std::size_t>::max())) {
      return 0;
    }
    return static_cast<std::size_t>(snapshot->SizeBytes());
  }

  std::shared_ptr<Model> model;
  std::unique_ptr<SessionSnapshot> snapshot;
  std::size_t position;
};

Qwen36TextRunnerState& RequireState(TextRunnerState& state) {
  auto* q36 = dynamic_cast<Qwen36TextRunnerState*>(&state);
  if (q36 == nullptr) {
    throw std::logic_error("text runner state is not Qwen3.6-35B-A3B");
  }
  return *q36;
}

const Qwen36TextRunnerState& RequireState(const TextRunnerState& state) {
  const auto* q36 = dynamic_cast<const Qwen36TextRunnerState*>(&state);
  if (q36 == nullptr) {
    throw std::logic_error("text runner state is not Qwen3.6-35B-A3B");
  }
  return *q36;
}

class Qwen36TextRunner final : public HipTextModelRunner {
public:
  Qwen36TextRunner(
      std::shared_ptr<Model> model, std::uint32_t max_context, bool use_mtp,
      std::uint32_t max_draft_tokens,
      std::optional<TextRunnerPersistenceDescriptor> persistence = std::nullopt)
      : model_(std::move(model)),
        max_context_(max_context),
        use_mtp_(use_mtp),
        max_draft_tokens_(max_draft_tokens),
        persistence_(std::move(persistence)) {}

  sampling::JsonConstraint::ToolFormat ToolFormat() const override {
    return sampling::JsonConstraint::ToolFormat::kQwen;
  }

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    return {
        .model_id = model_->ModelName(),
        .state_abi = std::string(kStateAbi),
        .max_context = max_context_,
        .capabilities =
            TextRunnerCapabilities{
                .incremental_prefill = true,
                .snapshot = true,
                .fork = true,
                .final_token_advance_required = false,
                .incremental_text_is_exact = true,
                .multi_token_decode = use_mtp_,
                .batched_multi_token_decode = use_mtp_,
                .batched_multi_token_decode_max_width = use_mtp_ ? 8u : 0u,
                .prefix_reuse = true,
            },
        .persistence = persistence_,
        .sampling_preset = sampling::TextModelPreset::kQwen36,
    };
  }

  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    std::optional<std::size_t> capacity;
    if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) {
      const auto deferred = model_->DeferredScratchBytes();
      capacity = free_bytes > deferred ? free_bytes - deferred : 0;
    }
    // Snapshots live in host memory, not in the device state pool.
    return {
        .resident_weights_bytes = model_->ResidentBytes(),
        .state_capacity_bytes = capacity,
        .per_request_state_bytes = model_->SessionBytes(
            use_mtp_ ? core::SessionMode::kSpeculative
                     : core::SessionMode::kAutoregressive,
            max_context_),
        // Runtime scratch is shared and already allocated at model load;
        // reserve its remaining lazy buffers once from aggregate capacity.
        .temporary_scratch_bytes = 0,
        .retained_snapshot_capacity_bytes = HostSnapshotBudgetBytes(),
        .retained_snapshot_ceiling_bytes = HostSnapshotCeilingBytes(),
        .requires_device_runtime_lock = true,
    };
  }

  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    std::vector<TextExecutionPlan> plans{
        {.kind = TextExecutionPlanKind::kSerial, .physical_width = 1}};
    for (std::size_t width = 2; width <= 8; ++width) {
      plans.push_back(
          {.kind = TextExecutionPlanKind::kBatched, .physical_width = width});
    }
    return plans;
  }

  [[nodiscard]] std::shared_ptr<const sampling::ConstraintVocabulary>
  BuildConstraintVocabulary() const override {
    return std::make_shared<const sampling::ConstraintVocabulary>(
        model_->tokenizer().GetVocabSize(), [this](std::uint32_t id) {
          const auto& tokenizer = model_->tokenizer();
          return sampling::ConstraintVocabulary::Piece{
              tokenizer.DecodeTokenCopy(id), tokenizer.IsStopToken(id)};
        });
  }
  [[nodiscard]] std::uint32_t VocabularySize() const override {
    return static_cast<std::uint32_t>(model_->tokenizer().GetVocabSize());
  }

  // As Flash-Next: llama.cpp's Qwen3-Coder format forces "\n</think>" and
  // also ends reasoning at a tool call.
  [[nodiscard]] ReasoningTags ReasoningBudgetTags() const override {
    return {"<think>", "\n</think>", {"</think>", "<tool_call>"}};
  }

  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    return model_->tokenizer().Encode(text);
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override {
    return tokenization::QwenChatTemplate::RenderAndTokenize(
        model_->tokenizer(), request.messages, RequestTools(request),
        QwenChatOptions(request, max_context_));
  }

  // Text only: Prepare gives the same tokens and stable cache prefix as for
  // Flash-Next, and an image is a request error.
  [[nodiscard]] std::optional<TextPreparedPrompt> PreparePrompt(
      const ChatRequest& request) const override {
    if (std::ranges::any_of(request.messages,
                            [](const auto& m) { return !m.images.empty(); })) {
      throw std::invalid_argument("Qwen3.6-35B-A3B serves text only");
    }
    auto prompt = models::qwen::vision::Prepare(
        model_->tokenizer(), request.messages, RequestTools(request),
        QwenChatOptions(request, max_context_), std::string_view{},
        max_context_, nullptr);
    const auto cache_prefix = prompt.stable_prefix_tokens;
    return TextPreparedPrompt{std::move(prompt.tokens), {}, cache_prefix};
  }

  [[nodiscard]] std::optional<std::size_t> CountPromptTokens(
      const ChatRequest& request) const override {
    const auto tokens = RenderAndTokenize(request);
    if (!tokens)
      return std::nullopt;
    return tokens->size();
  }

  [[nodiscard]] std::optional<std::string> RenderPrompt(
      const ChatRequest& request) const override {
    std::string error;
    auto rendered = tokenization::QwenChatTemplate::Render(
        request.messages, RequestTools(request),
        QwenChatOptions(request, max_context_), &error);
    if (!rendered)
      throw std::invalid_argument(error);
    return rendered;
  }

  [[nodiscard]] ext::InitialOutputState InitialOutputState(
      const ChatRequest& request) const override {
    return QwenChatOptions(request, max_context_).enable_thinking
               ? ext::InitialOutputState::kReasoning
               : ext::InitialOutputState::kContent;
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    return model_->tokenizer().Decode(tokens);
  }

  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    return std::make_unique<Qwen36TextRunnerState>(model_, max_context_,
                                                   use_mtp_);
  }

  void PreparePrefixReuse(
      TextRunnerState& state,
      std::span<const TextRunnerToken> prefix) const override {
    const auto& q36 = RequireState(state);
    if (q36.position() != prefix.size()) {
      throw std::logic_error(
          "Qwen3.6-35B-A3B reused prefix does not match checkpoint");
    }
    q36.session().ResetDraftPolicy();
  }

  [[nodiscard]] TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& q36 = RequireState(state);
    if (offset != q36.position()) {
      throw std::logic_error(
          "Qwen3.6-35B-A3B prefill offset does not match retained state");
    }
    if (offset >= prompt.size()) {
      throw std::logic_error("Qwen3.6-35B-A3B prefill has no remaining input");
    }
    const std::size_t consumed = std::min<std::size_t>(
        {max_input_tokens, prompt.size() - offset, model_->PrefillCapacity()});
    const std::size_t next_position = offset + consumed;
    const auto prefix = EngineTokens(prompt.first(next_position));
    std::string error;
    if (!q36.session().Sync(prefix, &error)) {
      q36.set_position(0);
      throw std::runtime_error(Message("Qwen3.6-35B-A3B prefill failed: ", error));
    }
    q36.set_position(next_position);
    return {
        .consumed_tokens = consumed,
        .decode_ready = next_position == prompt.size(),
    };
  }

  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    auto& q36 = RequireState(state);
    if (q36.position() >= max_context_) {
      return {.stop = true, .piece = {}};
    }
    const auto logits = q36.session().Logits();
    if (logits.empty()) {
      throw std::runtime_error(
          "Qwen3.6-35B-A3B token selection has no logits");
    }
    const auto token = static_cast<std::int32_t>(sampler.Sample(logits));
    const auto logprobs =
        sampling::CollectLogprobs(logits, token, sampler.config().logprobs);
    if (q36.stop_at_eos() && model_->IsStopToken(token)) {
      return {.stop = true, .token = 0, .piece = {}, .logprobs = logprobs};
    }
    return {
        .stop = false,
        .token = static_cast<TextRunnerToken>(token),
        .piece = model_->TokenText(token),
        .logprobs = logprobs,
    };
  }

  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    if (token > static_cast<TextRunnerToken>(
                    std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument(
          "Qwen3.6-35B-A3B token ID exceeds engine range");
    }
    auto& q36 = RequireState(state);
    std::string error;
    if (!q36.session().Evaluate(static_cast<std::int32_t>(token), &error)) {
      throw std::runtime_error(Message("Qwen3.6-35B-A3B decode failed: ", error));
    }
    q36.set_position(q36.position() + 1);
  }

  [[nodiscard]] std::optional<TextDecodeSelection> PreviewFirstToken(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    return SelectNext(state, sampler);
  }

  [[nodiscard]] TextDecodeStep DecodeStep(
      TextRunnerState& state, std::size_t max_tokens,
      sampling::SamplerState& sampler) const override {
    if (!use_mtp_ || max_tokens == 1) {
      return TextModelRunner::DecodeStep(state, max_tokens, sampler);
    }
    if (max_tokens == 0) {
      throw std::invalid_argument(
          "Qwen3.6-35B-A3B MTP decode budget must be at least one token");
    }
    auto& q36 = RequireState(state);
    if (q36.position() >= max_context_) {
      return {.selections = {}, .stop = true};
    }
    const auto stats_before = q36.session().Statistics();
    sampling::SamplerState working_sampler = sampler;
    Session::DecodeResult decoded;
    std::string error;
    const auto budget =
        std::min<std::size_t>(max_tokens, std::uint64_t{max_draft_tokens_} + 1);
    if (!q36.session().DecodeStep(budget, working_sampler, &decoded, &error,
                                  q36.stop_at_eos())) {
      throw std::runtime_error(
          Message("Qwen3.6-35B-A3B MTP decode failed: ", error));
    }
    // The pool accepts the returned tokens once. Publish the RNG and residual
    // draw so the next batch retains the rejection-conditioned distribution.
    sampler.CopyDrawStateFrom(working_sampler);
    TextDecodeStep step;
    step.stop = decoded.stop;
    step.selections.reserve(decoded.tokens.size());
    for (const std::int32_t token : decoded.tokens) {
      step.selections.push_back({
          .stop = false,
          .token = static_cast<TextRunnerToken>(token),
          .piece = model_->TokenText(token),
      });
    }
    q36.set_position(q36.session().Position());
    const auto stats_after = q36.session().Statistics();
    step.draft_rounds = stats_after.cycles - stats_before.cycles;
    step.draft_tokens = stats_after.drafted - stats_before.drafted;
    step.draft_accepted_tokens = stats_after.accepted - stats_before.accepted;
    return step;
  }

  void AdvanceBatch(
      std::span<const TextRunnerAdvance> advances) const override {
    if (advances.size() < 2) {
      return TextModelRunner::AdvanceBatch(advances);
    }
    std::vector<Session::BatchOutcome> outcomes(advances.size());
    std::vector<Session::AdvanceRequest> requests;
    for (const auto& advance : advances) {
      if (advance.token > static_cast<TextRunnerToken>(
                              std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument(
            Message(kShortName, " token exceeds engine range"));
      }
      requests.push_back({&RequireState(advance.state.get()).session(),
                          static_cast<std::int32_t>(advance.token),
                          &outcomes[requests.size()]});
    }
    std::string error;
    (void)Session::EvaluateBatch(requests, &error);
    for (std::size_t i = 0; i < advances.size(); ++i) {
      const auto& advance = advances[i];
      if (!outcomes[i].completed) {
        auto failure = std::make_exception_ptr(std::runtime_error(
            Message("Qwen3.6 advance failed: ", outcomes[i].error)));
        if (!advance.failure)
          std::rethrow_exception(failure);
        *advance.failure = failure;
        continue;
      }
      auto& state = RequireState(advance.state.get());
      state.set_position(state.session().Position());
    }
  }

  [[nodiscard]] std::vector<TextDecodeStep> DecodeBatch(
      std::span<const TextRunnerDecode> decodes) const override {
    if (decodes.size() < 2 || !use_mtp_) {
      return TextModelRunner::DecodeBatch(decodes);
    }
    const auto count = decodes.size();
    std::vector<sampling::SamplerState> samplers;
    std::vector<Session::DecodeResult> results(count);
    std::vector<Session::BatchOutcome> outcomes(count);
    std::vector<Session::SpeculativeStats> before;
    std::vector<Session::DecodeRequest> requests;
    samplers.reserve(count);
    before.reserve(count);
    requests.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      auto& state = RequireState(decodes[i].state.get());
      auto& session = state.session();
      auto& sampler = samplers.emplace_back(decodes[i].sampler.get());
      before.push_back(session.Statistics());
      requests.push_back(
          {&session,
           std::min<std::size_t>(decodes[i].max_tokens,
                                 std::uint64_t{max_draft_tokens_} + 1),
           &sampler, &results[i], state.stop_at_eos(), &outcomes[i]});
    }
    std::string error;
    (void)Session::DecodeBatch(requests, &error);
    const auto active_count = static_cast<std::size_t>(std::count_if(
        results.begin(), results.end(),
        [](const auto& result) { return !result.tokens.empty(); }));
    std::vector<TextDecodeStep> steps(count);
    for (std::size_t i = 0; i < count; ++i) {
      if (!outcomes[i].completed) {
        steps[i].failure = std::make_exception_ptr(std::runtime_error(
            Message("Qwen3.6 MTP failed: ", outcomes[i].error)));
        continue;
      }
      auto& state = RequireState(decodes[i].state.get());
      decodes[i].sampler.get().CopyDrawStateFrom(samplers[i]);
      state.set_position(state.session().Position());
      auto& step = steps[i];
      step.stop = results[i].stop;
      if (active_count > 1 && !results[i].tokens.empty()) {
        step.execution_plan = {.kind = TextExecutionPlanKind::kBatched,
                               .physical_width = active_count};
      }
      for (const auto token : results[i].tokens) {
        step.selections.push_back({.stop = false,
                                   .token = static_cast<TextRunnerToken>(token),
                                   .piece = model_->TokenText(token)});
      }
      const auto stats = state.session().Statistics();
      step.draft_rounds = stats.cycles - before[i].cycles;
      step.draft_tokens = stats.drafted - before[i].drafted;
      step.draft_accepted_tokens = stats.accepted - before[i].accepted;
    }
    return steps;
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return RequireState(state).position();
  }

  [[nodiscard]] std::size_t SnapshotPayloadBytes(
      const TextRunnerState& state) const override {
    const std::uint64_t bytes = RequireState(state).session().SnapshotBytes();
    if (bytes == 0 || bytes > static_cast<std::uint64_t>(
                                  std::numeric_limits<std::size_t>::max())) {
      throw std::overflow_error("Qwen3.6-35B-A3B snapshot size is unavailable");
    }
    return static_cast<std::size_t>(bytes);
  }

  [[nodiscard]] std::unique_ptr<TextRunnerSnapshot> Snapshot(
      const TextRunnerState& state) const override {
    const auto& q36 = RequireState(state);
    std::string error;
    auto snapshot = q36.session().SaveSnapshot(&error);
    if (snapshot == nullptr) {
      throw std::runtime_error(
          Message("Qwen3.6-35B-A3B snapshot failed: ", error));
    }
    return std::make_unique<Qwen36TextRunnerSnapshot>(
        model_, std::move(snapshot), q36.position());
  }

  void RestoreOrFork(TextRunnerState& state,
                     const TextRunnerSnapshot& snapshot) const override {
    const auto& owned = OwnedSnapshot(snapshot);
    auto& restored = RequireState(state);
    std::string error;
    if (!restored.session().RestoreSnapshot(*owned.snapshot, &error)) {
      restored.Invalidate();
      throw std::runtime_error(
          Message("Qwen3.6-35B-A3B snapshot restore failed: ", error));
    }
    restored.set_position(owned.position);
  }

  [[nodiscard]] std::size_t PersistentSnapshotPayloadBytes(
      const TextRunnerSnapshot& snapshot) const override {
    return OwnedSnapshot(snapshot).PayloadBytes();
  }

  [[nodiscard]] std::size_t SerializePersistentSnapshot(
      const TextRunnerSnapshot& snapshot,
      std::span<std::uint8_t> destination) const override {
    const auto& owned = OwnedSnapshot(snapshot);
    if (destination.size() != owned.PayloadBytes() ||
        !owned.snapshot->CopyTo(destination)) {
      throw std::invalid_argument(
          "Qwen3.6-35B-A3B persistent snapshot serialization failed");
    }
    return destination.size();
  }

  void StreamPersistentSnapshot(const TextRunnerSnapshot& snapshot,
                                const SnapshotSink& sink) const override {
    sink(OwnedSnapshot(snapshot).snapshot->bytes());
  }

  void RestorePersistentSnapshot(
      TextRunnerState& state,
      std::span<const std::uint8_t> payload) const override {
    auto& restored = RequireState(state);
    std::string error;
    if (!restored.session().RestoreSnapshot(payload, &error)) {
      restored.Invalidate();
      throw std::runtime_error(
          Message("Qwen3.6-35B-A3B persistent snapshot restore failed: ", error));
    }
    const std::uint32_t position = restored.session().Position();
    if (position == 0 || position > max_context_) {
      restored.Invalidate();
      throw std::runtime_error(
          "Qwen3.6-35B-A3B persistent snapshot restored an invalid position");
    }
    restored.set_position(position);
  }

private:
  const Qwen36TextRunnerSnapshot& OwnedSnapshot(
      const TextRunnerSnapshot& snapshot) const {
    const auto* owned = dynamic_cast<const Qwen36TextRunnerSnapshot*>(&snapshot);
    if (owned == nullptr || owned->model.get() != model_.get() ||
        owned->snapshot == nullptr) {
      throw std::invalid_argument(
          "Qwen3.6-35B-A3B snapshot does not belong to this model");
    }
    return *owned;
  }

  std::shared_ptr<Model> model_;
  std::uint32_t max_context_;
  bool use_mtp_;
  std::uint32_t max_draft_tokens_;
  std::optional<TextRunnerPersistenceDescriptor> persistence_;
};

}  // namespace

std::shared_ptr<ext::TextModelRunner> CreateQwen36TextRunner(
    std::shared_ptr<Model> model, std::uint32_t max_context,
    const ext::TextSpeculativeConfig& speculative_config,
    std::string artifact_fingerprint) {
  const bool use_mtp =
      speculative_config.backend == ext::TextSpeculativeBackend::kMtp;
  std::optional<TextRunnerPersistenceDescriptor> persistence;
  if (!artifact_fingerprint.empty()) {
    persistence = TextRunnerPersistenceDescriptor{
        .compatibility_identity = Qwen36CompatibilityIdentity(
            artifact_fingerprint, use_mtp, max_context,
            speculative_config.max_draft_tokens, model->DecodeConcurrency()),
        .payload_version = Session::kSnapshotPayloadVersion,
    };
  }
  return std::make_shared<Qwen36TextRunner>(
      std::move(model), max_context, use_mtp,
      speculative_config.max_draft_tokens, std::move(persistence));
}

std::shared_ptr<ext::TextModelRunner> CreateQwen36TextRunner(
    const ext::TextRunnerOptions& options, std::string* error) {
  const auto fail = [error](std::string message) {
    if (error)
      *error = std::move(message);
    return std::shared_ptr<ext::TextModelRunner>{};
  };
  std::shared_ptr<const core::GgufReader> reader = options.reader;
  if (!reader) {
    std::string load_error;
    auto owner = core::GgufReader::OpenFile(options.model_path, &load_error);
    if (!owner)
      return fail("Failed to open GGUF: " + load_error);
    reader = std::shared_ptr<const core::GgufReader>(std::move(owner));
  }
  if (reader->GetMetadataString("general.architecture").value_or("") !=
      "qwen35moe") {
    return fail("Qwen3.6-35B-A3B requires architecture qwen35moe");
  }
  const auto& speculative = options.speculative;
  const bool use_mtp = speculative.backend == ext::TextSpeculativeBackend::kMtp;
  if (speculative.backend != ext::TextSpeculativeBackend::kDisabled &&
      !use_mtp) {
    return fail(
        "Qwen3.6-35B-A3B HTTP models support only MTP speculative decoding "
        "(--speculative mtp)");
  }
  if (use_mtp && !speculative.draft_model_path.empty()) {
    return fail(
        "Qwen3.6-35B-A3B MTP is native to the model artifact; omit "
        "--mtp-model");
  }
  if (!options.vision_model_path.empty())
    return fail("Qwen3.6-35B-A3B serves text only; omit --mmproj");
  if (options.enable_yarn)
    return fail("Qwen3.6-35B-A3B does not support --enable-yarn");
  if (options.session_count == 0)
    return fail("HTTP session count must be at least one");
  if (use_mtp && (speculative.max_draft_tokens == 0 ||
                  speculative.min_draft_tokens != 1)) {
    return fail(
        "Qwen3.6-35B-A3B MTP drafts a confidence-gated chain; custom draft "
        "floors are unsupported");
  }
  std::string load_error;
  if (!tokenization::QwenChatTemplate::ValidateGgufTemplate(*reader,
                                                            &load_error)) {
    return fail("Unsupported Qwen3.6-35B-A3B chat template: " + load_error);
  }
  auto model = Model::Load(
      options.model_path,
      ModelOptions{
          .max_context = options.max_context,
          .mtp = use_mtp,
          .max_draft_tokens = speculative.max_draft_tokens,
          .decode_concurrency =
              DecodeConcurrencyForSessions(options.session_count),
      },
      &load_error);
  if (model == nullptr)
    return fail("Failed to create Qwen3.6-35B-A3B model: " + load_error);
  const std::uint32_t max_context =
      options.max_context == 0 ? model->MaxContext() : options.max_context;
  if (max_context == 0 || max_context > model->MaxContext()) {
    return fail(
        "HTTP context exceeds the loaded Qwen3.6-35B-A3B model context");
  }
  if (use_mtp && !model->HasMtp()) {
    return fail(
        "Qwen3.6-35B-A3B MTP requires a model loaded with its MTP block");
  }
  std::string fingerprint = options.disk_cache.model_artifact_fingerprint;
  if (!options.disk_cache.directory.empty()) {
    if (fingerprint.empty()) {
      try {
        fingerprint = core::GgufIdentityHex(*reader);
      } catch (const std::exception& e) {
        return fail(std::string("Qwen3.6-35B-A3B disk cache fingerprinting "
                                "failed: ") +
                    e.what());
      }
    }
    if (!IsSha256Hex(fingerprint) || options.disk_cache.capacity_bytes == 0) {
      return fail(
          "Qwen3.6-35B-A3B persistent disk cache configuration is invalid");
    }
  } else {
    fingerprint.clear();
  }
  return CreateQwen36TextRunner(std::move(model), max_context, speculative,
                                std::move(fingerprint));
}

}  // namespace gufo::models::qwen36_35b_a3b

#endif  // ENGINE_ENABLE_HIP
