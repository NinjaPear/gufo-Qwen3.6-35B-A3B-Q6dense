#include "src/models/qwen36_35b_a3b/continuation_identity.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>

#include "gufo/ext/continuation.hpp"
#include "src/families/qwen/text/chat_template.hpp"
#include "src/models/qwen36_35b_a3b/engine.hpp"

namespace gufo::models::qwen36_35b_a3b {

bool IsSha256Hex(std::string_view value) noexcept {
  return value.size() == 64 && std::ranges::all_of(value, [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

// Byte-identical to the identity the fork wrote before the extension port,
// so existing disk-cache entries stay valid.
std::vector<std::uint8_t> Qwen36CompatibilityIdentity(
    std::string_view artifact_fingerprint, bool has_mtp,
    std::uint32_t max_context, std::uint32_t max_draft_tokens,
    std::uint32_t decode_concurrency) {
  if (!IsSha256Hex(artifact_fingerprint)) {
    throw std::invalid_argument(
        "Qwen3.6-35B-A3B disk cache requires an artifact fingerprint");
  }
  std::ostringstream identity;
  identity << "schema=gufo-text-continuation-v2\n"
           << "model_kind=qwen36-35b-a3b\n"
           << "artifact_id=" << ext::kGgufIdentityScheme << ':'
           << artifact_fingerprint << '\n'
           << "tokenizer=embedded-in-artifact\n"
           << "chat_template=qwen36-compiled-v1\n"
           << "chat_template_reference_sha256="
           << tokenization::QwenChatTemplate::Qwen36OfficialTemplateSha256()
           << '\n'
           << "state_abi=qwen36-35b-a3b-rocm-session-v1\n"
           << "payload_layout=q36-rocm-session-snapshot-v"
           << Session::kSnapshotPayloadVersion << '\n'
           << "context_tokens=" << max_context << '\n'
           << "position_policy=absolute-v1\n"
           << "adapters=none\n";
  if (has_mtp) {
    identity << "draft_backend=q36-native-mtp-v1\n"
             << "draft_max_tokens=" << max_draft_tokens << '\n'
             << "draft_cost_concurrency=" << decode_concurrency << '\n';
  }
  const std::string canonical = identity.str();
  return {canonical.begin(), canonical.end()};
}

}  // namespace gufo::models::qwen36_35b_a3b
