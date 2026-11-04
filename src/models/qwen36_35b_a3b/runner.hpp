#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "gufo/ext/text.hpp"

namespace gufo::models::qwen36_35b_a3b {
class Model;

inline constexpr std::uint32_t DecodeConcurrencyForSessions(
    std::size_t session_count) {
  return static_cast<std::uint32_t>(
      std::clamp<std::size_t>(session_count, 1, 8));
}

#if defined(ENGINE_ENABLE_HIP)
std::shared_ptr<ext::TextModelRunner> CreateQwen36TextRunner(
    const ext::TextRunnerOptions& options, std::string* error);

std::shared_ptr<ext::TextModelRunner> CreateQwen36TextRunner(
    std::shared_ptr<Model> model, std::uint32_t max_context,
    const ext::TextSpeculativeConfig& speculative_config = {},
    std::string artifact_fingerprint = {});
#endif

}  // namespace gufo::models::qwen36_35b_a3b
