#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

namespace gufo::models::qwen36_35b_a3b {

[[nodiscard]] bool IsSha256Hex(std::string_view value) noexcept;

/// Qwen3.6-35B-A3B carries its MTP block in the main artifact, so the
/// artifact identity covers the draft weights too.
[[nodiscard]] std::vector<std::uint8_t> Qwen36CompatibilityIdentity(
    std::string_view artifact_fingerprint, bool has_mtp,
    std::uint32_t max_context, std::uint32_t max_draft_tokens,
    std::uint32_t decode_concurrency);

}  // namespace gufo::models::qwen36_35b_a3b
