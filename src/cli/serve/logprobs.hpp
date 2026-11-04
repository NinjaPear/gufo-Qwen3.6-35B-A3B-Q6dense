#ifndef GUFO_SERVER_LOGPROBS_HPP_
#define GUFO_SERVER_LOGPROBS_HPP_

#include <algorithm>
#include <cmath>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "src/cli/serve/sampling_request.hpp"
#include "src/core/utf8.hpp"

namespace gufo::server {

struct DecodedLogprob {
  sampling::TokenId id{0};
  std::string token;
  double logprob{0};
};

/// Sampled token first, then the unique top candidates in descending order,
/// as in vLLM v0.32.0's append_logprobs_for_next_position.
struct DecodedLogprobs {
  std::vector<DecodedLogprob> entries;
  [[nodiscard]] std::size_t StorageBytes() const {
    std::size_t bytes =
        sizeof(*this) + entries.capacity() * sizeof(DecodedLogprob);
    for (const auto& entry : entries)
      bytes += entry.token.capacity();
    return bytes;
  }
};

/// A borrowed event; text-only consumers retain the existing string-view API.
struct TokenPiece {
  std::string_view text;
  const DecodedLogprobs* logprobs{nullptr};
  TokenPiece(std::string_view value,
             const DecodedLogprobs* probabilities = nullptr)
      : text(value), logprobs(probabilities) {}
  TokenPiece(const std::string& value) : text(value) {}
  TokenPiece(const char* value) : text(value) {}
  operator std::string_view() const { return text; }
};

inline std::optional<SamplingRequestError> ParseLogprobs(
    const json::Value& body, bool chat, sampling::SamplingConfig* config) {
  config->logprobs.reset();
  const auto* enabled = body.find("logprobs");
  if (chat) {
    if (enabled && !enabled->is_null() && !enabled->is_bool())
      return SamplingRequestError{"'logprobs' must be a boolean",
                                  "invalid_logprobs"};
    std::uint32_t count = 0;
    if (auto error = detail::ReadSamplingInteger(body, "top_logprobs", 0U, 20U,
                                                 &count, true))
      return error;
    if (enabled && enabled->as_bool())
      config->logprobs = count;
    else if (count)
      return SamplingRequestError{"'top_logprobs' requires 'logprobs': true",
                                  "invalid_top_logprobs"};
  } else if (enabled && !enabled->is_null()) {
    std::uint32_t count = 0;
    if (auto error = detail::ReadSamplingInteger(body, "logprobs", 0U, 20U,
                                                 &count, true))
      return error;
    config->logprobs = count;
  }
  return std::nullopt;
}

/// Correct byte-fallback tokens against the last four sampled token pieces,
/// independently for every alternative (vLLM v0.32.0 LogprobsProcessor).
class LogprobDecoder {
public:
  template<typename Decode>
  DecodedLogprobs DecodeToken(const sampling::TokenLogprobs& values,
                              Decode decode) {
    DecodedLogprobs result;
    const auto raw = decode(values.selected.token);
    const auto append = [&](const sampling::TokenLogprob& entry) {
      const auto bytes = decode(entry.token);
      result.entries.push_back({entry.token, Correct(bytes), entry.logprob});
    };
    append(values.selected);
    for (const auto& entry : values.top)
      if (entry.token != values.selected.token)
        append(entry);
    history_.push_back(raw);
    if (history_.size() > 4)
      history_.erase(history_.begin());
    return result;
  }

private:
  std::string Correct(std::string_view raw) const {
    constexpr std::string_view replacement = "\xEF\xBF\xBD";
    auto text = core::Utf8Decoder{}.Push(raw, true);
    if (!text.ends_with(replacement))
      return text;
    for (std::size_t count = 1; count <= history_.size(); ++count) {
      const auto begin = history_.size() - count;
      std::string combined;
      for (auto i = begin; i < history_.size(); ++i)
        combined += history_[i];
      combined += raw;
      auto decoded = core::Utf8Decoder{}.Push(combined, true);
      if (decoded.ends_with(replacement))
        continue;
      auto clean_end = history_.size();
      while (clean_end > begin && core::Utf8Decoder{}
                                      .Push(history_[clean_end - 1], true)
                                      .ends_with(replacement))
        --clean_end;
      std::string clean;
      for (auto i = begin; i < clean_end; ++i)
        clean += history_[i];
      clean = core::Utf8Decoder{}.Push(clean, true);
      std::size_t common = 0;
      while (common < clean.size() && common < decoded.size() &&
             clean[common] == decoded[common])
        ++common;
      // Keep the substring on a Unicode scalar boundary.
      while (common < decoded.size() && common > 0 &&
             (static_cast<unsigned char>(decoded[common]) & 0xc0) == 0x80)
        --common;
      return decoded.substr(common);
    }
    return {};
  }
  std::vector<std::string> history_;
};

inline json::Value LogprobJson(const DecodedLogprob& value) {
  auto result = json::Value::object();
  result["token"] = value.token;
  result["logprob"] = std::max(value.logprob, -9999.0);
  result["bytes"] = json::Value::array();
  for (const unsigned char byte : value.token)
    result["bytes"].push_back(static_cast<int>(byte));
  return result;
}

inline json::Value ChatLogprobs(std::span<const DecodedLogprobs> tokens,
                                std::uint32_t count) {
  auto result = json::Value::object();
  result["content"] = json::Value::array();
  for (const auto& token : tokens) {
    auto entry = LogprobJson(token.entries.at(0));
    entry["top_logprobs"] = json::Value::array();
    for (std::size_t i = 0;
         i < std::min<std::size_t>(count, token.entries.size()); ++i)
      entry["top_logprobs"].push_back(LogprobJson(token.entries[i]));
    result["content"].push_back(std::move(entry));
  }
  return result;
}

inline std::size_t UnicodeLength(std::string_view text) {
  return std::count_if(text.begin(), text.end(), [](unsigned char byte) {
    return (byte & 0xc0) != 0x80;
  });
}

inline json::Value CompletionLogprobs(std::span<const DecodedLogprobs> tokens,
                                      std::size_t offset = 0) {
  auto result = json::Value::object();
  for (const auto* name :
       {"tokens", "token_logprobs", "top_logprobs", "text_offset"})
    result[name] = json::Value::array();
  for (const auto& token : tokens) {
    const auto& selected = token.entries.at(0);
    result["tokens"].push_back(selected.token);
    result["token_logprobs"].push_back(std::max(selected.logprob, -9999.0));
    result["text_offset"].push_back(offset);
    offset += UnicodeLength(selected.token);
    auto top = json::Value::object();
    for (const auto& entry : token.entries)
      top[entry.token] = std::max(entry.logprob, -9999.0);
    result["top_logprobs"].push_back(std::move(top));
  }
  return result;
}

}  // namespace gufo::server
#endif  // GUFO_SERVER_LOGPROBS_HPP_
