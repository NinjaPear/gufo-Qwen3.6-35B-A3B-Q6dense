#ifndef GUFO_SERVER_RESPONSES_LOGPROBS_HPP_
#define GUFO_SERVER_RESPONSES_LOGPROBS_HPP_

#include "src/cli/serve/logprobs.hpp"

namespace gufo::server {

inline std::optional<SamplingRequestError> ParseResponsesLogprobs(
    const json::Value& body, sampling::SamplingConfig* config) {
  config->logprobs.reset();
  bool enabled = false;
  if (const auto* include = body.find("include");
      include && !include->is_null()) {
    if (!include->is_array())
      return SamplingRequestError{"'include' must be an array of strings",
                                  "invalid_include"};
    for (const auto& value : include->items()) {
      if (!value.is_string())
        return SamplingRequestError{"'include' must be an array of strings",
                                    "invalid_include"};
      enabled |= value.str() == "message.output_text.logprobs";
    }
  }
  std::uint32_t count = 0;
  if (auto error = detail::ReadSamplingInteger(body, "top_logprobs", 0U, 20U,
                                               &count, true))
    return error;
  if (enabled)
    config->logprobs = count;
  return std::nullopt;
}

/// A range in the UTF-8-normalized, unfiltered model output.
struct TextSourceSpan {
  std::size_t begin;
  std::size_t size;
};

// Streaming text events omit the bytes field carried by output_text parts.
inline json::Value ResponsesStreamLogprobs(const json::Value& tokens) {
  auto result = json::Value::array();
  for (const auto& token : tokens.items()) {
    auto entry = json::Value::object();
    entry["token"] = *token.find("token");
    entry["logprob"] = *token.find("logprob");
    entry["top_logprobs"] = json::Value::array();
    for (const auto& top : token.find("top_logprobs")->items()) {
      auto alternative = json::Value::object();
      alternative["token"] = *top.find("token");
      alternative["logprob"] = *top.find("logprob");
      entry["top_logprobs"].push_back(std::move(alternative));
    }
    result.push_back(std::move(entry));
  }
  return result;
}

/// Join filtered output spans to generated tokens by source position, never
/// by matching text (a tool argument can repeat the visible answer verbatim).
/// A token crossing a text/framing boundary belongs to its first visible
/// text span. Pure reasoning, tool markup and hidden stop tokens are omitted.
class ResponsesLogprobs {
public:
  explicit ResponsesLogprobs(std::uint32_t count) : count_(count) {}

  void Push(const TokenPiece& piece, bool final = false) {
    if (piece.logprobs && !piece.text.empty())
      tokens_.push_back({.input_begin = input_bytes_,
                         .input_end = input_bytes_ + piece.text.size(),
                         .values = *piece.logprobs});
    input_bytes_ += piece.text.size();
    (void)decoder_.Push(piece.text, final, [&](auto input, auto output) {
      const auto end = consumed_bytes_ + input;
      for (auto i = mapped_; i < tokens_.size() && tokens_[i].input_begin < end;
           ++i) {
        auto& token = tokens_[i];
        if (!token.mapped)
          token.begin = decoded_bytes_;
        token.end = decoded_bytes_ + output;
        token.mapped = true;
      }
      while (mapped_ < tokens_.size() && tokens_[mapped_].input_end <= end)
        ++mapped_;
      consumed_bytes_ = end;
      decoded_bytes_ += output;
    });
  }

  json::Value Take(std::span<const TextSourceSpan> spans) {
    auto result = json::Value::array();
    for (const auto& span : spans) {
      if (!span.size)
        continue;
      // Unmapped tokens are only the trailing incomplete UTF-8 scalar.
      const auto mapped_end =
          tokens_.begin() + mapped_ +
          (mapped_ < tokens_.size() && tokens_[mapped_].mapped ? 1 : 0);
      auto it = std::lower_bound(
          tokens_.begin(), mapped_end, span.begin,
          [](const auto& token, auto begin) { return token.end <= begin; });
      for (; it != mapped_end && it->begin < span.begin + span.size; ++it) {
        if (it->emitted)
          continue;
        auto entries = ChatLogprobs(std::span(&it->values, 1), count_);
        result.push_back(entries.find("content")->items().front());
        it->emitted = true;
      }
    }
    return result;
  }

private:
  struct Token {
    std::size_t input_begin, input_end;
    DecodedLogprobs values;
    std::size_t begin{0}, end{0};
    bool mapped{false}, emitted{false};
  };
  std::uint32_t count_;
  core::Utf8Decoder decoder_;
  std::vector<Token> tokens_;
  std::size_t input_bytes_{0}, consumed_bytes_{0}, decoded_bytes_{0},
      mapped_{0};
};

}  // namespace gufo::server
#endif  // GUFO_SERVER_RESPONSES_LOGPROBS_HPP_
