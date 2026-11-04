#pragma once

#include <string_view>

#include "gufo/ext/gguf_reader.hpp"

namespace gufo::models::qwen36_35b_a3b {

[[nodiscard]] bool ProbeQwen36(std::string_view model_root);
[[nodiscard]] bool ProbeQwen36Gguf(const core::GgufReader& reader);

}  // namespace gufo::models::qwen36_35b_a3b
