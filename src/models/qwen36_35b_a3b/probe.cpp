#include "src/models/qwen36_35b_a3b/probe.hpp"

#include <filesystem>
#include <string>

namespace gufo::models::qwen36_35b_a3b {

bool ProbeQwen36Gguf(const core::GgufReader& reader) {
  return reader.GetMetadataString("general.architecture").value_or("") ==
         "qwen35moe";
}

bool ProbeQwen36(std::string_view model_root) {
  try {
    const std::filesystem::path path(model_root);
    if (!std::filesystem::is_regular_file(path))
      return false;
    std::string error;
    auto reader = core::GgufReader::OpenFile(path.string(), &error);
    return reader != nullptr && ProbeQwen36Gguf(*reader);
  } catch (...) {
    return false;
  }
}

}  // namespace gufo::models::qwen36_35b_a3b
