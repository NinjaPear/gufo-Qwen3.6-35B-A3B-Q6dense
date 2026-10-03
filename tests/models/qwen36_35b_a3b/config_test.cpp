#include "src/models/qwen36_35b_a3b/config.hpp"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace q36 = gufo::models::qwen36_35b_a3b;

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

using Value =
    std::variant<std::uint32_t, float, std::string, std::vector<std::uint64_t>,
                 std::vector<std::int64_t>, std::vector<double>>;
using Metadata = std::map<std::string, Value>;

// Minimal metadata-only GGUF: no model or GPU is needed to test the loader.
std::optional<q36::Config> Parse(const Metadata& fields,
                                 std::string* error = nullptr) {
  std::vector<std::uint8_t> bytes;
  const auto pod = [&]<class T>(T value) {
    const auto begin = bytes.size();
    bytes.resize(begin + sizeof(value));
    std::memcpy(bytes.data() + begin, &value, sizeof(value));
  };
  const auto text = [&](const std::string& value) {
    pod(std::uint64_t{value.size()});
    bytes.insert(bytes.end(), value.begin(), value.end());
  };
  pod(std::uint32_t{0x46554747});
  pod(std::uint32_t{3});
  pod(std::uint64_t{0});
  pod(std::uint64_t{fields.size()});
  for (const auto& [key, value] : fields) {
    text(key);
    std::visit(
        [&](const auto& item) {
          using T = std::decay_t<decltype(item)>;
          if constexpr (std::is_same_v<T, std::uint32_t>) {
            pod(std::uint32_t{4});
            pod(item);
          } else if constexpr (std::is_same_v<T, float>) {
            pod(std::uint32_t{6});
            pod(item);
          } else if constexpr (std::is_same_v<T, std::string>) {
            pod(std::uint32_t{8});
            text(item);
          } else {
            pod(std::uint32_t{9});
            using E = typename T::value_type;
            pod(std::uint32_t{std::is_same_v<E, std::uint64_t>  ? 10U
                              : std::is_same_v<E, std::int64_t> ? 11U
                                                                : 12U});
            pod(std::uint64_t{item.size()});
            for (const auto entry : item)
              pod(entry);
          }
        },
        value);
  }
  bytes.resize((bytes.size() + 31) / 32 * 32);
  std::string local;
  std::string& message = error != nullptr ? *error : local;
  const auto reader =
      gufo::core::GgufReader::OpenMemory(bytes.data(), bytes.size(), &message);
  Require(reader != nullptr && message.empty(), "metadata GGUF did not open");
  const auto result = q36::Config::FromGguf(*reader, &message);
  Require(result.has_value() == message.empty(),
          "a rejected config must explain itself");
  return result;
}

// The released Qwen3.6-35B-A3B metadata, MTP block included.
Metadata ValidMetadata() {
  Metadata fields{{"general.architecture", std::string{"qwen35moe"}}};
  for (const auto& [key, value] :
       std::initializer_list<std::pair<const char*, std::uint32_t>>{
           {"block_count", 41},
           {"nextn_predict_layers", 1},
           {"embedding_length", 2048},
           {"context_length", 262144},
           {"full_attention_interval", 4},
           {"attention.head_count", 16},
           {"attention.head_count_kv", 2},
           {"attention.key_length", 256},
           {"attention.value_length", 256},
           {"rope.dimension_count", 64},
           {"ssm.conv_kernel", 4},
           {"ssm.state_size", 128},
           {"ssm.group_count", 16},
           {"ssm.time_step_rank", 32},
           {"ssm.inner_size", 4096},
           {"expert_count", 256},
           {"expert_used_count", 8},
           {"expert_feed_forward_length", 512},
           {"expert_shared_feed_forward_length", 512}}) {
    fields[std::string{"qwen35moe."} + key] = value;
  }
  fields["qwen35moe.attention.layer_norm_rms_epsilon"] = 1e-6F;
  fields["qwen35moe.rope.freq_base"] = 1e7F;
  fields["qwen35moe.rope.dimension_sections"] =
      std::vector<std::uint64_t>{11, 11, 10, 0};
  return fields;
}

void CheckValidMetadata() {
  const auto c = Parse(ValidMetadata());
  Require(c.has_value(), "the released metadata must parse");
  Require(c->num_layers == 40 && c->num_layers_all == 41 && c->HasMtp(),
          "the MTP block is not a trunk layer");
  Require(c->SsmConvChannels() == 8192 && c->SsmValueDim() == 4096 &&
              c->AttentionQDim() == 4096 && c->AttentionKvDim() == 512,
          "derived projection widths changed");
  // Every fourth layer is full attention: 30 DeltaNet + 10 GQA.
  std::uint32_t linear = 0;
  for (std::uint32_t layer = 0; layer < c->num_layers; ++layer)
    linear += c->IsLinearLayer(layer) ? 1 : 0;
  Require(linear == 30 && !c->IsLinearLayer(3) && c->IsLinearLayer(4) &&
              !c->IsLinearLayer(39) && !c->IsLinearLayer(40),
          "layer schedule changed");

  // Artifacts exported without the MTP block keep the same trunk.
  auto trunk = ValidMetadata();
  trunk.erase("qwen35moe.nextn_predict_layers");
  trunk["qwen35moe.block_count"] = std::uint32_t{40};
  const auto plain = Parse(trunk);
  Require(plain && plain->num_layers == 40 && !plain->HasMtp(),
          "a trunk-only artifact must parse without MTP");
  // The value length may be omitted; it defaults to the key length.
  auto implicit = ValidMetadata();
  implicit.erase("qwen35moe.attention.value_length");
  Require(Parse(implicit).has_value(), "value length must default to keys");
}

void CheckMalformedMetadata() {
  const auto valid = ValidMetadata();
  const auto reject = [](const Metadata& fields, const std::string& what) {
    std::string error;
    Require(!Parse(fields, &error), "accepted " + what);
    Require(!error.empty(), "rejected " + what + " without a reason");
  };
  {
    auto wrong = valid;
    wrong["general.architecture"] = std::string{"qwen4exp"};
    reject(wrong, "another architecture");
  }
  for (const auto* key : {"block_count",
                          "embedding_length",
                          "context_length",
                          "full_attention_interval",
                          "attention.head_count",
                          "attention.head_count_kv",
                          "attention.key_length",
                          "rope.dimension_count",
                          "ssm.conv_kernel",
                          "ssm.state_size",
                          "ssm.group_count",
                          "ssm.time_step_rank",
                          "ssm.inner_size",
                          "expert_count",
                          "expert_used_count",
                          "expert_feed_forward_length",
                          "expert_shared_feed_forward_length",
                          "rope.dimension_sections",
                          "attention.layer_norm_rms_epsilon",
                          "rope.freq_base"}) {
    auto wrong = valid;
    wrong.erase(std::string{"qwen35moe."} + key);
    reject(wrong, std::string{"missing "} + key);
    wrong[std::string{"qwen35moe."} + key] = std::string{"4"};
    reject(wrong, std::string{"string "} + key);
  }
  for (const auto& sections :
       {std::vector<std::uint64_t>{11, 11, 10},
        std::vector<std::uint64_t>{11, 11, 10, 0, 0},
        std::vector<std::uint64_t>{11, 11, 10, 1},
        std::vector<std::uint64_t>{11, 11, 11, 0},
        std::vector<std::uint64_t>{10, 11, 11, 0},
        std::vector<std::uint64_t>{0, 16, 16, 0},
        std::vector<std::uint64_t>{UINT64_MAX, 11, 10, 0}}) {
    auto wrong = valid;
    wrong["qwen35moe.rope.dimension_sections"] = sections;
    reject(wrong, "mRoPE sections");
  }
  for (const Value& value :
       {Value{std::vector<std::int64_t>{-1, 11, 10, 0}},
        Value{std::vector<double>{11.0, 11.0, 10.0, 0.0}}}) {
    auto wrong = valid;
    wrong["qwen35moe.rope.dimension_sections"] = value;
    reject(wrong, "non-integer mRoPE sections");
  }
  for (const float value :
       {0.0F, -1e-6F, std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN()}) {
    for (const auto* key :
         {"attention.layer_norm_rms_epsilon", "rope.freq_base"}) {
      auto wrong = valid;
      wrong[std::string{"qwen35moe."} + key] = value;
      reject(wrong, std::string{"bad float "} + key);
    }
  }
  // Each kernel-selecting dimension is pinned to the validated profile, and
  // the MTP block must stay a single trailing block.
  for (const auto& [key, value] :
       std::initializer_list<std::pair<const char*, std::uint32_t>>{
           {"block_count", 49},
           {"nextn_predict_layers", 2},
           {"nextn_predict_layers", 41},
           {"embedding_length", 2560},
           {"context_length", 0},
           {"full_attention_interval", 0},
           {"full_attention_interval", 3},
           {"attention.head_count", 24},
           {"attention.head_count", 15},
           {"attention.head_count_kv", 4},
           {"attention.key_length", 128},
           {"attention.value_length", 128},
           {"rope.dimension_count", 128},
           {"ssm.conv_kernel", 3},
           {"ssm.state_size", 64},
           {"ssm.group_count", 32},
           {"ssm.time_step_rank", 48},
           {"ssm.inner_size", 6144},
           {"expert_count", 512},
           {"expert_used_count", 0},
           {"expert_used_count", 10},
           {"expert_used_count", 257},
           {"expert_feed_forward_length", 640},
           {"expert_shared_feed_forward_length", 0}}) {
    auto wrong = valid;
    wrong[std::string{"qwen35moe."} + key] = value;
    reject(wrong, std::string{key} + "=" + std::to_string(value));
  }
}

}  // namespace

int main() {
  try {
    CheckValidMetadata();
    CheckMalformedMetadata();
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
  std::cout << "Qwen3.6 metadata checks passed.\n";
}
