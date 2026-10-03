#include "src/models/qwen36_35b_a3b/config.hpp"

#include <cmath>
#include <limits>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace gufo::models::qwen36_35b_a3b {
namespace {

constexpr std::string_view kArchitecture = "qwen35moe";

struct Reader {
  const core::GgufReader& gguf;
  std::string* error;
  bool ok{true};

  void Fail(std::string message) {
    if (ok && error != nullptr) {
      *error = std::move(message);
    }
    ok = false;
  }

  std::uint32_t U32(std::string_view key, bool required = true,
                    std::uint32_t fallback = 0) {
    const auto value = gguf.GetMetadataUint32(key);
    if (value.has_value()) {
      return *value;
    }
    if (required || gguf.FindMetadata(key) != nullptr) {
      Fail("missing or invalid GGUF integer " + std::string(key));
    }
    return fallback;
  }

  float F32(std::string_view key, float fallback) {
    const auto value = gguf.GetMetadataFloat32(key);
    if (!value || !std::isfinite(*value))
      Fail("missing or invalid GGUF float " + std::string(key));
    return value.value_or(fallback);
  }

  /// Integer arrays of any width; GGUF writers pick the narrowest type.
  std::vector<std::uint64_t> U64Array(std::string_view key) {
    const auto* meta = gguf.FindMetadata(key);
    std::vector<std::uint64_t> out;
    if (meta == nullptr) {
      Fail("missing GGUF array " + std::string(key));
      return out;
    }
    if (const auto* u = std::get_if<std::vector<std::uint64_t>>(&meta->value)) {
      out = *u;
    } else if (const auto* s =
                   std::get_if<std::vector<std::int64_t>>(&meta->value)) {
      for (auto v : *s) {
        if (v < 0) {
          Fail("negative GGUF array value " + std::string(key));
          return {};
        }
        out.push_back(static_cast<std::uint64_t>(v));
      }
    } else {
      Fail("GGUF array must contain integers: " + std::string(key));
    }
    return out;
  }
};

}  // namespace

std::optional<Config> Config::FromGguf(const core::GgufReader& gguf,
                                       std::string* error_msg) {
  Reader r{gguf, error_msg};
  if (gguf.GetMetadataString("general.architecture") != kArchitecture) {
    r.Fail("GGUF architecture is not qwen35moe");
    return std::nullopt;
  }

  Config c;
  const std::string_view p = "qwen35moe.";
  auto key = [&](std::string_view suffix) {
    return std::string(p) + std::string(suffix);
  };

  c.num_layers_all = r.U32(key("block_count"));
  c.nextn_layers = r.U32(key("nextn_predict_layers"), false, 0);
  if (c.nextn_layers >= c.num_layers_all) {
    r.Fail("nextn_predict_layers must be below block_count");
    return std::nullopt;
  }
  c.num_layers = c.num_layers_all - c.nextn_layers;
  c.hidden_size = r.U32(key("embedding_length"));
  c.context_length = r.U32(key("context_length"));
  c.rms_eps = r.F32(key("attention.layer_norm_rms_epsilon"), 1e-6F);

  c.full_attention_interval = r.U32(key("full_attention_interval"));
  c.num_heads = r.U32(key("attention.head_count"));
  c.num_kv_heads = r.U32(key("attention.head_count_kv"));
  c.head_dim = r.U32(key("attention.key_length"));
  if (r.U32(key("attention.value_length"), false, c.head_dim) != c.head_dim)
    r.Fail("attention value length must match the supported key length");
  c.rotary_dim = r.U32(key("rope.dimension_count"));
  c.rope_theta = r.F32(key("rope.freq_base"), 10000000.0F);
  {
    const auto sections = r.U64Array(key("rope.dimension_sections"));
    if (sections.size() != 4 || sections[3] != 0) {
      r.Fail("rope.dimension_sections must contain three axes and a zero pad");
    } else {
      std::uint64_t sum = 0;
      for (std::size_t i = 0; i < 3; ++i) {
        if (sections[i] == 0 || sections[i] > c.rotary_dim / 2) {
          r.Fail("invalid mRoPE axis dimension");
          break;
        }
        sum += sections[i];
        c.rope_sections[i] = static_cast<std::uint32_t>(sections[i]);
      }
      if (sum != c.rotary_dim / 2)
        r.Fail("mRoPE sections must sum to half the rotary dimension");
    }
  }

  c.ssm_conv_kernel = r.U32(key("ssm.conv_kernel"));
  c.ssm_head_dim = r.U32(key("ssm.state_size"));
  c.ssm_num_k_heads = r.U32(key("ssm.group_count"));
  c.ssm_num_v_heads = r.U32(key("ssm.time_step_rank"));
  c.ssm_inner_size = r.U32(key("ssm.inner_size"));

  c.num_experts = r.U32(key("expert_count"));
  c.num_experts_used = r.U32(key("expert_used_count"));
  c.expert_ff = r.U32(key("expert_feed_forward_length"));
  c.shared_expert_ff = r.U32(key("expert_shared_feed_forward_length"));

  if (!r.ok) {
    return std::nullopt;
  }

  // Structural limits of this runtime, not of the format.
  if (c.context_length == 0 || c.context_length > INT32_MAX ||
      !std::isfinite(c.rms_eps) || c.rms_eps <= 0 ||
      !std::isfinite(c.rope_theta) || c.rope_theta <= 0 || c.hidden_size == 0 ||
      c.full_attention_interval == 0 || c.num_heads == 0 ||
      c.num_kv_heads == 0 || c.num_heads % c.num_kv_heads != 0 ||
      c.head_dim == 0 || c.rotary_dim == 0 || c.rotary_dim % 2 != 0 ||
      c.rotary_dim > c.head_dim || c.ssm_conv_kernel == 0 ||
      c.ssm_head_dim == 0 || c.ssm_num_k_heads == 0 || c.ssm_num_v_heads == 0 ||
      c.ssm_num_v_heads % c.ssm_num_k_heads != 0 ||
      c.ssm_inner_size != c.SsmValueDim() || c.num_experts == 0 ||
      c.num_experts_used == 0 || c.num_experts_used > c.num_experts ||
      c.expert_ff == 0 || c.shared_expert_ff == 0) {
    r.Fail("qwen35moe metadata describes an unsupported shape");
    return std::nullopt;
  }
  // These dimensions select specialized attention, DeltaNet and expert
  // kernels. Reject other profiles before allocating or launching;
  // satisfying divisibility alone does not make a shape executable.
  if (c.num_layers != 40 || c.nextn_layers > 1 || c.hidden_size != 2048 ||
      c.full_attention_interval != 4 || c.num_heads != 16 ||
      c.num_kv_heads != 2 || c.head_dim != 256 || c.rotary_dim != 64 ||
      c.rope_sections != std::array<std::uint32_t, 4>{11, 11, 10, 0} ||
      c.ssm_conv_kernel != 4 || c.ssm_head_dim != 128 ||
      c.ssm_num_k_heads != 16 || c.ssm_num_v_heads != 32 ||
      c.num_experts != 256 || c.num_experts_used != 8 || c.expert_ff != 512 ||
      c.shared_expert_ff != 512) {
    r.Fail("unsupported Qwen3.6-35B-A3B kernel geometry");
    return std::nullopt;
  }
  return c;
}

}  // namespace gufo::models::qwen36_35b_a3b
