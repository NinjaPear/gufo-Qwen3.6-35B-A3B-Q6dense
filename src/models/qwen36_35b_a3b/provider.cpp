#include <gufo/ext/generated/qwen36_35b_a3b_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

#include "src/models/qwen36_35b_a3b/probe.hpp"
#include "src/models/qwen36_35b_a3b/runner.hpp"

namespace gufo::models::qwen36_35b_a3b {

#if defined(ENGINE_ENABLE_HIP)
int RunQwen36Benchmark(const ext::TextBenchOptions& opt,
                       const std::shared_ptr<const core::GgufReader>& reader,
                       std::chrono::steady_clock::time_point model_load_start);

int RunQwen36Prompt(const ext::TextPromptOptions& opt,
                    const core::GgufReader& reader,
                    std::chrono::steady_clock::time_point model_load_start);

int RunQwen36Chat(const ext::TextPromptOptions& opt,
                  const core::GgufReader& reader,
                  std::chrono::steady_clock::time_point model_load_start);
#endif

}  // namespace gufo::models::qwen36_35b_a3b

void qwen36_35b_a3b_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(gufo::ext::generated::extension_qwen36_35b_a3b::
                           kProvider_qwen36_35b_a3b);
#if defined(ENGINE_ENABLE_HIP)
  registry.AddTextProvider({
      .id = "qwen36-35b-a3b",
      .probe = &gufo::models::qwen36_35b_a3b::ProbeQwen36,
      .probe_gguf = &gufo::models::qwen36_35b_a3b::ProbeQwen36Gguf,
      .create = &gufo::models::qwen36_35b_a3b::CreateQwen36TextRunner,
      .bench_driver = &gufo::models::qwen36_35b_a3b::RunQwen36Benchmark,
      .prompt_driver = &gufo::models::qwen36_35b_a3b::RunQwen36Prompt,
      .chat_driver = &gufo::models::qwen36_35b_a3b::RunQwen36Chat,
  });
#endif
}
