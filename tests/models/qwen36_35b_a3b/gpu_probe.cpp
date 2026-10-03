// Runs the ROCm engine over a prompt and reports the greedy continuation in
// reference_probe's format, so both can be compared against an independent
// implementation. --bench instead times prefill and decode.
//
//   gpu_probe --model MODEL.gguf [--prompt TEXT | --prompt-file PATH]
//             [--generate N | --follow ID,ID,...] [--top 5] [--mtp]
//             [--context N]
//   gpu_probe --model MODEL.gguf --bench --generate N --prompt-file PATH
//             [--prompt-tokens N] [--mtp] [--repeat R] [--show]
//   gpu_probe --model MODEL.gguf --streams N --prompt-dir DIR --generate N
//             [--mtp] [--context N]
//
// --streams serves every *.txt prompt in DIR with N concurrent streams, as
// the server schedules them: while any stream decodes, a pending prompt is
// fed in 512-token chunks between batched decode cycles.
//
// Without --prompt-tokens the prompt is used as is and decoding stops at a
// stop token, as a served request would. --follow evaluates the given
// continuation instead of the greedy one, so another implementation's
// distributions can be compared at identical prefixes.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/core/session_mode.hpp"
#include "src/models/qwen36_35b_a3b/engine.hpp"

namespace q = gufo::models::qwen36_35b_a3b;

namespace {

using Clock = std::chrono::steady_clock;

double Seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

void PrintTop(std::span<const float> logits, const q::Model& model,
              std::uint32_t top) {
  std::vector<std::uint32_t> order(logits.size());
  std::iota(order.begin(), order.end(), 0U);
  std::partial_sort(
      order.begin(), order.begin() + top, order.end(),
      [&](std::uint32_t a, std::uint32_t b) { return logits[a] > logits[b]; });
  const float max_logit = logits[order[0]];
  double denom = 0.0;
  for (float v : logits) {
    denom += std::exp(static_cast<double>(v - max_logit));
  }
  for (std::uint32_t i = 0; i < top; ++i) {
    const std::uint32_t id = order[i];
    const double p =
        std::exp(static_cast<double>(logits[id] - max_logit)) / denom;
    std::string text = model.TokenText(static_cast<std::int32_t>(id));
    std::replace(text.begin(), text.end(), '\n', ' ');
    std::printf("    %6u  %8.4f  %.8g  %s\n", id, logits[id], p, text.c_str());
  }
}

std::int32_t Argmax(std::span<const float> logits) {
  return static_cast<std::int32_t>(
      std::max_element(logits.begin(), logits.end()) - logits.begin());
}

/// One request of the --streams workload.
struct StreamRequest {
  std::string name;
  std::vector<std::int32_t> prompt;
  std::size_t fed{0};
  std::size_t produced{0};
  double arrive_s{0}, first_token_s{-1}, decode_start_s{-1}, done_s{-1};
  double prefill_s{0};
  std::uint64_t drafted0{0}, accepted0{0};  ///< session totals at arrival
  std::uint64_t drafted{0}, accepted{0};
};

int RunStreams(const std::string& model_path, const std::string& dir,
               int streams, int generate, bool mtp, std::uint32_t context) {
  constexpr std::size_t kChunk = 512;  // the server's decode-active budget
  std::vector<std::string> files;
  for (const auto& e : std::filesystem::directory_iterator(dir))
    if (e.path().extension() == ".txt")
      files.push_back(e.path().string());
  std::sort(files.begin(), files.end());
  std::string error;
  q::ModelOptions options;
  options.mtp = mtp;
  options.decode_concurrency = static_cast<std::uint32_t>(streams);
  options.max_context = context != 0 ? context : 50000;
  auto model = q::Model::Load(model_path, options, &error);
  if (!model) {
    std::fprintf(stderr, "load failed: %s\n", error.c_str());
    return 1;
  }
  std::deque<StreamRequest> queue;
  for (const auto& f : files) {
    std::ifstream in(f);
    std::stringstream ss;
    ss << in.rdbuf();
    StreamRequest r;
    r.name = std::filesystem::path(f).stem().string();
    r.prompt = model->Tokenize(ss.str());
    queue.push_back(std::move(r));
  }
  struct Slot {
    std::unique_ptr<q::Session> session;
    std::optional<StreamRequest> request;
    gufo::sampling::SamplerState sampler;
  };
  std::vector<Slot> slots(static_cast<std::size_t>(streams));
  for (auto& slot : slots) {
    slot.session =
        model->CreateSession(mtp ? gufo::core::SessionMode::kSpeculative
                                 : gufo::core::SessionMode::kAutoregressive,
                             options.max_context, &error);
    if (!slot.session) {
      std::fprintf(stderr, "session failed: %s\n", error.c_str());
      return 1;
    }
  }
  std::vector<StreamRequest> finished;
  const auto start = Clock::now();
  double decode_busy_s = 0, prefill_busy_s = 0;
  std::size_t decode_steps = 0, decode_rows = 0;
  while (true) {
    const double now = Seconds(start);
    for (auto& slot : slots) {
      if (!slot.request && !queue.empty()) {
        slot.request = std::move(queue.front());
        queue.pop_front();
        slot.request->arrive_s = now;
        slot.session->Reset();
        slot.session->ResetDraftPolicy();
        slot.sampler = {};
        slot.request->drafted0 = slot.session->Statistics().drafted;
        slot.request->accepted0 = slot.session->Statistics().accepted;
      }
    }
    bool any_decoding = false, any_active = false;
    for (auto& slot : slots) {
      if (!slot.request)
        continue;
      any_active = true;
      any_decoding |= slot.request->fed == slot.request->prompt.size();
    }
    if (!any_active)
      break;
    // One prefill step: the oldest pending prompt, bounded while others
    // decode.
    Slot* pending = nullptr;
    for (auto& slot : slots)
      if (slot.request && slot.request->fed < slot.request->prompt.size() &&
          (pending == nullptr ||
           slot.request->arrive_s < pending->request->arrive_s))
        pending = &slot;
    if (pending != nullptr) {
      auto& r = *pending->request;
      const std::size_t take = any_decoding
                                   ? std::min(kChunk, r.prompt.size() - r.fed)
                                   : r.prompt.size() - r.fed;
      const auto t0 = Clock::now();
      if (!pending->session->Sync(std::span(r.prompt).first(r.fed + take),
                                  &error)) {
        std::fprintf(stderr, "prefill failed: %s\n", error.c_str());
        return 1;
      }
      const double dt = Seconds(t0);
      r.prefill_s += dt;
      prefill_busy_s += dt;
      r.fed += take;
      if (r.fed == r.prompt.size()) {
        r.first_token_s = Seconds(start);
        r.decode_start_s = r.first_token_s;
      }
    }
    // One batched decode cycle for every decoding stream.
    std::vector<q::Session::DecodeRequest> batch;
    std::vector<q::Session::DecodeResult> results(slots.size());
    std::vector<q::Session::BatchOutcome> outcomes(slots.size());
    std::vector<Slot*> members;
    for (std::size_t i = 0; i < slots.size(); ++i) {
      auto& slot = slots[i];
      if (!slot.request || slot.request->fed < slot.request->prompt.size())
        continue;
      batch.push_back(
          {slot.session.get(),
           static_cast<std::size_t>(generate) - slot.request->produced,
           &slot.sampler, &results[i], true, &outcomes[i]});
      members.push_back(&slot);
    }
    if (!batch.empty()) {
      const auto t0 = Clock::now();
      if (!q::Session::DecodeBatch(batch, &error)) {
        std::fprintf(stderr, "decode batch failed: %s\n", error.c_str());
        return 1;
      }
      decode_busy_s += Seconds(t0);
      ++decode_steps;
      decode_rows += batch.size();
      const double t = Seconds(start);
      for (std::size_t i = 0; i < slots.size(); ++i) {
        auto& slot = slots[i];
        if (std::find(members.begin(), members.end(), &slot) == members.end())
          continue;
        auto& r = *slot.request;
        r.produced += results[i].tokens.size();
        if (results[i].stop ||
            r.produced >= static_cast<std::size_t>(generate)) {
          r.done_s = t;
          r.drafted = slot.session->Statistics().drafted - r.drafted0;
          r.accepted = slot.session->Statistics().accepted - r.accepted0;
          finished.push_back(std::move(r));
          slot.request.reset();
        }
      }
    }
  }
  const double wall = Seconds(start);
  std::size_t prompt_tokens = 0, out_tokens = 0;
  std::uint64_t drafted = 0, accepted = 0;
  double decode_speed = 0, ttft = 0;
  std::size_t speed_n = 0;
  std::sort(finished.begin(), finished.end(),
            [](const auto& a, const auto& b) { return a.name < b.name; });
  for (const auto& r : finished) {
    prompt_tokens += r.prompt.size();
    out_tokens += r.produced;
    const double decode_s = r.done_s - r.decode_start_s;
    const double speed = decode_s > 0 ? r.produced / decode_s : 0;
    // Nine-token answers measure one or two cycles; keep them out of the
    // mean decode speed.
    if (r.produced >= 32) {
      decode_speed += speed;
      ++speed_n;
    }
    ttft += r.first_token_s - r.arrive_s;
    drafted += r.drafted;
    accepted += r.accepted;
    std::printf(
        "  %-14s prompt %6zu out %5zu | wait+prefill %6.2f s | decode %6.1f "
        "tok/s | accepted %llu/%llu\n",
        r.name.c_str(), r.prompt.size(), r.produced,
        r.first_token_s - r.arrive_s, speed,
        static_cast<unsigned long long>(r.accepted),
        static_cast<unsigned long long>(r.drafted));
  }
  std::printf(
      "STREAMS %d: wall %.1f s | requests %zu | prompt %zu tok, out %zu tok\n"
      "  throughput: output %.1f tok/s, prompt %.1f tok/s (whole run)\n"
      "  per request: mean decode %.1f tok/s (outputs >= 32 tok), mean "
      "time to first token %.1f s\n"
      "  GPU split: prefill %.1f s, decode %.1f s (%zu batched cycles, mean "
      "%.2f streams) -> decode-phase output %.1f tok/s\n"
      "  MTP: accepted %llu of %llu drafts (%.0f%%)\n",
      streams, wall, finished.size(), prompt_tokens, out_tokens,
      out_tokens / wall, prompt_tokens / wall,
      speed_n ? decode_speed / speed_n : 0.0,
      finished.empty() ? 0.0 : ttft / finished.size(), prefill_busy_s,
      decode_busy_s, decode_steps,
      decode_steps ? static_cast<double>(decode_rows) / decode_steps : 0.0,
      decode_busy_s > 0 ? out_tokens / decode_busy_s : 0.0,
      static_cast<unsigned long long>(accepted),
      static_cast<unsigned long long>(drafted),
      drafted ? 100.0 * accepted / drafted : 0.0);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path;
  std::string prompt;
  std::string prompt_file;
  int generate = 0;
  std::uint32_t top = 5;
  std::uint32_t context = 0;
  std::uint32_t prompt_tokens = 0;
  int repeat = 1;
  bool mtp = false;
  bool bench = false;
  bool show = false;
  int streams = 0;
  std::string prompt_dir;
  std::vector<std::int32_t> follow;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> std::string {
      return i + 1 < argc ? argv[++i] : std::string();
    };
    if (arg == "--model") {
      model_path = next();
    } else if (arg == "--prompt") {
      prompt = next();
    } else if (arg == "--prompt-file") {
      prompt_file = next();
    } else if (arg == "--generate") {
      generate = std::atoi(next().c_str());
    } else if (arg == "--top") {
      top = static_cast<std::uint32_t>(std::atoi(next().c_str()));
    } else if (arg == "--context") {
      context = static_cast<std::uint32_t>(std::atoi(next().c_str()));
    } else if (arg == "--prompt-tokens") {
      prompt_tokens = static_cast<std::uint32_t>(std::atoi(next().c_str()));
    } else if (arg == "--repeat") {
      repeat = std::atoi(next().c_str());
    } else if (arg == "--mtp") {
      mtp = true;
    } else if (arg == "--bench") {
      bench = true;
    } else if (arg == "--show") {
      show = true;
    } else if (arg == "--streams") {
      streams = std::atoi(next().c_str());
    } else if (arg == "--prompt-dir") {
      prompt_dir = next();
    } else if (arg == "--follow") {
      std::stringstream ids(next());
      for (std::string id; std::getline(ids, id, ',');)
        follow.push_back(static_cast<std::int32_t>(std::atoi(id.c_str())));
      generate = static_cast<int>(follow.size());
    } else {
      std::fprintf(stderr, "unknown argument %s\n", arg.c_str());
      return 2;
    }
  }
  if (model_path.empty()) {
    std::fprintf(stderr, "--model is required\n");
    return 2;
  }
  if (!prompt_file.empty()) {
    std::ifstream in(prompt_file);
    std::stringstream ss;
    ss << in.rdbuf();
    prompt = ss.str();
  }

  if (streams > 0)
    return RunStreams(model_path, prompt_dir, streams, generate, mtp, context);

  std::string error;
  q::ModelOptions options;
  options.mtp = mtp;
  options.max_context =
      context != 0 ? context
                   : std::max<std::uint32_t>(
                         4096, prompt_tokens +
                                   static_cast<std::uint32_t>(generate + 64));
  const auto load_start = Clock::now();
  auto model = q::Model::Load(model_path, options, &error);
  if (!model) {
    std::fprintf(stderr, "load failed: %s\n", error.c_str());
    return 1;
  }
  std::printf("loaded in %.2f s, resident %.2f GiB, mtp %d\n",
              Seconds(load_start),
              static_cast<double>(model->ResidentBytes()) / (1u << 30),
              model->HasMtp() ? 1 : 0);
  auto session =
      model->CreateSession(mtp ? gufo::core::SessionMode::kSpeculative
                               : gufo::core::SessionMode::kAutoregressive,
                           options.max_context, &error);
  if (!session) {
    std::fprintf(stderr, "session failed: %s\n", error.c_str());
    return 1;
  }

  std::vector<std::int32_t> tokens = model->Tokenize(prompt);
  if (bench) {
    // Repeat the text to the requested length, as a fixed realistic prompt.
    if (tokens.empty()) {
      std::fprintf(stderr, "--bench needs a prompt\n");
      return 2;
    }
    if (prompt_tokens != 0) {
      std::vector<std::int32_t> base = tokens;
      tokens.clear();
      while (tokens.size() < prompt_tokens) {
        tokens.insert(tokens.end(), base.begin(), base.end());
      }
      tokens.resize(prompt_tokens);
    }
    const bool stop_at_eos = prompt_tokens == 0;
    for (int r = 0; r < repeat; ++r) {
      session->Reset();
      const auto prefill_start = Clock::now();
      if (!session->Sync(tokens, &error)) {
        std::fprintf(stderr, "prefill failed: %s\n", error.c_str());
        return 1;
      }
      const double prefill_s = Seconds(prefill_start);
      gufo::sampling::SamplerState sampler;
      std::size_t produced = 0;
      std::vector<std::int32_t> output;
      const auto decode_start = Clock::now();
      while (produced < static_cast<std::size_t>(generate)) {
        q::Session::DecodeResult result;
        if (!session->DecodeStep(generate - produced, sampler, &result, &error,
                                 stop_at_eos)) {
          std::fprintf(stderr, "decode failed: %s\n", error.c_str());
          return 1;
        }
        produced += result.tokens.size();
        output.insert(output.end(), result.tokens.begin(), result.tokens.end());
        if (result.stop) {
          break;
        }
      }
      const double decode_s = Seconds(decode_start);
      const auto& stats = session->Statistics();
      std::printf(
          "run %d: prefill %u tok in %.3f s = %.1f tok/s | decode %zu tok in "
          "%.3f s = %.1f tok/s | mtp drafted %llu accepted %llu\n",
          r, static_cast<unsigned>(tokens.size()), prefill_s,
          tokens.size() / prefill_s, produced, decode_s, produced / decode_s,
          static_cast<unsigned long long>(stats.drafted),
          static_cast<unsigned long long>(stats.accepted));
      if (show) {
        std::printf("output: %s\n", model->Decode(output).c_str());
      }
    }
    return 0;
  }

  std::printf("prompt tokens (%zu):", tokens.size());
  for (auto t : tokens) {
    std::printf(" %d", t);
  }
  std::printf("\n");
  if (!session->Sync(tokens, &error)) {
    std::fprintf(stderr, "prefill failed: %s\n", error.c_str());
    return 1;
  }
  std::vector<std::int32_t> generated;
  std::int32_t token = tokens.back();
  for (int step = 0; step <= generate; ++step) {
    const auto logits = session->Logits();
    std::printf("pos %zu token %d '%s'\n", tokens.size() + generated.size() - 1,
                token, model->TokenText(token).c_str());
    PrintTop(logits, *model, top);
    if (step == generate) {
      break;
    }
    token = follow.empty() ? Argmax(logits) : follow[step];
    generated.push_back(token);
    if (!session->Evaluate(token, &error)) {
      std::fprintf(stderr, "evaluate failed: %s\n", error.c_str());
      return 1;
    }
  }
  std::printf("generated: %s\n", model->Decode(generated).c_str());
  return 0;
}
