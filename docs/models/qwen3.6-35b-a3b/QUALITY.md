# Qwen3.6-35B-A3B quality

**Gufo matches llama.cpp on the same GGUF at 443 of 445 positions** (top-1),
mean KL 0.0020 and mean total variation 0.0064 over eight prompts, with
identical tokenization. Q6dense target with its native MTP block;
[identities](artifacts/model-identities.json). These checks compare two
implementations of one quantized file; they do not qualify the quantization
against the original BF16 model. Measured October 3, 2026.

| Check | Result |
| --- | --- |
| Upstream agreement, llama.cpp `b11069` | 8 prompts × up to 64 teacher-forced positions (prose, code, JSON extraction, math, translation, raw text, list, thinking). Top-1 443/445; KL(llama.cpp ‖ Gufo) mean 0.0020, p99 0.055, max 0.085; TV mean 0.0064, max 0.18. KL uses the union of both top-20 sets plus one residual bucket. [Evidence](artifacts/llama-parity.json). |
| Greedy MTP versus AR | Frontier logits, tokens and RNG are bit-identical at depths 16 and 4096 for greedy and three sampled configurations, and after a fresh model load with varied draft budgets |
| Batched sessions, C2/C4/C6/C8 | Logits, tokens, draft and acceptance counts, RNG, sampling history and full snapshot bytes match isolated execution, including ragged chains mixed with one-token decoding |
| Prefill chunking | Logits and sampled MTP replay are bit-identical across chunk boundaries 94–135 of 136 tokens, 1025 of 2048 and 2048 of 4096 |
| Seeded MTP cache rebuilding | Two seeds × 200 tokens replay exactly after different prefill splits and snapshot restore |
| Serving | 25 sampling strategies replay exactly through the HTTP backend, serially and in concurrent pairs, with exact AR/greedy prefixes for 1–3-token budgets; cancellation and invalid peers leave other sessions exact |
| Operators | FP64 references for prefill and 8-bit-KV decode attention (decode rows identical at widths 1–8, prefill identical across chunkings), DeltaNet, dense Q4_K/Q5_K/Q6_K decode projections (identical at widths 1–8), routed experts, router/alpha-beta projections, MTP hidden normalization and draft probability |

Decode attention reads an 8-bit copy of the K/V cache (one F16 scale per 32
values); prefill keeps F16 K/V and reads Q8_0 copies of the Q6_K matrices.
Both are covered by the llama.cpp comparison above. Batched prefill and
one-token decoding of the same prefix therefore differ slightly: with a
repeated synthetic prompt, `gufo bench --validate-prefill` reports the same
top-1 token with cosine 0.93–0.997 between the two logit vectors, below the
gate tuned for Flash-Next.

Sampled MTP can consume different RNG draws from AR. Seeded replay requires
the same build, request budget, capacity and sampling configuration.

## Task quality

Crawler extraction (company products and bios), thinking off, scored 0–10 by
`qwen/qwen3.8-max-0902` at temperature 0. Prompts and pages are private and
are not retained here.

| Comparison | Products | Bios |
| --- | ---: | ---: |
| Q6dense versus Unsloth UD-Q4_K_XL, same pages | 7.25 / 7.25 | 7.62 / 7.62 |
| Qwen3.6 Q6dense (Gufo) versus Qwen3.8-Flash-Next (OpenRouter), 6 + 6 new pages | 7.83 / 8.33 | 7.67 / 8.00 |

## Reproduce

Tests live in [`tests/models/qwen36_35b_a3b`](../../../tests/models/qwen36_35b_a3b).
`session_test` accepts `--batch-only`, `--prefill-only`, `--sampling-only` or
`--cache-only`.

```sh
nix develop -c cmake --build --preset gpu-test \
  --target qwen36_35b_a3b_tests qwen36_35b_a3b_model_tests qwen36_35b_a3b_gpu_probe
nix develop -c ctest --preset gpu-full -R '^qwen36_35b_a3b\.'
build/gpu-test/tests/models/qwen36_35b_a3b/qwen36_35b_a3b_session_test --model "$MODEL"
build/gpu-test/tests/models/qwen36_35b_a3b/qwen36_35b_a3b_snapshot_test --model "$MODEL"

# llama.cpp agreement, against a running llama-server on the same file
python3 tests/models/qwen36_35b_a3b/llama_parity.py \
  --probe build/gpu-test/tests/models/qwen36_35b_a3b/qwen36_35b_a3b_gpu_probe \
  --model "$MODEL" --url http://127.0.0.1:8080
```

## Benchmark method

All tables measured October 3, 2026 with the production `nix build` binary
and llama.cpp `b11069` on the same file; one warmed sample per point, greedy,
thinking off. The llama.cpp AR 12,288 row and the Gufo MTP 12,288 rows were
measured again after interference from the host's other GPU service.
Single-user uses pp2048/tg128 over a cached prefix; MTP pp is the maximum
across mixed/repetitive workloads. C1/2/4/6/8 use the same d0 prompts; every
session prefills before measured tg128. Gufo multi-user MTP completions match
its C1 AR references. Loading: cold model file (evicted with
`POSIX_FADV_DONTNEED`, zero resident pages verified), C1/MTP/capacity 262144.
Memory: C1/AR/capacity 133121, peak global HIP allocation including idle
memory. Commands and identities are in [artifacts](artifacts/bench.json) and
the [benchmark workflow](../../../.agents/skills/benchmark-model/SKILL.md).
