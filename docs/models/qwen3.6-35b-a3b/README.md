# Qwen3.6-35B-A3B

Hybrid Gated DeltaNet / gated-attention mixture-of-experts text model on
gfx1151: 40 layers (30 DeltaNet, 10 full attention), 256 experts with 8 routed
plus one shared, about 3B active parameters, and a native one-layer MTP block.
Supported target: the **Q6dense** GGUF — trunk dense weights Q6_K, routed
experts as in Unsloth UD-Q4_K_XL, token embeddings and the MTP block's dense
weights Q8_0 — and other `qwen35moe` GGUFs with Q8_0, Q4_K, Q5_K or Q6_K dense
weights. Text only.

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md)

## Load and run

```sh
nix develop -c hf download ligamentexceed/Qwen3.6-35B-A3B-Q6dense-GGUF \
  --revision cd429bbcdc0a2843bdc509c804088013ed7212a4 \
  --include "Qwen3.6-35B-A3B-Q6dense.gguf" \
  --local-dir models/qwen3.6-35b-a3b
nix build
MODEL=models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-Q6dense.gguf
./result/bin/gufo chat --model "$MODEL" --speculative mtp
./result/bin/gufo serve llm --model "$MODEL" --speculative mtp \
  --sessions 2 --context 32768
```

MTP uses the draft block stored in the same GGUF; there is no sidecar and
`--mtp-model` is rejected. Omit `--speculative` for AR: AR sessions allocate no
predictor state. Draft length adapts to acceptance and is capped by
`--draft-tokens` (1–7); drafts after the first are verified only while the
draft probability is at least 0.5. Greedy MTP output equals AR token for token.

The official template defaults to thinking on and drops earlier reasoning
unless `preserve_thinking` is set; it has no reasoning-effort control. Sampling
defaults follow the model card (thinking: temperature 1.0, top-p 0.95; off:
0.7, 0.8; top-k 20 and presence penalty 1.5 in both). Native context is 262144;
YaRN extension is unsupported.

Memory: weights take about 22.5 GiB. Each full-attention layer keeps an F16
K/V cache for prefill and an 8-bit copy for decode, about 31 KiB per token
over the ten trunk layers (34 KiB with the MTP block); recurrent state is fixed
per session. One 256K-token MTP session needs about 33 GiB; admission reserves
the configured capacity before creating sessions.

## Producing Q6dense

Quantize the BF16 GGUF of `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` (revision
`5bc3e238d916f48a861bac2f8a1990a0e9b7e98d`, both `BF16/` shards and
`imatrix_unsloth.gguf_file`) with llama.cpp `b11069` (`nix build
.#llama-cpp-reference`). `q6dense.types`, published with the weights, sets
every tensor's type explicitly:

```sh
llama-quantize --imatrix imatrix_unsloth.gguf_file \
  --tensor-type-file q6dense.types \
  Qwen3.6-35B-A3B-BF16-00001-of-00002.gguf Qwen3.6-35B-A3B-Q6dense.gguf \
  Q4_K_M 16
```

Lower dense precision (Q5_K) gave no speed gain; see
[Experiments](EXPERIMENTS.md).

## Tools and artifacts

Model tests are in `tests/models/qwen36_35b_a3b`: CPU metadata checks, HIP
operator tests against FP64 references, and the full-model `session_test` /
`snapshot_test` (MTP versus AR, batching, rollback, prefill chunking, cache
replay). `reference_probe` runs the CPU oracle; `gpu_probe` times single and
multi-stream decoding. Retained result summaries belong in `artifacts/`;
generated traces stay in the ignored top-level `artifacts/` tree.
