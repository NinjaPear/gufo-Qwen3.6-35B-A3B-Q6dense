#!/usr/bin/env python3
"""Compare Gufo's Qwen3.6 next-token distributions with llama.cpp.

llama-server (same GGUF, already running) produces a greedy continuation and
its top-K probabilities per step. gpu_probe then evaluates the same tokens
(teacher forcing) and prints its own top-K, so both distributions are taken
at identical prefixes. Reported per position: top-1 agreement, total
variation and KL(llama || Gufo) over the union of both top-K sets plus one
bucket for the remaining mass.

  llama_parity.py --probe build/gpu-test/.../qwen36_35b_a3b_gpu_probe \
      --model MODEL.gguf --url http://127.0.0.1:18080 [--tokens 64] [--top 20]
"""

import argparse
import json
import math
import re
import subprocess
import sys
import urllib.request

PROMPTS = {
    "prose": "<|im_start|>user\nWrite a short paragraph about how tides work."
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "code": "<|im_start|>user\nWrite a Python function that merges two sorted "
    "lists without using sort().<|im_end|>\n<|im_start|>assistant\n"
    "<think>\n\n</think>\n\n",
    "json": "<|im_start|>user\nExtract the products as JSON with name and "
    "price fields. Page: \"Acme Rocket Boots - $129. Acme Jet Gloves - $49. "
    "Free shipping over $100.\"<|im_end|>\n<|im_start|>assistant\n"
    "<think>\n\n</think>\n\n",
    "math": "<|im_start|>user\nA train travels 180 km in 2.5 hours. What is "
    "its average speed? Explain briefly.<|im_end|>\n<|im_start|>assistant\n"
    "<think>\n\n</think>\n\n",
    "multilingual": "<|im_start|>user\nTranslate into French and German: "
    "\"The library opens at nine.\"<|im_end|>\n<|im_start|>assistant\n"
    "<think>\n\n</think>\n\n",
    "raw": "The history of the printing press begins",
    "list": "<|im_start|>user\nList five uses of copper, one per line."
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "thinking": "<|im_start|>user\nIs 221 a prime number?<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n",
}


def post(url, path, body):
    request = urllib.request.Request(
        url + path, json.dumps(body).encode(),
        {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(request, timeout=600))


def llama(url, prompt, tokens, top):
    ids = post(url, "/tokenize", {"content": prompt, "add_special": False,
                                  "parse_special": True})["tokens"]
    reply = post(url, "/completion", {
        "prompt": prompt, "n_predict": tokens, "temperature": 0, "top_k": 1,
        "n_probs": top, "cache_prompt": False, "return_tokens": True,
        "post_sampling_probs": False})
    steps = [{p["id"]: math.exp(p["logprob"]) for p in s["top_logprobs"]}
             for s in reply["completion_probabilities"]]
    return ids, reply["tokens"], steps


def gufo(probe, model, prompt, follow, top):
    out = subprocess.run(
        [probe, "--model", model, "--prompt", prompt, "--top", str(top),
         "--follow", ",".join(map(str, follow))],
        capture_output=True, check=True).stdout.decode(errors="replace")
    header = re.search(r"^prompt tokens \(\d+\):(.*)$", out, re.M)
    if header is None:
        raise RuntimeError("gpu_probe printed no prompt tokens")
    ids = [int(t) for t in header.group(1).split()]
    steps = []
    for block in re.split(r"^pos \d+ token -?\d+ .*$", out, flags=re.M)[1:]:
        steps.append({int(m.group(1)): float(m.group(3)) for m in re.finditer(
            r"^\s+(\d+)\s+(-?[\d.]+)\s+([\d.eE+-]+)", block, re.M)})
    return ids, steps[:len(follow)]


def compare(p, q):
    keys = set(p) | set(q)
    rest_p = max(0.0, 1.0 - sum(p.get(k, 0.0) for k in keys))
    rest_q = max(0.0, 1.0 - sum(q.get(k, 0.0) for k in keys))
    pairs = [(p.get(k, 0.0), q.get(k, 0.0)) for k in keys] + [(rest_p, rest_q)]
    tv = 0.5 * sum(abs(a - b) for a, b in pairs)
    kl = sum(a * math.log(a / max(b, 1e-12)) for a, b in pairs if a > 0)
    return tv, kl, max(p, key=p.get) == max(q, key=q.get)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    rows, total = [], {"positions": 0, "top1": 0, "tv": [], "kl": []}
    for name, prompt in PROMPTS.items():
        llama_ids, continuation, reference = llama(
            args.url, prompt, args.tokens, args.top)
        gufo_ids, candidate = gufo(args.probe, args.model, prompt,
                                   continuation, args.top)
        tvs, kls, top1 = [], [], 0
        for p, q in zip(reference, candidate):
            tv, kl, same = compare(p, q)
            tvs.append(tv)
            kls.append(kl)
            top1 += same
        rows.append({"prompt": name, "tokenization_match": llama_ids == gufo_ids,
                     "positions": len(tvs), "top1_agreement": top1,
                     "mean_tv": sum(tvs) / len(tvs), "max_tv": max(tvs),
                     "mean_kl": sum(kls) / len(kls), "max_kl": max(kls)})
        total["positions"] += len(tvs)
        total["top1"] += top1
        total["tv"] += tvs
        total["kl"] += kls
        print(json.dumps(rows[-1]), file=sys.stderr)
    tv, kl = sorted(total["tv"]), sorted(total["kl"])
    summary = {
        "positions": total["positions"],
        "top1_agreement": total["top1"],
        "tokenization_match": all(r["tokenization_match"] for r in rows),
        "mean_tv": sum(tv) / len(tv), "p99_tv": tv[int(0.99 * (len(tv) - 1))],
        "max_tv": tv[-1], "mean_kl": sum(kl) / len(kl),
        "p99_kl": kl[int(0.99 * (len(kl) - 1))], "max_kl": kl[-1],
        "top_k": args.top, "tokens_per_prompt": args.tokens, "prompts": rows}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
