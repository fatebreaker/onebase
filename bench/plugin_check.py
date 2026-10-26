"""End-to-end check of the vLLM plugin (lowbit_lora/plugin.py): the same engine with and without LOWBIT_FUSED.

    python bench/plugin_check.py --model <dir> --adapter <dir> --out results/plugin_check/<tag>.json [--eager]
Run once with LOWBIT_FUSED=0 and once with LOWBIT_FUSED=1 (one process each); `--compare a.json b.json` reports:
  greedy agreement on 8 prompts x {adapter, no adapter, mixed batch}: share of identical completions and the first
  divergent position; log-prob fidelity on the same sequences: mean |delta log p| of the realised tokens and top-5 KL;
  throughput of the generation pass (descriptive). A second stock run gives the run-to-run floor.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

PROMPTS = [
    "Explain in two sentences why the sky is blue.",
    "A patient weighs 70 kg and is 175 cm tall. Compute the body mass index and say what category it falls in.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "What is the creatinine clearance formula of Cockcroft-Gault? Give the formula and the units.",
    "List three differences between a list and a tuple in Python.",
    "Translate to French: The meeting has been moved to Thursday afternoon.",
    "A train travels 300 km in 2.5 hours. What is its average speed in m/s?",
    "Summarize the main idea of the central limit theorem for a student.",
]


def run(a) -> dict:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(model=a.model, quantization=None if a.quantization in ("none", "auto") else a.quantization, enable_lora=True,
              max_lora_rank=a.rank, max_loras=2,
              max_model_len=4096, gpu_memory_utilization=0.85, seed=0, enforce_eager=a.eager,
              enable_prefix_caching=False)
    tok = llm.get_tokenizer()
    msgs = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            for p in PROMPTS]
    lora = LoRARequest("rl", 1, a.adapter)
    sp = SamplingParams(max_tokens=a.max_tokens, temperature=0.0)
    res = {"fused": os.environ.get("LOWBIT_FUSED", "0"), "eager": a.eager, "quantization": a.quantization,
           "gen": {}, "lp": {}}
    for kind, reqs in (("adapter", [lora] * len(msgs)), ("base", None),
                       ("mixed", [lora if i % 2 else None for i in range(len(msgs))])):
        t = time.perf_counter()
        outs = llm.generate(msgs, sp, lora_request=reqs, use_tqdm=False)
        dt = time.perf_counter() - t
        res["gen"][kind] = {"texts": [o.outputs[0].text for o in outs],
                            "ids": [list(o.outputs[0].token_ids) for o in outs],
                            "tok_s": sum(len(o.outputs[0].token_ids) for o in outs) / dt}
    if a.ref_seqs:                       # score the SAME token sequences in every run (a reference run's)
        seq_ids = json.loads(Path(a.ref_seqs).read_text())["seqs"]
    else:
        seq_ids = [list(o.prompt_token_ids) + ids for o, ids in zip(outs, res["gen"]["adapter"]["ids"])]
    res["seqs"] = seq_ids
    seqs = [{"prompt_token_ids": ids} for ids in seq_ids]
    lp_sp = SamplingParams(max_tokens=1, prompt_logprobs=5, temperature=1.0)
    for kind, req in (("adapter", lora), ("base", None)):
        outs = llm.generate(seqs, lp_sp, lora_request=req, use_tqdm=False)
        res["lp"][kind] = [[None if s is None else {int(k): v.logprob for k, v in s.items()} for s in o.prompt_logprobs]
                           for o in outs]
    return res


def compare(f1: str, f2: str) -> None:
    a, b = json.loads(Path(f1).read_text()), json.loads(Path(f2).read_text())
    print(f"{f1} (fused={a['fused']}, {a.get('quantization', 'fp8')}) vs {f2} (fused={b['fused']}, "
          f"{b.get('quantization', 'fp8')})")
    for kind in ("adapter", "base", "mixed"):
        ia, ib = a["gen"][kind]["ids"], b["gen"][kind]["ids"]
        same = sum(x == y for x, y in zip(ia, ib))
        div = [next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y))) for x, y in zip(ia, ib)]
        print(f"  greedy {kind:7s}: identical {same}/{len(ia)}; first divergence {div}; "
              f"tok/s {a['gen'][kind]['tok_s']:.0f} vs {b['gen'][kind]['tok_s']:.0f}")
    if a.get("seqs") != b.get("seqs"):
        print("  (log-prob comparison skipped: the two runs scored different sequences)")
        return
    for kind in ("adapter", "base"):
        d, kl, n = 0.0, 0.0, 0
        for seq, sa, sb in zip(a["seqs"], a["lp"][kind], b["lp"][kind]):
            for i, (pa, pb) in enumerate(zip(sa, sb)):
                if not pa or not pb:
                    continue
                t = str(seq[i]) if str(seq[i]) in pa else seq[i]        # the realised token (json keys are str)
                d += abs(pa.get(t, -30) - pb.get(t, -30))
                kl += sum(math.exp(pa[k]) * (pa[k] - pb.get(k, -30)) for k in pa)
                n += 1
        print(f"  logprobs {kind:7s}: mean |dlogp| of the realised token {d / max(n, 1):.4f}; "
              f"top-5 KL {kl / max(n, 1):.5f} over {n} positions")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--adapter")
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--ref-seqs", default="", help="a previous run's JSON: score its sequences")
    ap.add_argument("--quantization", default="fp8", help="fp8 (default), none (the bf16 model, the accuracy anchor), "
                                                         "or auto (from the checkpoint, e.g. a ModelOpt NVFP4 one)")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare)
        return 0
    res = run(a)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res))
    print("wrote", a.out, {k: round(v["tok_s"]) for k, v in res["gen"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
