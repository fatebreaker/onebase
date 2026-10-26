"""MATCHED-SPEED-HOPPER parity (docs/REGISTRY.md): greedy continuations of N DAPO-Math prompts with one random rank-64 adapter on
the matched engine, with its current W8A16 GEMM (--engine w16: lowbit_lora/w16_engine.py) or the swap-AB GEMM with the swept
tiles (--engine w16swap: lowbit_lora/w16_engine_swap.py), saved as token ids; --compare A.json B.json reports how many
continuations are identical and the mean length of the common prefix. The two GEMMs differ only in the order of fp32
additions inside an MMA, so the continuations should agree except where a near-tie flips a greedy token.
    VLLM_ENABLE_V1_MULTIPROCESSING=0 LOWBIT_FUSED=0 python bench/w16s_parity.py --engine w16swap --model M --adapter A --out o.json
    python bench/w16s_parity.py --compare a.json b.json"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def compare(fa: str, fb: str) -> dict:
    a, b = json.loads(Path(fa).read_text()), json.loads(Path(fb).read_text())
    same, pref = 0, []
    for x, y in zip(a["tokens"], b["tokens"]):
        same += x == y
        k = 0
        while k < min(len(x), len(y)) and x[k] == y[k]:
            k += 1
        pref.append(k)
    return {"n": len(pref), "identical": same, "mean_common_prefix": sum(pref) / len(pref), "max_tokens": a["max_tokens"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=("w16", "w16swap"))
    ap.add_argument("--model")
    ap.add_argument("--adapter")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--gpu-util", type=float, default=0.35)
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        print(json.dumps(compare(*a.compare)))
        return 0
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    if a.engine == "w16":
        from lowbit_lora import w16_engine as E
    else:
        from lowbit_lora import w16_engine_swap as E
        n = E.load_cfg(os.environ.get("LOWBIT_SWAP_CFG", str(Path(__file__).resolve().parents[1] / "results/w16_swap.jsonl")),
                       job=os.environ.get("LOWBIT_SWAP_JOB") or None)
        print(f"[lowbit] swap tiles loaded: {n}", flush=True)
    E.install()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from rl_rollout_bench3 import dapo_prompts
    tok = AutoTokenizer.from_pretrained(a.model)
    ps = dapo_prompts(tok, a.n)
    llm = LLM(model=a.model, max_model_len=4096, max_num_seqs=64, gpu_memory_utilization=a.gpu_util, seed=0, quantization="fp8",
              enable_lora=True, max_lora_rank=64, max_loras=2)
    outs = llm.generate([{"prompt_token_ids": p} for p in ps], SamplingParams(temperature=0.0, max_tokens=a.max_tokens),
                        lora_request=LoRARequest("pol", 1, a.adapter), use_tqdm=False)
    Path(a.out).write_text(json.dumps({"engine": a.engine, "max_tokens": a.max_tokens, "stats": E.stats,
                                       "tokens": [list(o.outputs[0].token_ids) for o in outs]}))
    print(f"[lowbit] {a.engine} layers: {E.stats}; wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
