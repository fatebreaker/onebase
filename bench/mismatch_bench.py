"""MISMATCH-BENCH (descriptive; RL-KERNELS): which trainer change moves the trainer-engine mismatch the RL logs report.

Phase gen: vLLM's online FP8 engine (or --engine bf16; WEIGHT-ONLY-FP8) with the plugin, built as train/grpo_q4.py builds it (LOWBIT_FUSED=1, enable_lora rank 64,
CUDA graphs, at most 128 sequences, 0.45 of the GPU), serves one RL adapter (--adapter) and samples --group completions for
each of --prompts DAPO-Math prompts (temperature 1, non-thinking, at most 2,048 tokens), keeping the engine's log-probability
of every sampled token; written to --data.
Phase score: one trainer variant loads the base and the same adapter (PEFT, fp32 adapter weights, as the trainer holds them)
and computes, under bf16 autocast, the log-probabilities of the sampled tokens; prints the mean |dlog p| per token against the
engine's (the RL logs' "mismatch"), overall and by position bucket. --variant: bf16 (no matching), fp8w (matched_fp8
parts="w", as the RL runs), fp8both (parts="both": W8A8 forward), fused (W8A8 forward, adapter projections through our
kernels: the RL-L40S-FAST trainer), mixed (parts="w" for the projections without an adapter, the fused W8A8 path for the
adapter projections, whose engine side runs the same K1 quantization). flash-linear-attention or the PyTorch fallback is chosen by PYTHONPATH (reported).
    python bench/mismatch_bench.py --phase gen --model M --adapter A --data D
    python bench/mismatch_bench.py --phase score --model M --adapter A --data D --variant fused --out results/mismatch_bench.jsonl"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "train"))
sys.path.insert(0, str(LB / "bench"))
DL = Path("$SCRATCH/data/std_bench_dl")


def gen(a) -> None:
    import pandas as pd
    import std_bench
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    df = pd.read_parquet(DL.parent / "dapo_dl/en/train.parquet")
    data = [r.prompt + std_bench.SUFFIX for r in df.itertuples()]
    random.Random(1234).shuffle(data)
    kw = {k: v for k, v in (("linear_backend", a.linear_backend), ("kv_cache_dtype", a.kv_cache_dtype),
                            ("mamba_ssm_cache_dtype", a.mamba_ssm_cache_dtype)) if v != "auto"}
    if a.topk:
        kw["max_logprobs"] = max(20, a.topk)
    llm = LLM(model=a.model, enable_lora=True, max_lora_rank=64, max_loras=1, max_cpu_loras=2, max_model_len=4096,
              gpu_memory_utilization=a.gpu_util, enforce_eager=False, max_num_seqs=128, seed=0,
              quantization=None if a.engine == "bf16" else "fp8", **kw)
    print("engine options:", json.dumps({"engine": a.engine, **kw}), flush=True)
    sp = SamplingParams(n=a.group, temperature=1.0, top_p=1.0, max_tokens=2048, seed=7, logprobs=a.topk)
    msgs = [[{"role": "user", "content": q}] for q in data[:a.prompts]]
    outs = llm.chat(msgs, sp, lora_request=LoRARequest("policy", 1, a.adapter), use_tqdm=False,
                    chat_template_kwargs={"enable_thinking": False})
    rows = []
    for o in outs:
        for c in o.outputs:
            row = {"prompt": list(o.prompt_token_ids), "completion": list(c.token_ids),
                   "elp": [d[t].logprob for d, t in zip(c.logprobs, c.token_ids)]}
            if a.topk:                                   # the sampler's top-k at every position (rank <= k; the sampled
                hk = [sorted((x for x in d.items() if x[1].rank is not None and x[1].rank <= a.topk),
                             key=lambda x: x[1].rank) for d in c.logprobs]   # token is added when outside them)
                row["tk_ids"] = [[t for t, _ in h] for h in hk]
                row["tk_lp"] = [[v.logprob for _, v in h] for h in hk]
            rows.append(row)
    Path(a.data).write_text(json.dumps(rows))
    print(f"wrote {len(rows)} sequences, {sum(len(r['completion']) for r in rows)} completion tokens to {a.data}", flush=True)


def score(a) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from token_lp import token_logprobs
    fla = importlib.util.find_spec("fla") is not None
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0})
    if a.variant != "bf16":
        from mlsys.matched_fp8 import patch_model
        patch_model(model, parts="w" if a.variant in ("fp8w", "mixed", "w16fused") else "both", use_scaled_mm=True)
    if a.variant in ("fused", "w16fused"):
        from lowbit_lora import fp8_train
        fp8_train.fp8_backward(model, kind="w8a16" if a.variant == "w16fused" else "w8a8")
    model = PeftModel.from_pretrained(model, a.adapter, is_trainable=True)
    if a.variant in ("fused", "mixed", "w16fused"):
        from lowbit_lora import train_lora
        train_lora.install(model, kind="w16" if a.variant == "w16fused" else "fused")
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.eval()
    rows = json.loads(Path(a.data).read_text())
    met = {"mismatch": [], "kl": [], "tv": [], "head": []}    # per-token tensors, concatenated at the end
    with torch.no_grad():
        for r in rows:
            p, c = r["prompt"], r["completion"]
            ids = torch.tensor([p + c], device="cuda:0")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
            lp = token_logprobs(logits, ids)[0, len(p) - 1:len(p) + len(c) - 1].float().cpu()
            e = torch.tensor(r["elp"])
            met["mismatch"].append((lp - e).abs())
            met["kl"].append(e - lp)                     # log q - log p at y ~ q: an unbiased per-token estimate of KL(q || p)
            if "tk_ids" in r:                            # the sampler's top-k head H against the trainer's full distribution
                K = max(len(x) for x in r["tk_ids"])
                idx = torch.tensor([x + [x[0]] * (K - len(x)) for x in r["tk_ids"]], device="cuda:0")
                msk = torch.tensor([[1.0] * len(x) + [0.0] * (K - len(x)) for x in r["tk_ids"]])
                qlp = torch.tensor([x + [0.0] * (K - len(x)) for x in r["tk_lp"]])
                plp = []
                for t0 in range(0, len(c), 512):         # fp32 log-softmax in chunks of positions
                    z = logits[0, len(p) - 1 + t0:len(p) - 1 + min(t0 + 512, len(c))].float()
                    plp.append(torch.log_softmax(z, dim=-1).gather(1, idx[t0:t0 + 512]).cpu())
                plp = torch.cat(plp)
                q, ph = qlp.exp() * msk, plp.exp() * msk
                qt, pt = 1 - q.sum(1), 1 - ph.sum(1)
                met["tv"].append(0.5 * ((q - ph).abs().sum(1) + (qt - pt).abs()))    # TV with the tail as one bucket
                met["head"].append(q.sum(1))
            del logits
    pos = lambda xs: {f"{k * 512}-{k * 512 + 511}" if k < 3 else "1536+": float(torch.cat([x[k * 512:(k + 1) * 512] if k < 3  # noqa: E731
                      else x[1536:] for x in xs]).mean()) for k in range(4) if any(len(x) > k * 512 for x in xs)}
    allx = torch.cat(met["mismatch"])
    row = {"variant": a.variant, "fla": fla, "gpu": torch.cuda.get_device_name(0), "label": a.label, "seqs": len(rows),
           "tokens": int(allx.numel()), "mismatch": float(allx.mean()), "by_pos": pos(met["mismatch"]),
           "kl": float(torch.cat(met["kl"]).mean()), "by_pos_kl": pos(met["kl"])}
    if met["tv"]:
        row.update(tv=float(torch.cat(met["tv"]).mean()), by_pos_tv=pos(met["tv"]), head=float(torch.cat(met["head"]).mean()),
                   head_min=float(torch.cat(met["head"]).min()))
    print(json.dumps(row), flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(row) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("gen", "score"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--variant", default="fp8w", choices=("bf16", "fp8w", "fp8both", "fused", "mixed", "w16fused"))
    ap.add_argument("--label", default="")
    ap.add_argument("--gpu-util", type=float, default=0.45, help="gen: the engine's share of the GPU (bf16 on a 48 GB GPU needs more)")
    ap.add_argument("--linear-backend", default="auto", help="gen: vLLM's --linear-backend (marlin: weight-only FP8)")
    ap.add_argument("--kv-cache-dtype", default="auto", help="gen: vLLM's KV cache dtype (fp8: an FP8 KV cache)")
    ap.add_argument("--mamba-ssm-cache-dtype", default="auto", help="gen: the recurrent state's dtype (Qwen3.5's config: float32)")
    ap.add_argument("--topk", type=int, default=0, help="gen: keep the sampler's top-k log-probs at every position")
    ap.add_argument("--engine", default="fp8", choices=("fp8", "bf16"),
                    help="gen: vLLM's online FP8 (its kernel, W8A8 or weight-only Marlin, from the environment) or bf16")
    ap.add_argument("--out", default=str(LB / "results/mismatch_bench.jsonl"))
    a = ap.parse_args()
    gen(a) if a.phase == "gen" else score(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
