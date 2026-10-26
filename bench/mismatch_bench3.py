"""MISMATCH-DECOMP-3 (the v5d mock review, R2 M3/M4): bench/mismatch_bench2.py's two phases unchanged, plus each prompt's sums
in the score row ("per_prompt": {sha1 of the prompt's token ids: [kl sum, |dlog p| sum, tokens]}), so ratios between engines
can be bootstrapped over prompts, paired (bench/score_decomp3.py). Rows -> results/mismatch_bench3.jsonl.
    python bench/mismatch_bench3.py --phase score --model M --adapter A --data D --variant w16fused --label L"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "train"))
sys.path.insert(0, str(LB / "bench"))
import mismatch_bench as MB  # noqa: E402


def boot(per: list[tuple[float, int]], n: int = 2000, seed: int = 0) -> list[float]:
    """95% interval of sum(s) / sum(k) over resampled (sum, count) units."""
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        pick = [per[rng.randrange(len(per))] for _ in per]
        vals.append(sum(s for s, _ in pick) / max(sum(k for _, k in pick), 1))
    vals.sort()
    return [vals[int(0.025 * n)], vals[int(0.975 * n) - 1]]


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
    groups: dict[tuple, dict] = {}                       # prompt token ids -> sums
    with torch.no_grad():
        for r in rows:
            p, c = r["prompt"], r["completion"]
            if not c:
                continue
            ids = torch.tensor([p + c], device="cuda:0")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
            lp = token_logprobs(logits, ids)[0, len(p) - 1:len(p) + len(c) - 1].float().cpu()
            e = torch.tensor(r["elp"])
            g = groups.setdefault(tuple(p), {"kl": 0.0, "mm": 0.0, "tv": 0.0, "n": 0, "ntv": 0})
            g["kl"] += float((e - lp).sum())
            g["mm"] += float((lp - e).abs().sum())
            g["n"] += len(c)
            if "tk_ids" in r:
                K = max(len(x) for x in r["tk_ids"])
                idx = torch.tensor([x + [x[0]] * (K - len(x)) for x in r["tk_ids"]], device="cuda:0")
                msk = torch.tensor([[1.0] * len(x) + [0.0] * (K - len(x)) for x in r["tk_ids"]])
                qlp = torch.tensor([x + [0.0] * (K - len(x)) for x in r["tk_lp"]])
                plp = []
                for t0 in range(0, len(c), 512):
                    z = logits[0, len(p) - 1 + t0:len(p) - 1 + min(t0 + 512, len(c))].float()
                    plp.append(torch.log_softmax(z, dim=-1).gather(1, idx[t0:t0 + 512]).cpu())
                plp = torch.cat(plp)
                q, ph = qlp.exp() * msk, plp.exp() * msk
                qt, pt = 1 - q.sum(1), 1 - ph.sum(1)
                g["tv"] += float((0.5 * ((q - ph).abs().sum(1) + (qt - pt).abs())).sum())
                g["ntv"] += len(c)
            del logits
    G = list(groups.values())
    tot = lambda k, n="n": sum(g[k] for g in G) / max(sum(g[n] for g in G), 1)  # noqa: E731
    import hashlib
    pp = {hashlib.sha1(json.dumps(list(k)).encode()).hexdigest()[:16]: [g["kl"], g["mm"], g["n"]]
          for k, g in groups.items()}
    row = {"variant": a.variant, "fla": fla, "gpu": torch.cuda.get_device_name(0), "label": a.label, "prompts": len(G),
           "seqs": len(rows), "tokens": sum(g["n"] for g in G),
           "kl": tot("kl"), "kl_ci95": boot([(g["kl"], g["n"]) for g in G]),
           "mismatch": tot("mm"), "mismatch_ci95": boot([(g["mm"], g["n"]) for g in G])}
    if any(g["ntv"] for g in G):
        row.update(tv=tot("tv", "ntv"), tv_ci95=boot([(g["tv"], g["ntv"]) for g in G]))
    row["per_prompt"] = pp
    print(json.dumps(row), flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(row) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("gen", "score"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--prompts", type=int, default=16)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--variant", default="w16fused", choices=("bf16", "fp8w", "fp8both", "fused", "mixed", "w16fused"))
    ap.add_argument("--label", default="")
    ap.add_argument("--gpu-util", type=float, default=0.45)
    ap.add_argument("--linear-backend", default="auto")
    ap.add_argument("--kv-cache-dtype", default="auto")
    ap.add_argument("--mamba-ssm-cache-dtype", default="auto")
    ap.add_argument("--topk", type=int, default=0)
    ap.add_argument("--engine", default="fp8", choices=("fp8", "bf16"))
    ap.add_argument("--out", default=str(LB / "results/mismatch_bench3.jsonl"))
    a = ap.parse_args()
    MB.gen(a) if a.phase == "gen" else score(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
