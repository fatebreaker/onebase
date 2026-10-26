"""MISMATCH-DECOMP-3 (docs/REGISTRY.md, registered before any data): MISMATCH-DECOMP-2's decomposition on the H100 with the
deployed matched engine (the swap-AB W8A16 GEMM, RL-MATCHED-H100's), five adapters and 128 DAPO-Math prompts x 4 samples (top-32),
every ratio with a paired 95% bootstrap interval over prompts. Rows: results/mismatch_bench3.jsonl (bench/mismatch_bench3.py),
labels decomp3_<adapter>_<engine>_<trainer>_<GPU>_<job>; engines bf16, w8a8 (our plugin), w8a8fq (the fake-quant control),
swap (the matched engine), marlin (vLLM's weight-only FP8); trainers bf16 and w16fused (the W8A16 trainer).
Per adapter (KL per token to the W8A16 trainer; "bf16" = the bf16 engine to the bf16 trainer):
  (i)   the W8A8 engine's KL at least 4x bf16's
  (ii)  the fake-quant control's excess over bf16's at least half the W8A8 engine's excess
  (iii) the matched engine's KL at most 1.5x bf16's
Each held for all five adapters: met; for none: not met; otherwise partly met, with the adapters listed.
Descriptive: Marlin's KL over bf16's and over the matched engine's; the W8A8 engine to the bf16 trainer over bf16's (weight and
activation rounding together); |dlog p|; every ratio's paired interval (2,000 resamples of the prompts).
    python3 bench/score_decomp3.py [--json results/std_bench/decomp3_verdict.json]"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
ADAPTERS = ("qwen9_dapo_S8TL_s0", "qwen9_dapo_S0G_s0", "qwen9_dapo_S8G_s0", "qwen9_dapo_S8MH_s0", "qwen9_dapo_S4TF_s0")
PAIRS = {"bf16": ("bf16", "bf16"), "w8a8": ("w8a8", "w16fused"), "w8a8_bf16tr": ("w8a8", "bf16"),
         "fq": ("w8a8fq", "w16fused"), "matched": ("swap", "w16fused"), "marlin": ("marlin", "w16fused")}


def load() -> dict:
    """{(adapter, engine, trainer): per_prompt dict} from each label's newest row."""
    out = {}
    f = LB / "results/mismatch_bench3.jsonl"
    for line in (f.read_text().splitlines() if f.exists() else []):
        r = json.loads(line)
        lab = r.get("label", "")
        if not lab.startswith("decomp3_") or "per_prompt" not in r:
            continue
        for ad in ADAPTERS:
            pre = f"decomp3_{ad}_"
            if lab.startswith(pre):
                eng, tr = lab[len(pre):].split("_")[:2]
                out[(ad, eng, tr)] = r["per_prompt"]
    return out


def kl(pp: dict, keys) -> float:
    return sum(pp[k][0] for k in keys) / max(sum(pp[k][2] for k in keys), 1)


def stats(d: dict, ad: str, n: int = 2000, seed: int = 0) -> dict | None:
    """Point estimates and paired bootstrap intervals (over the prompts common to every pair) of the ratios."""
    pp = {name: d.get((ad, e, t)) for name, (e, t) in PAIRS.items()}
    if any(v is None for v in pp.values()):
        return None
    keys = sorted(set.intersection(*(set(v) for v in pp.values())))
    def ratios(ks):
        k = {name: kl(v, ks) for name, v in pp.items()}
        ex8 = k["w8a8"] - k["bf16"]
        return {"w8a8_x": k["w8a8"] / k["bf16"], "fq_share": (k["fq"] - k["bf16"]) / ex8 if ex8 else float("nan"),
                "matched_x": k["matched"] / k["bf16"], "marlin_x": k["marlin"] / k["bf16"],
                "marlin_over_matched": k["marlin"] / k["matched"], "w8a8_bf16tr_x": k["w8a8_bf16tr"] / k["bf16"]}
    est = ratios(keys)
    rng = random.Random(seed)
    boots = {q: [] for q in est}
    for _ in range(n):
        ks = [keys[rng.randrange(len(keys))] for _ in keys]
        for q, x in ratios(ks).items():
            boots[q].append(x)
    ci = {}
    for q, xs in boots.items():
        xs.sort()
        ci[q] = [xs[int(0.025 * n)], xs[int(0.975 * n) - 1]]
    return {"prompts": len(keys), "kl_x1e4": {name: 1e4 * kl(v, keys) for name, v in pp.items()},
            "ratios": est, "ci95": ci}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    d = load()
    per = {ad: stats(d, ad) for ad in ADAPTERS}
    done = {ad: x for ad, x in per.items() if x}
    for ad, x in done.items():
        print(ad, json.dumps({"kl_x1e4": {k: round(v, 2) for k, v in x["kl_x1e4"].items()},
                              "ratios": {k: round(v, 3) for k, v in x["ratios"].items()},
                              "ci95": {k: [round(v[0], 3), round(v[1], 3)] for k, v in x["ci95"].items()}}))
    v = {"adapters_done": sorted(done)}
    if len(done) < len(ADAPTERS):
        v["status"] = f"incomplete ({len(done)} of {len(ADAPTERS)} adapters scored)"
    else:
        def rule(held):
            n = sum(held.values())
            return {"held_for": [ad for ad, h in held.items() if h],
                    "verdict": "MET" if n == len(held) else ("NOT MET" if n == 0 else "PARTLY MET")}
        v["i"] = rule({ad: x["ratios"]["w8a8_x"] >= 4 for ad, x in done.items()})
        v["ii"] = rule({ad: x["ratios"]["fq_share"] >= 0.5 for ad, x in done.items()})
        v["iii"] = rule({ad: x["ratios"]["matched_x"] <= 1.5 for ad, x in done.items()})
    print("verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps({"adapters": per, "verdict": v}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
