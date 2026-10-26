"""MISMATCH-DECOMP-3B (docs/REGISTRY.md, registered before any data): per adapter, the per-token KL of the deployed matched engine
(swap) to the bf16 trainer and to the W8A16 trainer, and of Marlin to the bf16 trainer, each over the same job's bf16 pair
(rows labelled decomp3b_<adapter>_<engine>_<trainer>_..., results/mismatch_bench3.jsonl), with paired bootstrap intervals over the
prompts common to every pair, as bench/score_decomp3.py.
  (i)  the matched engine's KL to the bf16 trainer at least 4x the bf16 pair's, every adapter
  (ii) the matched engine's KL to the W8A16 trainer at most 1.5x the bf16 pair's, every adapter
    python3 bench/score_decomp3b.py [--json results/std_bench/decomp3b_verdict.json]"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
ADAPTERS = ("qwen9_dapo_S8TL_s0", "qwen9_dapo_S0G_s0", "qwen9_dapo_S8G_s0", "qwen9_dapo_S8MH_s0", "qwen9_dapo_S4TF_s0")
PAIRS = {"bf16": ("bf16", "bf16"), "matched_bf16tr": ("swap", "bf16"), "matched": ("swap", "w16fused"),
         "marlin_bf16tr": ("marlin", "bf16")}


def load() -> dict:
    out = {}
    f = LB / "results/mismatch_bench3.jsonl"
    for line in (f.read_text().splitlines() if f.exists() else []):
        r = json.loads(line)
        lab = r.get("label", "")
        if not lab.startswith("decomp3b_") or "per_prompt" not in r:
            continue
        for ad in ADAPTERS:
            pre = f"decomp3b_{ad}_"
            if lab.startswith(pre):
                eng, tr = lab[len(pre):].split("_")[:2]
                out[(ad, eng, tr)] = r["per_prompt"]
    return out


def kl(pp: dict, keys) -> float:
    return sum(pp[k][0] for k in keys) / max(sum(pp[k][2] for k in keys), 1)


def stats(d: dict, ad: str, n: int = 2000, seed: int = 0) -> dict | None:
    pp = {name: d.get((ad, e, t)) for name, (e, t) in PAIRS.items()}
    if any(v is None for v in pp.values()):
        return None
    keys = sorted(set.intersection(*(set(v) for v in pp.values())))
    def ratios(ks):
        k = {name: kl(v, ks) for name, v in pp.items()}
        return {"matched_bf16tr_x": k["matched_bf16tr"] / k["bf16"], "matched_x": k["matched"] / k["bf16"],
                "marlin_bf16tr_x": k["marlin_bf16tr"] / k["bf16"]}
    est = ratios(keys)
    rng = random.Random(seed)
    boots = {q: [] for q in est}
    for _ in range(n):
        ks = [keys[rng.randrange(len(keys))] for _ in keys]
        for q, x in ratios(ks).items():
            boots[q].append(x)
    ci = {q: sorted(xs)[int(0.025 * n)] for q, xs in boots.items()}
    ci = {q: [sorted(boots[q])[int(0.025 * n)], sorted(boots[q])[int(0.975 * n) - 1]] for q in boots}
    return {"prompts": len(keys), "kl_x1e4": {name: 1e4 * kl(v, keys) for name, v in pp.items()}, "ratios": est, "ci95": ci}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    d = load()
    per = {ad: stats(d, ad) for ad in ADAPTERS}
    done = {ad: x for ad, x in per.items() if x}
    for ad, x in done.items():
        print(ad, json.dumps({"ratios": {k: round(v, 3) for k, v in x["ratios"].items()}}))
    v = {"adapters_done": sorted(done)}
    if len(done) < len(ADAPTERS):
        v["status"] = f"incomplete ({len(done)} of {len(ADAPTERS)} adapters scored)"
    else:
        def rule(held):
            n_ = sum(held.values())
            return {"held_for": [ad for ad, h in held.items() if h],
                    "verdict": "MET" if n_ == len(held) else ("NOT MET" if n_ == 0 else "PARTLY MET")}
        v["i"] = rule({ad: x["ratios"]["matched_bf16tr_x"] >= 4 for ad, x in done.items()})
        v["ii"] = rule({ad: x["ratios"]["matched_x"] <= 1.5 for ad, x in done.items()})
    print("verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps({"adapters": per, "verdict": v}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
