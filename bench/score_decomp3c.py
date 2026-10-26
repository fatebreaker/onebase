"""MISMATCH-DECOMP-3C (docs/REGISTRY.md, registered before any data; v21 mock review R2 M3 and missing experiment 3): does rounding the
activations alike on both sides remove the W8A8 engine's mismatch? MISMATCH-DECOMP-3's protocol (H100, the five adapters, 128
DAPO-Math prompts x 4 samples, top-32; bench/mismatch_bench2.py gen and bench/mismatch_bench3.py score, unchanged) with each W8A8
engine's samples scored by the weight-matched W8A16 trainer (w16fused) and by a trainer that also rounds its activations to FP8 as
the engine does (fp8both: the W8A8 forward), beside the bf16 pair on its own samples. Rows: results/mismatch_bench3.jsonl, labels
decomp3c_<adapter>_<engine>_<trainer>_<GPU>_<job> (mlsys jobs/pod_runs/decomp3c_pod.sh). Per adapter (KL per token):
  (i)  the W8A8 engine's KL to the rounding-alike trainer at least 4x the bf16 pair's (rounding alike leaves the mismatch far
       above an unrounded pair, where MISMATCH-DECOMP-3's matched engine sits at 1.0-1.3x)
  (ii) that KL at least 0.8x the W8A8 engine's KL to the W8A16 trainer (rounding alike removes at most a fifth of it)
Each held for all five adapters: met; for none: not met; otherwise partly met, with the adapters listed.
Descriptive: the rounding-alike over the W8A16-trainer ratio with its paired 95% bootstrap interval over prompts; |dlog p|.
    python3 bench/score_decomp3c.py [--json results/std_bench/decomp3c_verdict.json]"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
ADAPTERS = ("qwen9_dapo_S8TL_s0", "qwen9_dapo_S0G_s0", "qwen9_dapo_S8G_s0", "qwen9_dapo_S8MH_s0", "qwen9_dapo_S4TF_s0")
PAIRS = {"bf16": ("bf16", "bf16"), "w8a8": ("w8a8", "w16fused"), "alike": ("w8a8", "fp8both")}


def load() -> dict:
    """{(adapter, engine, trainer): per_prompt} from each label's newest row."""
    out = {}
    f = LB / "results/mismatch_bench3.jsonl"
    for line in (f.read_text().splitlines() if f.exists() else []):
        r = json.loads(line)
        lab = r.get("label", "")
        if not lab.startswith("decomp3c_") or "per_prompt" not in r:
            continue
        for ad in ADAPTERS:
            pre = f"decomp3c_{ad}_"
            if lab.startswith(pre):
                eng, tr = lab[len(pre):].split("_")[:2]
                out[(ad, eng, tr)] = r["per_prompt"]
    return out


def kl(pp: dict, keys) -> float:
    return sum(pp[k][0] for k in keys) / max(sum(pp[k][2] for k in keys), 1)


def absd(pp: dict, keys) -> float:
    return sum(pp[k][1] for k in keys) / max(sum(pp[k][2] for k in keys), 1)


def stats(d: dict, ad: str, n: int = 2000, seed: int = 0) -> dict | None:
    pp = {name: d.get((ad, e, t)) for name, (e, t) in PAIRS.items()}
    if any(v is None for v in pp.values()):
        return None
    keys = sorted(set.intersection(*(set(v) for v in pp.values())))

    def ratios(ks):
        k = {name: kl(v, ks) for name, v in pp.items()}
        return {"alike_x": k["alike"] / k["bf16"], "w8a8_x": k["w8a8"] / k["bf16"], "alike_over_w8a8": k["alike"] / k["w8a8"]}
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
            "absdlogp": {name: absd(v, keys) for name, v in pp.items()}, "ratios": est, "ci95": ci}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    d = load()
    per = {ad: stats(d, ad) for ad in ADAPTERS}
    done = {ad: x for ad, x in per.items() if x}
    for ad, x in done.items():
        print(ad, json.dumps({"ratios": {k: round(v, 3) for k, v in x["ratios"].items()},
                              "ci95": {k: [round(v[0], 3), round(v[1], 3)] for k, v in x["ci95"].items()}}))
    v: dict = {"adapters_done": sorted(done)}
    if len(done) < len(ADAPTERS):
        v["status"] = f"incomplete ({len(done)} of {len(ADAPTERS)} adapters scored)"
    else:
        def rule(held):
            n_ = sum(held.values())
            return {"held_for": [ad for ad, h in held.items() if h],
                    "verdict": "MET" if n_ == len(held) else ("NOT MET" if n_ == 0 else "PARTLY MET")}
        v["i"] = rule({ad: x["ratios"]["alike_x"] >= 4 for ad, x in done.items()})
        v["ii"] = rule({ad: x["ratios"]["alike_over_w8a8"] >= 0.8 for ad, x in done.items()})
    print("verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps({"adapters": per, "verdict": v}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
