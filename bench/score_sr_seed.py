"""SR-SEED (docs/REGISTRY.md, registered 2026-10-14 00:48 UTC, before any data): R of the STD-BENCH adapters' stochastic
merges drawn with seeds 1 and 2 (results/std_bench/mistral_<w>_srseed<k>) next to the registered draw (seed 0, in
mistral_<w>), against STD-BENCH's references, 10,000 resamples. Prediction: on GSM8K both further NVFP4 draws keep
R < 0.8; on MBPP both keep R >= 0.8.
    python bench/score_sr_seed.py [--json out.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import score_std_bench as sb

RES = sb.RES
JUDGED = {"math": "gsm8k", "code": "mbpp"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    out = {}
    for w, bs in sb.BENCHES.items():
        for b in bs:
            ref = {k: sb.load(RES / f"mistral_{w}" / k, b) for k in ("bf16_base", "bf16_exact", "nvfp4_base", "fp8_base")}
            for fmt in ("nvfp4", "fp8"):
                rs = []
                for seed in (0, 1, 2):
                    d = RES / (f"mistral_{w}" if seed == 0 else f"mistral_{w}_srseed{seed}") / f"{fmt}_srmerge"
                    x = sb.load(d, b)
                    if x is None or any(v is None for v in ref.values()):
                        rs.append(None)
                        continue
                    r = sb.rr.retention(ref["bf16_base"], ref["bf16_exact"], ref[f"{fmt}_base"], x, n_boot=10000)
                    rs.append({"seed": seed, "acc": sb.acc(x), "R": r["R"], "R_ci95": r["R_ci95"]})
                out[f"{b}/{fmt}"] = rs
                txt = "; ".join("--" if r is None else f"seed {r['seed']}: {r['acc']:.1f}% R = {r['R']:.2f} "
                                f"[{r['R_ci95'][0]:.2f}, {r['R_ci95'][1]:.2f}]" for r in rs)
                print(f"{b:9s} {fmt:5s} {txt}")
    verdict = {}
    for w, b in JUDGED.items():
        rs = [r for r in out[f"{b}/nvfp4"][1:] if r is not None]
        if len(rs) == 2:
            ok = all(r["R"] < 0.8 for r in rs) if b == "gsm8k" else all(r["R"] >= 0.8 for r in rs)
            verdict[b] = {"met": ok, "R": [r["R"] for r in rs]}
            print(f"prediction on {b} (NVFP4 draws 1, 2: {'R < 0.8' if b == 'gsm8k' else 'R >= 0.8'}): "
                  f"{'MET' if ok else 'NOT MET'} ({', '.join(f'{r:.2f}' for r in verdict[b]['R'])})")
    if a.json:
        Path(a.json).write_text(json.dumps({"rows": out, "verdict": verdict}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
