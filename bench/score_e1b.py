"""ICML-E1B-TF, its extensions and ICML-E1B-FP4 (docs/REGISTRY.md): per adapter, the registered predictions from the rows of
results/bias_noise_tf.jsonl (FP8, e4m3 activations) and results/bias_noise_tf_fp4.jsonl (NVFP4, W4A4 emulation):
  (i)   two round-to-nearest passes bit-identical (max |d log q| = 0)
  (ii)  the noise-averaged SR sampler's KL (Richardson estimate) at most 0.25x round to nearest's (FP8, ICML-E1B-TF), below
        round to nearest's (NVFP4, ICML-E1B-FP4)
  (iii) the per-draw SR KL at least round to nearest's
Reported with the Richardson ratio's spread over adapters and, where both formats exist for an adapter, the FP4/FP8 growth of
each KL (round to nearest's should grow about as the squared rounding step; SR's averaged KL, a second-order term, faster).
    python3 bench/score_e1b.py [--json results/std_bench/e1b_verdict.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LB = Path(__file__).resolve().parents[1]


def rows(name: str) -> dict:
    f = LB / "results" / name
    out = {}
    for line in (f.read_text().splitlines() if f.exists() else []):
        r = json.loads(line)
        out[r["adapter"].rstrip("/").rsplit("/", 1)[1]] = r          # the newest row per adapter
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    out = {}
    for fmt, name, bar in (("fp8", "bias_noise_tf.jsonl", 0.25), ("nvfp4", "bias_noise_tf_fp4.jsonl", 1.0)):
        for ad, r in rows(name).items():
            ratio = r["kl_sr_rich"] / r["kl_rtn"]
            v = {"tokens": r["tokens"], "kl_rtn": r["kl_rtn"], "kl_sr_draw": r["kl_sr_draw"], "kl_sr_rich": r["kl_sr_rich"],
                 "kl_sr_rich_ci95": r["kl_sr_rich_ci95"], "ratio": ratio,
                 "i": r["det_max_abs_dlogq"] == 0.0,
                 "ii": ratio <= bar if fmt == "fp8" else ratio < bar,
                 "iii": r["kl_sr_draw"] >= r["kl_rtn"]}
            out.setdefault(fmt, {})[ad] = v
            print(f"{fmt:5s} {ad:22s} rtn {1e4 * r['kl_rtn']:7.2f}  draw {1e4 * r['kl_sr_draw']:7.2f}  rich {1e4 * r['kl_sr_rich']:7.2f}"
                  f"  ratio {ratio:.2f}  (i) {'met' if v['i'] else 'NOT'}  (ii) {'met' if v['ii'] else 'NOT met'}"
                  f"  (iii) {'met' if v['iii'] else 'NOT met'}")
    f = LB / "results/bias_noise_tf_fmt.jsonl"                      # ICML-E1B-FMT: activation formats on FP8 weights
    for line in (f.read_text().splitlines() if f.exists() else []):
        r = json.loads(line)
        fmt = r["label"].split("_")[1]
        ad = r["adapter"].rstrip("/").rsplit("/", 1)[1]
        out.setdefault("fmt", {}).setdefault(ad, {})[fmt] = {"kl_rtn": r["kl_rtn"], "kl_sr_draw": r["kl_sr_draw"],
                                                              "kl_sr_rich": r["kl_sr_rich"], "ratio": r["kl_sr_rich"] / r["kl_rtn"],
                                                              "det": r["det_max_abs_dlogq"]}
    for ad, fm in out.get("fmt", {}).items():
        print(f"fmt   {ad:22s} " + "  ".join(f"{k}: rtn {1e4 * v['kl_rtn']:.1f} rich {1e4 * v['kl_sr_rich']:.1f} ratio {v['ratio']:.2f}"
                                             for k, v in fm.items()))
        r_ = {k: v["ratio"] for k, v in fm.items()}
        if all(k in r_ for k in ("e4m3", "e3m2", "e2m1")):
            fm["_i"] = r_["e4m3"] < r_["e3m2"] < r_["e2m1"] and r_["e2m1"] >= 1.5 * r_["e3m2"]
            fm["_ii"] = r_["e3m2"] < 1 <= r_["e2m1"]
            print(f"      (i) {'met' if fm['_i'] else 'NOT met'}  (ii) {'met' if fm['_ii'] else 'NOT met'}")
    both = sorted(set(out.get("fp8", {})) & set(out.get("nvfp4", {})))
    out["growth"] = {ad: {k: out["nvfp4"][ad][k] / out["fp8"][ad][k] for k in ("kl_rtn", "kl_sr_draw", "kl_sr_rich")} for ad in both}
    for ad, g in out["growth"].items():
        print(f"growth FP4/FP8 {ad}: rtn {g['kl_rtn']:.1f}x, draw {g['kl_sr_draw']:.1f}x, SR averaged {g['kl_sr_rich']:.1f}x")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
