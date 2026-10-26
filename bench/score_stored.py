"""STORED-STRONG (docs/REGISTRY.md): NVFP4-stored adapters of strong recent models, served exactly. For each adapter
(QMATH-CONV's Qwen3.5-9B S0G seeds 0-2 on our NVFP4 base; GRANITE-RL's Granite-4.2-8B seeds 0-2 on IBM's NVFP4 release),
MATH-500, OlympiadBench and Minerva pooled, every variant on one GPU:
  gate  judged if the bf16 gain (exact adapter minus base, bf16) is at least 3 points
  (i)   the NVFP4-stored adapter on the low-precision base keeps R >= 0.8 (R against that base, paired bootstrap interval)
  (ii)  its accuracy there minus the exact adapter's on the same base: paired 95% interval within +-2 points
Descriptive: the NVFP4-stored adapter on the bf16 base (Qwen3.5-9B), R of the exact adapter on the low-precision base.
Files: results/std_bench/<adapter>_stored/<variant>/<bench>.jsonl (Granite's release: <adapter>_stored_relnvfp4).
    python3 bench/score_stored.py [--json results/std_bench/stored_verdict.json]"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
RES = LB / "results/std_bench"
sys.path.insert(0, "$ROUNDING/analysis")
import read_retention as rr  # noqa: E402

BENCHES = ("math500", "olympiad", "minerva")


def load(d: Path) -> dict | None:
    out = {}
    for b in BENCHES:
        f = d / f"{b}.jsonl"
        if not f.exists():
            return None
        out.update({f"{b}:{r['id']}": int(r["correct"]) for r in map(json.loads, f.read_text().splitlines())})
    return out


def paired_diff(x: dict, y: dict) -> tuple[float, float, float]:
    """Accuracy of x minus y (points) over the common problems, with a normal 95% interval from the paired outcomes."""
    ids = sorted(set(x) & set(y))
    n = len(ids)
    b = sum(x[i] and not y[i] for i in ids)
    c = sum(y[i] and not x[i] for i in ids)
    d = (b - c) / n
    var = ((b + c) / n - d * d) / n
    h = 1.96 * math.sqrt(max(var, 0.0))
    return 100 * d, 100 * (d - h), 100 * (d + h)


def judge(ref_dir: Path, low_dir: Path, low_base: str, low_exact: str, low_stored: str, extra: dict) -> dict | None:
    """ref_dir holds bf16_base and bf16_exact (the gain's reference); low_dir the low-precision base's variants."""
    base, ex = load(ref_dir / "bf16_base"), load(ref_dir / "bf16_exact")
    lb, le, ls = (load(low_dir / v) for v in (low_base, low_exact, low_stored))
    if None in (base, ex, lb, le, ls):
        return None
    gain = 100 * (sum(ex.values()) / len(ex) - sum(base.values()) / len(base))
    out = {"n": len(base), "gain": gain, "judged": gain >= 3}
    for lab, v in (("exact", le), ("stored", ls)):
        r = rr.retention(base, ex, lb, v, n_boot=10000)
        out[f"R_{lab}"] = {"R": r["R"], "R_ci95": r["R_ci95"], "acc": 100 * r["acc"]["rl_q"]}
    for lab, (vb, v) in extra.items():
        if vb is not None and v is not None:
            r = rr.retention(base, ex, vb, v, n_boot=10000)
            out[f"R_{lab}"] = {"R": r["R"], "R_ci95": r["R_ci95"], "acc": 100 * r["acc"]["rl_q"]}
    d, lo, hi = paired_diff(ls, le)
    out["stored_minus_exact"] = [d, lo, hi]
    if out["judged"]:
        out["i"] = {"R": out["R_stored"]["R"], "met": out["R_stored"]["R"] >= 0.8}
        out["ii"] = {"diff": [d, lo, hi], "met": lo >= -2 and hi <= 2}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    res = {}
    for k in (0, 1, 2):
        d = RES / f"qwen9_dapo_S0G_s{k}_stored"
        e = judge(d, d, "nvfp4_base", "nvfp4_exact", "nvfp4_stored",
                  {"stored_on_bf16": (load(d / "bf16_base"), load(d / "bf16_stored"))})
        if e:
            res[f"qwen9_math_s{k}"] = e
        g = RES / f"granite8_dapo_s{k}_stored"
        e = judge(g, RES / f"granite8_dapo_s{k}_stored_relnvfp4", "bf16_base", "bf16_exact", "bf16_stored", {})
        if e:
            res[f"granite_s{k}"] = e
    for k, e in res.items():
        msg = f"{k}: gain {e['gain']:+.1f} ({'judged' if e['judged'] else 'not judged'}); " + ", ".join(
            f"{lab[2:]} R {v['R']:.2f} [{v['R_ci95'][0]:.2f}, {v['R_ci95'][1]:.2f}]" for lab, v in e.items() if lab.startswith("R_"))
        msg += f"; stored - exact {e['stored_minus_exact'][0]:+.1f} [{e['stored_minus_exact'][1]:+.1f}, {e['stored_minus_exact'][2]:+.1f}]"
        print(msg)
        for p in ("i", "ii"):
            if p in e:
                print(f"   ({p}) {'MET' if e[p]['met'] else 'NOT MET'}: {json.dumps(e[p])}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
