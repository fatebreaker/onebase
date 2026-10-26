"""SYNC-NVFP4, SYNC-NVFP4-M and SYNC-FP8 (docs/REGISTRY.md): SmolLM2 GRPO whose policy syncs into an NVFP4 (FP8) engine
are exact (arm A, --sync exact) or merges onto the frozen NVFP4 (FP8) grid (arm B, --sync nvfp4merge / fp8merge). Reads each run's training log
(logs/lb_syncnv_smol_gsm8k_<prefix>_<arm>_s<seed>.out) and its GSM8K evaluation (results/std_bench/smol_gsm8k_<prefix>_<arm>_s<seed>).
  sync      arm B's largest and median delivered share over all syncs (NVFP4: largest <= 0.05; FP8: median < 0.2)
  reward    mean rollout reward of steps 151-200 minus steps 1-50 (arm A > 0.05, arm B < 0.03 predicted)
  mismatch  mean trainer-engine |delta log p| over the first 10 steps (SYNC-NVFP4-M: arm A < 0.02 predicted)
  final     each final policy served exactly on the low-precision base (<fmt>_exact): accuracy, and arm A against arm B
    python3 bench/score_syncnv.py --prefix nvsync|nvsyncm|fp8sync --seeds 0 [--json out.json]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
RES = LB / "results/std_bench"
sys.path.insert(0, "$ROUNDING/analysis")
import read_retention as rr  # noqa: E402

FMT = {"nvsync": "nvfp4", "nvsyncm": "nvfp4", "fp8sync": "fp8"}


def load(f: Path) -> dict | None:
    return {r["id"]: int(r["correct"]) for r in map(json.loads, f.read_text().splitlines())} if f.exists() else None


def acc(x: dict) -> float:
    return 100 * sum(x.values()) / len(x)


MODEL = "smol"                                                   # --model mistral: SYNC-NVFP4-MIS's runs


def run(prefix: str, arm: str, seed: int) -> dict | None:
    fmt = FMT[prefix]
    name = f"{MODEL}_gsm8k_{prefix}_{'exact' if arm == 'A' else fmt + 'merge'}_s{seed}"
    f = LB / f"logs/lb_syncnv_{name}.out"
    if not f.exists():
        return None
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.startswith('{"step"')]
    if not rows or rows[-1]["step"] < 200:
        return None
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
    rew = {r["step"]: r["reward"] for r in rows}
    out = {"name": name,
           "reward_early": mean([rew[s] for s in range(1, 51) if s in rew]),
           "reward_late": mean([rew[s] for s in range(151, 201) if s in rew]),
           "mismatch_first10": mean([r["mismatch"] for r in rows if r["step"] <= 10 and r.get("mismatch") is not None]),
           "mismatch_last10": mean([r["mismatch"] for r in rows if r["step"] > 190 and r.get("mismatch") is not None]),
           "sync_gain_max": max((r["sync_gain"] for r in rows if r.get("sync_gain") is not None), default=None)}
    gains = sorted(r["sync_gain"] for r in rows if r.get("sync_gain") is not None)
    out["sync_gain_median"] = gains[len(gains) // 2] if gains else None
    out["reward_rise"] = out["reward_late"] - out["reward_early"]
    ev = {k: load(RES / name / k / "gsm8k.jsonl") for k in ("bf16_base", "bf16_exact", f"{fmt}_base", f"{fmt}_exact",
                                                             f"{fmt}_merge")}
    out["acc"] = {k: (acc(v) if v else None) for k, v in ev.items()}
    out["_ev"] = ev
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="nvsync", choices=tuple(FMT))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--json", default="")
    ap.add_argument("--model", default="smol", choices=("smol", "mistral"))
    a = ap.parse_args()
    global MODEL
    MODEL = a.model
    res = {}
    for s in a.seeds:
        for arm in ("A", "B"):
            r = run(a.prefix, arm, s)
            if r:
                res[f"{arm}{s}"] = r
                print(f"arm {arm} seed {s}: reward {r['reward_early']:.3f} -> {r['reward_late']:.3f} (rise {r['reward_rise']:+.3f}); "
                      f"mismatch first 10 steps {r['mismatch_first10']:.3f}, last 10 {r['mismatch_last10']:.3f}; sync gain max "
                      f"{r['sync_gain_max']}, median {r['sync_gain_median']}; GSM8K " + ", ".join(f"{k} {v:.1f}" for k, v in r["acc"].items() if v is not None))
        A, B = res.get(f"A{s}"), res.get(f"B{s}")
        fx = f"{FMT[a.prefix]}_exact"
        if A and B and A["_ev"][fx] and B["_ev"][fx]:
            x, y = A["_ev"][fx], B["_ev"][fx]
            ids = sorted(set(x) & set(y))
            w, l, p = rr.mcnemar([x[i] for i in ids], [y[i] for i in ids])
            res[f"AvsB{s}"] = {"wins": w, "losses": l, "p": p}
            print(f"seed {s}: final policies served exactly on the {FMT[a.prefix]} base, A vs B {w}:{l} p {p:.2g}")
    if a.json:
        Path(a.json).write_text(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "_ev"} for k, v in res.items()},
                                           indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
