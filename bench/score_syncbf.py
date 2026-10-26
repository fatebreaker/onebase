"""SYNC-BF16-S (docs/REGISTRY.md): SmolLM2 GRPO with a bf16 merge-sync (seeds 4, 5; logs/lb_syncnv_smol_gsm8k_bfsync_bf16merge_s<k>.out,
results/std_bench/smol_gsm8k_bfsync_bf16merge_s<k>) paired with GPTQ-SMOL-H100's exact-sync run of the same seed
(logs/lb_smolg_smol_gsm8k_rlb_s<k>.out, results/std_bench/smol_gsm8k_rlb_s<k>), same recipe and GPU type.
  (i)  the merge-sync's mean delivered share over syncs 1-10 is below 0.5 and over syncs 191-200 above 0.5
  (ii) its final policy's GSM8K accuracy (bf16 base, exact adapter) is within 3 points of the exact-sync run's
    python3 bench/score_syncbf.py [--json results/std_bench/syncbf_verdict.json]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
RES = LB / "results/std_bench"
sys.path.insert(0, "$ROUNDING/analysis")
import read_retention as rr  # noqa: E402


def log(f: Path) -> list[dict]:
    return [json.loads(x) for x in f.read_text().splitlines() if x.startswith('{"step"')] if f.exists() else []


def load(f: Path) -> dict | None:
    return {r["id"]: int(r["correct"]) for r in map(json.loads, f.read_text().splitlines())} if f.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[4, 5])
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    out = {}
    for s in a.seeds:
        mrg = log(LB / f"logs/lb_syncnv_smol_gsm8k_bfsync_bf16merge_s{s}.out")
        exa = log(LB / f"logs/lb_smolg_smol_gsm8k_rlb_s{s}.out")
        em = load(RES / f"smol_gsm8k_bfsync_bf16merge_s{s}" / "bf16_exact/gsm8k.jsonl")
        ee = load(RES / f"smol_gsm8k_rlb_s{s}" / "bf16_exact/gsm8k.jsonl")
        if not mrg or mrg[-1]["step"] < 200 or em is None or ee is None:
            print(f"seed {s}: not complete")
            continue
        g = {r["step"]: r["sync_gain"] for r in mrg if r.get("sync_gain") is not None}
        early = sum(g[t] for t in range(1, 11) if t in g) / max(1, sum(t in g for t in range(1, 11)))
        late = sum(g[t] for t in range(191, 201) if t in g) / max(1, sum(t in g for t in range(191, 201)))
        am, ae = 100 * sum(em.values()) / len(em), 100 * sum(ee.values()) / len(ee)
        ids = sorted(set(em) & set(ee))
        w, l, p = rr.mcnemar([em[i] for i in ids], [ee[i] for i in ids])
        rew = lambda rows, lo, hi: sum(r["reward"] for r in rows if lo <= r["step"] <= hi) / max(1, sum(lo <= r["step"] <= hi for r in rows))  # noqa: E731
        out[str(s)] = {"gain_syncs_1_10": early, "gain_syncs_191_200": late, "acc_merge_sync": am, "acc_exact_sync": ae,
                       "merge_vs_exact": {"wins": w, "losses": l, "p": p},
                       "reward_merge": [rew(mrg, 1, 50), rew(mrg, 151, 200)], "reward_exact": [rew(exa, 1, 50), rew(exa, 151, 200)],
                       "i": early < 0.5 and late > 0.5, "ii": abs(am - ae) <= 3}
        print(f"seed {s}: delivered {early:.3f} (syncs 1-10) -> {late:.3f} (191-200); final GSM8K merge-sync {am:.1f} vs exact "
              f"{ae:.1f} ({w}:{l} p {p:.2g}); (i) {'MET' if out[str(s)]['i'] else 'NOT MET'}, (ii) {'MET' if out[str(s)]['ii'] else 'NOT MET'}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
