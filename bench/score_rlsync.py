"""RL-SYNC-GSM8K (docs/REGISTRY.md, registered 2026-10-14 06:36 UTC; seed-2 amendment 07:43 UTC): GRPO on GSM8K with a
bf16 merge-sync (adapters mistral_gsm8k_rl_m<k>) against its exact-sync twin of the same seed (mistral_gsm8k_rl,
mistral_gsm8k_rl_s1, mistral_gsm8k_rl_s2). Each final policy is the bf16 base + its exact adapter on GSM8K test,
against the base evaluated in the same run.
  (i)  the merge-sync delivers less than half of the update over its first 10 syncs (median logged sync_gain < 0.5), and
       more over steps 191-200
  (ii) the merge-sync policy's gain is smaller than its exact twin's on at least 2 of the 3 seed pairs (on both, while
       only two exist); per pair a McNemar test of the two policies; pooled over the pairs, the mean difference of the
       two policies' accuracies with a bootstrap 95% interval over problems
    python bench/score_rlsync.py [--json out.json]"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
import runlog  # noqa: E402

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, "$ROUNDING/analysis")
import read_retention as rr  # noqa: E402

RES = LB / "results/std_bench"
AD = Path("$SCRATCH/std_bench/adapters")
EXACT = {0: "mistral_gsm8k_rl", 1: "mistral_gsm8k_rl_s1", 2: "mistral_gsm8k_rl_s2"}


def load(d: str, v: str) -> dict | None:
    f = RES / d / v / "gsm8k.jsonl"
    if not f.exists():
        return None
    return {r["id"]: int(r["correct"]) for r in map(json.loads, f.read_text().splitlines())}


def acc(x: dict) -> float:
    return 100 * sum(x.values()) / len(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    out, pairs = {}, []
    for k, ex in EXACT.items():
        m = f"mistral_gsm8k_rl_m{k}"
        log = AD / m / "log.jsonl"
        g = [r.get("sync_gain") for r in runlog.read(log)]   # one row per step (bench/runlog.py)
        early = [x for x in g[:11] if x is not None][:10]
        late = [x for x in g[190:200] if x is not None]
        row = {"sync_gain_first10": statistics.median(early) if early else None,
               "sync_gain_191_200": statistics.median(late) if late else None}
        e_base, e_pol, m_base, m_pol = load(ex, "bf16_base"), load(ex, "bf16_exact"), load(m, "bf16_base"), load(m, "bf16_exact")
        if None not in (e_base, e_pol, m_base, m_pol):
            row |= {"exact_gain": acc(e_pol) - acc(e_base), "merge_gain": acc(m_pol) - acc(m_base),
                    "exact_acc": acc(e_pol), "merge_acc": acc(m_pol)}
            ids = sorted(set(e_pol) & set(m_pol))
            w, l, p = rr.mcnemar([e_pol[i] for i in ids], [m_pol[i] for i in ids])
            row |= {"exact_wins": w, "merge_wins": l, "p": p}
            pairs.append((e_pol, m_pol))
        out[k] = row
        print(f"seed {k}: " + ", ".join(f"{key} {val:.3f}" if isinstance(val, float) else f"{key} {val}" for key, val in row.items()))
    done = [r for r in out.values() if "merge_gain" in r]
    early_ok = all(r["sync_gain_first10"] is not None and r["sync_gain_first10"] < 0.5 for r in out.values() if r["sync_gain_first10"] is not None)
    rise_ok = all(r["sync_gain_191_200"] is not None and r["sync_gain_191_200"] > r["sync_gain_first10"]
                  for r in out.values() if r["sync_gain_191_200"] is not None)
    v = {"i": early_ok and rise_ok, "pairs": len(done)}
    if done:
        smaller = sum(r["merge_gain"] < r["exact_gain"] for r in done)
        need = 2 if len(done) == 3 else len(done)
        v["ii"] = {"smaller": smaller, "of": len(done), "met": smaller >= need}
        rows = [(e[i], mm[i]) for e, mm in pairs for i in sorted(set(e) & set(mm))]
        rng = random.Random(0)
        diffs = []
        for _ in range(10000):
            s = [rows[rng.randrange(len(rows))] for _ in rows]
            diffs.append(100 * (sum(x for x, _ in s) - sum(y for _, y in s)) / len(s))
        diffs.sort()
        d0 = 100 * (sum(x for x, _ in rows) - sum(y for _, y in rows)) / len(rows)
        v["ii"]["pooled_exact_minus_merge"] = [d0, diffs[250], diffs[9750]]
        print(f"(i) {'MET' if v['i'] else 'NOT MET'}; (ii) merge-sync gain smaller on {smaller} of {len(done)} pairs -> "
              f"{'MET' if v['ii']['met'] else 'NOT MET'}; pooled exact - merge = {d0:+.1f} points [{diffs[250]:+.1f}, {diffs[9750]:+.1f}]")
    if a.json:
        Path(a.json).write_text(json.dumps({"seeds": out, "verdict": v}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
