"""EVAL-AVG8-H100ALL extension -CS (docs/REGISTRY.md, registered 2026-10-24 18:17 UTC before any avg@8 data of its policies;
descriptive, no predictions; this readout written after the first data, implementing only the registered comparisons): the 10-24
tests' 9B policies on the GH200's common bf16 engine with avg@8 (results/std_bench_avg8_gh200/<run>/bf16_exact/{olympiad,minerva}).
  CO-SERVE-MATH     S8CS9L seeds 0-2 minus S8S9L seeds 0-2
  MULTI-USER-RL     S8MU2H seeds 0-3 minus S8MU1H seeds 0-2 (amendment -K3's ONE)
  CO-SERVE-MATCHED  S8CM9L seeds 0-2 minus S8MS9L seeds 0-2           (once trained and scored)
  RL-SHARE45        S8MS45 seeds 0-2 minus S8ML and minus S8MS9L, seeds 0-2 (once trained and scored)
Each a Welch 95% interval beside the test's registered own-engine verdict. Policies scored by 2026-10-25 12:00 UTC count
(EVAL-AVG8-H100ALL's stopping rule).
    python3 bench/score_avg8_cs.py [--json results/std_bench/avg8_cs_verdict.json]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
from score_avg8_onecopy import score  # noqa: E402  (the GH200 directory and the stopping rule)
from score_rlmulti import welch  # noqa: E402

PAIRS = (("CO-SERVE-MATH", "S8CS9L", (0, 1, 2), "S8S9L", (0, 1, 2)),
         ("MULTI-USER-RL", "S8MU2H", (0, 1, 2, 3), "S8MU1H", (0, 1, 2)),
         ("CO-SERVE-MATCHED", "S8CM9L", (0, 1, 2), "S8MS9L", (0, 1, 2)),
         ("RL-SHARE45 vs S8ML", "S8MS45", (0, 1, 2), "S8ML", (0, 1, 2)),
         ("RL-SHARE45 vs S8MS9L", "S8MS45", (0, 1, 2), "S8MS9L", (0, 1, 2)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    out: dict = {}
    for name, arm, sa, ref, sr in PAIRS:
        xs = {s: score(arm, s) for s in sa}
        rs = {s: score(ref, s) for s in sr}
        v = {"arm": arm, "ref": ref, "arm_scores": xs, "ref_scores": rs}
        x, r = [t for t in xs.values() if t is not None], [t for t in rs.values() if t is not None]
        if len(x) == len(sa) and len(r) == len(sr):
            w = welch(x, r)
            w.pop("verdict")                          # score_rlmulti's +-5 rule is not this extension's (descriptive)
            v.update(w, arm_mean=sum(x) / len(x), ref_mean=sum(r) / len(r))
        else:
            v["missing"] = [f"{arm}_s{s}" for s, t in xs.items() if t is None] + [f"{ref}_s{s}" for s, t in rs.items() if t is None]
        out[name] = v
        print(name, json.dumps({k: v[k] for k in v if k not in ("arm_scores", "ref_scores")}, default=str))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
