"""CODE-Q-REGRADE (docs/REGISTRY.md): regrade saved LiveCodeBench responses (std_bench6.py's <variant>/lcb.jsonl) with fewer
programs at a time and a longer wall-clock limit, so the verdict depends on the program, not on the node's load. The
original grading ran 24 programs at a time with a 6 s wall-clock limit per test on pod nodes shared with training jobs, and
identical responses received different verdicts across runs. Here: --workers programs at a time (default 8), --timeout s
per test (default 30; the 10 s CPU limit of the sandbox is unchanged), every public and private test as before
(bench/code_exec.py). Writes <variant>/lcb_regrade.jsonl beside each lcb.jsonl (id, correct, k, testtype; no response).
    python3 bench/lcb_regrade.py results/std_bench/qwen9_code_s1 [...] [--workers 8] [--timeout 30]"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
import code_exec  # noqa: E402

DATA = Path("$SCRATCH/data")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="evaluation directories (results/std_bench/<run>)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=30.0)
    a = ap.parse_args()
    items = {it["id"]: it for it in map(json.loads, (DATA / "code_dl/lcb_eval.jsonl").read_text().splitlines())}
    for run in a.runs:
        for f in sorted(Path(run).glob("*/lcb.jsonl")):
            out = f.with_name("lcb_regrade.jsonl")
            if out.exists():
                print(f"{out}: done", flush=True)
                continue
            rows = [json.loads(x) for x in f.read_text().splitlines()]
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=a.workers) as ex:
                ok = list(ex.map(lambda r: code_exec.grade_tests(r["response"], items[r["id"]]["tests"], items[r["id"]]["starter"],
                                                                  timeout=a.timeout), rows))
            tmp = out.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text("".join(json.dumps({"id": r["id"], "correct": int(k), "k": 1, "testtype": r.get("testtype"),
                                               "orig": r["correct"]}) + "\n" for r, k in zip(rows, ok)))
            os.replace(tmp, out)
            flips = sum(int(k) != int(r["correct"]) for r, k in zip(rows, ok))
            print(json.dumps({"file": str(f), "n": len(rows), "acc_orig": round(100 * sum(r["correct"] for r in rows) / len(rows), 2),
                              "acc_regrade": round(100 * sum(ok) / len(ok), 2), "flips": flips, "s": round(time.time() - t0)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
