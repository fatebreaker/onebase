"""Data for CODE-Q (code RL on strong models), from the downloaded copies in data/code_dl (untrusted: read as data only):
  train  DeepCoder's PrimeIntellect subset (agentica-org/DeepCoder-Preview-Dataset, MIT; stdin/stdout tests): problems
         with 1-64 tests and at most 1 MB of test input whose first reference solution passes its first 8 tests under
         bench/code_exec.py (problems that accept several answers fail exact checking and are dropped), the first 8 tests
         kept -> data/code_dl/code_train.jsonl {id, q, tests}
  eval   LiveCodeBench code_generation_lite releases v5 and v6 (342 problems, 2024-10 to 2025-04), public and private
         tests -> data/code_dl/lcb_eval.jsonl {id, q, tests, starter, testtype, date, difficulty}
Prompts follow LiveCodeBench's: the problem, then for stdin problems an instruction to read stdin and write stdout, for
functional problems the starter code; the answer in a ```python block.
    python bench/code_prep.py --workers 24"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import code_exec as ce  # noqa: E402

D = Path("$SCRATCH/data/code_dl")
STDIN_SFX = ("\n\nRead the inputs from stdin, solve the problem and write the answer to stdout (do not directly test on "
             "the sample inputs). Enclose your code within delimiters as follows: ```python\n# YOUR CODE HERE\n```")
FUNC_SFX = ("\n\nYou will use the following starter code to write the solution to the problem and enclose your code within "
            "delimiters.\n```python\n{starter}\n```")


def _check(item):
    i, problem, tests, sol = item
    ok = ce.grade_tests(sol, tests, max_tests=8)
    return i, ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    import pandas as pd
    # eval: LiveCodeBench v5 + v6
    ev = []
    for f in ("test5.jsonl", "test6.jsonl"):
        for x in (D / "lcb" / f).read_text().splitlines():
            r = json.loads(x)
            tests = ce.load_lcb_tests(r)
            tt = tests[0]["testtype"]
            q = r["question_content"] + (FUNC_SFX.format(starter=r["starter_code"]) if tt == "functional" else STDIN_SFX)
            ev.append({"id": f"lcb-{r['question_id']}", "q": q, "tests": tests, "starter": r["starter_code"], "testtype": tt,
                       "date": r["contest_date"][:10], "difficulty": r["difficulty"]})
    (D / "lcb_eval.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ev))
    print("eval", len(ev), flush=True)
    # train: PrimeIntellect, reference-verified
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(D / "deepcoder/primeintellect/*.parquet")))])
    items = []
    for i, r in enumerate(df.itertuples()):
        tests = json.loads(r.tests) if isinstance(r.tests, str) else list(r.tests)
        tests = [{"input": t["input"], "output": t["output"], "testtype": "stdin"} for t in tests if t.get("type") == "stdin_stdout"]
        sols = json.loads(r.solutions) if isinstance(r.solutions, str) else list(r.solutions)
        if not (1 <= len(tests) <= 64) or not sols or sum(len(t["input"]) for t in tests) > 1 << 20:
            continue
        items.append((i, r.problem, tests, sols[0]))
    print("candidates", len(items), flush=True)
    keep = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, ok in ex.map(_check, items, chunksize=8):
            keep[i] = ok
    tr = []
    for i, problem, tests, _ in items:
        if keep.get(i):
            q = problem if "```python" in problem else problem + STDIN_SFX
            tr.append({"id": f"pi-{i}", "q": q, "tests": tests[:8]})
    (D / "code_train.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tr))
    print("train", len(tr), "of", len(items), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
