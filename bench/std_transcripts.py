"""STD-BENCH responses -> per-problem transcript JSONs for mlsys.fidelity (SR-MERGE-STD): the bf16 + adapter variant's
own greedy responses (results/std_bench/mistral_<w>/bf16_exact/<bench>.jsonl) with the benchmark's prompt, as
{"id", "transcript": [{"role": "user", ...}, {"role": "assistant", ...}]}, one file per problem.
    venv_vllm/bin/python -I bench/std_transcripts.py --which math --bench gsm8k --out <dir>"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", required=True, choices=("math", "code"))
    ap.add_argument("--model", default="mistral", choices=("mistral", "llama"))
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(LB / "bench"))
    import std_bench
    q = {it["id"]: it["q"] for it in std_bench.load_bench(a.bench)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for line in (LB / f"results/std_bench/{a.model}_{a.which}/bf16_exact/{a.bench}.jsonl").read_text().splitlines():
        r = json.loads(line)
        rec = {"id": r["id"], "transcript": [{"role": "user", "content": q[r["id"]]},
                                             {"role": "assistant", "content": r["response"]}]}
        (out / f"{r['id'].replace('/', '_')}.json").write_text(json.dumps(rec))
        n += 1
    print(f"{n} transcripts -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
