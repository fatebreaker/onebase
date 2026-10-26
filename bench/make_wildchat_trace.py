"""A realistic serving trace from WildChat-1M (Zhao et al., 2024; /datasets/ai/wildchat): the first user turn and the first assistant
reply of non-toxic, unredacted conversations, in vLLM's ShareGPT format (vllm bench serve --dataset-name sharegpt), so the
benchmark draws real prompt and output lengths (it drops pairs past 1,024 prompt or 2,048 total tokens). Shard 0, first N rows
that qualify, in timestamp order.
    python bench/make_wildchat_trace.py [--n 4000] -> results/traces/wildchat_sharegpt_<n>.json"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc

LB = Path(__file__).resolve().parents[1]
SHARD = Path("/datasets/ai/wildchat/allenai___wild_chat-1_m/default/0.0.0/8016827cee8c947143c498df467af52753e5181b/"
             "wild_chat-1_m-train-00000-of-00019.arrow")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    a = ap.parse_args()
    with pa.memory_map(str(SHARD)) as src:
        try:
            t = ipc.open_stream(src).read_all()
        except pa.ArrowInvalid:
            t = ipc.open_file(src).read_all()
    t = t.select(["conversation_hash", "timestamp", "conversation", "toxic", "redacted"])
    rows = sorted(t.to_pylist(), key=lambda r: str(r["timestamp"]))
    out = []
    for r in rows:
        if r["toxic"] or r["redacted"]:
            continue
        conv = r["conversation"]
        if isinstance(conv, str):
            try:
                conv = ast.literal_eval(conv)
            except (ValueError, SyntaxError):
                continue
        if len(conv) < 2 or conv[0].get("role") != "user" or conv[1].get("role") != "assistant":
            continue
        u, g = conv[0].get("content") or "", conv[1].get("content") or ""
        if not u.strip() or not g.strip():
            continue
        out.append({"id": r["conversation_hash"], "conversations": [{"from": "human", "value": u}, {"from": "gpt", "value": g}]})
        if len(out) >= a.n:
            break
    dst = LB / f"results/traces/wildchat_sharegpt_{a.n}.json"
    dst.write_text(json.dumps(out))
    print(f"{len(out)} conversations -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
