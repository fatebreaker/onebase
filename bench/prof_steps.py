"""PROFILE-STEP-H100 (docs/REGISTRY.md, registered 2026-10-25; written before any data): where each engine's batch-1 decode time
goes. Reads nsys traces exported to SQLite (nsys export --type sqlite) of one W1 run per engine (bench/*_nvtx.py: the two timed
generations as NVTX ranges W1_1 = 512-token prefill + 1 token and W1_257 = the same + 256 more), and per pair of ranges reports,
per decoded token ((W1_257 - W1_1) / 256, as TPOT is defined):
  wall_ms     the ranges' wall time (TPOT under the profiler)
  busy_ms     GPU time with at least one kernel, memcpy or memset running (the union of their intervals, any process)
  idle_ms     wall_ms - busy_ms: time the GPU waits on the host
  kernels     kernel launches (graph nodes included only with --cuda-graph-trace=node; with =graph, those outside graphs)
  graph_launches  CUDA graph executions (with --cuda-graph-trace=graph; their time counts as busy)
  lora_ms     busy time of kernels whose names match LORA_RE (vLLM's lora_shrink/expand, our fused and PDL kernels, SGLang's
              csgmv/sgemm LoRA kernels, TensorRT-LLM's LoRA grouped GEMMs), a lower bound on the LoRA path's GPU time
and the 15 kernels with the most time per token. Medians over the run's three pairs.
    python3 bench/prof_steps.py --label <label> --sqlite <trace.sqlite> --out results/prof_steps_h100.jsonl"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics as st
from pathlib import Path

LORA_RE = re.compile(r"lora|sgmv|bgmv|shrink|expand|punica|segment_gemm|sgemm_lora|grouped|lowbit|_k2|pdl", re.I)


def union(iv: list[tuple[int, int]]) -> int:
    tot, cur_s, cur_e = 0, None, None
    for s, e in sorted(iv):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot


def tables(con) -> set:
    return {r[0] for r in con.execute("select name from sqlite_master where type='table'")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--sqlite", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    con = sqlite3.connect(a.sqlite)
    have = tables(con)
    names = dict(con.execute("select id, value from StringIds")) if "StringIds" in have else {}
    nv = []
    for s, e, text, tid in con.execute("select start, end, text, textId from NVTX_EVENTS where end is not null"):
        t = text if text is not None else names.get(tid)
        if t in ("W1_1", "W1_257"):
            nv.append((s, e, t))
    nv.sort()
    pairs, pend = [], None
    for s, e, t in nv:
        if t == "W1_1":
            pend = (s, e)
        elif pend is not None:
            pairs.append((pend, (s, e)))
            pend = None
    if "CUPTI_ACTIVITY_KIND_KERNEL" not in have:      # no CUDA activity in the trace (e.g. a forked, untraced engine process)
        print(json.dumps({"label": a.label, "sqlite": a.sqlite, "no_cuda_trace": True}))
        with open(a.out, "a") as fh:
            fh.write(json.dumps({"label": a.label, "sqlite": a.sqlite, "no_cuda_trace": True}) + "\n")
        return 1
    kern = [(s, e, names.get(n, str(n))) for s, e, n in
            con.execute("select start, end, coalesce(shortName, demangledName) from CUPTI_ACTIVITY_KIND_KERNEL")]
    other = []
    for t in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        if t in have:
            other += [(s, e) for s, e in con.execute(f"select start, end from {t}")]
    graphs = []   # --cuda-graph-trace=graph: each graph launch's execution as one range (whatever the table's name in this nsys)
    for t in sorted(have):
        if "GRAPH" in t.upper() and {"start", "end"} <= {r[1] for r in con.execute(f"pragma table_info({t})")}:
            graphs += [(s, e) for s, e in con.execute(f"select start, end from {t} where end is not null")]
    other += graphs
    kern.sort()
    starts = [k[0] for k in kern]

    import bisect

    def window(s: int, e: int) -> dict:
        i = bisect.bisect_left(starts, s - 10_000_000)       # kernels starting up to 10 ms before the window may overlap it
        ks = [(max(ks_, s), min(ke, e), n) for ks_, ke, n in kern[i:bisect.bisect_right(starts, e)] if ke > s and ks_ < e]
        os_ = [(max(x, s), min(y, e)) for x, y in other if y > s and x < e]
        ng = sum(1 for x, y in graphs if y > s and x < e)
        by: dict = {}
        for x, y, n in ks:
            by[n] = by.get(n, 0) + (y - x)
        return {"wall": e - s, "busy": union([(x, y) for x, y, _ in ks] + os_), "n": len(ks), "g": ng,
                "lora": union([(x, y) for x, y, n in ks if LORA_RE.search(n)]), "by": by}

    per = []
    for (s1, e1), (s2, e2) in pairs:
        w1, w2 = window(s1, e1), window(s2, e2)
        d = {k: (w2[k] - w1[k]) / 256 for k in ("wall", "busy", "n", "g", "lora")}
        d["by"] = {n: (w2["by"].get(n, 0) - w1["by"].get(n, 0)) / 256 for n in set(w2["by"]) | set(w1["by"])}
        per.append(d)
    rec: dict = {"label": a.label, "sqlite": a.sqlite, "pairs": len(per)}
    if per:
        med = {k: st.median(p[k] for p in per) for k in ("wall", "busy", "n", "g", "lora")}
        rec.update({"wall_ms": med["wall"] / 1e6, "busy_ms": med["busy"] / 1e6, "idle_ms": (med["wall"] - med["busy"]) / 1e6,
                    "kernels": med["n"], "graph_launches": med["g"], "lora_ms": med["lora"] / 1e6,
                    "per_pair": [{k: (p[k] / 1e6 if k != "n" else p[k]) for k in ("wall", "busy", "n", "lora")} for p in per]})
        tot: dict = {}
        for p in per:
            for n, v in p["by"].items():
                tot[n] = tot.get(n, 0) + v / len(per)
        rec["top_kernels_us"] = [[n[:120], round(v / 1e3, 2)] for n, v in sorted(tot.items(), key=lambda x: -x[1])[:15]]
    print(json.dumps({k: v for k, v in rec.items() if k != "per_pair"})[:1500])
    with open(a.out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return 0 if per else 1


if __name__ == "__main__":
    raise SystemExit(main())
