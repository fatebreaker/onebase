"""CO-SERVE-MATCHED (docs/REGISTRY.md, registered before any data; this scorer written before any data): Figure 1b's thesis
configuration end to end on ONE L40S -- one engine, one FP8 copy, the matched W8A16 engine serving the tenants and the rollouts, and
RL-INTEGRATED's recipe training WITHOUT any correction (arm S8CM9L, mlsys jobs/pod_runs/coserve_matched_pod.sh), seeds 0-2, against
S8MS9L seeds 0-4 (the same recipe, engine and copy, RL alone). Per seed, from $SCRATCH/coserve/
run_cmatched_s<k> and the run's log and evaluations (CO-SERVE-MATH's definitions: scored chunks are the Y chunks inside the RL segments):
  (i)   every S8CM9L run completes its 200 steps, and at least 99.9% of the scored tenant requests complete
  (ii)  final accuracy (OlympiadBench + Minerva pooled, each policy on its own FP8 engine), S8CM9L minus S8MS9L: Welch 95% interval
        within +-5 points (else inconclusive)
  (iii) the tenants beside RL, pooled over every run's scored chunks: median time per output token at most 100 ms and p99 time to
        first token at most 5 s
  (iv)  co-located GPU-hours (one GPU for the RL phase, mean over runs) at most 0.9x separate deployments (2x S8MS9L's mean)
  (v)   serving tenants leaves the matched arithmetic intact: S8CM9L's mean trainer-engine mismatch over steps >= 3 at most 1.25x
        S8MS9L's (both uncorrected)
Descriptive: (iv) under consolidated accounting (mock review R3 v23: a GPU serving 1 request/s alone is mostly idle): separate =
S8MS9L's GPU-hours plus the tenants' share of a consolidated serving GPU, their output tokens served during the RL phase divided by
the same engine's tenant throughput with 64 requests at once (the W chunk); tenants alone (A) against beside RL (Y); RL step and
rollout time against S8MS9L's. The same consolidated accounting is reported for CO-SERVE-MATH (S8CS9L against S8S9L; descriptive,
amendment -CONSOLIDATED). Stopping: runs trained and evaluated by 2026-10-28 12:00 UTC.
    python3 bench/score_coserve_matched.py [--json results/std_bench/coserve_matched_verdict.json]"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
import score_rlqmath as Q  # noqa: E402
from score_coserve import peak_mem, phases, requests  # noqa: E402
from score_coserve_math import pooled, scored, segments  # noqa: E402
from score_rlmulti import welch  # noqa: E402

RUNS = Path("$SCRATCH/coserve")
for _a in ("S8CM9L", "S8MS9L", "S8CS9L", "S8S9L"):
    Q.EV.setdefault(_a, ("fp8eng_base", "fp8eng_exact"))


def tenant_tokens(d: Path, names: list) -> int:
    n = 0
    for k in names:
        f = d / f"{k}.json"
        if f.exists():
            n += sum(json.loads(f.read_text()).get("output_lens", []))
    return n


def arm(name: str, run_prefix: str, seeds) -> dict:
    runs, ys, as_ = {}, [], []
    for s in seeds:
        d = RUNS / f"{run_prefix}{s}"
        rs = Q.rows(name, s)
        x = Q.summary(name, s)
        rec = {"steps": rs[-1]["step"] if rs else 0}
        if (d / "phases.jsonl").exists():
            ph = phases(d)
            segs = segments(d)
            sc = scored(ph, segs)
            an = sorted((k for k in ph if k.startswith("A_") and "end" in ph[k]), key=lambda k: int(k[2:]))
            rec.update({"scored_chunks": len(sc), "Y": requests(d, sc), "A": requests(d, an), "segments": len(segs),
                        "mem": peak_mem(d, ph) if "A_1" in ph and "end" in ph.get("rl_y", {}) else None})
            if segs and segs[-1][2]:
                rec["rl_phase_h"] = sum(hi - lo for lo, hi, _ in segs) / 3600
            w = requests(d, ["W"]) if (d / "W.json").exists() else {}
            if w.get("tok_s") and w["tok_s"] == w["tok_s"]:
                rec["tenant_consolidated_h"] = tenant_tokens(d, sc) / w["tok_s"] / 3600
                rec["W_tok_s"] = w["tok_s"]
            ys.append((d, sc))
            as_.append((d, an))
            kv = re.findall(r"GPU KV cache size: ([0-9,]+) tokens", (d / "server.log").read_text(errors="replace")) if (d / "server.log").exists() else []
            rec["kv_tokens"] = int(kv[-1].replace(",", "")) if kv else None
        if rs:
            late = [r for r in rs if r["step"] >= 3]
            rec.update({"t_step": st.fmean(r["t"] for r in late) if late else None,
                        "t_gen": st.fmean(r["t_gen"] for r in late) if late else None,
                        "mismatch": st.fmean(r["mismatch"] for r in late if r.get("mismatch") is not None) if late else None})
        if x and x.get("hard_final") is not None:
            rec.update({"hard_final": x["hard_final"], "hard_base": x.get("hard_base")})
        runs[s] = rec
    return {"runs": runs, "ys": ys, "as": as_}


def refs_of(name: str, seeds) -> list:
    out = []
    for s in seeds:
        x = Q.summary(name, s)
        if x and x.get("hard_final") is not None:
            rs = Q.rows(name, s)
            late = [r for r in rs if r["step"] >= 3]
            out.append({"seed": s, "hard_final": x["hard_final"], "gpu_hours": x.get("gpu_hours"),
                        "t_step": st.fmean(r["t"] for r in late) if late else None,
                        "t_gen": st.fmean(r["t_gen"] for r in late) if late else None,
                        "mismatch": st.fmean(r["mismatch"] for r in late if r.get("mismatch") is not None) if late else None})
    return out


def consolidated(done: list, refs: list) -> dict | None:
    if not done or not refs or not all(r.get("tenant_consolidated_h") is not None for r in done):
        return None
    co = st.fmean(r["rl_phase_h"] for r in done)
    sep = st.fmean(r["gpu_hours"] for r in refs) + st.fmean(r["tenant_consolidated_h"] for r in done)
    return {"colocated_h": co, "separate_consolidated_h": sep, "ratio": co / sep,
            "tenant_consolidated_h": st.fmean(r["tenant_consolidated_h"] for r in done)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    seeds, ref_seeds = (0, 1, 2), (0, 1, 2, 3, 4)
    A = arm("S8CM9L", "run_cmatched_s", seeds)
    runs, refs = A["runs"], refs_of("S8MS9L", ref_seeds)
    done = [r for r in runs.values() if r.get("hard_final") is not None and r.get("rl_phase_h")]
    v: dict = {"n": len(done), "n_refs": len(refs)}
    if len(done) == len(seeds) and len(refs) == len(ref_seeds):
        yv = pooled(A["ys"], "Y")
        v["i"] = {"steps": [r["steps"] for r in runs.values()], "share_completed": yv["share_completed"],
                  "met": all(r["steps"] >= 200 for r in runs.values()) and yv["share_completed"] >= 0.999}
        v["ii"] = welch([r["hard_final"] for r in done], [r["hard_final"] for r in refs])
        v["iii"] = {"tpot_p50": yv["tpot_p50"], "ttft_p99": yv["ttft_p99"], "met": yv["tpot_p50"] <= 100 and yv["ttft_p99"] <= 5000}
        co, sep = st.fmean(r["rl_phase_h"] for r in done), 2 * st.fmean(r["gpu_hours"] for r in refs)
        v["iv"] = {"colocated_h": co, "separate_h": sep, "ratio": co / sep, "met": co <= 0.9 * sep}
        mm, mr = st.fmean(r["mismatch"] for r in done), st.fmean(r["mismatch"] for r in refs)
        v["v"] = {"mismatch": mm, "ref_mismatch": mr, "ratio": mm / mr, "met": mm <= 1.25 * mr}
        v["descriptive"] = {"A": pooled(A["as"], "A"), "Y": yv, "iv_consolidated": consolidated(done, refs),
                            "t_step": {"S8CM9L": st.fmean(r["t_step"] for r in done), "S8MS9L": st.fmean(r["t_step"] for r in refs)},
                            "t_gen": {"S8CM9L": st.fmean(r["t_gen"] for r in done), "S8MS9L": st.fmean(r["t_gen"] for r in refs)}}
    else:
        v["status"] = "incomplete"
    B = arm("S8CS9L", "run_math_s", (0, 1, 2))               # CO-SERVE-MATH, amendment -CONSOLIDATED (descriptive)
    bdone = [r for r in B["runs"].values() if r.get("hard_final") is not None and r.get("rl_phase_h")]
    out = {"runs": runs, "refs": refs, "verdict": v,
           "coserve_math_consolidated": consolidated(bdone, refs_of("S8S9L", (0, 1, 2))) if len(bdone) == 3 else None}
    print(json.dumps(out, indent=1, default=float))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=float) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
