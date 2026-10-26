"""CO-SERVE-MATH (docs/REGISTRY.md, registered before any data; written before any data): the 9B math RL recipe co-served with tenants
on ONE L40S, one FP8 copy for the engine, the trainer and every tenant (arm S8CS9L, mlsys jobs/pod_runs/coserve_math_pod.sh,
train/grpo_q18.py), seeds 0-2, against S8S9L seeds 0-2 (the same recipe and copy, RL alone; RL-SHARED-9B-FULL). Per seed, from
$SCRATCH/coserve/run_math_s<k> and the run's log and evaluations:
  (i)   every S8CS9L run completes its 200 steps, and at least 99.9% of the scored tenant requests complete (scored: the Y chunks
        that start and end inside the RL phase)
  (ii)  final accuracy (OlympiadBench + Minerva pooled, each policy on its own FP8 engine), S8CS9L minus S8S9L: Welch 95% interval
        within +-5 points (else inconclusive)
  (iii) the tenants beside RL, pooled over every run's scored chunks: median time per output token at most 100 ms and p99 time to
        first token at most 5 s
  (iv)  GPU-hours: co-located (one GPU for the RL phase, mean over runs) at most 0.9x separate deployments (one GPU for RL alone,
        S8S9L's mean GPU-hours, and one serving the tenants meanwhile: 2x)
Descriptive: tenants alone (A) against beside RL (Y): median and p99 TTFT and TPOT, tenant output tokens/s; RL step and rollout time
against S8S9L's; the trainer-engine mismatch; peak GPU memory per phase; the engine's KV cache. Stopping: runs trained and evaluated
by 2026-10-27 12:00 UTC.
Amendment -RESUME (10-23, before any resumed data; lent pods are preempted): a run restarted from its checkpoints has several RL
segments (each rl_y start in phases.jsonl opens one); a segment cut by a preemption ends at its last logged event. Scored chunks are
the Y chunks inside any segment; (iv) counts every segment's wall time, so steps a restart repeats count twice (conservative).
    python3 bench/score_coserve_math.py [--json results/std_bench/coserve_math_verdict.json]"""
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
from score_rlmulti import welch  # noqa: E402

RUNS = Path("$SCRATCH/coserve")
SEEDS = (0, 1, 2)
for _a in ("S8CS9L", "S8S9L"):
    Q.EV.setdefault(_a, ("fp8eng_base", "fp8eng_exact"))


def segments(d: Path) -> list:
    """(start, end, closed) per RL segment, in order (amendment -RESUME): an rl_y start opens one; an rl_y end closes it; a segment
    cut by a preemption ends at the last event logged before the next engine start."""
    out, cur, last = [], None, None
    for line in (d / "phases.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["phase"] == "engine" and r["event"] == "start" and cur is not None:
            out.append((cur, last, False))
            cur = None
        if r["phase"] == "rl_y" and r["event"] == "start":
            cur = r["t"]
        elif r["phase"] == "rl_y" and r["event"] == "end" and cur is not None:
            out.append((cur, r["t"], True))
            cur = None
        last = r["t"]
    if cur is not None:
        out.append((cur, last, False))
    return out


def scored(ph: dict, segs: list) -> list:
    """Y chunks that start and end inside an RL segment (the run's last segment closed: the run finished)."""
    if not segs or not segs[-1][2]:
        return []
    return sorted((k for k, v in ph.items() if k.startswith("Y_") and "end" in v
                   and any(lo <= v["start"] and v["end"] <= hi for lo, hi, _ in segs)), key=lambda k: int(k[2:]))


def pooled(d_list: list, which: str) -> dict:
    """Per-request TTFT/TPOT pooled over several runs' chunks (requests() pools within one directory)."""
    tt, tp, sent, done = [], [], 0, 0
    for d, names in d_list:
        for n in names:
            f = d / f"{n}.json"
            if not f.exists():
                sent += 1
                continue
            r = json.loads(f.read_text())
            errs = r.get("errors", [])
            sent += r.get("num_prompts", len(errs))
            done += r.get("completed", 0)
            for t, itl, e in zip(r.get("ttfts", []), r.get("itls", []), errs or [""] * len(r.get("ttfts", []))):
                if e:
                    continue
                tt.append(1e3 * t)
                if itl:
                    tp.append(1e3 * sum(itl) / len(itl))
    from score_coserve import pct
    return {"which": which, "sent": sent, "completed": done, "share_completed": done / sent if sent else float("nan"),
            "ttft_p50": pct(tt, 0.5), "ttft_p99": pct(tt, 0.99), "tpot_p50": pct(tp, 0.5), "tpot_p99": pct(tp, 0.99)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    runs, refs, ys, as_ = {}, [], [], []
    for s in SEEDS:
        d = RUNS / f"run_math_s{s}"
        rs = Q.rows("S8CS9L", s)
        x = Q.summary("S8CS9L", s)
        rec = {"steps": rs[-1]["step"] if rs else 0}
        if (d / "phases.jsonl").exists():
            ph = phases(d)
            segs = segments(d)
            sc = scored(ph, segs)
            an = sorted((k for k in ph if k.startswith("A_") and "end" in ph[k]), key=lambda k: int(k[2:]))
            rec.update({"scored_chunks": len(sc), "Y": requests(d, sc), "A": requests(d, an), "mem": peak_mem(d, ph) if "A_1" in ph and "end" in ph.get("rl_y", {}) else None})
            rec["segments"] = len(segs)
            if segs and segs[-1][2]:
                rec["rl_phase_h"] = sum(hi - lo for lo, hi, _ in segs) / 3600
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
    for s in SEEDS:
        x = Q.summary("S8S9L", s)
        if x and x.get("hard_final") is not None:
            rs = Q.rows("S8S9L", s)
            late = [r for r in rs if r["step"] >= 3]
            refs.append({"seed": s, "hard_final": x["hard_final"], "gpu_hours": x.get("gpu_hours"),
                         "t_step": st.fmean(r["t"] for r in late) if late else None,
                         "t_gen": st.fmean(r["t_gen"] for r in late) if late else None})
    done = [r for r in runs.values() if r.get("hard_final") is not None and r.get("rl_phase_h")]
    v: dict = {"n": len(done)}
    if len(done) == len(SEEDS) and len(refs) == len(SEEDS):
        yv = pooled(ys, "Y")
        v["i"] = {"steps": [r["steps"] for r in runs.values()], "share_completed": yv["share_completed"],
                  "met": all(r["steps"] >= 200 for r in runs.values()) and yv["share_completed"] >= 0.999}
        v["ii"] = welch([r["hard_final"] for r in done], [r["hard_final"] for r in refs])
        v["iii"] = {"tpot_p50": yv["tpot_p50"], "ttft_p99": yv["ttft_p99"],
                    "met": yv["tpot_p50"] <= 100 and yv["ttft_p99"] <= 5000}
        co = st.fmean(r["rl_phase_h"] for r in done)
        sep = 2 * st.fmean(r["gpu_hours"] for r in refs)
        v["iv"] = {"colocated_h": co, "separate_h": sep, "ratio": co / sep, "met": co <= 0.9 * sep}
        v["descriptive"] = {"A": pooled(as_, "A"), "Y": yv,
                            "t_step": {"S8CS9L": st.fmean(r["t_step"] for r in done), "S8S9L": st.fmean(r["t_step"] for r in refs)},
                            "t_gen": {"S8CS9L": st.fmean(r["t_gen"] for r in done), "S8S9L": st.fmean(r["t_gen"] for r in refs)}}
    else:
        v["status"] = "incomplete"
    out = {"runs": runs, "refs": refs, "verdict": v}
    print(json.dumps(out, indent=1, default=float))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=float) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
