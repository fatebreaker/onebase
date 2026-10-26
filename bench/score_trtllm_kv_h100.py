"""TRTLLM-KV-H100 (docs/REGISTRY.md, registered 2026-10-25; this scorer written before any data; mock reviews R1 v34 M5, R3 v35):
TensorRT-LLM's fastest 32-adapter setting on one H100 at KV-cache shares 0.6 and 0.7 (mlsys jobs/pod_runs/trtllm_kv_h100_pod.sh ->
results/trtllm_kv_h100.jsonl, three runs each) beside TRTLLM-H100's share 0.5 (results/trtllm_overhead_h100.jsonl), and the
plugin's decode path at 64:32 in the same job (results/e2e_trtllm_kv_h100.jsonl, three measured starts).
  (i) the plugin's W2 above TensorRT-LLM's best mean W2 over the shares 0.5, 0.6 and 0.7
Descriptive: each share's KV-cache tokens (from the run logs) and W2; a share whose executor did not start is reported as such.
    python3 bench/score_trtllm_kv_h100.py [--json results/trtllm_kv_h100_verdict.json]"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
RES = LB / "results"


def rows(f: Path) -> list[dict]:
    return [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []


def kv_tokens(log_glob: str) -> list[int]:
    out = []
    for f in sorted((LB / "logs").glob(log_glob)):
        m = re.findall(r"Allocated [0-9.]+ GiB for max tokens in paged KV cache \((\d+)\)", f.read_text(errors="replace"))
        if m:
            out.append(int(m[-1]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    trt: dict = {}
    for r in rows(RES / "trtllm_overhead_h100.jsonl"):
        if r["label"] == "fp8mo_trt_r64n32_overlap" and "W2_gen_tok_s" in r:
            trt.setdefault("0.5", []).append(r["W2_gen_tok_s"])
    for r in rows(RES / "trtllm_kv_h100.jsonl"):
        m = re.fullmatch(r"fp8mo_trt_r64n32_overlap_kv([0-9.]+)", r["label"])
        if m and "W2_gen_tok_s" in r:
            trt.setdefault(m.group(1), []).append(r["W2_gen_tok_s"])
    shares = {s: {"W2_mean": sum(v) / len(v), "runs": len(v), "W2_all": v} for s, v in trt.items()}
    shares.get("0.5", {})["kv_tokens"] = kv_tokens("trtllmh100/trt_fp8mo_trt_r64n32_overlap_*.log")
    for s in ("0.6", "0.7"):
        shares.setdefault(s, {"runs": 0})["kv_tokens"] = kv_tokens(f"trtllmkvh100/trt_fp8mo_trt_r64n32_overlap_kv{s}_*.log")
    plug = [r["W2_gen_tok_s"] for r in rows(RES / "e2e_trtllm_kv_h100.jsonl")
            if r["label"].rpartition("#")[2] in ("1", "2", "3") and r["label"].startswith("plugin-pdl+r64:n32")]
    out: dict = {"trtllm": shares, "plugin_W2": plug}
    ok = [s for s, v in shares.items() if v.get("runs")]
    if len(plug) == 3 and ok and all(shares.get(s, {}).get("runs", 0) >= 3 or s == "0.5" for s in ("0.6", "0.7")
                                    if shares.get(s, {}).get("runs", 0)):
        best = max(ok, key=lambda s: shares[s]["W2_mean"])
        p = sum(plug) / 3
        out["i"] = {"met": p > shares[best]["W2_mean"], "plugin": p, "trt_best_share": best, "trt_best": shares[best]["W2_mean"],
                    "ratio": p / shares[best]["W2_mean"]}
    out["complete"] = "i" in out and len(plug) == 3
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
