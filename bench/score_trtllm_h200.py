"""TRTLLM-H200 (docs/REGISTRY.md; this scorer written before any data): TensorRT-LLM's LoRA path against the plugin's decode path on one
H200, Qwen3-8B, FP8 base, rank-64 random adapters (mlsys jobs/pod_runs/trtllm_h200_pod.sh).
  (i)  batch 1, one adapter (W1): the plugin's time per output token (plugin-pdl+r64:n1, vLLM online FP8) below TensorRT-LLM's
       fastest configuration with the adapter on its ModelOpt FP8 checkpoint (the faster of LoRA on one stream and overlapped with
       the base; the run without CUDA graphs counts only if neither ran)
  (ii) each engine's adapter cost over its own base at batch 1: plugin-pdl+r64:n1 / vLLM nolora - 1 below TensorRT-LLM's fastest
       r64n1 / its nolora - 1 (the same checkpoint)
Descriptive: 32 tenants (W2 generated tokens/s) of both, at each engine's faster setting; no-LoRA TPOT and throughput of both; the
block-scaled checkpoint's rows; load times. Rows of the newest job; vLLM rows without the '#warm' suffix.
    python3 bench/score_trtllm_h200.py [--json results/trtllm_h200_verdict.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LB = Path(__file__).resolve().parents[1]


def rows(fn: str) -> list[dict]:
    f = LB / "results" / fn
    rs = [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []
    if not rs:
        return []
    job = max(str(r.get("job", "")) for r in rs)
    return [r for r in rs if str(r.get("job", "")) == job]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    T = {r["label"]: r for r in rows("trtllm_overhead_h200.jsonl")}
    V = {r["label"]: r for r in rows("e2e_trtllm_h200.jsonl") if not r["label"].endswith("#warm")}
    V = {k.rsplit("#", 1)[0]: r for k, r in V.items()}

    def best(labels: list[str], key: str, low: bool = True):
        got = [(T[l][key], l) for l in labels if l in T and isinstance(T[l].get(key), (int, float))]
        return (min(got) if low else max(got)) if got else (None, None)

    t1, t1l = best(["fp8mo_trt_r64n1", "fp8mo_trt_r64n1_overlap"], "W1_tpot_ms")
    if t1 is None:
        t1, t1l = best(["fp8mo_trt_r64n1_nograph"], "W1_tpot_ms")
    t0 = T.get("fp8mo_trt_nolora", {}).get("W1_tpot_ms")
    p1 = V.get("plugin-pdl+r64:n1", {}).get("W1_tpot_ms")
    p0 = V.get("nolora", {}).get("W1_tpot_ms")
    v: dict = {"trt_r64n1_ms": t1, "trt_config": t1l, "trt_nolora_ms": t0, "plugin_r64n1_ms": p1, "vllm_nolora_ms": p0}
    if t1 is not None and p1 is not None:
        v["i"] = {"met": p1 < t1, "ratio": round(p1 / t1, 3)}
    else:
        v["i"] = {"status": "missing"}
    if None not in (t1, t0, p1, p0):
        v["ii"] = {"met": p1 / p0 - 1 < t1 / t0 - 1, "plugin_cost": round(p1 / p0 - 1, 4), "trt_cost": round(t1 / t0 - 1, 4)}
    else:
        v["ii"] = {"status": "missing"}
    w_t, w_tl = best(["fp8mo_trt_r64n32", "fp8mo_trt_r64n32_overlap"], "W2_gen_tok_s", low=False)
    v["descriptive"] = {
        "W2_32_tenants": {"trt": w_t, "trt_config": w_tl, "plugin": V.get("plugin-pdl+r64:n32", {}).get("W2_gen_tok_s")},
        "W2_nolora": {"trt": T.get("fp8mo_trt_nolora", {}).get("W2_gen_tok_s"), "vllm": V.get("nolora", {}).get("W2_gen_tok_s")},
        "trt_rows": {l: {k: r.get(k) for k in ("W1_tpot_ms", "W2_gen_tok_s", "load_s", "quantization", "overlap", "cuda_graphs")}
                     for l, r in T.items()},
        "vllm_rows": {l: {k: r.get(k) for k in ("W1_tpot_ms", "W2_gen_tok_s")} for l, r in V.items()}}
    print(json.dumps(v, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(v, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
