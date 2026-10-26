"""bench/rl_rollout_bench3.py on the matched-precision engine with the swap-AB W8A16 GEMM on Hopper (MATCHED-HOPPER-KERNEL):
lowbit_lora/w16_engine_swap.py with the swept tiles of results/w16_swap.jsonl (LOWBIT_SWAP_CFG overrides the file,
LOWBIT_SWAP_JOB picks a sweep job). Arguments as bench/rl_rollout_bench3.py; use its --fp8 so the layers are FP8.
    VLLM_ENABLE_V1_MULTIPROCESSING=0 LOWBIT_FUSED=0 python bench/rl_rollout_w16s.py <bench/rl_rollout_bench3.py arguments>"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lowbit_lora import w16_engine_swap  # noqa: E402

n = w16_engine_swap.load_cfg(os.environ.get("LOWBIT_SWAP_CFG", str(Path(__file__).resolve().parents[1] / "results/w16_swap.jsonl")),
                             job=os.environ.get("LOWBIT_SWAP_JOB") or None)
print(f"[lowbit] swap tiles loaded: {n}", flush=True)
w16_engine_swap.install()
import rl_rollout_bench3  # noqa: E402

if __name__ == "__main__":
    rc = rl_rollout_bench3.main()
    print("[lowbit] w16 swap engine layers:", w16_engine_swap.stats, flush=True)
    raise SystemExit(rc)
