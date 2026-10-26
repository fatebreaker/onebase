"""bench/rl_rollout_bench3.py on the matched-precision engine (RL-MATCHED-KERNELS): vLLM's online FP8 layers compute W8A16 through our
kernel (lowbit_lora/w16_engine.py), the engine in this process. Arguments as bench/rl_rollout_bench3.py (rollout throughput (fixed work)); use its --fp8 /
--engine fp8 so the layers are FP8, with LOWBIT_FUSED=0.
    VLLM_ENABLE_V1_MULTIPROCESSING=0 LOWBIT_FUSED=0 python bench/rl_rollout_w16.py <bench/rl_rollout_bench3.py arguments>"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lowbit_lora import w16_engine  # noqa: E402

w16_engine.install()
import rl_rollout_bench3  # noqa: E402

if __name__ == "__main__":
    rc = rl_rollout_bench3.main()
    print("[lowbit] w16 engine layers:", w16_engine.stats, flush=True)
    raise SystemExit(rc)
