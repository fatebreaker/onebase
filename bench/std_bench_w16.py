"""bench/std_bench5.py on the matched-precision engine (RL-MATCHED): with --engine fp8, vLLM's online FP8 layers compute W8A16
through our kernel (lowbit_lora/w16_engine.py), the engine in this process; LOWBIT_FUSED=0. Arguments as bench/std_bench5.py.
    VLLM_ENABLE_V1_MULTIPROCESSING=0 LOWBIT_FUSED=0 python bench/std_bench_w16.py --engine fp8 <std_bench5 arguments>"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lowbit_lora import w16_engine  # noqa: E402

w16_engine.install()
import std_bench5  # noqa: E402

if __name__ == "__main__":
    rc = std_bench5.main()
    print("[lowbit] w16 engine layers:", w16_engine.stats, flush=True)
    raise SystemExit(rc)
