"""KERNEL-CONTROLS amendment -FIG4 (docs/REGISTRY.md; post hoc, registered before its data): KERNEL-CONTROLS (b)'s occupancy readout on
the cases of Fig. 4 / Table S-exposure (paper_mlsys/figs/make_fig_exposure.py's sources and filter: one adapter, bf16, M <= 16, each
file's newest job; the Hopper rows only, since the Blackwell slice keeps the whole main loop). Per case K2 (the LoRA GEMM) compiled with
the case's own tiles (pdl_best) as run_pdl launches it, and its resident thread blocks per SM from cuOccupancyMaxActiveBlocksPerMultiprocessor
(the compiled kernel's warps and shared memory). Occupancy is a property of the compiled kernel and the SM, so one sm_90 GPU serves the
H100, H200 and GH200 rows (132 SMs each). Times are not measured here: the window is recomputed from each case's recorded t_main.
    python bench/occupancy_cases.py --out results/occupancy_fig4.jsonl"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SRC = (("H100", "pdl_k2.jsonl"), ("H100", "pdl_k2_tp_h100.jsonl"), ("H200", "pdl_k2_h200.jsonl"), ("GH200", "pdl_k2_exposure_gh200.jsonl"))
NEED = ("t_k2_nolora", "t_shrink_only", "t_pdl", "t_pdl_noshrink")


def cases(repo: Path) -> list[dict]:
    out = []
    for gpu, fn in SRC:
        f = repo / "results" / fn
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []
        rows = [r for r in rows if all(isinstance(r.get(k), (int, float)) for k in NEED) and r.get("n") == 1
                and r.get("fmt", "bf16") == "bf16" and r.get("M", 99) <= 16]
        if not rows:
            continue
        job = max(str(r.get("job", "")) for r in rows)
        for r in rows:
            if str(r.get("job", "")) != job or r["t_pdl_noshrink"] - r["t_pdl"] > 5.0:
                continue
            cfg = r.get("pdl_best") or {}
            if not all(k in cfg for k in ("BLOCK_M", "BLOCK_N", "SPLIT_K")) or "N" not in r:
                continue
            out.append({"gpu": gpu, "file": fn, "model": r.get("model", "qwen3_8b"), "proj": r["proj"], "r": r["r"], "M": r["M"],
                        "N": r["N"], "cfg": cfg, "t_main": r["t_k2_nolora"], "t_shrink": r["t_shrink_only"],
                        "exposed": r["t_pdl"] - r["t_pdl_noshrink"],
                        "grid": -(-r["M"] // cfg["BLOCK_M"]) * -(-r["N"] // cfg["BLOCK_N"]) * cfg["SPLIT_K"]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    import pdl_k2_bench as B
    from lowbit_lora import hopper_pdl as H
    from lowbit_lora.kernels import fp8_lora_pdl as KP

    bank = B.Bank(16)
    env = B.Env()
    env.set_tuned(False)
    H.load_table(str(Path(os.environ.get("LOWBIT_PDL_CFG", B.REPO / "results/pdl_k2.jsonl"))))
    n_sm = torch.cuda.get_device_properties(0).multi_processor_count
    info = {"gpu_run": torch.cuda.get_device_name(), "n_sm_run": n_sm, "job": os.environ.get("SLURM_JOB_ID", ""),
            "torch": torch.__version__, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = Path(a.out)
    cuda = ctypes.CDLL("libcuda.so.1")
    grabbed: list = []
    orig_launch = H._launch

    def spy(fn, grid, args, kw, compile_only):
        k = orig_launch(fn, grid, args, kw, True)            # compile only: the kernel object, nothing launched
        if fn is KP._gemm_pdl:
            grabbed.append(k)
        return None

    H._launch = spy
    try:
        last = None
        for c in cases(B.REPO):
            if last is not None and last != c["model"]:
                bank.drop_ops()
            last = c["model"]
            op, a_st, b_st = bank.op(c["model"], c["proj"], c["r"], 8, "bf16")
            _, margs = B.make_meta(8, bank.max_m, c["M"], 1)
            x = torch.randn(c["M"], op.K, device="cuda").to(torch.bfloat16)
            grabbed.clear()
            row = {**info, **c}
            try:
                H.run_pdl(op, x, margs, cfg=c["cfg"])
                if not grabbed:
                    raise RuntimeError("no _gemm_pdl compiled")
                k = grabbed[-1]
                k._init_handles()
                nb = ctypes.c_int(0)
                warps, smem = int(k.metadata.num_warps), int(k.metadata.shared)
                rc = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(ctypes.byref(nb), ctypes.c_void_p(k.function),
                                                                      ctypes.c_int(32 * warps), ctypes.c_size_t(smem))
                wave = 132 * max(1, nb.value)
                row.update(blocks_per_sm=nb.value, cu_rc=rc, num_warps=warps, shared=smem, n_regs=int(k.n_regs),
                           first_wave_blocks=wave, window_one=round(c["t_main"] * min(1.0, 132 / c["grid"]), 2),
                           window_api=round(c["t_main"] * min(1.0, wave / c["grid"]), 2))
                row["pred_one"] = round(max(0.0, c["t_shrink"] - row["window_one"]), 2)
                row["pred_api"] = round(max(0.0, c["t_shrink"] - row["window_api"]), 2)
                row["pred_whole"] = round(max(0.0, c["t_shrink"] - c["t_main"]), 2)
            except Exception as e:   # noqa: BLE001
                torch.cuda.synchronize()
                row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            print(json.dumps({k: v for k, v in row.items() if k != "cfg"}), flush=True)
            with out.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            del x, margs
        bank.drop_ops()
    finally:
        H._launch = orig_launch
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
