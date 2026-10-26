"""KERNEL-CONTROLS (docs/REGISTRY.md; mock review R3 v42, W2/W3 and change 2): two controls of the decode path on one H100.
(a) FLOOR-SPLITK. SHRINK-FLOOR's cases (results/pdl_k2_floor_gemv_h100.jsonl: Qwen3-8B's qkv, o, gate_up, down at one token, one
    adapter of rank 16, 64 or 128, each row's own K2 configuration, pdl_best) with the decode path's shrink kernel at every split-K
    SPLIT_K1 in 1-32 that the shared scratch holds: t_shrink (the shrink alone, gemm=False) and t_serial (that shrink, then the GEMM
    without the PDL attribute: a tuned shrink serialized), and t_pdl (the decode path as configured) in the same process; the
    fastest serialized split against the decode path.
(b) OCCUPANCY. For every K2 configuration in FIRSTWAVE-GRID's H100 files (results/firstwave_grid_{defaults,swept}_h100_r*.jsonl),
    K2 (the LoRA GEMM) compiled with it as run_pdl launches it, and its resident thread blocks per SM from the CUDA occupancy API
    (cuOccupancyMaxActiveBlocksPerMultiprocessor with the compiled kernel's warps and shared memory); per row the first wave in
    blocks, n_sm x occupancy, and the first-wave window recomputed with it: t_main x min(1, n_sm x occupancy / grid).
    python bench/kernel_controls.py --out results/kernel_controls_h100.jsonl"""
from __future__ import annotations

import argparse
import ctypes
import glob
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", default="floor,occupancy")
    ap.add_argument("--gpu", default="h100", help="whose SHRINK-FLOOR and FIRSTWAVE-GRID rows to rerun (h100, h200)")
    a = ap.parse_args()

    import torch
    import pdl_k2_bench as B
    from kernel_bench import Timer
    from lowbit_lora import hopper_pdl as H

    timer = Timer(warm=5, reps=40)
    bank = B.Bank(16)
    env = B.Env()
    env.set_tuned(False)
    H.load_table(str(Path(os.environ.get("LOWBIT_PDL_CFG", B.REPO / "results/pdl_k2.jsonl"))))
    n_sm = torch.cuda.get_device_properties(0).multi_processor_count
    info = {"gpu": torch.cuda.get_device_name(), "n_sm": n_sm, "job": os.environ.get("SLURM_JOB_ID", ""),
            "torch": torch.__version__, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = Path(a.out)

    def emit(r: dict) -> None:
        print(json.dumps({k: v for k, v in r.items() if k not in ("cfg",)}), flush=True)
        with out.open("a") as fh:
            fh.write(json.dumps(r) + "\n")

    if "floor" in a.parts:
        src = [json.loads(s) for s in (B.REPO / f"results/pdl_k2_floor_gemv_{a.gpu}.jsonl").read_text().splitlines() if s.strip()]
        for s in src:
            p, r, M, bc = s["proj"], s["r"], s["M"], s["pdl_best"]
            op, a_st, b_st = bank.op("qwen3_8b", p, r, 8, "bf16")
            _, margs = B.make_meta(8, bank.max_m, M, 1)
            x = torch.randn(M, op.K, device="cuda").to(torch.bfloat16)
            row = {**info, "part": "floor_splitk", "model": "qwen3_8b", "proj": p, "r": r, "M": M, "cfg": bc,
                   "split_default": int(H.resolve_cfg(op, M, bc)["SPLIT_K1"])}
            try:
                y0 = H.run_pdl(op, x, margs, cfg=bc).float().clone()
                row["t_pdl"] = round(timer.graph_time(lambda: H.run_pdl(op, x, margs, cfg=bc)), 2)
                per = {}
                for sk in (1, 2, 4, 8, 16, 32):
                    c = dict(bc, SPLIT_K1=sk)
                    g = H._geometry(op, x, H.resolve_cfg(op, M, c), True)
                    if g["split1"] != sk:          # the shared scratch does not hold this split
                        continue
                    y = H.run_pdl(op, x, margs, cfg=c, pdl=False).float()
                    ok = float(((y - y0).norm() / y0.norm()).item()) <= 2e-2
                    per[sk] = {"t_shrink": round(timer.graph_time(lambda: H.run_pdl(op, x, margs, cfg=c, gemm=False)), 2),
                               "t_serial": round(timer.graph_time(lambda: H.run_pdl(op, x, margs, cfg=c, pdl=False)), 2), "ok": ok}
                row["splits"] = per
                good = {k: v for k, v in per.items() if v["ok"]}
                if good:
                    bs = min(good, key=lambda k: good[k]["t_serial"])
                    row["best_split"], row["t_serial_best"] = bs, good[bs]["t_serial"]
                    row["t_shrink_best"] = min(v["t_shrink"] for v in good.values())
                    row["serial_over_pdl"] = round(good[bs]["t_serial"] / row["t_pdl"], 4)
            except Exception as e:   # noqa: BLE001
                torch.cuda.synchronize()
                row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            emit(row)
            del x, margs
        bank.drop_ops()

    if "occupancy" in a.parts:
        cuda = ctypes.CDLL("libcuda.so.1")
        grabbed: list = []
        orig_launch = H._launch

        from lowbit_lora.kernels import fp8_lora_pdl as KP

        def spy(fn, grid, args, kw, compile_only):
            k = orig_launch(fn, grid, args, kw, True)            # compile only: the kernel object, nothing launched
            if fn is KP._gemm_pdl:
                grabbed.append(k)
            return None

        def occupancy(k) -> dict:
            k._init_handles()
            nb = ctypes.c_int(0)
            warps = int(k.metadata.num_warps)
            smem = int(k.metadata.shared)
            rc = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(ctypes.byref(nb), ctypes.c_void_p(k.function),
                                                                  ctypes.c_int(32 * warps), ctypes.c_size_t(smem))
            return {"blocks_per_sm": nb.value, "cu_rc": rc, "num_warps": warps, "shared": smem, "n_regs": int(k.n_regs),
                    "n_spills": int(k.n_spills)}

        files = (sorted(glob.glob(str(B.REPO / "results/firstwave_grid_defaults_h100_r*.jsonl"))
                        + glob.glob(str(B.REPO / "results/firstwave_grid_swept_h100_r*.jsonl"))) if a.gpu == "h100" else
                 [str(B.REPO / "results/firstwave_grid_h200.jsonl"), str(B.REPO / "results/firstwave_grid_swept_h200.jsonl")])
        H._launch = spy
        try:
            for f in files:
                rows = [json.loads(s) for s in Path(f).read_text().splitlines() if s.strip()]
                for s in rows:
                    if not s.get("ok") or "grid" not in s:
                        continue
                    op, a_st, b_st = bank.op("qwen3_8b", s["proj"], s["r"], 8, "bf16")
                    _, margs = B.make_meta(8, bank.max_m, s["M"], 1)
                    x = torch.randn(s["M"], op.K, device="cuda").to(torch.bfloat16)
                    grabbed.clear()
                    row = {**info, "part": "occupancy", "file": Path(f).name, "proj": s["proj"], "r": s["r"], "M": s["M"],
                           "cfg": s["cfg"], "grid": s["grid"], "t_main": s["t_main"], "t_shrink": s["t_shrink"],
                           "exposed": s["exposed"], "pred_firstwave": s["pred_firstwave"], "pred_wholeloop": s["pred_wholeloop"]}
                    try:
                        H.run_pdl(op, x, margs, cfg=s["cfg"])
                        if not grabbed:
                            raise RuntimeError("no _gemm_pdl compiled")
                        o = occupancy(grabbed[-1])
                        row.update(o)
                        wave = n_sm * max(1, o["blocks_per_sm"])
                        win = s["t_main"] * min(1.0, wave / s["grid"])
                        row["first_wave_blocks"] = wave
                        row["window_api"] = round(win, 2)
                        row["pred_api"] = round(max(0.0, s["t_shrink"] - win), 2)
                    except Exception as e:   # noqa: BLE001
                        torch.cuda.synchronize()
                        row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                    emit(row)
                    del x, margs
                bank.drop_ops()
        finally:
            H._launch = orig_launch
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
