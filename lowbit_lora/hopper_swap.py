"""The swap-AB K2 (kernels/fp8_lora_swap.py) as a path of the plugin's fused op on Hopper, without touching the registered
plugin code (kernels/fp8_lora_vllm.py): run_swap(op, x, meta_args) launches K1 exactly as VllmFusedFP8LoRA._run does (the
adapter shrinks over vLLM's sorted segments plus the all-token pass that writes x_q and the per-token scales in natural
order), then the swap-AB GEMM with the LoRA epilogue. Kernel benchmark first (bench/swap_k2_bench.py); the plugin's
timed dispatch gets this path only after that, through an opt-in patch registered before its data.
    from lowbit_lora.hopper_swap import run_swap; y = run_swap(op, x, meta.meta_args(M, False))
"""
from __future__ import annotations

import os

import torch
import triton

from lowbit_lora.kernels import fp8_lora_vllm as F
from lowbit_lora.kernels.fp8_lora_swap import gemm_swap

CFG: dict = {}   # (bucket of M, N, K, n_slices) -> swept tiles (bench/swap_k2_bench.py --tune), else the defaults


def run_swap(op, x, meta_args, cfg: dict | None = None, out=None, precise: bool | None = None):
    tok, srt, seg_n, seg_start, seg_ids = meta_args[:5]
    M = x.shape[0]
    sc = op.scratch
    hp, pmax, sxb = sc.hp, sc.pmax, sc.sx
    xq = sc.xq[: M * op.K].view(M, op.K)
    cs_t, split = F.shrink_choice(M, op.R, op.split_k, op.cc_major)
    while split > 1 and (split * op.S * M * op.R > hp.numel() or split * M > pmax.numel()):
        split //= 2                                                  # what the shared scratch holds
    prefill = M > 64
    ds, _ = F.pick_cfg(M, op.N, op.K, op.R, op.widths, op.cc_major)
    cs = cs_t or ds
    n_sr = op.S * op.R // cs["BLOCK_SR"]
    grid_s = (triton.cdiv(M, cs["BLOCK_M"]) * n_sr * split, 1, op.L + 2)
    F._shrink_vllm[grid_s](
        x, op.a_ptrs, op.sa_ptrs, op.ag, hp, pmax, xq, sxb, srt, seg_ids, seg_n, seg_start,
        M, op.K, op.R, op.S, op.L + 1, x.stride(0), op.stride_al, op.stride_am, xq.stride(0),
        SPLIT_K=split, WRITE_XQ=prefill, N_SR=n_sr, BLOCK_M=cs["BLOCK_M"], BLOCK_SR=cs["BLOCK_SR"],
        BLOCK_K=cs["BLOCK_K"], DOT_F32=op.dot_f32, RN=not op.dot_f32, A_FP8=op.lora_fp8, A_FP4=op.lora_fp4,
        num_warps=cs["num_warps"], num_stages=cs["num_stages"])
    y = out if out is not None else torch.empty(M, op.N, dtype=torch.bfloat16, device=x.device)
    if precise is None:
        precise = os.environ.get("LOWBIT_PRECISE", "1") != "0"
    if cfg is None:
        cfg = CFG.get((F._bucket(M), op.N, op.K, op.S))
    return gemm_swap(op, xq, sxb, hp, y, tok, seg_ids, seg_n, split, cfg=cfg, precise=precise)


def load_cfg(path: str | None = None, job: str | None = None) -> int:
    """Swept tiles from bench/swap_k2_bench.py --tune (results/swap_k2.jsonl, the newest job unless given): for each
    (bucket of M, N, K, slices), the one-adapter rank-64 row's best tiles. Returns the number of entries."""
    import json
    from pathlib import Path
    f = Path(path or Path(__file__).resolve().parents[1] / "results/swap_k2.jsonl")
    rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []
    rows = [r for r in rows if r.get("swap_best") and r.get("n") == 1 and r.get("r") == 64]
    if not rows:
        return 0
    job = job or max(r["job"] for r in rows)
    for r in rows:
        if r["job"] == job:
            S = 3 if r["proj"] == "qkv" else 2 if r["proj"] == "gate_up" else 1
            CFG[(F._bucket(r["M"]), r["N"], r["K"], S)] = r["swap_best"]
    return len(CFG)


def install() -> None:
    """vLLM general plugin (entry point in pylib_swap/; on PYTHONPATH only for engines that opt in) and LOWBIT_SWAP=1: on
    compute capability 9 the fused op's timed dispatch also times the swap path and serves it where it is fastest. The
    registered plugin module is patched in memory only; with LOWBIT_SWAP unset this does nothing."""
    if os.environ.get("LOWBIT_SWAP", "0") != "1" or getattr(F.VllmFusedFP8LoRA, "_swap_installed", False):
        return
    V = F.VllmFusedFP8LoRA
    n_cfg = load_cfg(os.environ.get("LOWBIT_SWAP_CFG") or None, os.environ.get("LOWBIT_SWAP_JOB") or None)
    orig_run, orig_call, orig_warm = V._run, V.__call__, V.warmup

    def _run(self, x, meta_args, out=None, mode=None, cfg_s=None, cfg_g=None, hybrid=False, split=None):
        if hybrid == "w":
            return run_swap(self, x, meta_args, out=out)
        return orig_run(self, x, meta_args, out=out, mode=mode, cfg_s=cfg_s, cfg_g=cfg_g, hybrid=hybrid, split=split)

    def __call__(self, x, meta_args, out=None, mode=None, cfg_s=None, cfg_g=None):
        if self.base_fn is not None and self.dispatch.get(F._bucket(x.shape[0])) == "swap":
            return run_swap(self, x, meta_args, out=out)
        return orig_call(self, x, meta_args, out=out, mode=mode, cfg_s=cfg_s, cfg_g=cfg_g)

    def tune_dispatch(self) -> dict:
        """VllmFusedFP8LoRA.tune_dispatch with one more candidate on compute capability 9, the swap path (dispatch key
        marked "swap", so tables never mix with the unpatched op's)."""
        if self.base_fn is None:
            return self.dispatch
        kd = int(os.environ.get("LOWBIT_DISPATCH_K", "1"))
        key = (torch.cuda.get_device_name(self.w_q.device), self.N, self.K, self.R, tuple(self.widths),
               "fp4" if self.lora_fp4 else "fp8" if self.lora_fp8 else "bf16", self.base_q_fn is not None, kd, "swap")
        tune_max = int(os.environ.get("LOWBIT_TUNE_MAX_M", "2048"))
        if key not in F._DISPATCH:
            table, prev = {}, None
            for b in F._BUCKETS:
                M = min(b, self.max_m)
                if M < 1 or (table and M == prev):
                    break
                if table and M > tune_max:
                    table[b] = table[max(table)]
                    prev = M
                    continue
                prev = M
                x = torch.randn(M, self.K, device=self.w_q.device).to(torch.bfloat16)
                metas = [self._dummy_meta(M, 0)] + ([self._dummy_meta_multi(M, kd)] if min(kd, self.L, M) > 1 else [])
                tm = lambda h: sum(F._graph_us(lambda m=m: self._run(x, m, hybrid=h)) for m in metas)   # noqa: E731
                t = {"fused": tm(False), "hybrid": tm(True)}
                if self.base_q_fn is not None:
                    t["hybridq"] = tm("q")
                if not (self.lora_fp8 or self.lora_fp4):
                    t["hybrids"] = tm("s")
                if self.cc_major == 9:
                    try:
                        t["swap"] = tm("w")
                    except Exception:   # noqa: BLE001 -- a shape the swap tiles do not cover: not a candidate
                        pass
                table[b] = min(t, key=t.get)
                del x, metas
            F._DISPATCH[key] = table
            torch.cuda.empty_cache()
        self.dispatch = F._DISPATCH[key]
        return self.dispatch

    def warmup(self) -> None:
        orig_warm(self)                                   # times the dispatch (patched) and compiles the other paths
        if self.cc_major != 9 or self.base_fn is None:
            return
        sizes = [m for m in (1, 2, 16, 17, 48, 49, 96, 97, 192, 193, 512, 513, 2048, 2049, self.max_m, self.max_m - 1)
                 if 1 <= m <= self.max_m]
        for M in sizes:
            x = torch.zeros(M, self.K, dtype=torch.bfloat16, device=self.w_q.device)
            for lora in (0, -1):
                try:
                    self._run(x, self._dummy_meta(M, lora), hybrid="w")
                except Exception:   # noqa: BLE001
                    break
        torch.cuda.synchronize()

    V._run, V.__call__, V.tune_dispatch, V.warmup = _run, __call__, tune_dispatch, warmup
    V._swap_installed = True
    print(f"[lowbit] swap-AB K2 installed for Hopper dispatch ({n_cfg} swept tile entries)", flush=True)
