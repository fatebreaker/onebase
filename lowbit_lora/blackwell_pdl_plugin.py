"""BLACKWELL-PDL-E2E's opt-in plugin path (docs/REGISTRY.md): the NVFP4 decode path with the adapters off the critical
path (blackwell_pdl.run_pdl4 over kernels/nvfp4_lora_pdl.py, measured in BLACKWELL-PDL-K2) as one more candidate of the
plugin's NVFP4 op's timed dispatch on Blackwell sm_120, without touching the registered plugin code: the counterpart of
hopper_pdl.install for VllmFusedNVFP4LoRA (kernels/nvfp4_lora_vllm.py), which the plugin builds for ModelOpt NVFP4 layers.

  install()     the vLLM general-plugin hook, entry point lowbit_pdl4 in pylib_pdl4/ (put on PYTHONPATH only for engines
                that opt in), and itself a no-op unless LOWBIT_PDL=1. It patches the NVFP4 op in memory: its timed dispatch
                (tune_dispatch: CUDA-graph replays of every candidate per bucket of M, the plugin's own timer) also times
                the PDL path for every decode bucket M <= LOWBIT_PDL_MAX_M (32), next to the fused path, the hybrid path
                (vLLM's NVFP4 GEMM + our LoRA kernels) and vLLM's own LoRA kernels ("hybrids", tuned when
                VLLM_TUNED_CONFIG_FOLDER is set), and keeps the fastest; the op's call (what the plugin's registered custom
                op, torch.ops.vllm.lowbit_fused_fp8_lora, runs: opaque to torch.compile, captured in vLLM's CUDA graphs
                with the PDL edge) serves the PDL path where the table says so; the warm-up compiles the PDL kernels of
                every bucket that chose them and grows the shared split-K workspace to each such bucket's largest M,
                before vLLM captures its graphs (nothing may grow inside them; the activation-code and LoRA scratch is
                the op's own, sized at construction for the engine's largest batch). Dispatch tables are keyed apart from
                the unpatched op's ("pdl").
  load_table4   configurations from bench/pdl_fp4_bench.py's rows (LOWBIT_PDL_CFG, else results/pdl_fp4.jsonl beside the
                package; the newest job unless LOWBIT_PDL_JOB): the swept best (pdl_best) of every one-adapter row, keyed
                by (bucket of M, N, K, slices, rank, adapter format).
  plugin_cfg4   per op and bucket, robust to what the sweep did not measure (as hopper_pdl.plugin_cfg): the exact entry;
                else the same bucket and shape at the nearest rank (this format, then bf16, then any); else the same shape
                at the nearest measured decode bucket, re-tiled (_retile4); else nvfp4_lora_pdl.default_cfg4's defaults.
                num_warps is always re-derived from the final tile and the op's rank (blackwell_pdl.resolve_cfg4).
LOWBIT_REPORT (as hopper_pdl): "pdl-dispatch-table N=.. K=.. R=.. widths=.. fmt=.. cfg_src={bucket: how} us={bucket:
{path: us}}" once per shape, "pdl-layer <n> N=.. K=.. R=.. pdl_dispatch=<bucket>:<path>,..." per layer, and
"pdl-candidate-failed ..." / "pdl-warmup-failed ..." if a PDL variant cannot run (it is then not a candidate). The
plugin's own per-layer line maps the PDL path to "f".
"""
from __future__ import annotations

import os

import torch

from lowbit_lora.blackwell_pdl import run_pdl4
from lowbit_lora.kernels import nvfp4_lora_vllm as F4
from lowbit_lora.kernels.nvfp4_lora_pdl import work

PDL_MAX_M = int(os.environ.get("LOWBIT_PDL_MAX_M", "32"))     # the dispatch times the PDL path for buckets M <= this
_TABLE: dict = {}      # (bucket, N, K, S, R, fmt) -> the swept best configuration (results/pdl_fp4.jsonl, one adapter)
_TABLE_SRC: dict = {}  # where the table came from (path, job, entries), for the install line and the report
SMEM = 96 * 1024       # sm_120's 99 KB per block, less the epilogue's share (bench/pdl_fp4_bench.smem_ok4)


def _buckets() -> tuple:
    from lowbit_lora.kernels.fp8_lora_vllm import _BUCKETS   # (lazy, as nvfp4_lora_vllm: that module imports it)
    return _BUCKETS


def _bucket(M: int) -> int:
    from lowbit_lora.kernels.fp8_lora_vllm import _bucket as b
    return b(M)


def decode_buckets() -> tuple:
    return tuple(b for b in _buckets() if b <= 32)


def _report(line: str) -> None:
    rep = os.environ.get("LOWBIT_REPORT")
    if rep:
        try:
            with open(rep, "a") as fh:
                fh.write(f"{os.getpid()} {line}\n")
        except OSError:
            pass


def load_table4(path: str | None = None, job: str | None = None) -> int:
    """The plugin path's configurations (see the module's doc). A missing or empty file leaves the table empty: every
    op then runs nvfp4_lora_pdl.default_cfg4's defaults. Returns the number of entries."""
    import json
    from pathlib import Path
    f = Path(path or os.environ.get("LOWBIT_PDL_CFG") or Path(__file__).resolve().parents[1] / "results/pdl_fp4.jsonl")
    _TABLE.clear()
    rows = []
    if f.exists():
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("pdl_best") and r.get("n") == 1 and all(k in r for k in ("M", "N", "K", "S", "r")):
                rows.append(r)
    if rows:
        job = job or max(str(r.get("job", "")) for r in rows)
        for r in rows:
            if str(r.get("job", "")) == job:
                _TABLE[(_bucket(r["M"]), r["N"], r["K"], r["S"], r["r"], r.get("fmt", "bf16"))] = dict(r["pdl_best"])
    _TABLE_SRC.update(path=str(f), job=job if rows else None, entries=len(_TABLE))
    return len(_TABLE)


def _smem_ok(c: dict) -> bool:
    bm, bn, bk = int(c.get("BLOCK_M", 16)), int(c.get("BLOCK_N", 64)), int(c.get("BLOCK_K", 256))
    st = int(c.get("num_stages", 4))
    xb = bm * bk * 2 if int(c.get("X_MODE", 1)) == 0 else bm * bk // 2 + bm * bk // 16
    return max(1, st - 1) * (bn * bk // 2 + bn * bk // 16 + xb) <= SMEM


def _retile4(c: dict, b: int) -> dict:
    """A configuration found for another batch bucket, re-tiled for bucket b as nvfp4_lora_pdl.default_cfg4 tiles:
    swap-AB (kept from the source up to 16 tokens) with 8-token tiles up to 8 tokens and 16 up to 16, the plugin's
    orientation with 16- / 32-token tiles; the pipeline shortened while x's tiles would not fit sm_120's shared memory."""
    c = {k: v for k, v in c.items() if k != "num_warps"}
    swap = bool(c.get("SWAP", True)) and b <= 16
    c["SWAP"] = swap
    c["BLOCK_M"] = (8 if b <= 8 else 16) if swap else (16 if b <= 16 else 32 if b <= 32 else 64)
    while int(c.get("num_stages", 4)) > 2 and not _smem_ok(c):
        c["num_stages"] = int(c.get("num_stages", 4)) - 1
    return c


def plugin_cfg4(op, M: int) -> dict:
    """The PDL configuration for this op's shape at M tokens (see the module's doc). Cached per (op, bucket); the source
    of each bucket's entry in op._pdl4_cfg_src (exact / rank R fmt / bucket B rank R fmt / default)."""
    import math
    b = _bucket(M)
    cache = op.__dict__.setdefault("_pdl4_cfgs", {})
    if b in cache:
        return cache[b]
    shape = (op.N, op.K, op.S)
    fmt = op.lora_fmt
    c, how = None, "default"
    if (b, *shape, op.R, fmt) in _TABLE:
        c, how = _TABLE[(b, *shape, op.R, fmt)], "exact"
    if c is None:   # the same bucket and shape, the nearest rank: this format, then bf16, then any
        for want in (fmt, "bf16", None):
            ks = [k for k in _TABLE if k[0] == b and k[1:4] == shape and (want is None or k[5] == want)]
            if ks:
                k = min(ks, key=lambda k: (abs(math.log2(k[4]) - math.log2(max(op.R, 1))), -k[4]))
                c, how = _TABLE[k], f"rank {k[4]} {k[5]}"
                break
    if c is None:   # the same shape at the nearest measured decode bucket (log distance; the larger on ties), re-tiled
        ks = [k for k in _TABLE if k[1:4] == shape and k[0] <= max(decode_buckets())]
        if ks:
            pref = [k for k in ks if k[4] == op.R and k[5] == fmt] or [k for k in ks if k[5] == fmt] or ks
            k = min(pref, key=lambda k: (abs(math.log2(k[0]) - math.log2(b)), -k[0], abs(k[4] - op.R)))
            c, how = _retile4(_TABLE[k], b), f"bucket {k[0]} rank {k[4]} {k[5]}"
    cfg = {k: v for k, v in c.items() if k != "num_warps"} if c is not None else {}
    cache[b] = cfg
    op.__dict__.setdefault("_pdl4_cfg_src", {})[b] = how
    return cfg


def install() -> None:
    """vLLM general plugin hook (see the module's doc). Opt-in with LOWBIT_PDL=1; patches VllmFusedNVFP4LoRA once; the
    PDL path is a candidate only on compute capability 12 (the op's cc_major), for buckets M <= LOWBIT_PDL_MAX_M."""
    if os.environ.get("LOWBIT_PDL", "0") != "1":
        return
    V = F4.VllmFusedNVFP4LoRA
    if getattr(V, "_pdl4_installed", False):
        return
    n_cfg = load_table4(None, os.environ.get("LOWBIT_PDL_JOB") or None)
    orig_call, orig_warm = V.__call__, V.warmup
    n_ops = [0]

    def __call__(self, x, meta_args, out=None, cfg_s=None, cfg_g=None, base_fn=None, hybrid_m=64, split=None,
                 use_dispatch=True):
        if (use_dispatch and base_fn is None and self.dispatch
                and self.dispatch.get(_bucket(x.shape[0])) == "pdl"):
            return run_pdl4(self, x, meta_args, cfg=plugin_cfg4(self, x.shape[0]), out=out)
        return orig_call(self, x, meta_args, out=out, cfg_s=cfg_s, cfg_g=cfg_g, base_fn=base_fn, hybrid_m=hybrid_m,
                         split=split, use_dispatch=use_dispatch)

    def tune_dispatch(self) -> dict:
        """VllmFusedNVFP4LoRA.tune_dispatch (the same candidates, inputs and timer: one adapter on every token, the
        plugin's CUDA-graph replays) with the PDL path as one more candidate for the decode buckets on compute
        capability 12; the table's key is marked "pdl", so tables never mix with the unpatched op's."""
        if self.base_fn is None:
            return self.dispatch
        from lowbit_lora.kernels import fp8_lora_vllm as F   # (F._graph_us looked up at call time)
        key = ("nvfp4", torch.cuda.get_device_name(self.w_q.device), self.N, self.K, self.R, tuple(self.widths),
               self.lora_fmt, "pdl", PDL_MAX_M)
        if key not in F4._DISPATCH4:
            import json
            table, prev, times = {}, None, {}
            for b in _buckets():
                M = min(b, self.max_m)
                if M < 1 or M == prev:
                    break
                prev = M
                x = torch.randn(M, self.K, device=self.w_q.device).to(torch.bfloat16)
                meta = self._dummy_meta(M, 0)
                t = {"hybrid": F._graph_us(lambda: self(x, meta, base_fn=self.base_fn, hybrid_m=0, use_dispatch=False))}
                if M <= 128:
                    t["fused"] = F._graph_us(lambda: self(x, meta, hybrid_m=1 << 30, use_dispatch=False))
                if self.lora_fmt == "bf16":
                    t["hybrids"] = F._graph_us(lambda: self._run_stock(x, meta))
                if self.cc_major == 12 and M <= PDL_MAX_M:
                    try:
                        t["pdl"] = F._graph_us(lambda: run_pdl4(self, x, meta, cfg=plugin_cfg4(self, M)))
                    except Exception as e:   # noqa: BLE001 -- not a candidate for this shape and bucket
                        torch.cuda.synchronize()
                        work(self.w_q.device).reset()
                        _report(f"pdl-candidate-failed N={self.N} K={self.K} M={M}: {type(e).__name__}: "
                                f"{str(e)[:200]}")
                table[b] = min(t, key=t.get)
                times[b] = {k: round(v, 2) for k, v in t.items()}
                del x, meta
            F4._DISPATCH4[key] = table
            torch.cuda.empty_cache()
            _report(f"pdl-dispatch-table N={self.N} K={self.K} R={self.R} widths={list(self.widths)} fmt={self.lora_fmt} "
                    f"cfg_src={json.dumps({b: self.__dict__.get('_pdl4_cfg_src', {}).get(b) for b in decode_buckets()})} "
                    f"us={json.dumps(times)}")
        self.dispatch = F4._DISPATCH4[key]
        return self.dispatch

    def warmup(self) -> None:
        orig_warm(self)                                   # times the dispatch (patched) and compiles the other paths
        n_ops[0] += 1
        if self.cc_major == 12 and self.base_fn is not None:
            # every bucket that chose the PDL path: its kernels compiled (with and without an adapter) and the shared
            # split-K workspace grown to the bucket's largest M, here: vLLM captures its CUDA graphs after the warm-up
            for b in decode_buckets():
                M = min(b, self.max_m)
                if M < 1 or b > PDL_MAX_M or self.dispatch.get(b) != "pdl":
                    continue
                x = torch.zeros(M, self.K, dtype=torch.bfloat16, device=self.w_q.device)
                for lora in (0, -1):
                    try:
                        run_pdl4(self, x, self._dummy_meta(M, lora), cfg=plugin_cfg4(self, M))
                    except Exception as e:   # noqa: BLE001
                        _report(f"pdl-warmup-failed N={self.N} K={self.K} M={M}: {type(e).__name__}: {str(e)[:200]}")
                        break
            torch.cuda.synchronize()
        d = self.dispatch or {}
        _report(f"pdl-layer {n_ops[0]} N={self.N} K={self.K} R={self.R} "
                f"pdl_dispatch={','.join(f'{b}:{v}' for b, v in sorted(d.items()))}")

    V.__call__, V.tune_dispatch, V.warmup = __call__, tune_dispatch, warmup
    V._pdl4_installed = True
    print(f"[lowbit] PDL decode path installed for Blackwell NVFP4 dispatch (decode buckets M <= {PDL_MAX_M}; {n_cfg} "
          f"swept configurations from {_TABLE_SRC.get('path')}, job {_TABLE_SRC.get('job')})", flush=True)
