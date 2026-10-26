"""vLLM plugin: serve LoRA layers whose base is online per-tensor FP8 through the fused kernels (fp8_lora_vllm.py).

Loaded by vLLM in every process through the `vllm.general_plugins` entry point declared in
lowbit_lora-*.dist-info/entry_points.txt at the repo root (the repo is on PYTHONPATH; nothing is installed).
It does nothing unless LOWBIT_FUSED=1. LOWBIT_LORA_FP8=1 also serves the adapters FP8-stored (E4M3, per-row scales),
and LOWBIT_LORA_FP4=1 NVFP4-stored (E2M1, E4M3 block scales of 16, a global per adapter and slice), from low-precision
copies that replace vLLM's bf16 slots (these shrink to one staging slot, _hook_slot_writes; LOWBIT_RELEASE_SLOTS=0
keeps both) and are refreshed whenever vLLM writes or clears a slot.

What it patches: BaseLinearLayerWithLoRA._apply_sync (and apply, to bypass the aux-stream path) for layers whose base
quant method is Fp8PerTensorOnlineLinearMethod on the CUTLASS kernel, or ModelOptNvFp4LinearMethod (an NVFP4
checkpoint; Blackwell sm_120: fused kernels for M <= 64, vLLM's NVFP4 GEMM plus our LoRA kernels above), at TP 1,
without bias. For NVFP4 it keeps the checkpoint-layout weights (codes, linear block scales) that vLLM's backends
re-layout at load time (process_weights_after_loading is wrapped). Every other layer, and
any call it cannot serve, falls back to vLLM's own path. The fused call is a registered custom op
(torch.ops.vllm.lowbit_fused_fp8_lora), opaque to torch.compile and capturable in CUDA graphs: it only reads vLLM's
static LoRA buffers and Punica metadata, and its scratch buffers are allocated once per layer.
"""
from __future__ import annotations

import os

import torch

_REGISTRY: list = []      # layer id -> VllmFusedFP8LoRA
_PENDING_WARMUP: list = []   # LOWBIT_WARMUP_DEFER=1: ops whose load-time warm-up waits until vLLM has created every LoRA module
_INSTALLED = False
_CALLED = [False]


def _cache_root() -> str:
    """vLLM's cache root for this plugin's graphs: VLLM_CACHE_ROOT/lowbit-<hash of the package's source and every
    LOWBIT_* setting but the report paths>, so no graph is shared with stock runs or with another plugin setting.
    Every vLLM process installs the plugin; children inherit the root already keyed and keep it."""
    import hashlib
    from pathlib import Path

    import vllm.envs as envs
    if envs._is_envs_cache_enabled():
        raise RuntimeError("lowbit: vLLM's environment cache is already on; the compile-cache root cannot be keyed")
    h = hashlib.sha256()
    pkg = Path(__file__).resolve().parent
    for f in sorted(pkg.rglob("*.py")):
        h.update(str(f.relative_to(pkg)).encode())
        h.update(f.read_bytes())
    for k in sorted(os.environ):
        if k.startswith("LOWBIT_") and k not in ("LOWBIT_REPORT", "LOWBIT_VERBOSE"):
            h.update(f"{k}={os.environ[k]}\n".encode())
    base = envs.VLLM_CACHE_ROOT
    key = f"lowbit-{h.hexdigest()[:12]}"
    return base if os.path.basename(base) == key else os.path.join(base, key)


def _lean_base(base, kern):
    """LOWBIT_LEAN_BASE=1: vLLM's per-token FP8 GEMM for the hybrid paths without its eager-mode Python. The kernel's
    QuantFP8 runs its torch.compile'd native forward, which inside our opaque op is called eagerly per layer and pays
    TorchDynamo's cache lookup and compiled-region call each time (about 14 ms of host time per four-tenant 2,048-token
    prefill step on the H100); this calls the CUDA op that QuantFP8.forward_cuda calls (ops.scaled_fp8_quant, per token)
    and the CUTLASS GEMM that apply_scaled_mm calls. None (keep vLLM's path) unless the layer needs neither padding."""
    from vllm import _custom_ops as ops
    q = getattr(kern, "quant_fp8", None)
    w, sw = base.weight, base.weight_scale
    if (q is None or q.static or not q.use_per_token_if_dynamic or q.num_token_padding is not None
            or getattr(kern, "logical_output_size", None) not in (None, w.shape[1])):
        return None

    def fn(x):
        xq, sx = ops.scaled_fp8_quant(x, None, use_per_token_if_dynamic=True)
        return ops.cutlass_scaled_mm(xq, w, out_dtype=torch.bfloat16, scale_a=sx, scale_b=sw, bias=None)
    return fn


def _fused(x: torch.Tensor, layer_id: int, tok: torch.Tensor, srt: torch.Tensor, seg_n: torch.Tensor,
           seg_start: torch.Tensor, seg_ids: torch.Tensor) -> torch.Tensor:
    if not _CALLED[0]:          # once per process: proof that the compiled graph calls this op (not a stock graph)
        _CALLED[0] = True
        rep = os.environ.get("LOWBIT_REPORT")
        if rep:
            with open(rep, "a") as fh:
                fh.write(f"{os.getpid()} op-called\n")
    return _REGISTRY[layer_id](x, (tok, srt, seg_n, seg_start, seg_ids))


def _fused_fake(x: torch.Tensor, layer_id: int, tok: torch.Tensor, srt: torch.Tensor, seg_n: torch.Tensor,
                seg_start: torch.Tensor, seg_ids: torch.Tensor) -> torch.Tensor:
    return torch.empty(x.shape[0], _REGISTRY[layer_id].N, dtype=torch.bfloat16, device=x.device)


def _hook_slot_writes(layer, op) -> None:
    """Per instance (subclasses override set_lora): requantize a slot's FP8 copy after vLLM writes or clears it.
    vLLM's set_lora calls reset_lora first, so a write requantizes twice (cheap, outside CUDA graphs).
    By default (LOWBIT_RELEASE_SLOTS=0 keeps vLLM's buffers) the low-precision copies are the only per-slot storage. vLLM's bf16 slots shrink to
    one staging slot that every write goes through (32 rank-64 slots of Llama-3.1-8B are 10.7 GB of bf16, which
    the KV cache gets back: vLLM sizes it after loading the model, and these hooks run during the load). Besides
    set_lora / reset_lora, vLLM reads only the buffers' trailing shapes (dummy LoRAs), and the fused op must then
    serve every call of the layer (_apply_sync)."""
    set_lora, reset_lora = layer.set_lora, layer.reset_lora
    if os.environ.get("LOWBIT_RELEASE_SLOTS", "1") not in ("0", "", "false", "False"):
        layer.lora_a_stacked = tuple(t[:1].clone() for t in layer.lora_a_stacked)
        layer.lora_b_stacked = tuple(t[:1].clone() for t in layer.lora_b_stacked)
        op.release(layer.lora_a_stacked, layer.lora_b_stacked)
        torch.cuda.empty_cache()
        writing = [False]

        def set_lora_s(index, *args, **kw):
            writing[0] = True
            try:
                set_lora(0, *args, **kw)      # its reset_lora(0) lands in reset_lora_s
            finally:
                writing[0] = False
            op.requantize(index)

        def reset_lora_s(index):
            reset_lora(0)
            if not writing[0]:
                op.requantize(index)          # a cleared slot: the copy of the zeroed staging slot

        layer.set_lora, layer.reset_lora = set_lora_s, reset_lora_s
        layer._lowbit_released = True
        return

    def set_lora_q(index, *args, **kw):
        set_lora(index, *args, **kw)
        op.requantize(index)

    def reset_lora_q(index):
        reset_lora(index)
        op.requantize(index)

    layer.set_lora, layer.reset_lora = set_lora_q, reset_lora_q


def install() -> None:
    global _INSTALLED
    if _INSTALLED or os.environ.get("LOWBIT_FUSED", "0") in ("0", "", "false", "False"):
        return
    # vLLM's torch.compile cache is keyed on model and config, not on this patch: a cached graph compiled without the
    # plugin would silently bypass it (seen 10-09: a fused run reproduced the stock run bit for bit), and a graph
    # compiled with it bakes in the fused layers' ids. So the plugin keys the cache itself (_cache_root). Disabling the
    # cache instead (LOWBIT_COMPILE_CACHE=0; the behavior up to 10-14) compiles inside vLLM's memory profiling at every
    # engine start, which leaves less KV cache (Qwen3.6-27B FP8 on an L40S: 7.1 GiB, against 9.1 for a cached graph).
    if os.environ.get("LOWBIT_COMPILE_CACHE", "1") in ("0", "", "false", "False"):
        os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    else:
        os.environ["VLLM_CACHE_ROOT"] = _cache_root()
    from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA
    from vllm.model_executor.kernels.linear.scaled_mm.cutlass import CutlassFP8ScaledMMLinearKernel
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4LinearMethod
    from vllm.model_executor.layers.quantization.online.fp8 import Fp8PerTensorOnlineLinearMethod
    from vllm.utils.torch_utils import direct_register_custom_op

    from lowbit_lora.kernels.fp8_lora_vllm import _BUCKETS, VllmFusedFP8LoRA
    from lowbit_lora.kernels.nvfp4_lora_vllm import VllmFusedNVFP4LoRA

    orig_nvfp4_process = ModelOptNvFp4LinearMethod.process_weights_after_loading

    def nvfp4_process(self, layer):
        """Keep the checkpoint layout (codes [N, K/2], E4M3 block scales [N, K/16], linear) before the backend
        re-layouts it; copies, since a backend may shuffle in place. Globals as vLLM takes them (max over shards).
        The codes are then shared with vLLM whenever its backend kept them where and as they were (CUTLASS at aligned
        shapes swizzles only the scales), so only the linear-layout scales cost memory (~3.5 GB of an 8B model's
        KV cache otherwise); LOWBIT_NVFP4_SHARE=0 keeps the copy."""
        layer._lowbit_nvfp4 = st = {"w_q": layer.weight.data.clone(),
                                    "w_sf": layer.weight_scale.data.view(torch.float8_e4m3fn).clone(),
                                    "input_scale": float(layer.input_scale.max()),
                                    "weight_scale_2": float(layer.weight_scale_2.max())}
        out = orig_nvfp4_process(self, layer)
        w = layer.weight.data
        if (os.environ.get("LOWBIT_NVFP4_SHARE", "1") != "0" and w.dtype == st["w_q"].dtype
                and w.shape == st["w_q"].shape and w.is_contiguous() and torch.equal(w, st["w_q"])):
            st["w_q"] = w
        return out

    ModelOptNvFp4LinearMethod.process_weights_after_loading = nvfp4_process

    direct_register_custom_op(op_name="lowbit_fused_fp8_lora", op_func=_fused, mutates_args=[],
                              fake_impl=_fused_fake)
    fused_op = torch.ops.vllm.lowbit_fused_fp8_lora
    orig_sync = BaseLinearLayerWithLoRA._apply_sync
    orig_apply = BaseLinearLayerWithLoRA.apply
    orig_set_mapping = BaseLinearLayerWithLoRA.set_mapping
    stats = {"fused_layers": 0, "fallback_layers": 0}

    def _warm(op) -> None:
        """The load-time warm-up (compile and time every kernel variant). By default at once, inside vLLM's layer-by-layer
        creation of the LoRA modules, where its freed temporaries' allocator blocks are reused by later layers' LoRA buffers
        (a 48 MiB buffer in a 64 MiB block; the KV-gap diagnostic of 10-24: 364k against 411k KV tokens without the warm-up).
        LOWBIT_WARMUP_DEFER=1 (opt-in, 2026-10-25; default off, so earlier runs are unchanged) queues it until vLLM has created
        every LoRA module, then runs it and returns the freed blocks to the driver before vLLM sizes its KV cache."""
        if os.environ.get("LOWBIT_WARMUP", "1") == "0":
            return
        if os.environ.get("LOWBIT_WARMUP_DEFER", "0") not in ("0", "", "false", "False"):
            _PENDING_WARMUP.append(op)
        else:
            op.warmup()

    def _build_nvfp4(layer, base, qm) -> int:
        st = getattr(base, "_lowbit_nvfp4", None)
        ok = (st is not None and getattr(layer, "tp_size", 1) == 1 and getattr(base, "bias", None) is None
              and getattr(layer, "lora_a_stacked", None) is not None and len(layer.lora_a_stacked) <= 4
              and all(t.dtype == torch.bfloat16 for t in (*layer.lora_a_stacked, *layer.lora_b_stacked)))
        if not ok:
            return -1
        widths = [t.shape[2] for t in layer.lora_b_stacked]
        w_q, w_sf = st["w_q"], st["w_sf"]
        if not (w_q.shape[0] == sum(widths) and 2 * w_q.shape[1] == layer.lora_a_stacked[0].shape[3]
                and w_q.shape[1] % 32 == 0 and all(w % 64 == 0 for w in widths)
                and layer.lora_a_stacked[0].shape[2] % 16 == 0):
            return -1
        max_m = layer.punica_wrapper._token_lora_indices.numel()
        on = lambda v: os.environ.get(v, "0") not in ("0", "", "false", "False")   # noqa: E731
        fmt = "nvfp4" if on("LOWBIT_LORA_FP4") else "fp8" if on("LOWBIT_LORA_FP8") else "bf16"
        op = VllmFusedNVFP4LoRA(w_q, w_sf, 1.0 / st["weight_scale_2"], 1.0 / st["input_scale"], layer.lora_a_stacked,
                                layer.lora_b_stacked, max_m, lora_fmt=fmt)
        if fmt != "bf16":
            _hook_slot_writes(layer, op)
        op.base_fn = lambda x, _b=base, _qm=qm: _qm.apply(_b, x)       # vLLM's NVFP4 GEMM, for M > 64 (hybrid)
        _warm(op)
        _REGISTRY.append(op)
        return len(_REGISTRY) - 1

    def _build(layer) -> int:
        """Decide and construct once per layer, eagerly: vLLM calls set_mapping right after wrapping a layer, after the
        base weights were quantized and before torch.compile traces the forward (Dynamo cannot trace data_ptr())."""
        base = layer.base_layer
        qm = getattr(base, "quant_method", None)
        if isinstance(qm, ModelOptNvFp4LinearMethod):
            lid = _build_nvfp4(layer, base, qm)
            stats["fused_layers" if lid >= 0 else "fallback_layers"] += 1
            rep = os.environ.get("LOWBIT_REPORT")
            if rep:
                with open(rep, "a") as fh:
                    d = _REGISTRY[lid].dispatch if lid >= 0 else {}
                    dtxt = ",".join(f"{b}:{ {'hybrid': 'h', 'hybrids': 's'}.get(v, 'f') }"
                                    for b, v in sorted(d.items())) or "default"
                    fh.write(f"{os.getpid()} {type(layer).__name__} {'fused-nvfp4' if lid >= 0 else 'fallback'} "
                             f"{getattr(base, 'prefix', '')} dispatch={dtxt}\n")
            return lid
        # LOWBIT_TP=1 (opt-in, 2026-10-25; mock reviews R1 v34 / R3 v35): also tensor-parallel shards and layers with a bias. vLLM's
        # (not fully sharded) LoRA layers already shard per rank as the fused op computes: column-parallel layers keep the whole
        # shrink and slice B by output, row-parallel ones slice A by input and add their partial LoRA term before the layer's
        # all-reduce; a bias is added after the fused call. Off by default, so every earlier run is unchanged.
        tp_ok = os.environ.get("LOWBIT_TP", "0") not in ("0", "", "false", "False")
        ok = (isinstance(qm, Fp8PerTensorOnlineLinearMethod) and isinstance(qm.fp8_linear, CutlassFP8ScaledMMLinearKernel)
              and (getattr(layer, "tp_size", 1) == 1 or tp_ok) and (getattr(base, "bias", None) is None or tp_ok)
              and base.weight.dtype == torch.float8_e4m3fn and base.weight_scale.numel() == 1
              and getattr(layer, "lora_a_stacked", None) is not None and len(layer.lora_a_stacked) <= 4
              and all(t.dtype == torch.bfloat16 for t in (*layer.lora_a_stacked, *layer.lora_b_stacked)))
        if ok:
            w_q = base.weight.t()                                       # [N, K], the storage vLLM quantized into
            widths = [t.shape[2] for t in layer.lora_b_stacked]
            ok = (w_q.is_contiguous() and w_q.shape[0] == sum(widths) and w_q.shape[1] == layer.lora_a_stacked[0].shape[3]
                  and all(w % 64 == 0 for w in widths) and layer.lora_a_stacked[0].shape[2] % 16 == 0)
        lid = -1
        if ok:
            max_m = layer.punica_wrapper._token_lora_indices.numel()
            lora8 = os.environ.get("LOWBIT_LORA_FP8", "0") not in ("0", "", "false", "False")
            lora4 = os.environ.get("LOWBIT_LORA_FP4", "0") not in ("0", "", "false", "False")
            op = VllmFusedFP8LoRA(w_q, base.weight_scale.reshape(1).float(), layer.lora_a_stacked,
                                  layer.lora_b_stacked, max_m, lora_fp8=lora8, lora_fp4=lora4)
            # LOWBIT_DISPATCH: auto (default) times, per bucket of M, our GEMM with the LoRA epilogue against vLLM's
            # GEMM plus our LoRA kernels, and keeps the faster; fused / hybrid force one path
            disp = os.environ.get("LOWBIT_DISPATCH", "auto")
            if disp in ("auto", "hybrid"):
                op.base_fn = lambda x, _b=base, _qm=qm: _qm.apply(_b, x)
                if os.environ.get("LOWBIT_LEAN_BASE", "0") not in ("0", "", "false", "False"):
                    op.base_fn = _lean_base(base, qm.fp8_linear) or op.base_fn
                if os.environ.get("LOWBIT_HYBRIDQ", "0") not in ("0", "", "false", "False"):
                    # vLLM's CUTLASS call on our codes and scales (bit-identical to its own quantization): the hybrid
                    # path minus vLLM's quantize kernel, a third candidate for the timed dispatch
                    op.base_q_fn = lambda xq, sx, _b=base, _k=qm.fp8_linear: _k.apply_scaled_mm(
                        A=xq, B=_b.weight, out_dtype=torch.bfloat16, As=sx, Bs=_b.weight_scale, bias=None,
                        output_shape=[xq.shape[0], _b.weight.shape[1]])
            if lora8 or lora4:
                _hook_slot_writes(layer, op)
            _warm(op)                                                   # compile all variants before serving
            if disp == "hybrid":
                op.dispatch = {b: "hybrid" for b in _BUCKETS}
            _REGISTRY.append(op)
            lid = len(_REGISTRY) - 1
            stats["fused_layers"] += 1
        else:
            stats["fallback_layers"] += 1
        rep = os.environ.get("LOWBIT_REPORT")
        if rep:   # one line per layer decision (pid-tagged; workers are separate processes)
            d = _REGISTRY[lid].dispatch if lid >= 0 else {}
            dtxt = ",".join(f"{b}:{ {'hybrid': 'h', 'hybridq': 'q', 'hybrids': 's'}.get(v, 'f') }"
                            for b, v in sorted(d.items())) or "fused"
            with open(rep, "a") as fh:
                a16 = _REGISTRY[lid].a16_max_m if lid >= 0 else 0
                fh.write(f"{os.getpid()} {type(layer).__name__} {'fused' if lid >= 0 else 'fallback'} "
                         f"{getattr(base, 'prefix', '')} dispatch={dtxt}" + (f" w8a16<={a16}" if a16 else "") + "\n")
        return lid

    def set_mapping(self, punica_wrapper):
        orig_set_mapping(self, punica_wrapper)
        if "_lowbit_id" not in self.__dict__:
            self._lowbit_id = _build(self)

    def _note_fallback(layer, why: str) -> None:
        rep = os.environ.get("LOWBIT_REPORT")
        if rep and not layer.__dict__.get("_lowbit_fb"):
            layer.__dict__["_lowbit_fb"] = True
            with open(rep, "a") as fh:
                fh.write(f"{os.getpid()} {type(layer).__name__} runtime-fallback {why}\n")

    tp_bias = os.environ.get("LOWBIT_TP", "0") not in ("0", "", "false", "False")

    def _apply_sync(self, x, bias=None):
        lid = getattr(self, "_lowbit_id", -1)
        if lid < 0:
            return orig_sync(self, x, bias)
        if bias is not None and tp_bias and x.ndim == 2 and x.is_contiguous() and x.dtype == torch.bfloat16:
            return _apply_sync(self, x) + bias                       # LOWBIT_TP=1: the fused call, then the layer's bias
        if bias is not None or x.ndim != 2 or not x.is_contiguous() or x.dtype != torch.bfloat16:
            if self.__dict__.get("_lowbit_released"):   # vLLM's path has no adapters left to read: reshape, or refuse
                if bias is not None or x.dtype != torch.bfloat16:
                    raise RuntimeError(f"lowbit: released LoRA slots cannot serve bias={bias is not None} "
                                       f"dtype={x.dtype}; rerun with LOWBIT_RELEASE_SLOTS=0")
                x2 = x.reshape(-1, x.shape[-1]).contiguous()
                return _apply_sync(self, x2).reshape(*x.shape[:-1], -1)
            if not torch.compiler.is_compiling():   # file I/O cannot be traced; eager runs report the reason
                _note_fallback(self, f"bias={bias is not None} ndim={x.ndim} contiguous={x.is_contiguous()} "
                                     f"dtype={x.dtype} shape={tuple(x.shape)} stride={tuple(x.stride())}")
            return orig_sync(self, x, bias)
        tok, srt, seg_n, seg_start, seg_ids = self.punica_wrapper.token_mapping_meta.meta_args(
            x.size(0), self.lora_config.specialize_active_lora)[:5]
        return fused_op(x, lid, tok, srt, seg_n, seg_start, seg_ids)

    def apply(self, x, bias=None):
        if getattr(self, "_lowbit_id", -1) >= 0:
            return _apply_sync(self, x, bias)
        return orig_apply(self, x, bias)

    if os.environ.get("LOWBIT_WARMUP_DEFER", "0") not in ("0", "", "false", "False"):
        from vllm.lora.model_manager import LoRAModelManager
        orig_create = LoRAModelManager._create_lora_modules

        def _create_lora_modules(self):
            orig_create(self)
            if _PENDING_WARMUP:
                for op in _PENDING_WARMUP:
                    op.warmup()
                _PENDING_WARMUP.clear()
                torch.cuda.empty_cache()
                if os.environ.get("LOWBIT_VERBOSE"):
                    print("[lowbit] deferred warm-up done", flush=True)

        LoRAModelManager._create_lora_modules = _create_lora_modules
    BaseLinearLayerWithLoRA.set_mapping = set_mapping
    BaseLinearLayerWithLoRA._apply_sync = _apply_sync
    BaseLinearLayerWithLoRA.apply = apply
    _INSTALLED = True
    if os.environ.get("LOWBIT_VERBOSE"):
        print("[lowbit] fused FP8 + LoRA plugin installed", flush=True)
