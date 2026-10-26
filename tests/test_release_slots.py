"""CPU check of LOWBIT_RELEASE_SLOTS (plugin._hook_slot_writes): with vLLM's bf16 slots shrunk to one staging slot,
writing every adapter through vLLM's own set_lora / reset_lora must leave the FP8- and NVFP4-stored copies
bit-identical to the copies of fully populated bf16 slots, for both fused ops and every stored format.

    TRITON_INTERPRET=1 PYTHONPATH=.:../mlsys python tests/test_release_slots.py
"""
from __future__ import annotations

import os
import types

import torch

os.environ["LOWBIT_RELEASE_SLOTS"] = "1"
from lowbit_lora import plugin  # noqa: E402
from lowbit_lora.kernels import fp8_lora_vllm as F8  # noqa: E402
from lowbit_lora.kernels import nvfp4_lora_vllm as F4  # noqa: E402


def fake_layer(a_st, b_st):
    """An object with vLLM's own slot methods: MergedColumnParallelLinearWithLoRA.set_lora (reset_lora(index) via the
    instance, then copy into [index, 0, :rows, :cols] per slice) and BaseLinearLayerWithLoRA.reset_lora."""
    from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA
    from vllm.lora.layers.column_parallel_linear import MergedColumnParallelLinearWithLoRA
    lay = types.SimpleNamespace(lora_a_stacked=a_st, lora_b_stacked=b_st, n_slices=len(a_st), tp_size=1)
    lay.reset_lora = types.MethodType(BaseLinearLayerWithLoRA.reset_lora, lay)
    lay.set_lora = types.MethodType(MergedColumnParallelLinearWithLoRA.set_lora, lay)
    return lay


def stored(op) -> list[torch.Tensor]:
    st = getattr(op, "store", None)
    if st is not None:
        ts = [*st.a_q, *st.a_s, *st.b_q, *st.b_s, st.ag, st.bg]
    elif op.lora_fp4:
        ts = [*op.a4, *op.a4s, *op.b4, *op.b4s, op.ag, op.bg]
    else:
        ts = [*op.a8, *op.b8, *op.sa, *op.sb]
    return [t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t for t in ts]


def make(kind, fmt, w, a_st, b_st):
    if kind == "fp8":
        w_q = (w / (w.abs().max() / 448)).to(torch.float8_e4m3fn)
        return F8.VllmFusedFP8LoRA(w_q, torch.ones(1), a_st, b_st, max_m=32, split_k=2,
                                   lora_fp8=fmt == "fp8", lora_fp4=fmt == "nvfp4")
    g = F4.global_scale_for(w)
    w_q, w_sf = F4.nvfp4_quant_torch(w, g)
    return F4.VllmFusedNVFP4LoRA(w_q, w_sf, g, 1.0, a_st, b_st, max_m=32, split_k=2, lora_fmt=fmt)


def main() -> int:
    torch.manual_seed(0)
    K, widths, R, L = 128, (64, 32, 32), 32, 4
    N, S = sum(widths), len(widths)
    w = torch.randn(N, K) * 0.05
    A = [[(torch.randn(R, K) / K ** 0.5).to(torch.bfloat16) for _ in range(S)] for _ in range(L)]
    B = [[(torch.randn(n, R) * 0.05).to(torch.bfloat16) for n in widths] for _ in range(L)]
    A[2][1] = A[2][1] * 0                                     # a zero slice inside a slot
    fails = 0
    for kind in ("fp8", "nvfp4"):
        for fmt in ("fp8", "nvfp4"):
            full_a = tuple(torch.stack([A[i][s] for i in range(L)])[:, None] for s in range(S))
            full_b = tuple(torch.stack([B[i][s] for i in range(L)])[:, None] for s in range(S))
            ref = make(kind, fmt, w, full_a, full_b)          # copies of fully populated bf16 slots
            lay = fake_layer(tuple(torch.zeros_like(t) for t in full_a), tuple(torch.zeros_like(t) for t in full_b))
            op = make(kind, fmt, w, lay.lora_a_stacked, lay.lora_b_stacked)
            plugin._hook_slot_writes(lay, op)
            assert all(t.shape[0] == 1 for t in (*lay.lora_a_stacked, *lay.lora_b_stacked)), "slots not released"
            for i in (3, 0, 2, 1):                            # any order; each write goes through the staging slot
                lay.set_lora(i, list(A[i]), list(B[i]))
            ok = all(torch.equal(x, y) for x, y in zip(stored(op), stored(ref)))
            # clearing slot 1 must equal the copies of slots whose slot 1 is zero
            lay.reset_lora(1)
            za = tuple(t.clone() for t in full_a)
            zb = tuple(t.clone() for t in full_b)
            for t in (*za, *zb):
                t[1] = 0
            ok_reset = all(torch.equal(x, y) for x, y in zip(stored(op), stored(make(kind, fmt, w, za, zb))))
            print(f"{kind} op, {fmt}-stored: writes {'OK' if ok else 'MISMATCH'}, clear {'OK' if ok_reset else 'MISMATCH'}")
            fails += (not ok) + (not ok_reset)
    print("release-slots test", "PASSED" if not fails else f"FAILED ({fails})")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
