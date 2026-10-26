"""CPU check of BLACKWELL-PDL-E2E's plugin wiring (lowbit_lora/blackwell_pdl_plugin.install), no GPU: Triton's
interpreter, the plugin's NVFP4 timed dispatch with a fake timer (as tests/pdl_plugin_check.py for HOPPER-PDL-E2E).
  * install() is a no-op without LOWBIT_PDL=1 and patches VllmFusedNVFP4LoRA once with it; pylib_pdl4's entry point is
    found; the Hopper op is left alone;
  * the patched tune_dispatch times the PDL path only for decode buckets (M <= 32) next to the fused, hybrid and vLLM-kernel
    ("hybrids", emulated here) paths, keeps the fastest, and keys its table apart from the unpatched op's;
  * the op's call (what the plugin's custom op runs) serves the PDL path where the table says so and matches the fp32
    reference (one adapter, several adapters, tokens without one, a no-adapter batch after an adapter batch); buckets
    above 32 never use it; without the compute capability the PDL path is never a candidate;
  * the warm-up grows the shared split-K workspace for every bucket that chose the PDL path (vLLM captures its graphs
    after it) and writes the per-layer report lines, which bench/e2e_outcheck.layer_report counts (with the plugin's own
    NVFP4 layer lines);
  * plugin_cfg4: exact entries, the nearest rank, the nearest decode bucket re-tiled (shared memory kept within sm_120's),
    defaults for unknown shapes; results/pdl_fp4.jsonl (BLACKWELL-PDL-K2's job) gives every Qwen3-8B shape an entry at
    every decode bucket (exact at 1/4/8/16/32 for rank 64, re-tiled at 2).
    TRITON_INTERPRET=1 python tests/pdl4_plugin_check.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bench"))
os.environ.setdefault("TRITON_INTERPRET", "1")

BAD: list = []


def check(name, ok, msg=""):
    print(("  ok   " if ok else "  FAIL ") + name + (f"  {msg}" if msg else ""), flush=True)
    if not ok:
        BAD.append(name)


def test_layer_report() -> None:
    from e2e_outcheck import layer_report
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("11 pdl-dispatch-table N=6144 K=4096 R=64 widths=[4096, 1024, 1024] fmt=bf16 cfg_src={} us={}\n"
                "11 pdl-layer 1 N=6144 K=4096 R=64 pdl_dispatch=1:pdl,2:pdl,4:hybrids,64:hybrid\n"
                "11 MergedQKVParallelLinearWithLoRA fused-nvfp4 model.layers.0.self_attn.qkv_proj dispatch=1:f,2:f,4:s\n"
                "11 pdl-layer 2 N=4096 K=4096 R=64 pdl_dispatch=1:pdl,2:fused,4:hybrids,64:hybrid\n"
                "11 RowParallelLinearWithLoRA fused-nvfp4 model.layers.0.self_attn.o_proj dispatch=1:f,2:f,4:s\n"
                "11 op-called\n")
    r = layer_report(f.name)
    os.unlink(f.name)
    check("layer_report: NVFP4 layer lines and PDL lines counted", r["fused_layers"] == 2 and r["fallback_layers"] == 0
          and r["pdl_layers"] == 2 and r["pdl_dispatch"]["1"] == {"pdl": 2}
          and r["pdl_dispatch"]["2"] == {"pdl": 1, "fused": 1} and r["op_called"], json.dumps(r))


def test_real_table() -> None:
    """The table the E2E job will load (results/pdl_fp4.jsonl), on Qwen3-8B's four projection shapes."""
    from lowbit_lora import blackwell_pdl_plugin as P
    from lowbit_lora.blackwell_pdl import resolve_cfg4
    path = REPO / "results/pdl_fp4.jsonl"
    if not path.exists():
        check("results/pdl_fp4.jsonl present", False)
        return
    n = P.load_table4(str(path))
    check("real table: loads one job's one-adapter rows", n > 0 and P._TABLE_SRC["job"] is not None, str(P._TABLE_SRC))
    shapes = {"qkv": (6144, 4096, 3), "o": (4096, 4096, 1), "gate_up": (24576, 4096, 2), "down": (4096, 12288, 1)}
    widths = {"qkv": (4096, 1024, 1024), "o": (4096,), "gate_up": (12288, 12288), "down": (4096,)}
    for proj, (N, K, S) in shapes.items():
        for R in (64, 16):
            op = types.SimpleNamespace(N=N, K=K, S=S, R=R, lora_fmt="bf16", widths=list(widths[proj]),
                                       w_q=types.SimpleNamespace(device=types.SimpleNamespace(type="cpu")))
            cfgs = {b: P.plugin_cfg4(op, b) for b in P.decode_buckets()}
            src = op._pdl4_cfg_src
            if R == 64:
                ok = all(src[b] == "exact" for b in (1, 4, 8, 16, 32)) and src[2].startswith("bucket ")
            else:
                ok = src[1] == "exact" and src[16] == "exact" and all(src[b] != "default" for b in cfgs)
            res = {b: resolve_cfg4(op, b, c) for b, c in cfgs.items()}
            fits = all(P._smem_ok(c) and all(w % c["BLOCK_N"] == 0 for w in widths[proj]) for c in res.values())
            swap_ok = all((not c.get("SWAP")) or b <= 16 for b, c in res.items())
            check(f"real table: {proj} r{R} has an entry for every decode bucket", ok and fits and swap_ok,
                  json.dumps(src))


def test_plugin() -> None:
    import torch
    from vllm.lora.ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta

    from lowbit_lora import blackwell_pdl_plugin as P
    from lowbit_lora.blackwell_pdl import resolve_cfg4
    from lowbit_lora.kernels import fp8_lora_vllm as F
    from lowbit_lora.kernels import nvfp4_lora_vllm as F4
    from lowbit_lora.kernels.nvfp4_lora_pdl import work

    # pylib_pdl4's entry point
    import importlib
    from importlib.metadata import entry_points
    sys.path.insert(0, str(REPO / "pylib_pdl4"))
    importlib.invalidate_caches()
    eps = {e.name: e.value for e in entry_points(group="vllm.general_plugins")}
    check("entry point lowbit_pdl4 -> lowbit_lora.blackwell_pdl_plugin:install",
          eps.get("lowbit_pdl4") == "lowbit_lora.blackwell_pdl_plugin:install", str(eps))
    # not installed without LOWBIT_PDL=1
    os.environ.pop("LOWBIT_PDL", None)
    P.install()
    check("install() without LOWBIT_PDL=1: no patch", not getattr(F4.VllmFusedNVFP4LoRA, "_pdl4_installed", False))
    # a configuration table: one exact entry, one at another rank, one at another bucket, ignored rows
    torch.manual_seed(0)
    K, widths, R, L = 256, (64, 64), 16, 4
    N, S = sum(widths), len(widths)
    tab = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    base = {"X_MODE": 1, "SWAP": True, "BLOCK_M": 8, "BLOCK_N": 64, "BLOCK_K": 64, "SPLIT_K": 2, "num_stages": 1,
            "SPLIT_K1": 2, "S_BLOCK_K": 32, "S_STAGES": 1, "QBM": 16, "QBK": 64}
    rows = [{"job": "7", "M": 1, "N": N, "K": K, "S": S, "r": R, "n": 1, "fmt": "bf16", "pdl_best": base},
            {"job": "7", "M": 4, "N": N, "K": K, "S": S, "r": 64, "n": 1, "fmt": "bf16",
             "pdl_best": {**base, "SPLIT_K": 1}},
            {"job": "7", "M": 16, "N": N, "K": K, "S": S, "r": R, "n": 1, "fmt": "bf16",
             "pdl_best": {**base, "X_MODE": 0, "BLOCK_M": 16}},
            {"job": "7", "M": 16, "N": N, "K": K, "S": S, "r": R, "n": 8, "fmt": "bf16",   # several adapters: ignored
             "pdl_best": {**base, "BLOCK_K": 128}},
            {"job": "6", "M": 2, "N": N, "K": K, "S": S, "r": R, "n": 1, "fmt": "bf16",    # an older job: ignored
             "pdl_best": {**base, "BLOCK_K": 128}}]
    tab.write("\n".join(json.dumps(r) for r in rows) + "\n")
    tab.close()
    rep = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    rep.close()
    os.environ.update(LOWBIT_PDL="1", LOWBIT_PDL_CFG=tab.name, LOWBIT_REPORT=rep.name)
    hopper_call = F.VllmFusedFP8LoRA.__call__
    P.install()
    V = F4.VllmFusedNVFP4LoRA
    check("install() with LOWBIT_PDL=1 patches the NVFP4 op once", getattr(V, "_pdl4_installed", False))
    patched_call = V.__call__
    P.install()
    check("install() twice is a no-op", V.__call__ is patched_call)
    check("the FP8 (Hopper) op is untouched", F.VllmFusedFP8LoRA.__call__ is hopper_call)
    check("table: newest job, one-adapter rows only", P._TABLE_SRC["job"] == "7" and len(P._TABLE) == 3,
          str(P._TABLE_SRC))

    # the op as the plugin builds it, on the CPU (fp32 dots in the interpreter), with vLLM's NVFP4 GEMM and its LoRA
    # kernels emulated (the hybrid path's base and the "hybrids" candidate)
    w = (torch.randn(N, K) * 0.05).to(torch.bfloat16)
    w_global = F4.global_scale_for(w.float())
    w_q, w_sf = F4.nvfp4_quant_torch(w, w_global)
    a_st = tuple((torch.randn(L, 1, R, K) / K ** 0.5).to(torch.bfloat16) for _ in range(S))
    b_st = tuple((torch.randn(L, 1, n, R) * 0.05).to(torch.bfloat16) for n in widths)
    A = torch.cat([t[:, 0] for t in a_st], dim=1)
    B = torch.cat([t[:, 0] for t in b_st], dim=1)
    x_cal = (torch.randn(64, K) * 1.5).to(torch.bfloat16)
    x_global = F4.global_scale_for(x_cal.float())                 # a static calibration, as a checkpoint's input_scale
    op = V(w_q, w_sf, w_global, x_global, a_st, b_st, max_m=64)
    op.dot_f32 = True
    op.cc_major = 12

    def ref(xx, tok):
        return F4.reference(xx, tok, w_q, w_sf, w_global, x_global, A, B, widths)

    last = {}

    def vendor(xx):
        return ref(xx, torch.full((xx.shape[0],), -1, dtype=torch.int32)).to(torch.bfloat16)
    op.base_fn = vendor

    def stock(self, x, meta_args):   # the vendor GEMM, then vLLM's shrink + expand, emulated
        last["p"] = "hybrids"
        return ref(x, meta_args[0][: x.shape[0]]).to(torch.bfloat16)
    V._run_stock = stock

    def spy_call(self, x, meta_args, out=None, cfg_s=None, cfg_g=None, base_fn=None, hybrid_m=64, split=None,
                 use_dispatch=True):
        if not use_dispatch:
            last["p"] = "hybrid" if (base_fn is not None and hybrid_m == 0) else "fused"
        return patched_call(self, x, meta_args, out=out, cfg_s=cfg_s, cfg_g=cfg_g, base_fn=base_fn, hybrid_m=hybrid_m,
                            split=split, use_dispatch=use_dispatch)
    V.__call__ = spy_call
    real_run = P.run_pdl4

    def spy_pdl(*a, **kw):
        last["p"] = "pdl"
        return real_run(*a, **kw)
    P.run_pdl4 = spy_pdl
    fake = {"pdl": 1.0, "hybrids": 2.0, "hybrid": 3.0, "fused": 4.0}

    def graph_us(fn, reps=10):
        last.clear()
        fn()
        return fake[last["p"]]
    F._graph_us = graph_us
    torch.cuda.get_device_name = lambda *a: "NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb"
    torch.cuda.synchronize = lambda *a: None
    torch.cuda.empty_cache = lambda: None
    op.warmup()
    d = op.dispatch
    check("dispatch: the PDL path wins every timed decode bucket", all(d.get(b) == "pdl" for b in (1, 2, 4, 8, 16, 32)),
          str(d))
    check("dispatch: no PDL above M = 32", d.get(64) == "hybrids", str(d))
    keys = [k for k in F4._DISPATCH4 if "pdl" in k]
    check("dispatch table keyed apart (marked pdl)", len(keys) == 1, str(list(F4._DISPATCH4)))
    sks = {M: resolve_cfg4(op, M, P.plugin_cfg4(op, M))["SPLIT_K"] for M in (1, 2, 4, 8, 16, 32)}
    need = max(sk * M * N if sk > 1 else 1 for M, sk in sks.items())   # (blackwell_pdl.run_pdl4's work.ensure)
    have = work(torch.device("cpu")).p.numel()
    check("warm-up grew the split-K workspace before any capture", have >= need, f"{have} >= {need}")
    lines = open(rep.name).read().splitlines()
    check("report: the dispatch table and one pdl-layer line", sum("pdl-dispatch-table" in x for x in lines) == 1
          and sum(" pdl-layer " in x and "pdl_dispatch=1:pdl" in x for x in lines) == 1, "\n".join(lines)[:600])
    check("report: no failed PDL candidate or warm-up", not any("pdl-candidate-failed" in x or "pdl-warmup-failed" in x
                                                                for x in lines), "\n".join(lines)[:600])
    # plugin_cfg4 fallbacks
    src = op.__dict__.get("_pdl4_cfg_src", {})
    c1, c2, c4, c8, c32 = (P.plugin_cfg4(op, M) for M in (1, 2, 4, 8, 32))
    check("plugin_cfg4: exact entry (bucket 1)", c1 == base and src.get(1) == "exact", f"{c1} {src.get(1)}")
    check("plugin_cfg4: same bucket, nearest rank (bucket 4: rank 64)", c4.get("SPLIT_K") == 1
          and "rank 64" in src.get(4, ""), f"{c4} {src.get(4)}")
    check("plugin_cfg4: bucket 2 from bucket 1 (this rank first), 8-token swap-AB tiles", "bucket 1" in src.get(2, "")
          and c2.get("SWAP") is True and c2.get("BLOCK_M") == 8, f"{c2} {src.get(2)}")
    check("plugin_cfg4: bucket 8 from bucket 16, re-tiled to 8-token tiles", "bucket 16" in src.get(8, "")
          and c8.get("BLOCK_M") == 8 and c8.get("X_MODE") == 0, f"{c8} {src.get(8)}")
    check("plugin_cfg4: bucket 32 from bucket 16, the plugin's orientation with 32-token tiles",
          c32.get("BLOCK_M") == 32 and c32.get("SWAP") is False and c32.get("X_MODE") == 0
          and "bucket 16" in src.get(32, ""), f"{c32} {src.get(32)}")
    big = P._retile4({"X_MODE": 0, "SWAP": True, "BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 256, "num_stages": 4,
                      "num_warps": 8}, 32)
    check("_retile4: pipeline shortened to fit sm_120's shared memory", P._smem_ok(big) and big["num_stages"] < 4
          and "num_warps" not in big, str(big))
    other = types.SimpleNamespace(N=999, K=999, S=1, R=64, lora_fmt="bf16")
    check("plugin_cfg4: an unknown shape falls back to the defaults", P.plugin_cfg4(other, 1) == {})
    # serving through the op's call (what the plugin's registered custom op runs)
    V.__call__ = patched_call
    meta = LoRAKernelMeta.make(L, 64, device="cpu")
    worst = 0.0
    for M, kind in ((1, "one"), (5, "mixed"), (16, "several"), (13, "none"), (40, "above 32")):
        x = (torch.randn(M, K) * 1.5).to(torch.bfloat16)
        if kind == "one":
            tok = torch.full((M,), 2, dtype=torch.int32)
        elif kind == "mixed":
            tok = torch.tensor([0, -1, 3, 0, 1], dtype=torch.int32)
        elif kind == "several":
            tok = (torch.arange(M) % L).to(torch.int32)
        elif kind == "none":
            meta.prepare_tensors(torch.zeros(M, dtype=torch.int32))
            tok = torch.full((M,), -1, dtype=torch.int32)
        else:
            tok = torch.randint(-1, L, (M,), dtype=torch.int32)
        meta.prepare_tensors(tok)
        last.clear()
        y = op(x, meta.meta_args(M, False)).float()
        r = ref(x, tok)
        err = float((y - r).norm() / r.norm())
        worst = max(worst, err)
        via = op.dispatch.get(P._bucket(M))
        check(f"op call M={M} ({kind}): {via}, rel err {err:.1e}", err < 1e-2 and (via == "pdl") == (M <= 32))
    print(f"worst rel err {worst:.2e}")
    # without compute capability 12 the PDL path is never a candidate
    F4._DISPATCH4.clear()
    op2 = V(w_q, w_sf, w_global, x_global, a_st, b_st, max_m=16)
    op2.dot_f32 = True
    op2.cc_major = 9
    op2.base_fn = vendor
    V.__call__ = spy_call
    op2.tune_dispatch()
    check("other compute capabilities: no PDL candidate", "pdl" not in op2.dispatch.values(), str(op2.dispatch))
    V.__call__ = patched_call
    os.unlink(tab.name)
    os.unlink(rep.name)


if __name__ == "__main__":
    assert os.environ.get("TRITON_INTERPRET") == "1"
    test_layer_report()
    test_real_table()
    test_plugin()
    print("PASS" if not BAD else f"FAIL {len(BAD)}: {BAD}")
    raise SystemExit(1 if BAD else 0)
