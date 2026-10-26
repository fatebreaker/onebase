"""CPU check of HOPPER-PDL-E2E's plugin wiring (lowbit_lora/hopper_pdl.install) and of its output check
(bench/e2e_outcheck.compare), no GPU: Triton's interpreter, the plugin's timed dispatch with a fake timer.
  * install() is a no-op without LOWBIT_PDL=1 and patches VllmFusedFP8LoRA once with it; pylib_pdl's entry point is found;
  * the patched tune_dispatch times the PDL path only for decode buckets (M <= 32) next to fused, hybrid and vLLM's kernels
    ("hybrids", emulated here), keeps the fastest, and keys its table apart from the unpatched op's;
  * the op's call (what the plugin's custom op runs) serves the PDL path where the table says so and matches an fp32
    reference (one adapter, several adapters, tokens without one, a no-adapter batch); buckets above 32 never use it;
  * the warm-up grows the shared split-K workspace for every bucket that chose the PDL path (vLLM captures its graphs
    after it) and writes the per-layer report lines;
  * plugin_cfg: exact entries, the nearest rank, the nearest decode bucket re-tiled, defaults for unknown shapes;
  * compare(): identical tokens pass, divergences pass within 2e-2 mean |dlogp| up to each first divergence, larger
    differences or a reference token outside the top 5 fail.
    TRITON_INTERPRET=1 python tests/pdl_plugin_check.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
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


def test_compare() -> None:
    from e2e_outcheck import compare
    ref = {"tokens": [[1, 2, 3], [4, 5, 6]],   # position 1 of prompt 0: a near tie between tokens 2 and 9
           "logprobs": [[{"1": -0.1, "9": -2.0}, {"2": -0.69, "9": -0.70}, {"3": -0.3}],
                        [{"4": -0.4}, {"5": -0.5}, {"6": -0.6}]]}
    same = json.loads(json.dumps(ref))
    r = compare(same, ref)
    check("compare: identical tokens pass", r["ok"] and r["tokens_identical"] and r["mean_abs_dlogp"] == 0, str(r))
    near = {"tokens": [[1, 9, 9], [4, 5, 6]],          # diverges at position 1 where stock's token 2 is a near tie
            "logprobs": [[{"1": -0.11, "9": -2.0}, {"9": -0.69, "2": -0.70}, {"9": -0.1}],
                         [{"4": -0.41}, {"5": -0.51}, {"6": -0.61}]]}
    r = compare(near, ref)
    check("compare: a near-tie divergence passes on log-probs", r["ok"] and not r["tokens_identical"]
          and r["prompts_diverged"] == 1 and r["positions"] == 5, str(r))
    far = json.loads(json.dumps(near))
    far["logprobs"][0][1]["2"] = -1.5                   # stock's token far less likely here: |dlogp| 0.81
    r = compare(far, ref)
    check("compare: a large log-prob difference fails", not r["ok"], str(r))
    gone = json.loads(json.dumps(near))
    del gone["logprobs"][0][1]["2"]
    r = compare(gone, ref)
    check("compare: stock's token outside the top 5 fails", not r["ok"] and r["missing_from_top5"] == 1, str(r))


def test_layer_report() -> None:
    from e2e_outcheck import layer_report
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("11 MergedQKVParallelLinearWithLoRA fused model.layers.0.qkv dispatch=1:f,2:f,4:s\n"
                "11 RowParallelLinearWithLoRA fallback model.layers.0.ba dispatch=fused\n"
                "11 pdl-dispatch-table N=6144 K=4096 R=64 widths=[4096, 1024, 1024] fmt=bf16 cfg_src={} us={}\n"
                "11 pdl-layer 1 N=6144 K=4096 R=64 pdl_dispatch=1:pdl,2:pdl,4:hybrids,64:hybrid\n"
                "11 pdl-layer 2 N=4096 K=4096 R=64 pdl_dispatch=1:pdl,2:fused,4:hybrids,64:hybrid\n"
                "11 op-called\n")
    r = layer_report(f.name)
    os.unlink(f.name)
    check("layer_report: counts", r["fused_layers"] == 1 and r["fallback_layers"] == 1 and r["pdl_layers"] == 2
          and r["pdl_dispatch"]["1"] == {"pdl": 2} and r["pdl_dispatch"]["2"] == {"pdl": 1, "fused": 1}
          and r["op_called"], json.dumps(r))


def test_plugin() -> None:
    import torch
    from vllm.lora.ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta

    from lowbit_lora import hopper_pdl as H
    from lowbit_lora.kernels import fp8_lora_multi as FM
    from lowbit_lora.kernels import fp8_lora_vllm as F
    from lowbit_lora.kernels.fp8_lora_pdl import work

    # pylib_pdl's entry point
    from importlib.metadata import entry_points
    sys.path.insert(0, str(REPO / "pylib_pdl"))
    import importlib
    importlib.invalidate_caches()
    eps = {e.name: e.value for e in entry_points(group="vllm.general_plugins")}
    check("entry point lowbit_pdl -> lowbit_lora.hopper_pdl:install", eps.get("lowbit_pdl") == "lowbit_lora.hopper_pdl:install",
          str(eps))
    # not installed without LOWBIT_PDL=1
    os.environ.pop("LOWBIT_PDL", None)
    H.install()
    check("install() without LOWBIT_PDL=1: no patch", not getattr(F.VllmFusedFP8LoRA, "_pdl_installed", False))
    # a configuration table: one exact entry, one at another rank, one at another bucket
    torch.manual_seed(0)
    K, widths, R, L = 256, (64, 32, 32), 16, 4
    N, S = sum(widths), len(widths)
    tab = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    base = {"X_MODE": 1, "BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 64, "SPLIT_K": 2, "num_warps": 4, "num_stages": 1,
            "SPLIT_K1": 2, "S_BLOCK_K": 32, "S_STAGES": 1}
    rows = [{"job": "7", "M": 1, "N": N, "K": K, "S": S, "r": R, "n": 1, "fmt": "bf16", "pdl_best": base},
            {"job": "7", "M": 4, "N": N, "K": K, "S": S, "r": 64, "n": 1, "fmt": "bf16",
             "pdl_best": {**base, "SPLIT_K": 1}},
            {"job": "7", "M": 16, "N": N, "K": K, "S": S, "r": R, "n": 1, "fmt": "bf16",
             "pdl_best": {**base, "X_MODE": 0}},
            {"job": "7", "M": 16, "N": N, "K": K, "S": S, "r": R, "n": 8, "fmt": "bf16",   # several adapters: ignored
             "pdl_best": {**base, "BLOCK_K": 128}},
            {"job": "6", "M": 2, "N": N, "K": K, "S": S, "r": R, "n": 1, "fmt": "bf16",    # an older job: ignored
             "pdl_best": {**base, "BLOCK_K": 128}}]
    tab.write("\n".join(json.dumps(r) for r in rows) + "\n")
    tab.close()
    rep = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    rep.close()
    os.environ.update(LOWBIT_PDL="1", LOWBIT_PDL_CFG=tab.name, LOWBIT_REPORT=rep.name)
    H.install()
    V = F.VllmFusedFP8LoRA
    check("install() with LOWBIT_PDL=1 patches once", getattr(V, "_pdl_installed", False))
    run_patched = V._run
    H.install()
    check("install() twice is a no-op", V._run is run_patched)
    check("table: newest job, one-adapter rows only", H._TABLE_SRC["job"] == "7" and len(H._TABLE) == 3, str(H._TABLE_SRC))

    # the op as the plugin builds it, on the CPU (fp32 dots in the interpreter), with a vendor GEMM (torch) and vLLM's
    # LoRA kernels emulated (the "hybrids" candidate)
    w = torch.randn(N, K) * 0.05
    s_w = (w.abs().max() / 448.0).reshape(1).float()
    w_q = (w / s_w).to(torch.float8_e4m3fn)
    a_st = tuple((torch.randn(L, 1, R, K) / K ** 0.5).to(torch.bfloat16) for _ in range(S))
    b_st = tuple((torch.randn(L, 1, n, R) * 0.05).to(torch.bfloat16) for n in widths)
    A = torch.cat([t[:, 0] for t in a_st], dim=1)
    B = torch.cat([t[:, 0] for t in b_st], dim=1)
    op = F.VllmFusedFP8LoRA(w_q, s_w, a_st, b_st, max_m=64)
    op.dot_f32 = True
    op.cc_major = 9

    def vendor(xx):
        return FM.reference_multi(xx, torch.full((xx.shape[0],), -1, dtype=torch.int32), w_q, s_w, A, B, widths,
                                  1.0).to(torch.bfloat16)
    op.base_fn = vendor

    def stock(self, x, meta_args):   # vLLM's shrink + expand after the vendor GEMM, emulated
        tok = meta_args[0][: x.shape[0]]
        return FM.reference_multi(x, tok, w_q, s_w, A, B, widths, 1.0).to(torch.bfloat16)
    V._run_stock = stock
    last = {}
    patched_run = V._run

    def spy(self, x, meta_args, out=None, mode=None, cfg_s=None, cfg_g=None, hybrid=False, split=None):
        last["h"] = hybrid
        return patched_run(self, x, meta_args, out=out, mode=mode, cfg_s=cfg_s, cfg_g=cfg_g, hybrid=hybrid, split=split)
    V._run = spy
    fake = {"p": 1.0, "s": 2.0, True: 3.0, False: 4.0, "q": 5.0}

    def graph_us(fn, reps=10):
        fn()
        return fake[last["h"]]
    F._graph_us = graph_us
    torch.cuda.get_device_name = lambda *a: "NVIDIA H100 80GB HBM3"
    torch.cuda.synchronize = lambda *a: None
    torch.cuda.empty_cache = lambda: None
    op.warmup()
    d = op.dispatch
    check("dispatch: the PDL path wins every timed decode bucket", all(d[b] == "pdl" for b in (1, 2, 4, 8, 16, 32)), str(d))
    check("dispatch: no PDL above M = 32", d.get(64) == "hybrids", str(d))
    keys = [k for k in F._DISPATCH if "pdl" in k]
    check("dispatch table keyed apart (marked pdl)", len(keys) == 1, str(list(F._DISPATCH)))
    # workspace grown for every PDL bucket (split-K 2 at bucket 1, the bucket-16 entry and the re-tiled bucket 32)
    need = max(H.plugin_cfg(op, M).get("SPLIT_K", 1) * M * N for M in (1, 2, 4, 8, 16, 32))
    check("warm-up grew the split-K workspace before any capture", work(torch.device("cpu")).p.numel() >= need,
          f"{work(torch.device('cpu')).p.numel()} >= {need}")
    lines = open(rep.name).read().splitlines()
    check("report: the dispatch table and one pdl-layer line", sum("pdl-dispatch-table" in x for x in lines) == 1
          and sum(" pdl-layer " in x and "pdl_dispatch=1:pdl" in x for x in lines) == 1, "\n".join(lines)[:600])
    # plugin_cfg fallbacks
    src = op.__dict__.get("_pdl_cfg_src", {})
    c1, c2, c4, c8, c32 = (H.plugin_cfg(op, M) for M in (1, 2, 4, 8, 32))
    check("plugin_cfg: exact entry (bucket 1)", c1 == base and src.get(1) == "exact", f"{c1} {src.get(1)}")
    check("plugin_cfg: same bucket, nearest rank (bucket 4: rank 64)", c4.get("SPLIT_K") == 1 and "rank 64" in src.get(4, ""),
          f"{c4} {src.get(4)}")
    check("plugin_cfg: nearest bucket (2 -> 1, then 4 at the other rank... bucket 1 first)", "bucket 1" in src.get(2, ""),
          f"{c2} {src.get(2)}")
    check("plugin_cfg: bucket 32 from bucket 16 re-tiled to 32-token tiles", c32.get("BLOCK_M") == 32
          and c32.get("X_MODE") == 0 and "bucket 16" in src.get(32, ""), f"{c32} {src.get(32)}")

    class Other:   # a shape the table lacks
        N, K, S, R, lora_fp8, lora_fp4 = 999, 999, 1, 64, False, False
    check("plugin_cfg: an unknown shape falls back to the defaults", H.plugin_cfg(Other(), 1) == {})
    # serving through the op's call (what the plugin's registered custom op runs)
    meta = LoRAKernelMeta.make(L, 64, device="cpu")
    worst = 0.0
    for M, kind in ((1, "one"), (5, "mixed"), (16, "several"), (13, "none"), (40, "above 32")):
        x = torch.randn(M, K).to(torch.bfloat16)
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
        ref = FM.reference_multi(x, tok, w_q, s_w, A, B, widths, 1.0)
        err = float((y - ref).norm() / ref.norm())
        worst = max(worst, err)
        via = "pdl" if op.dispatch.get(F._bucket(M)) == "pdl" else op.dispatch.get(F._bucket(M))
        check(f"op call M={M} ({kind}): {via}, rel err {err:.1e}", err < 1e-2 and (via == "pdl") == (M <= 32))
    print(f"worst rel err {worst:.2e}")
    os.unlink(tab.name)
    os.unlink(rep.name)


if __name__ == "__main__":
    assert os.environ.get("TRITON_INTERPRET") == "1"
    test_compare()
    test_layer_report()
    test_plugin()
    print("PASS" if not BAD else f"FAIL {len(BAD)}: {BAD}")
    raise SystemExit(1 if BAD else 0)
