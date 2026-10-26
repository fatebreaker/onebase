"""STOCK-TUNED (docs/REGISTRY.md): vLLM's own LoRA kernels (lora_shrink, lora_expand) tuned for this GPU and model, so that
the stock path is compared at its best (MLSys v1 mock review). Two steps, each its own process:
  record  vLLM with its engine in this process (VLLM_ENABLE_V1_MULTIPROCESSING=0) and vLLM's own LoRA path (LOWBIT_FUSED=0),
          LoRA on --model with 32 slots of --adapter's rank, CUDA graphs captured and a short generation with adapters: the
          shape of every shrink and expand call (each slice's weight shape, the output width and offset, add_inputs, the
          dtypes), read from the call's frame when it asks get_lora_op_configs for its configuration -> --out/shapes.json
  tune    for each recorded call shape, rank in --ranks, slot count in --max-loras and token count in --ms, every candidate
          configuration (block sizes, split-K for the shrink, warps, stages; vLLM's default always among them) timed with
          CUDA graphs (bench/kernel_bench.Timer, L2 flushed) on synthetic tensors whose tokens cycle over min(slots, tokens)
          adapters; the fastest kept whose output matches the default configuration's (relative error below 1e-2). Calls
          that share a key of vLLM's lookup get the configuration with the least summed time.
Writes vLLM's tuned-config files to --out (<gpu>_SHRINK.json, <gpu>_EXPAND_<TRUE|FALSE>.json, nested max_loras ->
num_slices -> m -> k -> n as vllm/lora/ops/triton_ops/utils.py reads them when VLLM_TUNED_CONFIG_FOLDER=--out) and every
key's best and default times to --out/tuning.jsonl.
    python bench/tune_vllm_lora.py record --model <dir> --quantization fp8 --adapter <rank-r adapter> --out <dir>
    python bench/tune_vllm_lora.py tune --out <dir> [--ranks 16,64] [--max-loras 1,32] [--ms 1,8,16,32,64,128,256,512,2048,8192]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ["LOWBIT_FUSED"] = "0"

sys.path.insert(0, str(Path(__file__).resolve().parent))


def dt(t) -> str:
    return str(t).split(".")[-1]


def record(a) -> int:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    import vllm.lora.ops.triton_ops.lora_expand_op as eo
    import vllm.lora.ops.triton_ops.lora_shrink_op as so
    orig = so.get_lora_op_configs
    shapes: dict = {}

    def rec(op_type, *args, **kw):
        f = sys._getframe(1).f_locals
        if op_type == "shrink":
            d = {"op": "shrink", "w": [list(w.shape) for w in f["lora_a_weights"]], "in_dtype": dt(f["inputs"].dtype),
                 "out_dtype": dt(f["output_tensor"].dtype)}
        else:
            d = {"op": "expand", "w": [list(w.shape) for w in f["lora_b_weights"]], "in_dtype": dt(f["inputs"].dtype),
                 "out_dtype": dt(f["output_tensor"].dtype), "out_cols": int(f["output_tensor"].shape[1]),
                 "offset_start": int(f["offset_start"]), "add_inputs": bool(f["add_inputs"])}
        k = json.dumps(d, sort_keys=True)
        if k not in shapes:
            shapes[k] = {**d, "max_loras_seen": int(f["MAX_LORAS"]), "calls": 0}
        shapes[k]["calls"] += 1
        return orig(op_type, *args, **kw)

    so.get_lora_op_configs = rec
    eo.get_lora_op_configs = rec
    kw = {"quantization": a.quantization} if a.quantization else {}
    # E2E-72B-H200 -TILES2 (2026-10-25): --record-loras slots (default 32, as every earlier run); 32 slots of a 72B do not fit one H200
    nl = a.record_loras
    llm = LLM(model=a.model, enable_lora=True, max_lora_rank=a.rank, max_loras=nl, max_cpu_loras=nl, max_model_len=4096,
              gpu_memory_utilization=0.85, enable_prefix_caching=False, seed=0, **kw)
    loras = [LoRARequest(f"a{i}", i + 1, a.adapter) for i in range(nl)]
    sp = SamplingParams(max_tokens=16, ignore_eos=True, temperature=0.0)
    ps = [{"prompt_token_ids": [1000 + (7 * i + j) % 5000 for j in range(64 + 32 * i)]} for i in range(32)]
    llm.generate(ps, sp, lora_request=[loras[i % nl] for i in range(32)], use_tqdm=False)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = sorted(shapes.values(), key=lambda d: (d["op"], json.dumps(d["w"])))
    (out / "shapes.json").write_text(json.dumps({"model": a.model, "quantization": a.quantization, "rank": a.rank,
                                                 "shapes": rows}, indent=1))
    for d in rows:
        print(json.dumps(d))
    return 0


def candidates(op: str, M: int, r: int, default: dict):
    seen = [default]
    yield default
    if op == "shrink":   # about 30 per key: Triton compiles each per specialization, which dominates the tuning time
        bms = (16, 32) if M <= 64 else (32, 64)
        bns = (16,) if r <= 16 else (16, 32)
        bks = (128, 256) if M <= 128 else (32, 64, 128)
        sks = (8, 16, 32, 64) if M <= 128 else (1, 2, 4, 8)
        for bm, bn, bk, sk in itertools.product(bms, bns, bks, sks):
            c = {**default, "block_m": bm, "block_n": bn, "block_k": bk, "split_k": sk, "num_warps": 4, "num_stages": 2}
            if c not in seen:
                seen.append(c)
                yield c
    else:
        bms = (16, 32) if M <= 64 else (32, 64)
        bks = (16, 32) if r >= 32 else (16,)
        for bm, bn, bk, nw in itertools.product(bms, (64, 128, 256), bks, (4, 8)):
            c = {**default, "block_m": bm, "block_n": bn, "block_k": bk, "num_warps": nw, "num_stages": 2}
            if c not in seen:
                seen.append(c)
                yield c


def tune(a) -> int:
    import torch
    from kernel_bench import Timer
    import vllm.lora.ops.triton_ops.lora_expand_op as eo
    import vllm.lora.ops.triton_ops.lora_shrink_op as so
    from vllm.lora.ops.triton_ops import LoRAKernelMeta, lora_expand, lora_shrink
    from vllm.lora.ops.triton_ops.utils import get_lora_op_configs

    out = Path(a.out)
    rec = json.loads((out / "shapes.json").read_text())
    ranks = [int(x) for x in a.ranks.split(",")]
    slots = [int(x) for x in a.max_loras.split(",")]
    ms = [int(x) for x in a.ms.split(",")]
    gpu = torch.cuda.get_device_name().replace(" ", "_").replace("-", "_")
    timer = Timer(warm=3, reps=a.reps)
    cur: list = [None]
    so.get_lora_op_configs = eo.get_lora_op_configs = lambda *x, **k: cur[0]
    torch.manual_seed(0)
    rows = []
    groups: dict = {}   # vLLM's lookup key -> [(shape, r, L, M)]
    for d in rec["shapes"]:
        S = len(d["w"])
        for r, L, M in itertools.product(ranks, slots, ms):
            if d["op"] == "shrink":
                K = d["w"][0][-1]
                key = ("shrink", None, L + 1, S, M, K, r)
            else:
                key = ("expand", d["add_inputs"], L + 1, S, M, r, max(w[-2] for w in d["w"]))
            groups.setdefault(key, []).append((d, r, L, M))
    print(f"{len(rec['shapes'])} call shapes, {len(groups)} keys", flush=True)
    t_start = time.time()
    meta_cache: dict = {}
    for gi, (key, members) in enumerate(sorted(groups.items(), key=lambda kv: str(kv[0]))):
        op, add, Lk, S, M, k, n = key
        default = get_lora_op_configs(op, max_loras=Lk, batch=M, hidden_size=(k if op == "shrink" else n),
                                      rank=(n if op == "shrink" else k), num_slices=S,
                                      **({"add_inputs": add} if op == "expand" else {}))
        r = n if op == "shrink" else k
        fns = []
        for d, r_, L, M_ in members:
            if (L, M_) not in meta_cache:
                meta = LoRAKernelMeta.make(L, max(ms), device="cuda")
                meta.prepare_tensors((torch.arange(M_, device="cuda") % min(L, M_)).to(torch.int32))
                meta_cache[(L, M_)] = (meta, meta.meta_args(M_, False))
            margs = meta_cache[(L, M_)][1]
            if op == "shrink":
                Kd = d["w"][0][-1]
                x = torch.randn(M_, Kd, device="cuda").to(getattr(torch, d["in_dtype"]))
                ws = tuple((torch.randn(*([L] + list(w[1:-2]) + [r_, Kd]), device="cuda") / Kd ** 0.5).to(torch.bfloat16)
                           for w in d["w"])
                o = torch.empty(S, M_, r_, device="cuda", dtype=getattr(torch, d["out_dtype"]))
                fns.append((lambda x=x, ws=ws, o=o, margs=margs: lora_shrink(x, ws, o, *margs, 1.0), lambda o=o: o.clone(),
                            lambda: None))
            else:
                h = torch.randn(S, M_, r_, device="cuda").to(getattr(torch, d["in_dtype"]))
                ws = tuple((torch.randn(*([L] + list(w[1:-2]) + [w[-2], r_]), device="cuda") * 0.02).to(torch.bfloat16)
                           for w in d["w"])
                y0 = torch.randn(M_, d["out_cols"], device="cuda").to(getattr(torch, d["out_dtype"]))
                y = y0.clone()
                fns.append((lambda h=h, ws=ws, y=y, margs=margs, off=d["offset_start"], add_=d["add_inputs"]:
                            lora_expand(h, ws, y, *margs, offset_start=off, add_inputs=add_),
                            lambda y=y: y.clone(), lambda y=y, y0=y0: y.copy_(y0)))
        refs = []
        cur[0] = default
        for fn, get, reset in fns:
            reset()
            fn()
            refs.append(get().float())
        best, t_def, n_ok = None, None, 0
        for c in candidates(op, M, r, default):
            cur[0] = c
            try:
                ok = True
                for (fn, get, reset), ref in zip(fns, refs):
                    reset()
                    fn()
                    err = float((get().float() - ref).norm() / ref.norm().clamp_min(1e-12))
                    ok &= err < 1e-2
                if not ok:
                    continue
                t = sum(timer.graph_time(fn) for fn, _, _ in fns)
            except Exception:   # noqa: BLE001 -- shared memory or an unsupported tile
                continue
            n_ok += 1
            if c is default:
                t_def = t
            if best is None or t < best[0]:
                best = (t, c)
        row = {"op": op, "add_inputs": add, "max_loras": Lk, "num_slices": S, "m": M, "k": k, "n": n, "shapes": len(members),
               "candidates_ok": n_ok, "t_default_us": t_def, "t_best_us": best[0] if best else None,
               "best": best[1] if best else None}
        rows.append(row)
        if gi % 20 == 0:
            print(json.dumps({**row, "elapsed_s": round(time.time() - t_start)}), flush=True)
    files: dict = {}
    for row in rows:
        if not row["best"]:
            continue
        fn = f"{gpu}_SHRINK.json" if row["op"] == "shrink" else f"{gpu}_EXPAND_{str(row['add_inputs']).upper()}.json"
        node = files.setdefault(fn, {})
        for kk in (row["max_loras"], row["num_slices"], row["m"], row["k"]):
            node = node.setdefault(str(kk), {})
        node[str(row["n"])] = row["best"]
    for fn, data in files.items():
        (out / fn).write_text(json.dumps(data, indent=1))
    with open(out / "tuning.jsonl", "w") as fh:
        for row in rows:
            fh.write(json.dumps({**row, "gpu": gpu}) + "\n")
    tot_d = sum(r_["t_default_us"] or 0 for r_ in rows)
    tot_b = sum(r_["t_best_us"] or 0 for r_ in rows)
    print(json.dumps({"keys": len(rows), "files": sorted(files), "sum_default_us": round(tot_d, 1), "sum_best_us": round(tot_b, 1),
                      "elapsed_s": round(time.time() - t_start)}), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("record", "tune"))
    ap.add_argument("--record-loras", type=int, default=32, help="record: adapter slots of the engine (and distinct adapters)")
    ap.add_argument("--model", default="")
    ap.add_argument("--quantization", default="fp8")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--rank", type=int, default=64, help="record: the adapter's rank (the slots' max rank)")
    ap.add_argument("--ranks", default="16,64")
    ap.add_argument("--max-loras", default="1,32")
    ap.add_argument("--ms", default="1,8,16,32,64,128,256,512,2048,8192")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    return record(a) if a.step == "record" else tune(a)


if __name__ == "__main__":
    raise SystemExit(main())
