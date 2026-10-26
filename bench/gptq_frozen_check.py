"""GPTQ-FROZEN (docs/REGISTRY.md): can an error-compensated merge (GPTQ onto the checkpoint's frozen grid, mlsys
gptq_merge.gptq_frozen) deliver an update that a nearest frozen merge erases? Real inputs of Mistral-7B-Instruct with
an adapter: H from 64 GSM8K training question+answer sequences, the output gain and noise measured on 64 other
sequences, for every adapted module of layers 0, 15 and 31, frozen NVFP4 and NF4 (QLoRA's block absmax kept) grids;
nearest, stochastic (seed 1) and GPTQ merges.
    venv_vllm/bin/python bench/gptq_frozen_check.py --model <dir> --adapter <dir> --out <json>"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
DL = Path("$SCRATCH/data/std_bench_dl")
LAYERS = (0, 15, 31)
MODS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj",
        "mlp.up_proj", "mlp.down_proj")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-cal", type=int, default=64)
    ap.add_argument("--n-eval", type=int, default=64)
    a = ap.parse_args()
    import pandas as pd
    from peft import PeftModel
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from mlsys import gptq_merge as G, merge, quant
    import std_bench

    tok = AutoTokenizer.from_pretrained(a.model)
    df = pd.read_parquet(DL / "gsm8k_train.parquet")
    idx = list(range(len(df)))
    random.Random(0).shuffle(idx)
    texts = [tok.apply_chat_template([{"role": "user", "content": df.question[i] + std_bench.SUFFIX},
                                      {"role": "assistant", "content": df.answer[i]}], tokenize=False)
             for i in idx[: a.n_cal + a.n_eval]]
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    cfg = json.loads((Path(a.adapter) / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    fa = safe_open(str(Path(a.adapter) / "adapter_model.safetensors"), framework="pt")
    keys = list(fa.keys())

    # inputs of each adapted module (the base layer of the PEFT wrapper sees the module's input)
    phase = {"which": "cal"}
    H = {}
    Xe = {}
    hooks = []
    for L in LAYERS:
        for m in MODS:
            mod = model.base_model.model.model.layers[L].get_submodule(m)

            def hook(_mod, inp, _out, name=f"{L}.{m}"):
                x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                if phase["which"] == "cal":
                    H[name] = H.get(name, 0) + x.T @ x
                else:
                    Xe.setdefault(name, []).append(x[:: 4].cpu())       # every 4th position: memory
            hooks.append(mod.register_forward_hook(hook))
    with torch.no_grad():
        for k, t in enumerate(texts):
            phase["which"] = "cal" if k < a.n_cal else "eval"
            ids = tok(t, return_tensors="pt").input_ids[:, :1024].cuda()
            model(input_ids=ids)
    for h in hooks:
        h.remove()

    rows = []
    for L in LAYERS:
        for m in MODS:
            name = f"{L}.{m}"
            mod = model.base_model.model.model.layers[L].get_submodule(m)
            W = mod.base_layer.weight.detach().float()
            ka = [k for k in keys if k.endswith(f"layers.{L}.{m}.lora_A.weight")][0]
            kb = [k for k in keys if k.endswith(f"layers.{L}.{m}.lora_B.weight")][0]
            d = merge.lora_delta(fa.get_tensor(ka).cuda(), fa.get_tensor(kb).cuda(), scale)
            X = torch.cat(Xe[name]).cuda()
            Y = X @ d.T
            for fmt in ("nvfp4", "nf4"):
                t0 = time.time()
                bq, gq = G.gptq_frozen(W, d, H[name], fmt=fmt)
                t_gptq = time.time() - t0
                if fmt == "nvfp4":
                    _, nq = merge.apply_scheme(W, d, "nvfp4_frozen")
                    _, sq = merge.apply_scheme(W, d, "nvfp4_srfrozen", seed=1)
                else:
                    g, _ = quant._grouped(W, 64)
                    amax = g.abs().amax(-1, keepdim=True)
                    amax = torch.where(amax > 0, amax, torch.ones_like(amax))
                    gen = torch.Generator(device=W.device).manual_seed(1)

                    def onf4(x, mode, gen=None):
                        gg, _ = quant._grouped(x, 64)
                        return quant._ungrouped(quant._round_to_levels(gg / amax, quant.NF4_LEVELS, mode, gen) * amax, 0)
                    nq = onf4(bq + d, "nearest")
                    sq = onf4(bq + d, "stochastic", gen)
                rec = {"module": name, "fmt": fmt, "t_gptq_s": round(t_gptq, 1),
                       "update_rel": float(d.norm() / W.norm())}
                for lab, Q in (("nearest", nq), ("sr", sq), ("gptq", gq)):
                    D = Q - bq
                    Z = X @ D.T
                    rec[lab] = {"out_gain": float((Z * Y).sum() / (Y * Y).sum()),
                                "out_noise": float((Z - Y).norm() / Y.norm()),
                                "w_gain": float((D * d).sum() / (d * d).sum()),
                                "changed": float((D != 0).float().mean())}
                rows.append(rec)
                print(json.dumps(rec), flush=True)
            del X, Y
    Path(a.out).write_text(json.dumps({"adapter": a.adapter, "layers": LAYERS, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
