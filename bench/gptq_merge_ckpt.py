"""Build a GPTQ-merged checkpoint on a frozen grid (mlsys gptq_merge.gptq_frozen) for GPTQ-MERGE-BEHAVIOR: every adapted
projection of the base is replaced by the error-compensated merge of the adapter's update onto the base's own NVFP4 or
NF4 grid (the checkpoint's block scales or block absmax kept), as bf16 values of that grid, the way std_bench.py's
weights-only variants are served. H per module input from n_cal GSM8K training question+answer sequences run through
the bf16 model with the exact adapter. Every merged weight is checked to lie on the frozen grid.
    venv_vllm/bin/python bench/gptq_merge_ckpt.py --model <dir> --adapter <dir> --fmt nvfp4 nf4 --out-root <dir>
writes <out-root>/<fmt>/ (config, tokenizer, model.safetensors) and <out-root>/<fmt>/gptq_merge.json"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import torch

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
DL = Path("$SCRATCH/data/std_bench_dl")
MODS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj",
        "mlp.up_proj", "mlp.down_proj")
SHARED = {"self_attn.k_proj": "self_attn.q_proj", "self_attn.v_proj": "self_attn.q_proj",
          "mlp.up_proj": "mlp.gate_proj"}                  # modules reading the same input share one H


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--fmt", nargs="+", default=["nvfp4", "nf4"])
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--n-cal", type=int, default=128)
    ap.add_argument("--calib", default="gsm8k", choices=("gsm8k", "magicoder"),
                    help="magicoder: out-of-domain calibration (GPTQ-CALIB), Magicoder-OSS problem+solution pairs")
    a = ap.parse_args()
    import pandas as pd
    from peft import PeftModel
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from mlsys import gptq_merge as G, merge, quant
    import std_bench

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    if a.calib == "gsm8k":
        df = pd.read_parquet(DL / "gsm8k_train.parquet")
        pairs = [(q + std_bench.SUFFIX, ans) for q, ans in zip(df.question, df.answer)]
    else:
        rows = [json.loads(x) for x, _ in zip(open(DL / "magicoder_oss_75k.jsonl"), range(5000))]
        pairs = [(r["problem"], r["solution"]) for r in rows]
    idx = list(range(len(pairs)))
    random.Random(0).shuffle(idx)
    texts = [tok.apply_chat_template([{"role": "user", "content": pairs[i][0]}, {"role": "assistant", "content": pairs[i][1]}],
                                     tokenize=False) for i in idx[: a.n_cal]]
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    layers = model.base_model.model.model.layers
    H, hooks = {}, []
    for L, layer in enumerate(layers):
        for m in MODS:
            if m in SHARED:
                continue

            def hook(_mod, inp, _out, name=f"{L}.{m}"):
                x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                H[name] = H.get(name, 0) + x.T @ x
            hooks.append(layer.get_submodule(m).register_forward_hook(hook))
    with torch.no_grad():
        for t in texts:
            model(input_ids=tok(t, return_tensors="pt").input_ids[:, :1024].cuda())
    for h in hooks:
        h.remove()
    print(f"H from {len(texts)} sequences in {time.time() - t0:.0f}s", flush=True)

    cfg = json.loads((Path(a.adapter) / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    fa = safe_open(str(Path(a.adapter) / "adapter_model.safetensors"), framework="pt")
    keys = list(fa.keys())
    base = model.unload()                                   # the bf16 base with the LoRA layers removed (not merged)
    for fmt in a.fmt:
        out = Path(a.out_root) / fmt
        stats = {"num": 0.0, "den": 0.0, "changed": 0, "n": 0, "modules": 0, "t_gptq_s": 0.0}
        new = {}
        for L, layer in enumerate(base.model.layers):
            for m in MODS:
                lin = layer.get_submodule(m)
                W = lin.weight.detach().float()
                ka = [k for k in keys if k.endswith(f"layers.{L}.{m}.lora_A.weight")][0]
                kb = [k for k in keys if k.endswith(f"layers.{L}.{m}.lora_B.weight")][0]
                d = merge.lora_delta(fa.get_tensor(ka).cuda(), fa.get_tensor(kb).cuda(), scale)
                t1 = time.time()
                bq, q = G.gptq_frozen(W, d, H[f"{L}.{SHARED.get(m, m)}"], fmt=fmt)
                stats["t_gptq_s"] += time.time() - t1
                if fmt == "nvfp4":                          # every merged entry is a code of the frozen grid
                    _, sc, _ = quant.fp4_quant_with_scale(W, "nvfp4")
                    assert torch.equal(quant.fp4_on_scale(q, "nvfp4", sc), q), f"{L}.{m} off the NVFP4 grid"
                elif fmt == "fp8":
                    _, sc = quant.fp8_quant(W)
                    assert torch.equal(quant.fp8_quant(q, scale=sc)[0], q), f"{L}.{m} off the FP8 grid"
                else:
                    _, sc, _, lv = G.frozen_grid(W, "nf4")
                    g, _ = quant._grouped(q, 64)
                    back = quant._ungrouped(quant._round_to_levels(g / sc.unsqueeze(-1), lv.to(q.device), "nearest",
                                                                   None) * sc.unsqueeze(-1), 0)
                    assert torch.equal(back, q), f"{L}.{m} off the NF4 grid"
                D = q - bq
                stats["num"] += float((D.double() * d.double()).sum())
                stats["den"] += float((d.double() ** 2).sum())
                stats["changed"] += int((D != 0).sum())
                stats["n"] += D.numel()
                stats["modules"] += 1
                new[f"model.layers.{L}.{m}.weight"] = q.to(torch.bfloat16).cpu()
                del W, d, bq, q, D
        sd = {k: v.detach().cpu() for k, v in base.state_dict().items()}
        missing = [k for k in new if k not in sd]
        if missing:
            raise SystemExit(f"names not in the base state dict: {missing[:3]}")
        sd.update(new)
        out.mkdir(parents=True, exist_ok=True)
        base.save_pretrained(str(out), state_dict=sd, safe_serialization=True)
        tok.save_pretrained(str(out))
        for f in ("generation_config.json",):
            if (Path(a.model) / f).exists():
                shutil.copy(Path(a.model) / f, out / f)
        rec = {"adapter": a.adapter, "fmt": fmt, "n_cal": len(texts), "calib": a.calib, "weight_gain": stats["num"] / stats["den"],
               "changed_share": stats["changed"] / stats["n"], "modules": stats["modules"],
               "t_gptq_s": round(stats["t_gptq_s"], 1), "t_total_s": round(time.time() - t0, 1)}
        (out / "gptq_merge.json").write_text(json.dumps(rec, indent=1))
        print(json.dumps(rec), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
