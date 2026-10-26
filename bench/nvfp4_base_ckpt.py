"""NVFP4-BASE-CHECK (docs/REGISTRY.md): the weights-only NVFP4 base as a bf16 checkpoint, in two tensor-scale layouts.
NVFP4 has a per-tensor scale. The GPTQ merges (bench/gptq_merge_ckpt*.py) quantize each projection on its own (one tensor
scale per projection, "perproj"); the evaluation engine's variant worker (mlsys vllm_ext.chunked_scheme, nvfp4_fresh) gets
vLLM's packed q/k/v and gate/up weights and computes one tensor scale per 4,096-row chunk ("packed"). This writes either
layout for every projection of every layer; evaluated with std_bench.py --variants bf16_base it gives the accuracy of that
base, to compare with the engine's nvfp4_base.
    venv_vllm/bin/python bench/nvfp4_base_ckpt.py --model <dir> --layout perproj|packed --out <dir>"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

PACKS = (("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"), ("mlp.gate_proj", "mlp.up_proj"),
         ("self_attn.o_proj",), ("mlp.down_proj",))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layout", required=True, choices=("perproj", "packed"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from mlsys import merge
    from mlsys.vllm_ext import chunked_scheme
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0})
    n = 0
    with torch.no_grad():
        for layer in model.model.layers:
            for pack in PACKS:
                mods = [layer.get_submodule(m) for m in pack]
                Ws = [m.weight.data.float() for m in mods]
                if a.layout == "perproj":
                    qs = [merge.apply_scheme(W, torch.zeros_like(W), "nvfp4_fresh")[0] for W in Ws]
                else:
                    q = chunked_scheme(torch.cat(Ws), None, "nvfp4_fresh", "base", seed=0)
                    qs = list(torch.split(q, [W.shape[0] for W in Ws]))
                for m, q in zip(mods, qs):
                    m.weight.data.copy_(q.to(torch.bfloat16))
                    n += 1
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)
    AutoTokenizer.from_pretrained(a.model).save_pretrained(str(out))
    if (Path(a.model) / "generation_config.json").exists():
        shutil.copy(Path(a.model) / "generation_config.json", out / "generation_config.json")
    (out / "nvfp4_base.json").write_text(json.dumps({"model": a.model, "layout": a.layout, "projections": n}))
    print(json.dumps({"layout": a.layout, "projections": n, "out": str(out)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
