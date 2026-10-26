"""Write a ModelOpt-format NVFP4 (W4A4) checkpoint of a local bf16 HF model (no download), loadable by vLLM's
ModelOptNvFp4LinearMethod.

  * Weights: round-to-nearest NVFP4 (E2M1, E4M3 block scales of 16 along the input dim), vLLM's recipe
    (lowbit_lora/kernels/nvfp4_lora_vllm.nvfp4_quant_torch). weight_scale_2 = amax / (6 * 448), with one global shared
    by the projections vLLM fuses (q/k/v, gate/up), as vLLM recommends.
  * Activations: a static input_scale per layer = max |x| / (6 * 448), calibrated over local text (the task statements of
    agent_harness medcalc_train_1000), shared within each fused group (they read the same input anyway).
  * lm_head, embeddings and norms stay bf16. This is ModelOpt's default NVFP4 PTQ recipe (max calibration), done locally.
    python tools/make_nvfp4_ckpt.py --model <bf16 dir> --out <dir> [--n-calib 256] [--max-len 1024]
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from lowbit_lora.kernels.nvfp4_lora_vllm import nvfp4_quant_torch

QUANT = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
GROUP = {"q_proj": "qkv", "k_proj": "qkv", "v_proj": "qkv", "gate_proj": "gate_up", "up_proj": "gate_up"}
CALIB = Path("$AGENT_HARNESS/benchmarks/medcalc_train_1000/problems")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-calib", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--shard-gb", type=float, default=4.0)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    src, out = Path(a.model), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    amax: dict[str, torch.Tensor] = {}

    def hook(name):
        def f(_mod, inp):
            v = inp[0].detach().abs().amax().float()
            amax[name] = torch.maximum(amax[name], v) if name in amax else v
        return f

    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and name.rsplit(".", 1)[-1] in QUANT:
            mod.register_forward_pre_hook(hook(name))
    texts = [json.loads((d / "eval_config.json").read_text())["task"] for d in sorted(CALIB.iterdir())[: a.n_calib]]
    t0 = time.time()
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=a.max_len).input_ids.cuda()
            model(ids)
    print(f"calibrated {len(amax)} layers on {len(texts)} texts in {time.time() - t0:.0f}s", flush=True)
    sd = model.state_dict()
    gw: dict[tuple[str, str], float] = {}
    gx: dict[tuple[str, str], float] = {}
    for name in amax:
        layer, proj = name.rsplit(".", 1)
        key = (layer, GROUP.get(proj, proj))
        gw[key] = max(gw.get(key, 0.0), float(sd[name + ".weight"].float().abs().amax()))
        gx[key] = max(gx.get(key, 0.0), float(amax[name]))
    tensors: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        mod = k[: -len(".weight")] if k.endswith(".weight") else None
        if mod is not None and mod in amax:
            layer, proj = mod.rsplit(".", 1)
            key = (layer, GROUP.get(proj, proj))
            ws2 = gw[key] / (6.0 * 448.0)
            packed, sf = nvfp4_quant_torch(v, 1.0 / ws2)
            tensors[k] = packed.cpu()
            tensors[mod + ".weight_scale"] = sf.cpu()
            tensors[mod + ".weight_scale_2"] = torch.tensor(ws2, dtype=torch.float32)
            tensors[mod + ".input_scale"] = torch.tensor(gx[key] / (6.0 * 448.0), dtype=torch.float32)
        else:
            tensors[k] = v.detach().cpu().contiguous()
    # shards + index
    limit = int(a.shard_gb * 2 ** 30)
    shards, cur, size = [], {}, 0
    for k in sorted(tensors):
        nb = tensors[k].numel() * tensors[k].element_size()
        if cur and size + nb > limit:
            shards.append(cur)
            cur, size = {}, 0
        cur[k] = tensors[k]
        size += nb
    shards.append(cur)
    weight_map = {}
    for i, sh in enumerate(shards):
        fn = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(sh, str(out / fn), metadata={"format": "pt"})
        weight_map.update({k: fn for k in sh})
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}, indent=1))
    for f in src.iterdir():   # config, tokenizer, generation config, chat template
        if f.is_file() and not f.name.endswith(".safetensors") and f.name != "model.safetensors.index.json":
            shutil.copy2(f, out / f.name)
    cfg = json.loads((out / "config.json").read_text())
    cfg["quantization_config"] = {"quant_method": "modelopt", "quant_algo": "NVFP4", "group_size": 16,
                                  "ignore": ["lm_head"]}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "hf_quant_config.json").write_text(json.dumps({
        "producer": {"name": "modelopt", "version": "local-lowbit-lora"},
        "quantization": {"quant_algo": "NVFP4", "kv_cache_quant_algo": None, "group_size": 16,
                         "exclude_modules": ["lm_head"]}}, indent=2))
    (out / "lowbit_nvfp4_calib.json").write_text(json.dumps({
        "source": str(src), "n_calib": len(texts), "max_len": a.max_len, "calib_data": str(CALIB),
        "input_amax": {f"{l}.{g}": v for (l, g), v in gx.items()}}, indent=1))
    print(f"wrote {len(shards)} shards, {sum(t.numel() * t.element_size() for t in tensors.values()) / 2 ** 30:.2f} GiB "
          f"to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
