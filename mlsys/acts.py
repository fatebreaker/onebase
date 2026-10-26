"""Capture real input activations of the LoRA-targeted projections, for on-policy output SNR in the census.

Renders K agentic transcripts from a agent_harness results dir (per-problem JSONs with a 'transcript'), runs the base model
forward (transformers; CPU or GPU), and keeps N random token positions of the input to every target module of the
chosen layers. Output: a torch file {module name: [N, in_features] float32}.

    venv_vllm/bin/python -m mlsys.acts --model <base dir> --results <dir of problem JSONs> --layers 0,18,35 \
        --k 6 --max-tokens 1536 --keep 2048 --out results/acts/qwen3_8b.pt
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import torch

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def render(tok, rec: dict) -> str:
    msgs = [{"role": "user", "content": f"[task {rec.get('id', '')}]"}]
    for m in rec.get("transcript") or []:
        role = m.get("role")
        text = m.get("content") or m.get("sent") or ""
        if not text:
            continue
        msgs.append({"role": "assistant" if role == "assistant" else "user", "content": str(text)})
    try:
        return tok.apply_chat_template(msgs, tokenize=False)
    except Exception:   # templates that require alternating roles (Mistral): merge consecutive same-role turns
        merged: list[dict] = []
        for m in msgs:
            if merged and merged[-1]["role"] == m["role"]:
                merged[-1] = {"role": m["role"], "content": merged[-1]["content"] + "\n" + m["content"]}
            else:
                merged.append(dict(m))
        return tok.apply_chat_template(merged, tokenize=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--results", required=True, nargs="+", help="dirs of per-problem result JSONs")
    ap.add_argument("--layers", default="0,18,35")
    ap.add_argument("--k", type=int, default=6, help="transcripts per results dir")
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--keep", type=int, default=2048, help="token positions kept per module")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_grad_enabled(False)
    rng = random.Random(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=getattr(torch, a.dtype)).to(a.device).eval()
    print(f"loaded {a.model} in {time.time() - t0:.0f} s", flush=True)
    layers = {int(x) for x in a.layers.split(",")}
    store: dict[str, list[torch.Tensor]] = {}

    def hook(name):
        def fn(_mod, inp):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float().cpu()
            store.setdefault(name, []).append(x)
        return fn

    handles = []
    for name, mod in model.named_modules():
        m = re.search(r"layers\.(\d+)\.(.+)$", name)
        if m and int(m.group(1)) in layers and name.rsplit(".", 1)[-1] in TARGETS:
            key = name[name.index("layers."):]
            handles.append(mod.register_forward_pre_hook(hook(f"model.{key}")))
    texts = []
    for d in a.results:
        files = sorted(p for p in Path(d).glob("*.json") if p.name not in ("summary.json", "harness_v2.json"))
        rng.shuffle(files)
        n = 0
        for f in files:
            rec = json.loads(f.read_text())
            if rec.get("transcript"):
                texts.append(render(tok, rec))
                n += 1
            if n >= a.k:
                break
    for i, text in enumerate(texts):
        ids = tok(text, return_tensors="pt", truncation=True, max_length=a.max_tokens).input_ids.to(a.device)
        t1 = time.time()
        model(ids)
        print(f"transcript {i + 1}/{len(texts)}: {ids.shape[1]} tokens, {time.time() - t1:.0f} s", flush=True)
    for h in handles:
        h.remove()
    out = {}
    for name, xs in store.items():
        x = torch.cat(xs)
        idx = torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(a.seed))[: a.keep]
        out[name] = x[idx].contiguous()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"acts": out, "model": a.model, "results": a.results, "n_texts": len(texts)}, a.out)
    print(f"saved {len(out)} modules -> {a.out}  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
