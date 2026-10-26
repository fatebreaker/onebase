"""A text-only checkpoint from a Qwen3.5-architecture multimodal release (Qwen3.6/3.8-27B: Qwen3_5ForConditionalGeneration),
as the existing Qwen3.6-27B-text was made: the language model's tensors renamed model.language_model.* -> model.*,
lm_head kept, the vision tower (model.visual.*) and the multi-token-prediction head (mtp.*) dropped; the config is the
release's text_config with architectures [Qwen3_5ForCausalLM] and torch_dtype bfloat16; tokenizer, chat template and
generation config copied. Shards are rewritten one input shard at a time (bounded memory).
--arch sets the text architecture (default Qwen3_5ForCausalLM; Gemma4ForCausalLM for Gemma 4, whose LM head is tied).
    python bench/extract_text.py --src <multimodal dir> --out <text dir> [--arch Gemma4ForCausalLM]"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="Qwen3_5ForCausalLM")
    a = ap.parse_args()
    from safetensors import safe_open
    from safetensors.torch import save_file
    src, out = Path(a.src), Path(a.out)
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    idx = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    files = sorted(set(idx.values()))
    wmap, n, total = {}, 0, 0
    for i, f in enumerate(files):
        tensors = {}
        with safe_open(str(src / f), framework="pt") as h:
            for k in h.keys():
                if k.startswith("model.language_model."):
                    nk = "model." + k[len("model.language_model."):]
                elif k == "lm_head.weight":
                    nk = k
                else:
                    continue                                    # model.visual.*, mtp.*
                t = h.get_tensor(k)
                tensors[nk] = t
                total += t.numel() * t.element_size()
        if tensors:
            name = f"model-{i + 1:05d}-of-{len(files):05d}.safetensors"
            save_file(tensors, str(tmp / name), metadata={"format": "pt"})
            wmap.update({k: name for k in tensors})
            n += len(tensors)
        print(f"{f}: {len(tensors)} tensors kept", flush=True)
    (tmp / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": wmap}, indent=1))
    cfg = json.loads((src / "config.json").read_text())
    tc = dict(cfg["text_config"])
    tc["architectures"] = [a.arch]
    tc["torch_dtype"] = "bfloat16"
    (tmp / "config.json").write_text(json.dumps(tc, indent=2))
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja", "generation_config.json",
              "tokenizer.model", "special_tokens_map.json"):
        if (src / f).exists():
            shutil.copy(src / f, tmp / f)
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    print(f"wrote {out}: {n} tensors, {total / 1e9:.1f} GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
