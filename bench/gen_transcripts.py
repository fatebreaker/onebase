"""Own transcripts for an adapter that has none (P0-ACCURACY (b3), registry addendum 7): the adapter, served unmerged
in bf16 on its base by vLLM, answers fixed prompts greedily; each answer is written as one problem JSON in the harness
transcript format that mlsys.fidelity reads ({"id", "transcript": [{"role", "content"}, ...]}). GPU.

    python bench/gen_transcripts.py --model <base dir> --adapter <PEFT dir> --prompts bench/data/public_adapter_prompts.json \
        --out <dir> [--max-tokens 512] [--max-lora-rank 128]

strip: also writes a copy of a PEFT adapter without the weights of the given modules (e.g. lm_head, which the merge
arm cannot merge, or a sequence-classification head), with target_modules and modules_to_save updated to match:
    python bench/gen_transcripts.py strip --src <dir> --dst <dir> --modules lm_head
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def strip(src: str, dst: str, modules: list[str]) -> dict:
    from safetensors.torch import load_file, save_file
    s, d = Path(src), Path(dst)
    d.mkdir(parents=True, exist_ok=True)
    sd = load_file(str(s / "adapter_model.safetensors"))
    keep = {k: v for k, v in sd.items() if not any(f".{m}." in k or k.endswith(f".{m}") for m in modules)}
    save_file(keep, str(d / "adapter_model.safetensors"))
    cfg = json.loads((s / "adapter_config.json").read_text())
    tm = cfg.get("target_modules")
    if isinstance(tm, list):
        cfg["target_modules"] = [t for t in tm if t not in modules]
    ms = [t for t in (cfg.get("modules_to_save") or []) if t not in modules]
    cfg["modules_to_save"] = ms or None                 # vLLM serves only adapters without modules_to_save
    (d / "adapter_config.json").write_text(json.dumps(cfg, indent=1))
    for f in s.iterdir():
        if f.is_file() and f.name not in ("adapter_model.safetensors", "adapter_config.json"):
            shutil.copy2(f, d / f.name)
    return {"removed": len(sd) - len(keep), "kept": len(keep)}


def generate(a) -> int:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    prompts = json.loads(Path(a.prompts).read_text())
    llm = LLM(model=a.model, enable_lora=True, max_lora_rank=a.max_lora_rank, max_loras=1, max_model_len=4096,
              gpu_memory_utilization=a.gpu_util, seed=0)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)
    msgs = [[{"role": "user", "content": p}] for p in prompts]
    outs = llm.chat(msgs, sp, lora_request=LoRARequest("pub", 1, a.adapter), use_tqdm=False)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, (p, o) in enumerate(zip(prompts, outs)):
        text = o.outputs[0].text
        rec = {"id": f"pub-{i:02d}", "transcript": [{"role": "user", "content": p},
                                                    {"role": "assistant", "content": text}]}
        (out / f"pub-{i:02d}.json").write_text(json.dumps(rec))
    lens = [len(o.outputs[0].token_ids) for o in outs]
    print(json.dumps({"wrote": len(outs), "out": str(out), "tokens_mean": sum(lens) / len(lens),
                      "tokens_max": max(lens), "empty": sum(1 for x in lens if x == 0)}))
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "strip":
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--src", required=True)
        ap.add_argument("--dst", required=True)
        ap.add_argument("--modules", nargs="+", required=True)
        a = ap.parse_args()
        print(json.dumps(strip(a.src, a.dst, a.modules)))
        return 0
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--gpu-util", type=float, default=0.8)
    ap.add_argument("--max-lora-rank", type=int, default=128)
    return generate(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
