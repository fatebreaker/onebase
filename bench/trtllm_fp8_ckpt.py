"""TRTLLM-H200's base (docs/REGISTRY.md): a per-tensor FP8 checkpoint of a bf16 model for TensorRT-LLM, with NVIDIA ModelOpt's
default FP8 recipe (FP8_DEFAULT_CFG: per-tensor E4M3 weights and static per-tensor activation scales; lm_head kept in bf16),
calibrated on the first user turns of the local WildChat trace (results/traces/wildchat_sharegpt_4000.json; no download), then
exported as a Hugging Face checkpoint with hf_quant_config.json. Runs inside the TensorRT-LLM release container on one GPU.
    python3 bench/trtllm_fp8_ckpt.py --model <bf16 dir> --out <dir> [--calib 256 --max-len 512]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

LB = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--trace", default=str(LB / "results/traces/wildchat_sharegpt_4000.json"))
    a = ap.parse_args()
    import modelopt
    import modelopt.torch.quantization as mtq
    import torch
    from modelopt.torch.export import export_hf_checkpoint
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()
    convs = json.loads(Path(a.trace).read_text())
    texts = []
    for c in convs:
        turns = c.get("conversations") or c.get("conversation") or []
        first = next((t.get("value") or t.get("content") for t in turns if t.get("from", t.get("role")) in ("human", "user")), None)
        if first and len(first) > 32:
            texts.append(first)
        if len(texts) >= a.calib:
            break
    print(f"calibration texts: {len(texts)}", flush=True)

    def forward_loop(m):
        with torch.no_grad():
            for i in range(0, len(texts), 8):
                b = tok(texts[i:i + 8], return_tensors="pt", padding=True, truncation=True, max_length=a.max_len)
                m(input_ids=b.input_ids.cuda(), attention_mask=b.attention_mask.cuda())

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop)
    out = Path(a.out)
    tmp = out.with_name(f"{out.name}.tmp{os.getpid()}")   # export beside, then rename: a partial export never looks complete
    tmp.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        export_hf_checkpoint(model, export_dir=str(tmp))
    tok.save_pretrained(str(tmp))
    for f in ("generation_config.json",):
        if (Path(a.model) / f).exists() and not (tmp / f).exists():
            shutil.copy(Path(a.model) / f, tmp / f)
    (tmp / ".complete").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    if (out / ".complete").exists():
        print(f"{out} is complete (another run finished first); keeping it, removing {tmp}", flush=True)
        shutil.rmtree(tmp)
    else:
        if out.exists():   # an earlier export of ours that never completed
            shutil.rmtree(out)
        os.rename(tmp, out)
    print(json.dumps({"modelopt": modelopt.__version__, "src": a.model, "out": str(out), "calib": len(texts),
                      "s": round(time.time() - t0, 1),
                      "hf_quant_config": json.loads((out / "hf_quant_config.json").read_text())
                      if (out / "hf_quant_config.json").exists() else None}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
