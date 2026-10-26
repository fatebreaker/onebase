"""W1-OFFICIAL part 2: logit gain of merges into an officially released block-FP8 checkpoint vs the exact adapter.

  build   write a text-only FP8 checkpoint from the (multimodal) release: language-model tensors renamed to the text
          model's names, block-FP8 codes and 128x128 dequantization scales kept, vision/MTP tensors dropped, the bf16
          text model's config plus the release's quantization_config. --merge frozen|absmax also merges the adapter's
          update into every LoRA module first (frozen: the release's scales kept; absmax: block scales recomputed from
          the merged weights), as bench/official_fp8_census.py does per module.
  score   log-probs of the realised policy tokens of the adapter's own transcripts under a served model, with or
          without the adapter (vLLM, LoRA path for the exact adapter).
  report  logit gain g = <dq, d> / <d, d>, d = lp(bf16 + adapter) - lp(bf16), dq = lp(variant) - lp(release base).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path

import torch

FP8_MAX, BLK = 448.0, 128


def _expand(s, shape):
    n, k = shape
    return s.repeat_interleave(BLK, 0)[:n].repeat_interleave(BLK, 1)[:, :k]


def _block_amax(w):
    n, k = w.shape
    pn, pk = (-n) % BLK, (-k) % BLK
    wp = torch.nn.functional.pad(w.abs(), (0, pk, 0, pn))
    return wp.view((n + pn) // BLK, BLK, (k + pk) // BLK, BLK).amax(dim=(1, 3))


def build(a) -> int:
    from safetensors import safe_open
    from safetensors.torch import save_file
    rel, out, txt = Path(a.release), Path(a.out), Path(a.text_model)
    out.mkdir(parents=True, exist_ok=True)
    rcfg = json.loads((rel / "config.json").read_text())
    qc = dict(rcfg["quantization_config"])
    qc["modules_to_not_convert"] = sorted({m.replace("model.language_model.", "model.") for m in
                                           qc.get("modules_to_not_convert", []) if "visual" not in m})
    cfg = json.loads((txt / "config.json").read_text())
    cfg["quantization_config"] = qc
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    for f in txt.iterdir():          # tokenizer, chat template, generation config: the text model's
        if f.is_file() and not f.name.endswith(".safetensors") and f.name not in ("config.json",
                                                                                   "model.safetensors.index.json"):
            shutil.copy(f, out / f.name)
    upd = {}
    if a.merge != "none":
        acfg = json.loads((Path(a.adapter) / "adapter_config.json").read_text())
        sc = acfg["lora_alpha"] / acfg["r"]
        fa = safe_open(str(Path(a.adapter) / "adapter_model.safetensors"), framework="pt")
        pat = re.compile(r"layers\.(\d+)\.(.+)\.lora_A\.weight$")
        for k in fa.keys():
            m = pat.search(k)
            if m:
                upd[f"model.layers.{m.group(1)}.{m.group(2)}"] = (k, k.replace("lora_A", "lora_B"), sc)
    wmap = json.loads((rel / "model.safetensors.index.json").read_text())["weight_map"]
    new_map, merged = {}, 0
    for shard in sorted(set(wmap.values())):
        f = safe_open(str(rel / shard), framework="pt")
        tensors = {}
        for k in f.keys():
            if "visual" in k or k.startswith("mtp"):
                continue
            nk = k.replace("model.language_model.", "model.")
            tensors[nk] = f.get_tensor(k)
        for nk in [x for x in tensors if x.endswith(".weight") and x[:-7] in upd]:
            base = nk[:-7]
            codes, s = tensors[nk].to(torch.float32), tensors[base + ".weight_scale_inv"].to(torch.float32)
            W = codes * _expand(s, codes.shape)
            ka, kb, sc = upd[base]
            A, B = fa.get_tensor(ka).float(), fa.get_tensor(kb).float()
            Wn = W + sc * (B @ A)
            if a.merge == "absmax":
                s = _block_amax(Wn) / FP8_MAX
            q = torch.clamp(Wn / _expand(s, Wn.shape), -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            tensors[nk], tensors[base + ".weight_scale_inv"] = q, s.to(tensors[base + ".weight_scale_inv"].dtype)
            merged += 1
        if tensors:
            name = f"model-{shard.replace('.safetensors', '')}.safetensors"
            save_file({k: v.contiguous() for k, v in tensors.items()}, str(out / name), metadata={"format": "pt"})
            new_map.update({k: name for k in tensors})
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": new_map}, indent=1))
    print(json.dumps({"out": str(out), "merge": a.merge, "merged_modules": merged, "tensors": len(new_map)}))
    if a.merge != "none" and merged == 0:
        raise SystemExit("no LoRA module matched the release's tensors")
    return 0


def score(a) -> int:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from mlsys.fidelity import _steps, build as build_seq, realised
    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model)
    files = sorted(p for p in Path(a.results).glob("*.json") if p.name not in ("summary.json", "harness_v2.json"))
    random.Random(0).shuffle(files)
    seqs, masks = [], []
    for f in files:
        rec = json.loads(f.read_text())
        if rec.get("transcript"):
            ids, mask = build_seq(tok, rec, a.max_tokens, a.template)
            if sum(mask) >= 32:
                seqs.append(ids)
                masks.append(mask)
        if len(seqs) >= a.n:
            break
    llm = LLM(model=a.model, tokenizer=a.tokenizer or a.model, enable_lora=bool(a.adapter), max_lora_rank=128,
              max_loras=1, max_model_len=a.max_tokens + 8, gpu_memory_utilization=a.gpu_util,
              max_num_batched_tokens=2048, enable_chunked_prefill=True, max_num_seqs=16)
    kw = {"lora_request": LoRARequest("a", 1, a.adapter)} if a.adapter else {}
    outs = llm.generate([{"prompt_token_ids": s} for s in seqs], SamplingParams(max_tokens=1, prompt_logprobs=1),
                        use_tqdm=False, **kw)
    lp = [realised(_steps(o), s, m) for o, s, m in zip(outs, seqs, masks)]
    Path(a.out).write_text(json.dumps({"model": a.model, "adapter": a.adapter, "seqs": seqs, "masks": masks, "lp": lp}))
    print(json.dumps({"model": a.model, "adapter": bool(a.adapter), "seqs": len(seqs), "tokens": sum(map(sum, masks))}))
    return 0


def report(a) -> int:
    flat = lambda p: [x for s in json.loads(Path(p).read_text())["lp"] for x in s]  # noqa: E731
    ref, base, base_q = flat(a.ref), flat(a.base), flat(a.base_q)
    d = [r - b for r, b in zip(ref, base)]
    dd = sum(x * x for x in d)
    res = {}
    for v in a.variants:
        name, path = v.split("=", 1)
        x = flat(path)
        dq = [p - q for p, q in zip(x, base_q)]
        res[name] = {"g": sum(p * q for p, q in zip(dq, d)) / dd, "tokens": len(d)}
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    pb = sp.add_parser("build")
    pb.add_argument("--release", required=True)
    pb.add_argument("--text-model", required=True)
    pb.add_argument("--out", required=True)
    pb.add_argument("--merge", default="none", choices=("none", "frozen", "absmax"))
    pb.add_argument("--adapter", default="")
    ps = sp.add_parser("score")
    ps.add_argument("--model", required=True)
    ps.add_argument("--tokenizer", default="")
    ps.add_argument("--adapter", default="")
    ps.add_argument("--results", required=True)
    ps.add_argument("--out", required=True)
    ps.add_argument("--n", type=int, default=24)
    ps.add_argument("--max-tokens", type=int, default=4096)
    ps.add_argument("--template", default="chatml")
    ps.add_argument("--gpu-util", type=float, default=0.85)
    pr = sp.add_parser("report")
    pr.add_argument("--ref", required=True)
    pr.add_argument("--base", required=True)
    pr.add_argument("--base-q", required=True)
    pr.add_argument("--variants", nargs="+", required=True)
    pr.add_argument("--out", default="")
    a = ap.parse_args()
    return {"build": build, "score": score, "report": report}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
