"""Trainer-sampler agreement for matched-precision LoRA RL (lowbit-lora P2): log-probs of the same realised tokens under
the rollout engine (vLLM, bf16 or online FP8 with whatever plugin the environment enables) and under the trainer (HF +
PEFT, bf16 or matched FP8, mlsys/matched_fp8.py). GPU; run each stage in its own process (vLLM and HF never share one).

    python -m mlsys.check_matched engine  --model M --adapter A --results R --out e_fp8.json [--fp8] [--n 8]
    python -m mlsys.check_matched trainer --model M --adapter A --engine e_fp8.json --out t_fp8.json [--matched]
    python -m mlsys.check_matched report  --pairs t_bf16.json:e_fp8.json t_fp8.json:e_fp8.json ... [--out r.json]

report prints, per trainer:engine pair, the mean |log p_trainer - log p_engine| over the policy's own tokens.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def engine(a) -> int:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from mlsys.fidelity import _steps, build, realised
    tok = AutoTokenizer.from_pretrained(a.model)
    files = sorted(p for p in Path(a.results).glob("*.json") if p.name not in ("summary.json", "harness_v2.json"))
    random.Random(0).shuffle(files)
    seqs, masks = [], []
    for f in files:
        rec = json.loads(f.read_text())
        if rec.get("transcript"):
            ids, mask = build(tok, rec, a.max_tokens, a.template)
            if sum(mask) >= 32:
                seqs.append(ids)
                masks.append(mask)
        if len(seqs) >= a.n:
            break
    kw = {"quantization": "fp8"} if a.fp8 else {}
    llm = LLM(model=a.model, enable_lora=True, max_lora_rank=128, max_loras=1, max_model_len=a.max_tokens + 8,
              gpu_memory_utilization=a.gpu_util, max_num_batched_tokens=2048, enable_chunked_prefill=True, **kw)
    outs = llm.generate([{"prompt_token_ids": s} for s in seqs], SamplingParams(max_tokens=1, prompt_logprobs=1),
                        lora_request=LoRARequest("a", 1, a.adapter), use_tqdm=False)
    lp = [realised(_steps(o), s, m) for o, s, m in zip(outs, seqs, masks)]
    Path(a.out).write_text(json.dumps({"fp8": a.fp8, "seqs": seqs, "masks": masks, "lp": lp}))
    print(json.dumps({"engine": "fp8" if a.fp8 else "bf16", "seqs": len(seqs), "tokens": sum(map(sum, masks))}))
    return 0


def trainer(a) -> int:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    e = json.loads(Path(a.engine).read_text())
    info = {}
    if a.shared_url:      # lowbit P5's S9 trainer: host load, the engine's FP8 base mapped over CUDA IPC, then PEFT
        from mlsys.ipc_share import map_shared_base
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cpu")
        info = map_shared_base(model, a.shared_url)
    else:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda")
    if a.matched and not a.shared_url:
        from mlsys.matched_fp8 import patch_model
        info = patch_model(model, parts=a.parts)
    model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    inner = model.get_base_model()
    lps = []
    with torch.no_grad():
        for ids, mask in zip(e["seqs"], e["masks"]):
            x = torch.tensor([ids], device="cuda")
            h = inner.model(input_ids=x, use_cache=False).last_hidden_state[0]
            pos = [t for t, m in enumerate(mask) if m]
            out = []
            for i in range(0, len(pos), 1024):
                chunk = [t for t in pos[i:i + 1024] if t >= 1]
                if not chunk:
                    out.extend([None] * len(pos[i:i + 1024]))
                    continue
                lg = inner.lm_head(h[[t - 1 for t in chunk]]).float().log_softmax(-1)
                got = lg[torch.arange(len(chunk), device="cuda"), torch.tensor([ids[t] for t in chunk], device="cuda")]
                vals = dict(zip(chunk, got.tolist()))
                out.extend(vals.get(t) for t in pos[i:i + 1024])
            lps.append(out)
    Path(a.out).write_text(json.dumps({"matched": a.matched, "shared": a.shared_url, "patch": info, "lp": lps}))
    print(json.dumps({"trainer": "shared" if a.shared_url else "matched-fp8" if a.matched else "bf16", "patch": info}))
    return 0


def report(a) -> int:
    res = {}
    for pair in a.pairs:
        t, e = pair.split(":")
        lt, le = json.loads(Path(t).read_text())["lp"], json.loads(Path(e).read_text())["lp"]
        d = [abs(x - y) for st, se in zip(lt, le) for x, y in zip(st, se) if x is not None and y is not None]
        res[pair] = {"mean_abs_dlogp": sum(d) / max(len(d), 1), "tokens": len(d),
                     "p99_abs_dlogp": sorted(d)[int(0.99 * (len(d) - 1))] if d else None}
        print(pair, json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in res[pair].items()}))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pe = sub.add_parser("engine")
    pt = sub.add_parser("trainer")
    for p in (pe, pt):
        p.add_argument("--model", required=True)
        p.add_argument("--adapter", required=True)
        p.add_argument("--out", required=True)
    pe.add_argument("--results", required=True)
    pe.add_argument("--n", type=int, default=8)
    pe.add_argument("--max-tokens", type=int, default=4096)
    pe.add_argument("--template", default="chatml")
    pe.add_argument("--gpu-util", type=float, default=0.6)
    pe.add_argument("--fp8", action="store_true")
    pt.add_argument("--engine", required=True)
    pt.add_argument("--matched", action="store_true")
    pt.add_argument("--parts", default="both", choices=("both", "w", "a"))
    pt.add_argument("--shared-url", default="", help="map the FP8 base of this engine (SharedBaseWorker) instead")
    pr = sub.add_parser("report")
    pr.add_argument("--pairs", nargs="+", required=True)
    pr.add_argument("--out", default="")
    a = ap.parse_args()
    return {"engine": engine, "trainer": trainer, "report": report}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
