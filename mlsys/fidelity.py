"""Logit-level fidelity of served variants to the true bf16 RL policy, on the policy's own transcripts (GPU).

Reference = pristine base + the RL adapter in activation space (scheme none + LoRA). Every variant scores the same
token sequences with prompt_logprobs. Over the policy's own tokens (assistant-message content), it reports:
  vs the reference:  mean |delta log p| of the realised tokens, the share of positions whose top-1 token changes,
                     and a top-k KL(ref || variant) restricted to the reference's top-k support;
  logit gain:        for an RL variant paired with its base_q variant, the RL shift it keeps,
                     g = <dq, d> / <d, d> with d = log p_RL - log p_base and dq = log p_RLq - log p_baseq per
                     realised token (1 = the RL policy's log-prob shift arrives in full); also corr(dq, d).
                     A 95% interval resamples sequences.
This is the model-level analogue of the census gain (weights) and its output gain (one layer).

    venv_vllm/bin/python -m mlsys.fidelity --model <base dir> --adapter <run>/step100 \
        --results <dir of problem JSONs> --variants none:base bf16_merge:rl int4g128_fresh:base int4g128_fresh:rl \
        int4g128_fresh:base+lora --out results/fidelity/<tag>.json
Variant syntax: scheme:role (role base or rl; rl merges the adapter) or scheme:base+lora (rounded base + adapter), or
lora=<dir> (the pristine base with ANOTHER adapter in activation space, e.g. a scaled copy; paired with none:base), or
scheme:base+lora=<dir> (a rounded base with another adapter, e.g. a low-precision copy; paired with scheme:base).
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

# RL variant -> the base_q it is paired with (same treatment of the base; on-grid merges keep the bf16 base)
PAIR = {"bf16_merge:rl": "none:base", "bf16_sr_merge:rl": "none:base", "fp16_merge:rl": "none:base",
        "fp8_fresh:rl": "fp8_fresh:base", "int4g128_fresh:rl": "int4g128_fresh:base",
        "int4g128_frozen:rl": "int4g128_fresh:base", "int4g128_fresh:base+lora": "int4g128_fresh:base",
        "fp8_fresh:base+lora": "fp8_fresh:base", "int8_fresh:rl": "int8_fresh:base",
        "nvfp4_fresh:rl": "nvfp4_fresh:base", "mxfp4_fresh:rl": "mxfp4_fresh:base",
        **{f"e8m{m}_frozen:rl": f"e8m{m}_frozen:base" for m in range(2, 7)},
        "nvfp4_frozen:rl": "nvfp4_fresh:base", "mxfp4_frozen:rl": "mxfp4_fresh:base",   # native FP4 checkpoint merge
        "nvfp4_fresh:base+lora": "nvfp4_fresh:base", "mxfp4_fresh:base+lora": "mxfp4_fresh:base",
        # frozen-grid merges into an FP8 checkpoint, and SR merges onto frozen grids (merge "<fmt>_srfrozen")
        "fp8_frozen:rl": "fp8_fresh:base", "fp8_srfrozen:rl": "fp8_fresh:base",
        "nvfp4_srfrozen:rl": "nvfp4_fresh:base", "mxfp4_srfrozen:rl": "mxfp4_fresh:base"}


def topk_kl(ref: dict[int, float], var: dict[int, float], floor: float = -30.0) -> float:
    """KL(ref || var) over ref's top-k tokens (renormalised); var log-probs missing from its top-k use `floor`."""
    z = math.log(sum(math.exp(v) for v in ref.values()))
    kl = 0.0
    for t, lp in ref.items():
        p = math.exp(lp - z)
        kl += p * ((lp - z) - var.get(t, floor))
    return kl


def compare(ref_steps: list, var_steps: list, tokens: list[int], mask: list[int] | None = None) -> dict:
    """ref_steps / var_steps: per position {token_id: logprob} (vLLM prompt_logprobs, first entry None)."""
    dl, flips, kls, n = 0.0, 0, 0.0, 0
    for pos, tok in enumerate(tokens):
        if mask is not None and not mask[pos]:
            continue
        r, v = ref_steps[pos], var_steps[pos]
        if r is None or v is None or tok not in r or tok not in v:
            continue
        dl += abs(r[tok] - v[tok])
        flips += max(r, key=r.get) != max(v, key=v.get)
        kls += topk_kl(r, v)
        n += 1
    return {"positions": n, "mean_abs_dlogp": dl / max(n, 1), "top1_flip_share": flips / max(n, 1),
            "topk_kl": kls / max(n, 1)}


def realised(steps: list, tokens: list[int], mask: list[int]) -> list[float | None]:
    """log p of each realised token at the masked positions (None where unavailable)."""
    out = []
    for pos, tok in enumerate(tokens):
        if not mask[pos]:
            continue
        s = steps[pos]
        out.append(None if s is None or tok not in s else s[tok])
    return out


def logit_gain(d_rl: list[list], d_base: list[list], d_rlq: list[list], d_baseq: list[list],
               n_boot: int = 2000, seed: int = 0) -> dict:
    """Per sequence lists of realised log-probs (aligned). g = <dq, d>/<d, d>; bootstrap over sequences."""
    per = []
    for a, b, c, e in zip(d_rl, d_base, d_rlq, d_baseq):
        xs = [(x - y, z - w) for x, y, z, w in zip(a, b, c, e) if None not in (x, y, z, w)]
        per.append((sum(d * q for d, q in xs), sum(d * d for d, _ in xs), sum(q * q for _, q in xs),
                    sum(d for d, _ in xs), sum(q for _, q in xs), len(xs)))

    def stat(idx):
        sdq, sdd, sqq, sd, sq, n = (sum(per[i][k] for i in idx) for k in range(6))
        g = sdq / sdd if sdd else float("nan")
        cov = sdq - sd * sq / max(n, 1)
        vd, vq = sdd - sd * sd / max(n, 1), sqq - sq * sq / max(n, 1)
        return g, cov / math.sqrt(vd * vq) if vd > 0 and vq > 0 else float("nan"), n

    g, corr, n = stat(range(len(per)))
    rng = random.Random(seed)
    boots = sorted(stat([rng.randrange(len(per)) for _ in per])[0] for _ in range(n_boot))
    return {"gain": g, "gain_ci95": [boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]], "corr": corr,
            "positions": n}


def build(tok, rec: dict, max_tokens: int, template: str = "chatml") -> tuple[list[int], list[int]]:
    """Token ids of a transcript in the model's chat format and a mask of the policy's own tokens (assistant content).
    template: chatml (Qwen), llama3 (Llama 3.x) or mistral (Mistral instruct; consecutive user turns are joined)."""
    ids, mask = [], []

    def add(text: str, own: bool) -> None:
        t = tok(text, add_special_tokens=False).input_ids
        ids.extend(t)
        mask.extend([int(own)] * len(t))

    msgs = [("user", f"[task {rec.get('id', '')}]")]
    for m in rec.get("transcript") or []:
        text = m.get("content") or m.get("sent") or ""
        if text:
            msgs.append(("assistant" if m.get("role") == "assistant" else "user", str(text)))
    if template == "chatml":                       # token-identical to the builder the mlsys results used
        add(f"<|im_start|>user\n{msgs[0][1]}<|im_end|>\n", False)
        for role, text in msgs[1:]:
            add(f"<|im_start|>{role}\n", False)
            add(text, role == "assistant")
            add("<|im_end|>\n", False)
    elif template == "llama3":
        add("<|begin_of_text|>", False)
        for role, text in msgs:
            add(f"<|start_header_id|>{role}<|end_header_id|>\n\n", False)
            add(text, role == "assistant")
            add("<|eot_id|>", False)
    elif template == "mistral":
        merged: list[list[str]] = []
        for role, text in msgs:
            if merged and merged[-1][0] == role:
                merged[-1][1] += "\n\n" + text
            else:
                merged.append([role, text])
        add("<s>", False)
        for role, text in merged:
            if role == "user":
                add(f"[INST] {text} [/INST]", False)
            else:
                add(text, True)
                add("</s>", False)
    else:
        raise ValueError(f"unknown template {template!r}")
    return ids[:max_tokens], mask[:max_tokens]


def _steps(out) -> list:
    return [None if s is None else {t: lp.logprob for t, lp in s.items()} for s in out.prompt_logprobs]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--results", required=True, help="dir of per-problem JSONs of the RL policy (its own transcripts)")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--variants", nargs="+", required=True, help="scheme:role or scheme:base+lora")
    ap.add_argument("--gpu-util", type=float, default=0.7)
    ap.add_argument("--batch-tokens", type=int, default=2048,
                    help="prefill chunk: prompt_logprobs materialises [tokens x vocab] fp32 logits per chunk")
    ap.add_argument("--template", default="chatml", choices=("chatml", "llama3", "mistral"))
    ap.add_argument("--max-lora-rank", type=int, default=128)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer
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
    llm = LLM(model=a.model, enable_lora=True, max_lora_rank=a.max_lora_rank, max_loras=2, max_model_len=a.max_tokens + 8,
              gpu_memory_utilization=a.gpu_util, max_num_batched_tokens=a.batch_tokens, enable_chunked_prefill=True,
              worker_extension_cls="mlsys.vllm_ext.VariantWorker")
    sp = SamplingParams(max_tokens=1, prompt_logprobs=a.k, temperature=1.0)
    lora = LoRARequest("rl", 1, a.adapter)
    prompts = [{"prompt_token_ids": s} for s in seqs]

    def score(spec: dict, use_lora, ) -> list:
        info = llm.collective_rpc("apply_variant", args=(json.dumps(spec),))
        llm.reset_prefix_cache()
        req = use_lora if isinstance(use_lora, LoRARequest) else (lora if use_lora else None)
        outs = llm.generate(prompts, sp, lora_request=req)
        return [_steps(o) for o in outs], info

    ref, _ = score({"scheme": "none"}, True)
    lp = {"ref": [realised(r, s, m) for r, s, m in zip(ref, seqs, masks)]}
    res = {"model": a.model, "adapter": a.adapter, "results": a.results, "n_seqs": len(seqs),
           "policy_tokens": sum(map(sum, masks)), "variants": {}, "logit_gain": {}}
    extra_id = 2
    for v in a.variants:
        if v.startswith("lora="):                         # another adapter (e.g. scaled) on the pristine base
            req = LoRARequest(f"x{extra_id}", extra_id, v.split("=", 1)[1])
            extra_id += 1
            spec, use_lora = {"scheme": "none"}, req
            PAIR.setdefault(v, "none:base")
        elif ":base+lora=" in v:                          # another adapter on a rounded base: scheme:base+lora=<dir>
            scheme, d = v.split(":base+lora=", 1)
            req = LoRARequest(f"x{extra_id}", extra_id, d)
            extra_id += 1
            spec, use_lora = {"scheme": scheme, "role": "base", "adapter": a.adapter}, req
            PAIR.setdefault(v, f"{scheme}:base")
        else:
            scheme, role = v.split(":")
            use_lora = role == "base+lora"
            spec = {"scheme": scheme, "role": "base" if use_lora else role, "adapter": a.adapter}
        var, info = score(spec, use_lora)
        per = [compare(r, x, s, m) for r, x, s, m in zip(ref, var, seqs, masks)]
        n = sum(p["positions"] for p in per)
        res["variants"][v] = {k: sum(p[k] * p["positions"] for p in per) / max(n, 1)
                              for k in ("mean_abs_dlogp", "top1_flip_share", "topk_kl")} | {"positions": n,
                                                                                          "apply": info}
        lp[v] = [realised(x, s, m) for x, s, m in zip(var, seqs, masks)]
        del var
        print(v, json.dumps({k: (round(x, 5) if isinstance(x, float) else x) for k, x in res["variants"][v].items()
                             if k != "apply"}), flush=True)
    if "none:base" in lp:
        for v, b in PAIR.items():
            if v in lp and b in lp:
                res["logit_gain"][v] = logit_gain(lp["ref"], lp["none:base"], lp[v], lp[b]) | {"base_q": b}
                print("logit gain", v, "vs", b, json.dumps(res["logit_gain"][v]), flush=True)
    llm.collective_rpc("apply_variant", args=(json.dumps({"scheme": "none"}),))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
