"""ICML E1b (docs/REGISTRY.md: ICML-E1B-TF): is a low-precision sampler's mismatch bias or noise, measured in ONE path.

ICML-E1 (bench/mismatch_sr.py) failed its method check: the sampled token's probability came from vLLM's decode path and the
K re-evaluations from its prefill path, which differ even for round-to-nearest, and log of a K-sample mean underestimates the
log of the mean. Here every distribution comes from the same teacher-forced PyTorch forward of the weight-matched model on
the same contexts, and every KL is computed exactly over the full vocabulary at each completion position:
  p        the trainer: vLLM's per-tensor FP8 weight codes dequantized per call, bf16 activations (mlsys.matched_fp8, parts="w")
  q_rtn    the same with each FP8 layer's input rounded per token to e4m3, round to nearest (MISMATCH-DECOMP-2's fake-quant
           control; the LoRA branch sees the unrounded input, as in vLLM)
  q_sr,k   the same with stochastic rounding, fresh noise in every forward (lowbit_lora/w16_engine_fqsr.sr_rows), k = 1..K
Per position: KL(q_rtn || p); the per-draw KL mean_k KL(q_sr,k || p); the bias KL(qbar_K || p) of the noise-averaged sampler
(qbar_K = mean_k q_sr,k), which overestimates KL(qbar || p) by O(1/K) (convexity), so also the Richardson estimate
2 KL(qbar_K || p) - mean(KL(qbar_{K/2,a} || p), KL(qbar_{K/2,b} || p)) and the curve over K = 1, 2, 4, ...; and a determinism
check (two round-to-nearest passes, max |dlog q|). Means over completion tokens with 95% bootstrap intervals over prompts.
Contexts: ICML-E1's bf16-engine samples (16 DAPO-Math prompts x 4, S8TL seed 0's adapter).
    PYTHONPATH=$LB/pylib_fla:$LB:$ML python bench/bias_noise_tf.py --model M --adapter A \
        --data results/icml_e1_bf16_65181900.json --K 32 --label e1b --out results/bias_noise_tf.jsonl
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))

MODE = {"m": "none"}


def fq_rtn(x2):
    """Per-token e4m3 round trip, round to nearest (vLLM's per-token scale rule, as w16_engine_fq)."""
    import torch
    xf = x2.float()
    s = (xf.abs().amax(dim=-1, keepdim=True) / 448.0).clamp(min=1.0 / (448.0 * 512.0))
    q = (xf / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return (q.float() * s).to(torch.bfloat16)


def hook(module, args):
    if MODE["m"] == "none":
        return None
    from lowbit_lora.w16_engine_fqsr import sr_rows
    x = args[0]
    x2 = x.reshape(-1, x.shape[-1])
    xd = fq_rtn(x2) if MODE["m"] == "rtn" else sr_rows(x2)
    return (xd.reshape(x.shape),) + tuple(args[1:])


def kl_rows(lq, lp):
    """Per-row KL(q || p) from log-probabilities [T, V] (fp32)."""
    return (lq.exp() * (lq - lp)).sum(-1)


def main() -> int:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from mismatch_bench2 import boot
    from mlsys.matched_fp8 import patch_model
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--max-seqs", type=int, default=0)
    ap.add_argument("--label", default="e1b")
    ap.add_argument("--out", default=str(LB / "results/bias_noise_tf.jsonl"))
    a = ap.parse_args()
    assert a.K >= 2 and a.K % 2 == 0
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0})
    info = patch_model(model, parts="w", use_scaled_mm=False)
    model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    fp8 = [m for m in model.modules() if hasattr(m, "w_codes")]
    for m in fp8:
        m.register_forward_pre_hook(hook)
    print(f"weight-matched model: {info}; {len(fp8)} FP8 layers hooked; fla {importlib.util.find_spec('fla') is not None}",
          flush=True)
    d = json.loads(Path(a.data).read_text())
    rows = [r for r in d["rows"] if r["completion"]]
    if a.max_seqs:
        rows = rows[: a.max_seqs]
    curve_ks = [k for k in (1, 2, 4, 8, 16, 32, 64, 128) if k <= a.K]
    groups: dict = {}
    det_max = 0.0
    for i, r in enumerate(rows):
        p, c = r["prompt"], r["completion"]
        ids = torch.tensor([p + c], device="cuda:0")
        pos = slice(len(p) - 1, len(p) + len(c) - 1)

        def logp(mode):
            MODE["m"] = mode
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0, pos]
            MODE["m"] = "none"
            return torch.log_softmax(lg.float(), dim=-1)
        lp = logp("none")
        lr = logp("rtn")
        lr2 = logp("rtn")
        det_max = max(det_max, float((lr2 - lr).abs().max()))
        kl_rtn = kl_rows(lr, lp)
        del lr, lr2
        acc = [torch.zeros_like(lp), torch.zeros_like(lp)]            # the two halves' summed probabilities
        draw = torch.zeros(lp.shape[0], device=lp.device)
        curve = {}
        for k in range(a.K):
            lk = logp("sr")
            draw += kl_rows(lk, lp)
            acc[0 if k < a.K // 2 else 1] += lk.exp()
            del lk
            n_k = k + 1
            if n_k in curve_ks and n_k <= a.K // 2:                    # prefix of the first half
                qb = acc[0] / n_k
                curve[n_k] = (qb * (qb.clamp_min(1e-30).log() - lp)).sum(-1)
        kl_half = []
        for h in acc:
            qb = h / (a.K // 2)
            kl_half.append((qb * (qb.clamp_min(1e-30).log() - lp)).sum(-1))
        qb = (acc[0] + acc[1]) / a.K
        kl_k = (qb * (qb.clamp_min(1e-30).log() - lp)).sum(-1)
        curve[a.K] = kl_k
        del acc, qb
        g = groups.setdefault(tuple(p), {"n": 0, "rtn": 0.0, "draw": 0.0, "biasK": 0.0, "half": 0.0, "rich": 0.0,
                                         **{f"c{k}": 0.0 for k in curve_ks}})
        T = int(kl_rtn.shape[0])
        g["n"] += T
        g["rtn"] += float(kl_rtn.sum())
        g["draw"] += float(draw.sum()) / a.K
        g["biasK"] += float(kl_k.sum())
        half = 0.5 * (kl_half[0] + kl_half[1])
        g["half"] += float(half.sum())
        g["rich"] += float((2 * kl_k - half).sum())
        for k in curve_ks:
            if k in curve:
                g[f"c{k}"] += float(curve[k].sum())
        if i % 8 == 0:
            print(f"seq {i + 1}/{len(rows)}: tokens {T}, KL x1e4 rtn {1e4 * float(kl_rtn.mean()):.2f}, sr draw "
                  f"{1e4 * float(draw.mean()) / a.K:.2f}, sr bias(K={a.K}) {1e4 * float(kl_k.mean()):.2f}, rich "
                  f"{1e4 * float((2 * kl_k - half).mean()):.2f}; det max {det_max:.2e}", flush=True)
        del lp, kl_rtn, draw, kl_k, half, curve
        torch.cuda.empty_cache()
    G = list(groups.values())
    tot = lambda k: sum(g[k] for g in G) / sum(g["n"] for g in G)  # noqa: E731
    ci = lambda k: boot([(g[k], g["n"]) for g in G])  # noqa: E731
    row = {"label": a.label, "model": a.model, "adapter": a.adapter, "data": a.data, "K": a.K, "prompts": len(G),
           "tokens": sum(g["n"] for g in G), "gpu": torch.cuda.get_device_name(0), "fp8_layers": len(fp8),
           "det_max_abs_dlogq": det_max,
           "kl_rtn": tot("rtn"), "kl_rtn_ci95": ci("rtn"), "kl_sr_draw": tot("draw"), "kl_sr_draw_ci95": ci("draw"),
           "kl_sr_biasK": tot("biasK"), "kl_sr_biasK_ci95": ci("biasK"), "kl_sr_half": tot("half"),
           "kl_sr_rich": tot("rich"), "kl_sr_rich_ci95": ci("rich"),
           "curve": {str(k): tot(f"c{k}") for k in curve_ks}}
    print(json.dumps(row), flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
