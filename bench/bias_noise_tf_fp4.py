"""ICML E1b at NVFP4 (docs/REGISTRY.md: ICML-E1B-FP4): bench/bias_noise_tf.py's bias-versus-noise measurement with a 4-bit
sampler, W4A4 in emulation, in the same single teacher-forced path.
  p        the trainer: every linear layer vLLM quantizes (bias_noise_tf's set: all nn.Linear but lm_head, conv1d, embeddings)
           with its weight rounded to NVFP4 (mlsys.quant.nvfp4_quant: E2M1 in blocks of 16, E4M3 block scales, a float32
           tensor scale; our NVFP4 base) and bf16 activations
  q_rtn    the same with each such layer's input rounded to NVFP4, round to nearest (per token: a float32 scale mapping the
           token's max to 448 x 6, E4M3 scales per block of 16, E2M1 codes; ties to even, saturating); the LoRA branch sees
           the unrounded input, as in vLLM
  q_sr,k   the same with the E2M1 codes stochastically rounded (fresh noise in every forward; scales as round to nearest)
Every KL exact over the full vocabulary at each completion position, as bias_noise_tf.py: KL(q_rtn || p), the per-draw KL, the
noise-averaged sampler's KL(qbar_K || p), its Richardson estimate, the curve over K, and a determinism check.
    PYTHONPATH=$LB/pylib_fla:$LB:$ML python bench/bias_noise_tf_fp4.py --model M --adapter A \
        --data results/icml_e1_bf16_65181900.json --K 32 --label e1b4 --out results/bias_noise_tf_fp4.jsonl
CPU check of the activation quantizer: python bench/bias_noise_tf_fp4.py --selftest"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))

MODE = {"m": "none"}
SKIP = re.compile(r"(^|\.)(lm_head|conv1d|embed_tokens|score)$")      # mlsys.matched_fp8.SKIP


def nvfp4_rows(x2, mode: str):
    """Per-token NVFP4 round trip of x2 [T, K]: a float32 scale per row (its max to 448 x 6), E4M3 scales per block of 16
    along K, E2M1 codes rounded to nearest (ties to even) or stochastically; the same arithmetic as mlsys.quant.nvfp4_quant
    with the tensor scale taken per row."""
    import torch
    from mlsys.quant import E2M1_MAX, FP8_E4M3_MAX, _e2m1
    x = x2.float()
    T, K = x.shape
    pad = (-K) % 16
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    g = x.reshape(T, -1, 16)
    ts = (x.abs().amax(dim=-1) / (FP8_E4M3_MAX * E2M1_MAX)).clamp_min(1e-30)[:, None, None]
    block = (g.abs().amax(dim=-1, keepdim=True) / (E2M1_MAX * ts)).clamp(max=FP8_E4M3_MAX).to(torch.float8_e4m3fn).float()
    scale = (block * ts).clamp_min(1e-30)
    q = _e2m1(g / scale, "nearest" if mode == "rtn" else "stochastic", None) * scale
    q = q.reshape(T, -1)
    return (q[:, :K] if pad else q).to(torch.bfloat16)


def hook(module, args):
    if MODE["m"] == "none":
        return None
    x = args[0]
    x2 = x.reshape(-1, x.shape[-1])
    return (nvfp4_rows(x2, MODE["m"]).reshape(x.shape),) + tuple(args[1:])


def kl_rows(lq, lp):
    return (lq.exp() * (lq - lp)).sum(-1)


def selftest() -> int:
    """The quantizer on random rows: every output on the row's NVFP4 grid neighbours, SR unbiased to Monte Carlo error."""
    import torch
    torch.manual_seed(0)
    x = torch.randn(4, 100) * torch.tensor([1.0, 0.01, 30.0, 3.0])[:, None]
    x[3, :10] = 0.0
    rtn = nvfp4_rows(x, "rtn").float()
    S = 2000
    mean = sum(nvfp4_rows(x, "sr").float() for _ in range(S)) / S
    err_sr = (mean - x).abs().mean() / x.abs().mean()
    err_rn = (rtn - x).abs().mean() / x.abs().mean()
    print(f"relative mean |E[q] - x|: SR {float(err_sr):.4f} (Monte Carlo), round to nearest {float(err_rn):.4f}")
    assert float(err_sr) < 0.25 * float(err_rn), "SR not unbiased"
    assert bool((rtn[3, :10] == 0).all()), "zeros"
    print("ok")
    return 0


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from mismatch_bench2 import boot
    from mlsys.quant import nvfp4_quant
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--max-seqs", type=int, default=0)
    ap.add_argument("--label", default="e1b4")
    ap.add_argument("--out", default=str(LB / "results/bias_noise_tf_fp4.jsonl"))
    a = ap.parse_args()
    assert a.K >= 2 and a.K % 2 == 0
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0})
    lins = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and not SKIP.search(n)]
    with torch.no_grad():
        for _, m in lins:
            m.weight.copy_(nvfp4_quant(m.weight.detach(), "nearest").to(m.weight.dtype))
    for _, m in lins:
        m.register_forward_pre_hook(hook)
    model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    print(f"NVFP4 weights and hooks on {len(lins)} linear layers; fla {importlib.util.find_spec('fla') is not None}", flush=True)
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
        acc = [torch.zeros_like(lp), torch.zeros_like(lp)]
        draw = torch.zeros(lp.shape[0], device=lp.device)
        curve = {}
        for k in range(a.K):
            lk = logp("sr")
            draw += kl_rows(lk, lp)
            acc[0 if k < a.K // 2 else 1] += lk.exp()
            del lk
            n_k = k + 1
            if n_k in curve_ks and n_k <= a.K // 2:
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
    row = {"label": a.label, "format": "nvfp4 (W4A4 emulation)", "model": a.model, "adapter": a.adapter, "data": a.data, "K": a.K,
           "prompts": len(G), "tokens": sum(g["n"] for g in G), "gpu": torch.cuda.get_device_name(0), "layers": len(lins),
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
