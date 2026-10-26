"""E2E-TP2's parity check (docs/REGISTRY.md), as amended -PARITY2 before any data: the plugin under vLLM's tensor parallelism
(LOWBIT_TP=1) against vLLM's own LoRA path on the same engine settings. Run once per mode in its own process:
    LOWBIT_FUSED=0 python tests/tp_parity_check.py --model <dir> --adapter <dir> --tp 2 --out .../parity_<tag>_ref.json
    LOWBIT_FUSED=1 LOWBIT_TP=1 python tests/tp_parity_check.py --model <dir> --adapter <dir> --tp 2 --out .../parity_<tag>_ours.json
    python tests/tp_parity_check.py --compare ref.json ours.json   -> PASS / FAIL (exit 0 / 1)
-PARITY2 (why: a token-match rule fails on rounding alone; HOPPER-PDL-E2E's checks diverged on 5-9 of 16 batched prompts within 32
tokens even between vLLM's own kernel configurations): a strong adapter (the given random adapter's B scaled 20x, b_std 2e-2 as
HOPPER-PDL-E2E's strong check, so that a wrong LoRA term moves the logits); 8 prompts, 32 greedy tokens each (tokens recorded,
agreement descriptive); then teacher-forced log-probabilities of the REFERENCE's greedy tokens: the reference run scores its own
sequences, the plugin's run (whose --out ends in _ours.json) scores the reference's (read from the _ref.json beside it).
PASS if the mean |delta log p| over the 8 x 32 completion tokens is at most 0.05 and the largest at most 1.0."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

PROMPTS = ["Explain why the sky is blue in two sentences.", "Write a Python function that reverses a string.",
           "What is 17 times 23? Show your work.", "List three uses of a hash table.",
           "Translate 'good morning' into French and Spanish.", "Summarize the plot of Hamlet in one sentence.",
           "Give the derivative of x^3 + 2x.", "Name the largest planet in the solar system and one fact about it."]
B_SCALE = 20.0
MEAN_TOL, MAX_TOL = 0.05, 1.0


def strong_adapter(src: str) -> str:
    """A copy of the PEFT adapter at SRC with every lora_B scaled by B_SCALE (deterministic: both modes get the same one)."""
    import torch
    from safetensors.torch import load_file, save_file
    out = Path(tempfile.mkdtemp(prefix="tp_parity_strong_"))
    for f in Path(src).iterdir():
        if f.is_file() and f.suffix != ".safetensors":
            shutil.copy(f, out / f.name)
    sd = load_file(str(Path(src) / "adapter_model.safetensors"))
    sd = {k: (v.float() * B_SCALE).to(v.dtype) if "lora_B" in k else v for k, v in sd.items()}
    save_file(sd, str(out / "adapter_model.safetensors"))
    return str(out)


def run(a) -> int:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    adapter = strong_adapter(a.adapter)
    llm = LLM(model=a.model, quantization="fp8", tensor_parallel_size=a.tp, enable_lora=True, max_lora_rank=64, max_loras=2,
              max_model_len=2048, gpu_memory_utilization=0.85, seed=0)
    lr = LoRARequest("a", 1, adapter)
    outs = llm.generate(PROMPTS, SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True), lora_request=lr)
    res = {"tp": a.tp, "b_scale": B_SCALE, "tokens": [list(o.outputs[0].token_ids) for o in outs],
           "prompt_ids": [list(o.prompt_token_ids) for o in outs], "text": [o.outputs[0].text for o in outs]}
    ref_path = Path(a.out.replace("_ours.json", "_ref.json")) if a.out.endswith("_ours.json") else None
    src = json.loads(ref_path.read_text()) if (ref_path and ref_path.exists()) else res
    res["tf_source"] = "ref" if src is not res else "self"
    seqs = [p + t for p, t in zip(src["prompt_ids"], src["tokens"])]
    tf = llm.generate([{"prompt_token_ids": s} for s in seqs], SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0),
                      lora_request=lr)
    res["tf_lp"] = []
    for o, p, s in zip(tf, src["prompt_ids"], seqs):
        lps = o.prompt_logprobs            # one entry per position (None first): {token id: Logprob}
        res["tf_lp"].append([float(lps[i][s[i]].logprob) for i in range(len(p), len(s))])
    shutil.rmtree(adapter, ignore_errors=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh)
    print(json.dumps({"prompts": len(PROMPTS), "tf_source": res["tf_source"], "first": res["text"][0][:80]}))
    return 0


def compare(f1: str, f2: str) -> int:
    r, o = json.load(open(f1)), json.load(open(f2))
    same24 = sum(a[:24] == b[:24] for a, b in zip(r["tokens"], o["tokens"]))
    same32 = sum(a == b for a, b in zip(r["tokens"], o["tokens"]))
    ok_src = o.get("tf_source") == "ref" and r.get("tf_source") == "self"
    d = [abs(x - y) for lr, lo in zip(r.get("tf_lp", []), o.get("tf_lp", [])) for x, y in zip(lr, lo)]
    mean = sum(d) / len(d) if d else float("inf")
    mx = max(d) if d else float("inf")
    ok = ok_src and len(d) == 8 * 32 and mean <= MEAN_TOL and mx <= MAX_TOL
    print(json.dumps({"prompts_matching_24": same24, "prompts_matching_32": same32, "n": len(r["tokens"]), "tf_positions": len(d),
                      "mean_abs_dlogp": round(mean, 4), "max_abs_dlogp": round(mx, 4), "tf_sources_ok": ok_src,
                      "verdict": "PASS" if ok else "FAIL"}))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--out", default="")
    ap.add_argument("--compare", nargs=2, default=None)
    a = ap.parse_args()
    if a.compare:
        return compare(*a.compare)
    return run(a)


if __name__ == "__main__":
    raise SystemExit(main())
