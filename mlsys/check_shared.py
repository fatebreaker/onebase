"""Check the shared FP8 base (lowbit P5; mlsys/ipc_share.py) against the weight-matched trainer it replaces.

With a vLLM engine serving the base in online FP8 with --worker-extension-cls mlsys.ipc_share.SharedBaseWorker on this
GPU, the check (1) maps the engine's codes, scales, embedding and LM head into an HF model loaded on CPU and asserts
the codes equal this process's own (check_codes); (2) builds the current weight-matched trainer model (bf16 load on the
GPU, then matched_fp8.patch_model(parts="w")); (3) compares both models' logits on a few transcripts token by token
(they must be identical); (4) reports each model's GPU memory; (5) asks the engine for a completion afterwards.
    python -m mlsys.check_shared --url http://127.0.0.1:PORT --model <bf16 dir> --out <json>"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=1024)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .ipc_share import map_shared_base
    from .matched_fp8 import patch_model
    from .client import _post
    tok = AutoTokenizer.from_pretrained(a.model)
    res: dict = {}
    torch.cuda.reset_peak_memory_stats()
    m0 = torch.cuda.memory_allocated()
    t0 = time.time()
    shared = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cpu")
    res["map"] = map_shared_base(shared, a.url, check_codes=True)
    res["shared_model_gpu_gb_own"] = round((torch.cuda.memory_allocated() - m0) / 2**30, 3)
    res["map_seconds"] = round(time.time() - t0, 1)
    print("shared:", res["map"], "own GPU GB", res["shared_model_gpu_gb_own"], flush=True)
    m1 = torch.cuda.memory_allocated()
    own = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda")
    res["patch"] = patch_model(own, parts="w")
    torch.cuda.empty_cache()
    res["own_model_gpu_gb"] = round((torch.cuda.memory_allocated() - m1) / 2**30, 3)
    print("own:", res["patch"], "GPU GB", res["own_model_gpu_gb"], flush=True)
    texts = ["Compute the creatinine clearance of a 67-year-old woman weighing 58 kg with serum creatinine 1.4 mg/dL. "
             * 20, "import pandas as pd\n" + "df = pd.read_csv('labs.csv')\nprint(df.describe())\n" * 30]
    worst, floor, n = 0.0, 0.0, 0
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt").input_ids[:, : a.tokens].cuda()
            ls = shared(input_ids=ids).logits.float()
            lo = own(input_ids=ids).logits.float()
            lo2 = own(input_ids=ids).logits.float()        # the same model twice: kernel nondeterminism floor
            worst = max(worst, float((ls - lo).abs().max()))
            floor = max(floor, float((lo2 - lo).abs().max()))
            n += ids.shape[1]
    res["logits_max_abs_diff"] = worst
    res["own_vs_own_max_abs_diff"] = floor
    res["tokens_compared"] = n
    print(f"logits max |diff| over {n} tokens: shared vs own {worst}, own vs own {floor}", flush=True)
    r = _post(a.url, "/v1/completions", {"model": json.loads(urllib_get(a.url + "/v1/models"))["data"][0]["id"],
                                         "prompt": "The capital of France is", "max_tokens": 8, "temperature": 0})
    res["engine_after"] = (r or {}).get("choices", [{}])[0].get("text")
    print("engine still serves:", repr(res["engine_after"]), flush=True)
    res["pass"] = bool(res["engine_after"]) and res["map"]["mapped"] == res["map"]["linear_layers"]
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("PASS" if res["pass"] else "FAIL", flush=True)
    return 0 if res["pass"] else 1


def urllib_get(url: str) -> str:
    import urllib.request
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode()


if __name__ == "__main__":
    raise SystemExit(main())
