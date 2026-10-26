"""Why does the matched-FP8 trainer not reproduce vLLM's online-FP8 engine (P2 early gate, 2026-10-11 15:16)?
Stage `engine`: ask the running vLLM engine (online FP8) which modules it quantizes, with which quant method, weight
dtype and weight scale, and print its compilation pass config (fusions change activation codes).
Stage `compare`: recompute mlsys.matched_fp8's per-group scales from the HF checkpoint and compare them with the
engine's, module by module. GPU (engine stage).

    python -m mlsys.diag_fp8 engine --model M --out eng.json
    python -m mlsys.diag_fp8 compare --model M --engine eng.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _collect(self):                                   # runs on the vLLM worker (self = worker)
    import torch
    model = self.model_runner.model
    out = []
    for n, m in model.named_modules():
        qm = getattr(m, "quant_method", None)
        if qm is None:
            continue
        w = getattr(m, "weight", None)
        ws = getattr(m, "weight_scale", None)
        out.append({"name": n, "cls": type(m).__name__, "qm": type(qm).__name__,
                    "dtype": str(w.dtype) if isinstance(w, torch.Tensor) else None,
                    "shape": list(w.shape) if isinstance(w, torch.Tensor) else None,
                    "scale": ws.detach().float().flatten()[:4].tolist() if isinstance(ws, torch.Tensor) else None,
                    "act_key": str(getattr(qm, "activation_quant_key", "")),
                    "kernel": type(getattr(qm, "fp8_linear", None)).__name__})
    return out


def engine(a) -> int:
    from vllm import LLM
    llm = LLM(model=a.model, quantization="fp8", max_model_len=4096, gpu_memory_utilization=0.6, enforce_eager=False)
    mods = llm.collective_rpc(_collect)[0]
    pc = llm.llm_engine.vllm_config.compilation_config
    info = {"pass_config": str(getattr(pc, "pass_config", "")), "custom_ops": str(getattr(pc, "custom_ops", "")),
            "mode": str(getattr(pc, "mode", getattr(pc, "level", ""))), "modules": mods}
    Path(a.out).write_text(json.dumps(info, indent=1))
    from collections import Counter
    print("pass_config:", info["pass_config"][:600])
    print("custom_ops:", info["custom_ops"][:300])
    print("quant methods:", Counter((x["cls"], x["qm"], x["dtype"]) for x in mods).most_common())
    print("activation keys / kernels:", Counter((x["act_key"], x["kernel"]) for x in mods if x["qm"] != "UnquantizedLinearMethod").most_common(4))
    return 0


def compare(a) -> int:
    import re
    from safetensors import safe_open
    eng = json.loads(Path(a.engine).read_text())
    idx = json.loads((Path(a.model) / "model.safetensors.index.json").read_text())["weight_map"]
    handles = {}

    def get(name):
        f = idx[name]
        if f not in handles:
            handles[f] = safe_open(str(Path(a.model) / f), framework="pt")
        return handles[f].get_tensor(name).float()
    fused = {"qkv_proj": ["q_proj", "k_proj", "v_proj"], "gate_up_proj": ["gate_proj", "up_proj"],
             "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"], "in_proj_ba": ["in_proj_b", "in_proj_a"]}
    bad, n = 0, 0
    for x in eng["modules"]:
        if x["scale"] is None or x["qm"] == "UnquantizedLinearMethod":
            continue
        parent, _, leaf = x["name"].rpartition(".")
        parts = fused.get(leaf, [leaf])
        names = [f"{parent}.{p}.weight" for p in parts]
        names = [nm if nm in idx else nm.replace("model.", "model.language_model.", 1) for nm in names]
        if not all(nm in idx for nm in names):
            print("no checkpoint tensor for", x["name"], names[:1]); continue
        amax = max(float(get(nm).abs().max()) for nm in names)
        mine, theirs = amax / 448.0, x["scale"][0]
        n += 1
        if abs(mine - theirs) > 1e-6 * max(abs(theirs), 1e-12):
            bad += 1
            if bad <= 12:
                print(f"scale differs {x['name']}: engine {theirs:.6e} ours {mine:.6e} (ratio {theirs / mine:.4f}) parts {parts}")
    print(f"compared {n} quantized modules, {bad} with a different weight scale")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pe = sub.add_parser("engine"); pe.add_argument("--model", required=True); pe.add_argument("--out", required=True)
    pc = sub.add_parser("compare"); pc.add_argument("--model", required=True); pc.add_argument("--engine", required=True)
    a = ap.parse_args()
    return {"engine": engine, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
