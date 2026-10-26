"""ICML-E1B-ARCH (docs/REGISTRY.md): ICML-E1's contexts for another model family. Each row's prompt (Qwen3.5's chat template) is
decoded, its user message taken and re-rendered with the target model's chat template (generation prompt added); the completion is
decoded and re-encoded with the target tokenizer (no special tokens). The texts are the same problems and completions, so
bench/bias_noise_tf.py measures the same teacher-forced contexts on a dense model.
    python3 bench/e1b_retok.py --data results/icml_e1_bf16_65181900.json --src <Qwen3.5 dir> --dst <model dir> --out <json>"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ts, td = AutoTokenizer.from_pretrained(a.src), AutoTokenizer.from_pretrained(a.dst)
    d = json.loads(Path(a.data).read_text())
    rows = []
    for r in d["rows"]:
        if not r["completion"]:
            continue
        p = ts.decode(r["prompt"], skip_special_tokens=False)
        i = p.index("<|im_start|>user\n") + len("<|im_start|>user\n")
        user = p[i:p.index("<|im_end|>", i)]
        text = td.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True, tokenize=False)
        ids = td(text, add_special_tokens=False).input_ids              # the template's own special tokens are in the text
        comp = td(ts.decode(r["completion"], skip_special_tokens=True), add_special_tokens=False).input_ids
        rows.append({"prompt": list(ids), "completion": list(comp)})
    out = {"source": a.data, "src_tokenizer": a.src, "dst_tokenizer": a.dst, "rows": rows}
    Path(a.out).write_text(json.dumps(out))
    print(f"{len(rows)} rows, {sum(len(r['completion']) for r in rows)} completion tokens -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
