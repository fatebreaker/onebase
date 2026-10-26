"""BFCL (Berkeley Function-Calling Leaderboard, bfcl-eval 2026.3.23) for TOOL-RL, run in its own venv
($SCRATCH/venv_bfcl: bfcl-eval without its model-API and embedding dependencies).
Generation happens elsewhere (bench/std_bench3.py in the vLLM venv); this script builds the prompts and scores the
responses with BFCL's own code, as its leaderboard does for a local model in prompting mode:

  prep-bfcl     the test entries of the single-turn categories (load_dataset_entry, with BFCL's language hints), each
                turned into chat messages by the local handlers' prompting pre-processing
                (system_prompt_pre_processing_chat_model: BFCL's default system prompt listing the functions) -> JSONL
                {id, category, messages}
  prep-toolace  ToolACE (Team-ACE/ToolACE, Apache-2.0) examples whose first assistant turn is a function call, as RL
                prompts in the same format (its functions with BFCL's Python hint, BFCL's system prompt, the first user
                turn) with the reference calls decoded by BFCL's decoder -> JSONL {id, messages, gold}; examples whose
                reference does not decode (e.g. function names with spaces) are dropped
  score         a responses file {id, response} -> {id, category, correct}: AST categories through BFCL's
                default_decode_ast_prompting, is_function_calling_format_output and ast_checker; irrelevance categories
                correct when the response holds no decodable call (BFCL's relevance rule)
  check-decoder bench/toolcall.py's decoder (the trainer's reward, standard library only) against BFCL's on ToolACE's
                reference calls

ast_checker imports BFCL's model registry only to ask whether a model's API rewrites "." in function names; a local
model in prompting mode does not, so a stub registry answering "no" is installed before the import (no BFCL code is
changed).
    venv_bfcl/bin/python -I bench/bfcl_tools.py prep-bfcl --out <data>/bfcl_dl/bfcl_prompts.jsonl
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
CATS = ("simple_python", "multiple", "parallel", "parallel_multiple", "live_simple", "live_multiple", "live_parallel",
        "live_parallel_multiple", "irrelevance", "live_irrelevance")


def _stub_registry() -> None:
    class _Cfg:
        underscore_to_dot = False
    m = types.ModuleType("bfcl_eval.constants.model_config")
    m.MODEL_CONFIG_MAPPING = collections.defaultdict(_Cfg)
    sys.modules["bfcl_eval.constants.model_config"] = m


def prep_bfcl(a) -> int:
    from bfcl_eval.model_handler.utils import system_prompt_pre_processing_chat_model
    from bfcl_eval.utils import load_dataset_entry
    n = collections.Counter()
    with open(a.out, "w") as f:
        for cat in CATS:
            for e in load_dataset_entry(cat):
                msgs = system_prompt_pre_processing_chat_model(copy.deepcopy(e["question"][0]), e["function"], e["id"])
                f.write(json.dumps({"id": e["id"], "category": cat, "messages": msgs}) + "\n")
                n[cat] += 1
    print(json.dumps(n), sum(n.values()))
    return 0


def prep_toolace(a) -> int:
    from bfcl_eval.model_handler.utils import default_decode_ast_prompting, system_prompt_pre_processing_chat_model
    from bfcl_eval.utils import _func_doc_language_specific_pre_processing
    data = json.loads(Path(a.src).read_text())
    key = "Here is a list of functions in JSON format that you can invoke:"
    stats, kept = collections.Counter(), 0
    with open(a.out, "w") as f:
        for i, r in enumerate(data):
            cs = r["conversations"]
            if len(cs) < 2 or cs[0]["from"] != "user" or cs[1]["from"] != "assistant":
                stats["not user->assistant"] += 1
                continue
            ref = cs[1]["value"].strip()
            if not ref.startswith("["):
                stats["first reply not a call"] += 1
                continue
            try:                                                  # the JSON list after the key (text may follow it)
                tail = r["system"].split(key, 1)[1]
                funcs, _ = json.JSONDecoder().raw_decode(tail[tail.index("["):])
            except (IndexError, ValueError):
                stats["functions not parsed"] += 1
                continue
            try:
                gold = default_decode_ast_prompting(ref)
                funcs = _func_doc_language_specific_pre_processing(copy.deepcopy(funcs), "simple_python")
            except Exception:                                    # noqa: BLE001 -- the reference is not BFCL-decodable
                stats["reference not decodable"] += 1
                continue
            if not gold or not all(isinstance(g, dict) and len(g) == 1 for g in gold):
                stats["reference empty"] += 1
                continue
            msgs = system_prompt_pre_processing_chat_model([{"role": "user", "content": cs[0]["value"]}], funcs, f"toolace_{i}")
            f.write(json.dumps({"id": f"toolace-{i}", "messages": msgs, "gold": gold, "n_calls": len(gold)}) + "\n")
            kept += 1
    print(json.dumps(stats), "kept", kept, "of", len(data))
    return 0


def score(a) -> int:
    _stub_registry()
    from bfcl_eval.constants.enums import Language
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
    from bfcl_eval.model_handler.utils import default_decode_ast_prompting
    from bfcl_eval.utils import (extract_test_category_from_id, is_empty_output, is_function_calling_format_output,
                                 load_dataset_entry, load_file, POSSIBLE_ANSWER_PATH, VERSION_PREFIX)
    resp = {r["id"]: r["response"] for r in map(json.loads, Path(a.raw).read_text().splitlines())}
    cats = sorted({extract_test_category_from_id(i) for i in resp})
    out, acc = [], collections.defaultdict(list)
    for cat in cats:
        entries = {e["id"]: e for e in load_dataset_entry(cat, include_prereq=False, include_language_specific_hint=False)}
        answers = {} if "irrelevance" in cat else {x["id"]: x["ground_truth"] for x in
                                                   load_file(POSSIBLE_ANSWER_PATH / f"{VERSION_PREFIX}_{cat}.json")}
        for i, e in entries.items():
            if i not in resp:
                continue
            text = resp[i]
            if "irrelevance" in cat:
                try:
                    dec = default_decode_ast_prompting(text)
                    ok = is_empty_output(dec)
                except Exception:                                # noqa: BLE001 -- no decodable call: correct here
                    ok = True
            else:
                try:
                    dec = default_decode_ast_prompting(text)
                    ok = is_function_calling_format_output(dec) and ast_checker(
                        e["function"], dec, answers[i], Language.PYTHON, cat, "local-prompting")["valid"]
                except Exception:                                # noqa: BLE001 -- undecodable: wrong
                    ok = False
            out.append({"id": i, "category": cat, "correct": int(bool(ok))})
            acc[cat].append(int(bool(ok)))
    tmp = Path(a.out).with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in out))
    tmp.replace(a.out)
    print(json.dumps({c: [round(100 * sum(v) / len(v), 1), len(v)] for c, v in acc.items()}))
    return 0


def check_decoder(a) -> int:
    sys.path.insert(0, str(HERE))
    import toolcall
    from bfcl_eval.model_handler.utils import default_decode_ast_prompting
    rows = [json.loads(x) for x in Path(a.toolace).read_text().splitlines()]
    data = json.loads(Path(a.src).read_text())
    same = diff = 0
    ex = []
    for r in rows:
        ref = data[int(r["id"].split("-")[1])]["conversations"][1]["value"]
        theirs = default_decode_ast_prompting(ref)
        try:
            ours = toolcall.decode(ref)
        except Exception as e:                                   # noqa: BLE001
            ours = f"error: {e}"
        if toolcall.match(ours, theirs) and json.dumps(ours, sort_keys=True, default=str) == json.dumps(theirs, sort_keys=True, default=str):
            same += 1
        else:
            diff += 1
            if len(ex) < 5:
                ex.append((ref[:160], ours, theirs))
    print(json.dumps({"same": same, "different": diff}))
    for e in ex:
        print(e)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep-bfcl")
    p.add_argument("--out", required=True)
    p = sub.add_parser("prep-toolace")
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("score")
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("check-decoder")
    p.add_argument("--toolace", required=True)
    p.add_argument("--src", required=True)
    a = ap.parse_args()
    return {"prep-bfcl": prep_bfcl, "prep-toolace": prep_toolace, "score": score, "check-decoder": check_decoder}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
