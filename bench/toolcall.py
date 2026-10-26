"""Tool-call parsing and matching for function-calling RL (TOOL-RL), standard library only, so the trainer's venv needs
nothing new. The format is BFCL's prompting format, which ToolACE's data uses: a Python list of calls,
    [func_name1(param1=value1, param2=value2), func_name2(param=value)]
decode() mirrors BFCL's default_decode_ast_prompting (bfcl_eval/model_handler/utils.py: strip backticks and newlines, add
missing brackets, ast.parse in eval mode, each call to {dotted name: {keyword: value}}); bench/bfcl_tools.py
check-decoder compares the two on ToolACE's reference calls. match() is the RL reward: 1 when the predicted calls equal
the reference calls as a multiset (same names, same keyword set, equal values: strings compared after BFCL's
standardize_string, numbers by value, lists and dicts element-wise), else 0.
"""
from __future__ import annotations

import ast
import re


def _name(node: ast.AST) -> str:
    """The dotted name of a call target, as BFCL's resolve_ast_call builds it (attributes, then a base name if any)."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _value(node: ast.AST):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)) and isinstance(node.operand, ast.Constant):
        v = node.operand.value
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [_value(e) for e in node.elts]
    if isinstance(node, ast.Dict):
        return {_value(k): _value(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.Name):                              # a bare identifier (True/False/None are Constants)
        return node.id
    if isinstance(node, ast.Call):                              # a nested call, kept as text
        return ast.unparse(node)
    return ast.unparse(node)


def decode(text: str) -> list[dict]:
    """The calls in a model response, as BFCL's prompting decoder reads them; raises on anything else."""
    s = text.strip("`\n ")
    if not s.startswith("["):
        s = "[" + s
    if not s.endswith("]"):
        s = s + "]"
    tree = ast.parse(s.strip().strip("'"), mode="eval")
    body = tree.body
    calls = [body] if isinstance(body, ast.Call) else list(getattr(body, "elts", []))
    if not isinstance(body, ast.Call) and not isinstance(body, (ast.List, ast.Tuple)):
        raise ValueError("not a list of calls")
    out = []
    for c in calls:
        if not isinstance(c, ast.Call):
            raise ValueError("list element is not a call")
        # positional arguments are ignored, as BFCL's resolve_ast_call ignores them
        out.append({_name(c.func): {kw.arg: _value(kw.value) for kw in c.keywords}})
    return out


def _std(s: str) -> str:
    """BFCL's standardize_string: drop spaces and ,./-_*^, lower-case, single quotes for double."""
    return re.sub(r"[ ,./\-_*^]", "", s).lower().replace("'", '"')


def _eq(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b if isinstance(a, bool) and isinstance(b, bool) else False
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    if isinstance(a, str) and isinstance(b, str):
        return _std(a) == _std(b)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_eq(a[k], b[k]) for k in a)
    return a == b


def _call_eq(p: dict, g: dict) -> bool:
    (pn, pa), = p.items()
    (gn, ga), = g.items()
    return pn == gn and pa.keys() == ga.keys() and all(_eq(pa[k], ga[k]) for k in ga)


def match(pred: list[dict], gold: list[dict]) -> int:
    """1 if the predicted calls equal the reference calls as a multiset, else 0."""
    if len(pred) != len(gold):
        return 0
    left = list(gold)
    for p in pred:
        for i, g in enumerate(left):
            if _call_eq(p, g):
                del left[i]
                break
        else:
            return 0
    return 1


def reward(text: str, gold: list[dict]) -> int:
    try:
        return match(decode(text), gold)
    except Exception:                                            # noqa: BLE001 -- unparsable output earns nothing
        return 0
