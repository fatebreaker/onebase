"""Execution-based grading for competitive-programming code (CODE-Q): a candidate program passes a problem if it passes
every test, run in bench/std_bench.py's sandbox (isolated interpreter, temporary directory, CPU and memory limits,
destructive os/shutil/subprocess functions disabled).
  stdin tests       the program reads the test's input on stdin; its stdout must equal the expected output up to
                    trailing whitespace on each line and blank trailing lines, or token by token with numbers equal
                    within 1e-6 (absolute or relative)
  functional tests  LeetCode style (LiveCodeBench): the program defines class Solution; the starter code names the
                    method; each test's input is one JSON value per line (the arguments), its output a JSON value; all
                    tests of a problem run in one process, which prints how many passed
LiveCodeBench stores private tests as base64(zlib(pickle(json string))); they are unpickled with a restricted
unpickler that resolves no globals, so only plain values can load.
    from code_exec import grade_tests, load_lcb_tests"""
from __future__ import annotations

import base64
import io
import json
import pickle
import re
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from std_bench import GUARD, _limits  # noqa: E402


class _Plain(pickle.Unpickler):
    def find_class(self, module, name):                          # no globals: plain values only
        raise pickle.UnpicklingError(f"refused global {module}.{name}")


def _safe_unpickle(b: bytes):
    return _Plain(io.BytesIO(b)).load()


def load_lcb_tests(row: dict) -> list[dict]:
    """A LiveCodeBench row's public and private tests: [{"input", "output", "testtype"}]."""
    tests = json.loads(row["public_test_cases"])
    priv = row.get("private_test_cases") or ""
    if priv:
        try:
            tests += json.loads(priv)
        except json.JSONDecodeError:
            tests += json.loads(_safe_unpickle(zlib.decompress(base64.b64decode(priv.encode()))))
    return tests


def extract_code(resp: str) -> str:
    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", resp, re.S)
    return m[-1] if m else ""


def _norm(s: str) -> str:
    return "\n".join(x.rstrip() for x in s.strip().splitlines())


def _same(out: str, want: str) -> bool:
    """Equal up to trailing whitespace, or token by token with numbers equal within 1e-6 (absolute or relative)."""
    if _norm(out) == _norm(want):
        return True
    a, b = out.split(), want.split()
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x == y:
            continue
        try:
            fx, fy = float(x), float(y)
        except ValueError:
            return False
        if abs(fx - fy) > 1e-6 * max(1.0, abs(fy)):
            return False
    return True


def _run(src: str, stdin: str, timeout: float) -> tuple[int, str]:
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "prog.py"
        f.write_text(GUARD + src)
        try:
            r = subprocess.run([sys.executable, "-I", str(f)], cwd=d, input=stdin, capture_output=True, text=True,
                               timeout=timeout, preexec_fn=_limits, env={"PATH": "/usr/bin:/bin"})
            return r.returncode, r.stdout
        except subprocess.TimeoutExpired:
            return -9, ""


def _functional(code: str, starter: str, tests: list[dict], timeout: float) -> bool:
    m = re.search(r"def\s+(\w+)\s*\(\s*self", starter or "")
    if not m:
        return False
    harness = code + "\n\nimport json as _j\n_tests = " + repr([(t["input"], t["output"]) for t in tests]) + f"""
_ok = 0
for _inp, _out in _tests:
    try:
        _args = [_j.loads(x) for x in _inp.strip().split("\\n") if x.strip()]
        _res = getattr(Solution(), {m.group(1)!r})(*_args)
        if _res == _j.loads(_out):
            _ok += 1
        else:
            break
    except Exception:
        break
print("PASSED", _ok)
"""
    rc, out = _run(harness, "", timeout * max(1, len(tests)) ** 0.5 + 5)
    mm = re.search(r"PASSED (\d+)", out)
    return bool(mm) and int(mm.group(1)) == len(tests)


def grade_tests(resp: str, tests: list[dict], starter: str = "", timeout: float = 6.0, max_tests: int = 0) -> int:
    """1 if the response's last code block passes every test (or the first max_tests), else 0."""
    code = extract_code(resp)
    if not code.strip():
        return 0
    if max_tests:
        tests = tests[:max_tests]
    if not tests:
        return 0
    if tests[0].get("testtype") == "functional":
        return int(_functional(code, starter, tests, timeout))
    for t in tests:
        rc, out = _run(code, t["input"], timeout)
        if rc != 0 or not _same(out, t["output"]):
            return 0
    return 1
