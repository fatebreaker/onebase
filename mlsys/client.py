"""Tiny stdlib client for a vLLM server started with mlsys.vllm_ext.VariantWorker (VLLM_SERVER_DEV_MODE=1).

    python -m mlsys.client URL wait [SECONDS]
    python -m mlsys.client URL info
    python -m mlsys.client URL apply '{"scheme": "int4g128_fresh", "role": "base"}'     # also resets the prefix cache
    python -m mlsys.client URL load-lora NAME ADAPTER_DIR
    python -m mlsys.client URL unload-lora NAME
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request


def _post(url: str, path: str, body: dict | None = None, timeout: float = 3600) -> dict | None:
    req = urllib.request.Request(url.rstrip("/") + path, data=json.dumps(body or {}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    if not raw.strip():
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:                 # e.g. /v1/load_lora_adapter answers 200 with plain text
        return {"text": raw.decode(errors="replace")}


def wait(url: str, seconds: float = 1800) -> bool:
    t0 = time.time()
    while time.time() - t0 < seconds:
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/v1/models", timeout=10) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(5)
    return False


def rpc(url: str, method: str, *args: str) -> list:
    out = _post(url, "/collective_rpc", {"method": method, "args": list(args)})
    return (out or {}).get("results", [])


def apply(url: str, spec: dict) -> list:
    res = rpc(url, "apply_variant", json.dumps(spec))
    _post(url, "/reset_prefix_cache", timeout=120)
    return res


def load_lora(url: str, name: str, path: str) -> None:
    _post(url, "/v1/load_lora_adapter", {"lora_name": name, "lora_path": path}, timeout=600)


def unload_lora(url: str, name: str) -> None:
    _post(url, "/v1/unload_lora_adapter", {"lora_name": name}, timeout=600)


def main(argv: list[str]) -> int:
    url, cmd, rest = argv[1], argv[2], argv[3:]
    if cmd == "wait":
        ok = wait(url, float(rest[0]) if rest else 1800)
        print("up" if ok else "TIMEOUT")
        return 0 if ok else 1
    if cmd == "info":
        print(json.dumps(rpc(url, "variant_info")))
    elif cmd == "apply":
        print(json.dumps(apply(url, json.loads(rest[0]))))
    elif cmd == "load-lora":
        load_lora(url, rest[0], rest[1])
        print(f"loaded {rest[0]}")
    elif cmd == "unload-lora":
        unload_lora(url, rest[0])
        print(f"unloaded {rest[0]}")
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
