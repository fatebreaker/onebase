"""An RL run's per-step log (<adapters>/<run>/log.jsonl), read as docs/REGISTRY.md's "Resumed runs" note (2026-10-20) says. A run
that stops and resumes from its last checkpoint appends again the steps its stopped attempt had run past that checkpoint, so a step
can appear more than once. The run is the lineage of its latest attempt: the stopped attempt's rows from the resume point on are
dropped (time lost to the stop, reported apart and not charged to the run). For a finished run this is each step's last
occurrence; for a running one it also drops the stopped attempt's steps that the new attempt has not reached yet."""
from __future__ import annotations

import json
from pathlib import Path


def raw(f: Path) -> list[dict]:
    """Every row, in file order ([] if the log does not exist)."""
    return [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []


def read(f: Path) -> list[dict]:
    """One row per step, in step order: a row whose step is not above the previous one's starts a resumed attempt, which replaces
    the earlier rows from its step on."""
    out: list[dict] = []
    for r in raw(f):
        while out and out[-1]["step"] >= r["step"]:
            out.pop()
        out.append(r)
    return out


def lost_hours(f: Path) -> float:
    """Logged step time (hours) of the rows that a later attempt repeated."""
    return (sum(r["t"] for r in raw(f)) - sum(r["t"] for r in read(f))) / 3600
