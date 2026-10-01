"""Offline, LangSmith-shaped run tracer.

``TraceCollector`` is a standard LangChain callback handler: it records
on_llm_*/on_tool_*/on_chain_* events into a run tree with the same shape
LangSmith uses (run id, parent, type, inputs/outputs, duration). In production
you set LANGCHAIN_TRACING_V2=true and LANGSMITH_API_KEY and the SAME graph
emits these runs to LangSmith automatically — no code change.
"""

from __future__ import annotations

import time
import uuid

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class TraceCollector(BaseCallbackHandler):
    def __init__(self):
        self.runs: list[dict] = []
        self._stack: list[str] = []

    # -- run-tree plumbing ---------------------------------------------------
    def _push(self, run_type: str, name: str, inputs, run_id=None, parent_run_id=None) -> str:
        # prefer the callback manager's ids so start/end pair correctly
        run_id = str(run_id) if run_id is not None else str(uuid.uuid4())
        parent = (
            str(parent_run_id)
            if parent_run_id is not None
            else (self._stack[-1] if self._stack else None)
        )
        self.runs.append(
            {
                "id": run_id,
                "parent_id": parent,
                "name": name,
                "run_type": run_type,
                "start_ms": int(time.time() * 1000),
                "inputs": _sanitize(inputs),
                "outputs": None,
                "end_ms": None,
            }
        )
        self._stack.append(run_id)
        return run_id

    def _pop(self, run_id, outputs) -> None:
        run_id = str(run_id) if run_id is not None else (self._stack[-1] if self._stack else None)
        if run_id is None:
            return
        for run in self.runs:
            if run["id"] == run_id:
                run["outputs"] = _sanitize(outputs)
                run["end_ms"] = int(time.time() * 1000)
                break
        if self._stack and self._stack[-1] == run_id:
            self._stack.pop()

    # -- chain events ---------------------------------------------------------
    def on_chain_start(self, serialized, inputs, **kwargs) -> None:
        name = serialized.get("name", "chain") if serialized else "chain"
        self._push(
            "chain", name, inputs,
            run_id=kwargs.get("run_id"), parent_run_id=kwargs.get("parent_run_id"),
        )

    def on_chain_end(self, outputs, **kwargs) -> None:
        self._pop(kwargs.get("run_id"), outputs)

    # -- llm events -----------------------------------------------------------
    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        name = serialized.get("name", "llm") if serialized else "llm"
        self._push(
            "llm", name, {"prompts": list(prompts)},
            run_id=kwargs.get("run_id"), parent_run_id=kwargs.get("parent_run_id"),
        )

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        generations = [
            {"text": g.text, "tool_calls": getattr(g.message, "tool_calls", None)}
            for gen in response.generations
            for g in gen
        ]
        self._pop(kwargs.get("run_id"), {"generations": generations})

    # -- tool events ----------------------------------------------------------
    def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        name = serialized.get("name", "tool") if serialized else "tool"
        self._push(
            "tool", name, {"input": input_str},
            run_id=kwargs.get("run_id"), parent_run_id=kwargs.get("parent_run_id"),
        )

    def on_tool_end(self, output, **kwargs) -> None:
        self._pop(kwargs.get("run_id"), {"output": output})

    # -- rendering -------------------------------------------------------------
    def render(self) -> str:
        lines = []
        for run in self.runs:
            depth = self._depth(run)
            dur = (
                f"{run['end_ms'] - run['start_ms']}ms"
                if run["end_ms"] is not None
                else "in-flight"
            )
            lines.append(f"{'  ' * depth}[{run['run_type']}] {run['name']} ({dur})")
        return "\n".join(lines)

    def _depth(self, run: dict) -> int:
        depth = 0
        parent = run["parent_id"]
        by_id = {r["id"]: r for r in self.runs}
        while parent is not None:
            depth += 1
            parent = by_id.get(parent, {}).get("parent_id")
        return depth


def _sanitize(obj):
    """Keep the trace JSON-safe and bounded (no message blobs)."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(v, (str, int, float, bool, type(None))):
                out[k] = v
            elif isinstance(v, (list, tuple)):
                out[k] = [_sanitize(x) for x in v][:5]
            else:
                out[k] = str(v)[:200]
        return out
    if isinstance(obj, (list, tuple)):
        return [_sanitize(x) for x in obj][:5]
    if isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)[:200]
