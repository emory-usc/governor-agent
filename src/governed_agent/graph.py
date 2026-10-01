"""The agent graph.

agent -> tools (human approval interrupt) -> agent -> END

Checkpointing: every step is persisted per thread, so an interrupted run can
be resumed from any process after any amount of time — kill the process mid-run
and pick up exactly where it stopped.
"""

from __future__ import annotations

import sqlite3
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from governed_agent import tools
from governed_agent.model import build_model


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    caller: str


def _model(config):
    """Model from configurable override (tests) or env-driven default."""
    cfg = config.get("configurable", {}) if config else {}
    if "model" in cfg:
        return cfg["model"]  # scripted model: tool_calls arrive pre-scripted
    return build_model(tools.TOOLS)  # real models bind tools inside


def agent_node(state: AgentState, config) -> dict:
    response = _model(config).invoke(state["messages"])
    return {"messages": [response]}


def tool_node(state: AgentState, config) -> dict:
    """Interrupt for human approval before any tool executes.

    The run pauses at the checkpoint; resume with
    ``graph.invoke(Command(resume=True), config)`` to approve or
    ``Command(resume="reject")`` to refuse.
    """
    last = state["messages"][-1]
    calls = list(getattr(last, "tool_calls", None) or [])
    if not calls:
        return {}
    approval = interrupt(
        {
            "question": "Approve the following tool calls?",
            "tool_calls": [
                {"name": c["name"], "args": c.get("args", {})} for c in calls
            ],
        }
    )
    if approval in (False, "reject", "no"):
        return {
            "messages": [
                ToolMessage(
                    content="Tool call rejected by the human.",
                    tool_call_id=c["id"],
                )
                for c in calls
            ]
        }
    return ToolNode(tools.TOOLS).invoke({"messages": state["messages"]})


def _route_after_agent(state: AgentState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


def build_graph(checkpointer=None):
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", _route_after_agent, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)


def make_checkpointer(path: str | None = None) -> SqliteSaver:
    """Durable SQLite checkpointer. ``path=None`` keeps it in-memory."""
    conn = sqlite3.connect(path or ":memory:", check_same_thread=False)
    return SqliteSaver(conn)


def get_interrupt(result: dict) -> dict | None:
    """langgraph 1.x returns the state dict with an ``__interrupt__`` key
    instead of raising. Return the first interrupt payload, if any."""
    intr = result.get("__interrupt__") or []
    return intr[0].value if intr else None
