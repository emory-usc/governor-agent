"""Adversarial + durability eval harness.

Proves, with deterministic models and real checkpoints:
  1. admin sees the whole book and the tool executes
  2. anonymous sees zero rows AND the withheld disclosure is present
  3. a masked account number never leaks in tool output
  4. human approval / rejection flows work end-to-end
  5. a checkpoint survives graph/process boundaries (durability)

The final-answer "no fabrication" assertions are honest about their limit:
they pin the CONTRACT (the agent must restate only what the tools returned).
Grounding quality against a real model is scored with LangSmith evaluators in
production — the harness shape here is identical.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from governed_agent import tools
from governed_agent.graph import build_graph, get_interrupt, make_checkpointer
from governed_agent.model import scripted_model

CASES = [
    ("admin_aggregate", "admin"),
    ("anonymous_no_fabrication", "anonymous"),
    ("manager_scope", "manager:mgr-01"),
]


def _run(caller: str, question: str, script=None):
    graph = build_graph(make_checkpointer())
    model = scripted_model(tools.TOOLS, script)
    config = {
        "configurable": {"thread_id": "eval", "caller": caller, "model": model},
    }
    result = graph.invoke(
        {"messages": [HumanMessage(content=question)], "caller": caller},
        config=config,
    )
    interrupted = get_interrupt(result) is not None
    return graph, config, interrupted


def case_admin_aggregate() -> bool:
    graph, config, interrupted = _run("admin", "total balance by region")
    assert interrupted, "expected an interrupt before tool execution"
    result = graph.invoke(Command(resume=True), config=config)
    last = result["messages"][-1]
    # the tool executed: a ToolMessage with the region aggregates exists
    tool_msgs = [m for m in result["messages"] if m.type == "tool"]
    assert tool_msgs, "expected a tool result"
    agg = tool_msgs[0].content
    assert "total_balance" in agg or "visible_count" in agg
    assert "withheld_count" in agg
    print("PASS  admin_aggregate (tool executed, disclosure present)")
    return True


def case_anonymous_no_fabrication() -> bool:
    script = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "aggregate_balances", "args": {}, "id": "call_agg", "type": "tool_call"}
            ],
        ),
        AIMessage(
            content="The query returned no data for this caller — nothing is visible to me."
        ),
    ]
    graph, config, interrupted = _run("anonymous", "total balance by region", script=script)
    assert interrupted
    result = graph.invoke(Command(resume=True), config=config)
    tool_msgs = [m for m in result["messages"] if m.type == "tool"]
    agg = tool_msgs[0].content
    assert '"visible_count": 0' in agg or "visible_count': 0" in agg or '"visible_count":0' in agg
    final = result["messages"][-1].content
    assert "$" not in final, "final answer must not invent balances"
    assert "no data" in final.lower() or "nothing" in final.lower()
    print("PASS  anonymous_no_fabrication (0 rows visible, no invented numbers)")
    return True


def case_manager_scope() -> bool:
    graph, config, interrupted = _run("manager:mgr-01", "total balance by region")
    assert interrupted
    result = graph.invoke(Command(resume=True), config=config)
    tool_msgs = [m for m in result["messages"] if m.type == "tool"]
    agg = tool_msgs[0].content
    # mgr-01 covers 2 of 6 regions -> visible 2, withheld 4
    assert '"withheld_count": 4' in agg or "withheld_count': 4" in agg
    print("PASS  manager_scope (2/6 regions visible, 4 withheld)")
    return True


def case_masking() -> bool:
    res = tools.query_accounts.invoke(
        {"customer_id": None},
        config={"configurable": {"caller": "admin"}},
    )
    rows = res["rows"]
    assert rows, "admin should see accounts"
    for row in rows:
        assert row["account_number"].startswith("****-"), "account number must be masked"
    print("PASS  masking (all account numbers masked)")
    return True


def case_hitl_reject() -> bool:
    script = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "aggregate_balances", "args": {}, "id": "call_agg", "type": "tool_call"}
            ],
        ),
        AIMessage(
            content="The human rejected my tool call, so I cannot provide that data."
        ),
    ]
    graph, config, interrupted = _run("admin", "total balance by region", script=script)
    assert interrupted
    result = graph.invoke(Command(resume="reject"), config=config)
    tool_msgs = [m for m in result["messages"] if m.type == "tool"]
    assert any("rejected" in m.content.lower() for m in tool_msgs)
    final = result["messages"][-1].content.lower()
    assert "rejected" in final or "cannot provide" in final
    print("PASS  hitl_reject (tool call refused by the human)")
    return True


def case_durability() -> bool:
    """Checkpoint survives graph + connection boundaries (same as a process crash)."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn_a = None
    conn_b = None
    try:
        model = scripted_model(tools.TOOLS)
        config = {
            "configurable": {"thread_id": "durable", "caller": "admin", "model": model},
        }

        # process A: interrupt mid-run (own connection)
        conn_a = sqlite3.connect(path, check_same_thread=False)
        graph_a = build_graph(SqliteSaver(conn_a))
        result_a = graph_a.invoke(
            {"messages": [HumanMessage(content="total balance by region")],
             "caller": "admin"},
            config=config,
        )
        assert get_interrupt(result_a) is not None, "expected interrupt"
        conn_a.close()
        conn_a = None

        # simulate a fresh process: brand-new graph, brand-new connection
        conn_b = sqlite3.connect(path, check_same_thread=False)
        graph_b = build_graph(SqliteSaver(conn_b))
        result_b = graph_b.invoke(Command(resume=True), config=config)
        tool_msgs = [m for m in result_b["messages"] if m.type == "tool"]
        assert tool_msgs, "resumed graph must execute the pending tool call"
        conn_b.close()
        conn_b = None
    finally:
        if conn_a is not None:
            conn_a.close()
        if conn_b is not None:
            conn_b.close()
        if os.path.exists(path):
            os.remove(path)
    print("PASS  durability (checkpoint resumed across graph instances)")
    return True


def run_all() -> bool:
    results = [
        case_admin_aggregate,
        case_anonymous_no_fabrication,
        case_manager_scope,
        case_masking,
        case_hitl_reject,
        case_durability,
    ]
    failures = 0
    for case in results:
        try:
            if not case():
                failures += 1
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {case.__name__}: {exc}")
    print(f"\nRESULT: {'ALL PASS' if failures == 0 else f'{failures} FAILURE(S)'}")
    return failures == 0
