"""Tests for the agent graph: routing, HITL, tracing."""

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from governed_agent import tools
from governed_agent.graph import build_graph, get_interrupt
from governed_agent.model import scripted_model
from governed_agent.tracing import TraceCollector


def _run(caller="admin", script=None):
    graph = build_graph(MemorySaver())
    model = scripted_model(tools.TOOLS, script)
    config = {
        "configurable": {"thread_id": "t1", "caller": caller, "model": model},
    }
    result = graph.invoke(
        {"messages": [HumanMessage(content="total balance by region")], "caller": caller},
        config=config,
    )
    interrupted = get_interrupt(result) is not None
    return graph, config, interrupted


def test_agent_interrupts_before_tool_execution():
    graph, config, interrupted = _run()
    assert interrupted


def test_approve_completes_with_tool_result():
    graph, config, interrupted = _run()
    assert interrupted
    result = graph.invoke(Command(resume=True), config=config)
    types = [m.type for m in result["messages"]]
    assert types == ["human", "ai", "tool", "ai"]


def test_reject_returns_refusal():
    graph, config, interrupted = _run()
    assert interrupted
    result = graph.invoke(Command(resume="reject"), config=config)
    tool_msgs = [m for m in result["messages"] if m.type == "tool"]
    assert any("rejected" in m.content.lower() for m in tool_msgs)


def test_anonymous_tool_result_has_zero_rows():
    graph, config, interrupted = _run(caller="anonymous")
    assert interrupted
    result = graph.invoke(Command(resume=True), config=config)
    tool_msgs = [m for m in result["messages"] if m.type == "tool"]
    assert tool_msgs
    assert '"visible_count": 0' in tool_msgs[0].content


def test_trace_collector_builds_tree():
    graph = build_graph(MemorySaver())
    collector = TraceCollector()
    model = scripted_model(tools.TOOLS)
    config = {
        "configurable": {"thread_id": "t2", "caller": "admin", "model": model},
        "callbacks": [collector],
    }
    result = graph.invoke(
        {"messages": [HumanMessage(content="total balance by region")], "caller": "admin"},
        config=config,
    )
    assert get_interrupt(result) is not None
    result = graph.invoke(Command(resume=True), config=config)
    types = {r["run_type"] for r in collector.runs}
    assert "chain" in types
    assert "llm" in types
    assert "tool" in types
    # every non-root run has a parent
    for r in collector.runs:
        if r["parent_id"] is not None:
            assert any(x["id"] == r["parent_id"] for x in collector.runs)
