"""Durability tests: the checkpoint survives graph + connection boundaries."""

import os
import sqlite3
import tempfile

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from governed_agent import tools
from governed_agent.graph import build_graph, get_interrupt, make_checkpointer
from governed_agent.model import scripted_model


def test_checkpoint_survives_new_graph_instance():
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
        assert get_interrupt(result_a) is not None
        conn_a.close()
        conn_a = None

        # process B: fresh graph, fresh connection, same DB file
        conn_b = sqlite3.connect(path, check_same_thread=False)
        graph_b = build_graph(SqliteSaver(conn_b))
        result_b = graph_b.invoke(Command(resume=True), config=config)
        tool_msgs = [m for m in result_b["messages"] if m.type == "tool"]
        assert tool_msgs, "pending tool call must execute after cross-instance resume"
        conn_b.close()
        conn_b = None
    finally:
        if conn_a is not None:
            conn_a.close()
        if conn_b is not None:
            conn_b.close()
        if os.path.exists(path):
            os.remove(path)


def test_second_thread_is_independent():
    graph = build_graph(make_checkpointer())
    model = scripted_model(tools.TOOLS)
    config = {"configurable": {"thread_id": "thread-x", "caller": "admin", "model": model}}
    result = graph.invoke(
        {"messages": [HumanMessage(content="total balance by region")], "caller": "admin"},
        config=config,
    )
    assert get_interrupt(result) is not None
    # a different thread starts clean, not mid-interrupt
    other = {"configurable": {"thread_id": "thread-y", "caller": "admin", "model": model}}
    result_other = graph.invoke(
        {"messages": [HumanMessage(content="hi")], "caller": "admin"},
        config=other,
    )
    assert get_interrupt(result_other) is None
