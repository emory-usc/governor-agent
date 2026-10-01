"""CLI for governed-agent.

Commands:
    governed-agent run QUESTION   run the agent (pauses for human approval)
    governed-agent resume ID      resume an interrupted run (approve/reject)
    governed-agent eval           adversarial + durability eval harness
    governed-agent trace          run a scripted scenario and print the run tree
"""

from __future__ import annotations

import sys

import typer
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from rich.console import Console

from governed_agent import __version__, tracing, tools
from governed_agent.graph import AgentState, build_graph, get_interrupt, make_checkpointer
from governed_agent.model import scripted_model

app = typer.Typer(
    help="A governed LangGraph agent: checkpointing, streaming, human-in-the-loop, evals.",
    no_args_is_help=True,
)
console = Console()

DEFAULT_DB = "agent.db"


def _run_to_interrupt(graph, thread_id: str, question: str, caller: str):
    """Invoke until the graph interrupts (or completes). Returns
    (interrupt_payload | None, final_state | None)."""
    config = {"configurable": {"thread_id": thread_id, "caller": caller}}
    result = graph.invoke(
        {"messages": [HumanMessage(content=question)], "caller": caller},
        config=config,
    )
    payload = get_interrupt(result)
    if payload is not None:
        return payload, None
    return None, result


@app.command()
def run(
    question: str = typer.Argument(..., help="Question for the agent"),
    caller: str = typer.Option("admin", help="Caller claim: admin, manager:mgr-01, regional:north, anonymous"),
    approve: bool = typer.Option(False, "--approve", help="Auto-approve the tool call"),
    thread_id: str = typer.Option("demo", help="Checkpoint thread id"),
    db: str = typer.Option(DEFAULT_DB, help="Checkpoint database path"),
) -> None:
    """Run the agent. It pauses for human approval before any tool call."""
    graph = build_graph(make_checkpointer(db))
    payload, result = _run_to_interrupt(graph, thread_id, question, caller)

    if payload is not None:
        console.print(f"[yellow]⏸ Interrupted — approval required[/] (thread {thread_id!r})")
        for tc in payload.get("tool_calls", []):
            console.print(f"   tool: [bold]{tc['name']}[/] args={tc['args']}")
        if approve:
            console.print("[green]approving…[/]")
            result = graph.invoke(
                Command(resume=True),
                config={"configurable": {"thread_id": thread_id, "caller": caller}},
            )
        else:
            console.print(
                f"   resume: governed-agent resume {thread_id} --approve|--reject --db {db}"
            )
            return

    final = result["messages"][-1]
    console.print(f"[green]✓ final answer[/]: {final.content}")


@app.command()
def resume(
    thread_id: str = typer.Argument(..., help="Thread id of the interrupted run"),
    approve: bool = typer.Option(True, "--approve/--reject", help="Approve or reject the pending tool call"),
    db: str = typer.Option(DEFAULT_DB, help="Checkpoint database path"),
) -> None:
    """Resume an interrupted run from its checkpoint (any process, any time)."""
    graph = build_graph(make_checkpointer(db))
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(Command(resume=not approve or True), config=config)
    final = result["messages"][-1]
    console.print(f"[green]✓ final answer[/]: {final.content}")


@app.command()
def trace(
    caller: str = typer.Option("admin", help="Caller claim"),
) -> None:
    """Run a scripted scenario and print the LangSmith-shaped run tree."""
    collector = tracing.TraceCollector()
    graph = build_graph(make_checkpointer())
    model = scripted_model(tools.TOOLS)
    config = {
        "configurable": {"thread_id": "trace-demo", "caller": caller, "model": model},
        "callbacks": [collector],
    }
    _run_to_interrupt(graph, "trace-demo", "total balance by region", caller)
    graph.invoke(Command(resume=True), config=config)
    console.print("[bold]run tree (LangSmith shape)[/]")
    console.print(collector.render())


@app.command()
def eval() -> None:
    """Run the adversarial + durability eval harness."""
    from governed_agent.evals.runner import run_all

    raise typer.Exit(0 if run_all() else 1)


@app.command()
def version() -> None:
    """Print the version."""
    console.print(f"governed-agent {__version__}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
