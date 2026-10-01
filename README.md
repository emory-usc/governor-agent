# Governed Agent

[![CI](https://github.com/emory-usc/governed-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/emory-usc/governed-agent/actions/workflows/ci.yml)

A governed LangGraph agent with the pieces production teams actually need:
checkpointing, streaming, human-in-the-loop approval, tracing, and an eval
harness that proves the guardrails hold. The tools are the governed, read-only
query functions from `governed-mcp` — row-level security enforced on every
tool call, so the agent can only ever see what its caller is allowed to see.

This repo is a cookbook, not a toy: every pattern here (durable checkpoints,
interrupt/resume approval, LangSmith-shaped tracing, adversarial evals) is
reusable against your own tools and data.

## What it demonstrates

| Capability | How |
|---|---|
| **Tool calling with enforced boundaries** | Tools are wrapped governed-mcp queries: fail-closed RLS + masking on every call |
| **Human-in-the-loop** | The graph interrupts before tool execution; a human approves or rejects, and the run resumes from the checkpoint |
| **Checkpointing / durability** | SQLite checkpointer persists state per thread; kill the process mid-run and resume exactly where it stopped |
| **Streaming** | Token-level LLM streaming plus per-node state updates |
| **Tracing** | An offline, LangSmith-shaped run tree (spans, tool calls, timings) — swap one env var to stream the same runs into LangSmith |
| **Evals** | Adversarial cases proving the agent does not fabricate data the tools withheld, plus interrupt and durability cases |

## Quick start

```bash
pip install -e ".[dev]"

# deterministic offline demo (no API keys, scripted model)
governed-agent run "What is the total balance by region?"

# same graph, real model (set one of:)
#   OPENAI_API_KEY=...          -> gpt-4o-mini via langchain-openai
#   ANTHROPIC_API_KEY=...       -> claude-3-5-haiku via langchain-anthropic

governed-agent eval            # adversarial + durability eval harness
governed-agent trace           # print the offline run tree
```

## Architecture

```
user ──► agent (LLM + tools) ──► interrupt (HITL approval)
              │                        │ approve / reject
              ▼                        ▼
          tools node ◄── checkpoint (SQLite, per thread)
              │
              ▼
         final answer
```

`docs/architecture.md` has the full production story: how the checkpoint
survives restarts, how to wire LangSmith tracing, and how the approval gate
maps onto a real review workflow.

## Consuming the MCP server (alternative tool path)

The default tools call the governed-mcp functions in-process. To consume the
governed-mcp **MCP server** over the wire instead, install the `mcp` extra and
build the agent with `tools=load_mcp_tools()`:

```python
from governed_agent.tools import load_mcp_tools  # via langchain-mcp-adapters

# point it at a running governed-mcp server (see governed-mcp repo)
```

## Honest notes

- The bundled default is a **scripted deterministic model** so CI, tests, and
  demos run without API keys. The graph, checkpoints, interrupts, and tracing
  are identical with a real model — only the model binding changes.
- Synthetic data only; nothing here is real customer data or financial advice.
