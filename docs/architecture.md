# Governor Agent — architecture and production notes

## The graph

```
         ┌────────────────────────────────────────────────────────────┐
         │                     LangGraph StateGraph                  │
         │                                                            │
START ─► agent ──(tool_calls?)──► tools ──► agent ──(no calls)──► END │
           ▲                       │ ▲                                │
           │                       │ │ interrupt(resume=approve/reject)│
           └───────────────────────┘ │                                │
                                     ▼                                │
                            human approval (HITL)                     │
         └────────────────────────────────────────────────────────────┘
                  all state persisted per thread (SQLite checkpointer)
```

- **agent node** — one LLM call bound to the governed tools. With a real
  provider this streams tokens; with the bundled scripted model it is fully
  deterministic (tests/CI/demos).
- **tools node** — `interrupt()` BEFORE any tool executes. The run pauses at
  the checkpoint; a human resumes with `Command(resume=True)` (approve) or
  `Command(resume="reject")`.
- **routing** — agent → tools while the last message carries tool calls,
  otherwise END.
- **tools** — the governor-mcp query layer, ported in-process (see
  `governed.py`). Row-level security + masking + withheld disclosure on every
  call. The real governor-mcp MCP server can be consumed instead via
  `tools.load_mcp_tools()` (langchain-mcp-adapters).

## Durability

Every super-step is checkpointed to SQLite per `thread_id`. `tests/test_durability.py`
and the eval harness prove the strongest version of this: interrupt mid-run in
one graph instance, close the connection, open a brand-new graph instance on
the same file, resume — the pending tool call executes and the run completes.
That is the process-crash recovery story.

## Human-in-the-loop

The interrupt payload carries the proposed tool calls; the resume value is the
decision. A rejection produces a `ToolMessage` refusal — the agent's next
message must acknowledge it (the eval pins this contract).

## Tracing and observability

`TraceCollector` is a plain LangChain callback handler building a LangSmith-
shaped run tree (run id, parent, type, inputs/outputs, durations). In
production:

```bash
LANGCHAIN_TRACING_V2=true LANGSMITH_API_KEY=lsv2_... LANGSMITH_PROJECT=governor-agent
```

The same graph emits the same runs to LangSmith with zero code changes, where
you add offline evaluators (correctness, groundedness, tool-call accuracy) on
top of the harness shape in `evals/runner.py`.

## Deployment

The production path for a LangGraph agent is **LangGraph Platform** (managed
cloud or self-hosted): the graph is deployed as-is, checkpoints move to
Postgres, and the interrupt surface becomes a review queue (the "Inbox").
The Dockerfile here is the CI/worker form — it runs the eval harness so every
image is verified before use.

## Security notes

- Checkpoints contain full conversation history — treat the database as
  sensitive (encrypted volume, retention policy) when self-hosting.
- The bundled default is a scripted model; wire real provider keys only via
  environment, never in code or images.
- Caller identity is injected per-run via `config["configurable"]["caller"]`;
  in production that value comes from your auth layer, never from the client
  unchecked.
