"""Model binding.

Real providers via env vars; a deterministic scripted model otherwise, so CI,
tests, and offline demos run with zero API keys. The graph code is identical
either way — only the binding changes.
"""

from __future__ import annotations

import os

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


def build_model(tools: list):
    """Return a chat model bound to the agent's tools.

    Order of preference:
      1. OPENAI_API_KEY  -> langchain-openai (default gpt-4o-mini)
      2. ANTHROPIC_API_KEY -> langchain-anthropic (default claude-3-5-haiku-latest)
      3. neither -> a deterministic scripted model (demo/tests only)
    """
    if os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini")).bind_tools(tools)
    if os.environ.get("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")
        ).bind_tools(tools)
    return scripted_model(tools)


def scripted_model(tools: list, script: list[AIMessage] | None = None):
    """Deterministic model for tests/demos: pops the next scripted message per call.

    The default script calls the aggregate tool, then answers with the region
    totals. Tests pass their own scripts (e.g. the anonymous-case refusal).

    Note: the fake model does not support bind_tools in langchain-core 1.x —
    it doesn't need it: scripted messages carry their tool_calls directly.
    """
    if script is None:
        script = DEFAULT_SCRIPT
    return GenericFakeChatModel(messages=iter(script))


DEFAULT_SCRIPT: list[AIMessage] = [
    AIMessage(
        content="",
        tool_calls=[
            {
                "name": "aggregate_balances",
                "args": {},
                "id": "call_aggregate",
                "type": "tool_call",
            }
        ],
    ),
    AIMessage(
        content="Here are the balances by region I can see: "
        "the aggregate returned the region totals."
    ),
]

FINAL_APPROVED = AIMessage(
    content="Here are the balances by region I can see: "
    "the aggregate returned the region totals."
)
FINAL_REJECTED = AIMessage(
    content="The human rejected my tool call, so I cannot provide that data."
)


def resume_model(tools: list, rejected: bool = False):
    """Model for resuming an interrupted run.

    Real providers are stateless, so a fresh model is correct. For the
    scripted demo model we start AFTER the tool call — the pending call has
    already been made and the resume only needs the final response.
    """
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"):
        return build_model(tools)
    return GenericFakeChatModel(messages=iter([FINAL_REJECTED if rejected else FINAL_APPROVED]))
