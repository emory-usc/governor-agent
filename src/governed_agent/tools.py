"""LangChain tools over the governed query layer.

Every tool enforces the caller's scope (fail-closed row-level security) and
returns a withheld_count disclosure, so the agent can distinguish "no data"
from "data I am not allowed to see".

Caller identity flows in through the RunnableConfig: ``config["configurable"]["caller"]``.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from governed_agent import governed


def _caller(config: RunnableConfig | None) -> str:
    if config is None:
        return "anonymous"
    return config.get("configurable", {}).get("caller", "anonymous")


@tool
def query_customers(config: RunnableConfig, region: str | None = None) -> dict:
    """List customers visible to the current caller, optionally filtered by region."""
    return governed.query_customers(_caller(config), region=region)


@tool
def query_accounts(config: RunnableConfig, customer_id: str | None = None) -> dict:
    """List accounts visible to the current caller. Account numbers are masked."""
    return governed.query_accounts(_caller(config), customer_id=customer_id)


@tool
def aggregate_balances(config: RunnableConfig) -> dict:
    """Aggregate total balance and account count by region, over rows the caller can see."""
    return governed.aggregate_balances(_caller(config))


TOOLS = [query_customers, query_accounts, aggregate_balances]


def load_mcp_tools(server_url: str = "http://localhost:8000/mcp") -> list:
    """Consume the real governed-mcp MCP server instead of the in-process layer.

    Requires the ``mcp`` extra (langchain-mcp-adapters). The returned tools
    carry the same governance, enforced server-side.
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {"governed-mcp": {"url": server_url, "transport": "streamable_http"}}
    )

    class _LazyTools:
        """Loads tools once (async) and caches them."""

        def __init__(self):
            self._tools = None

        async def get(self):
            if self._tools is None:
                async with client:
                    self._tools = await client.get_tools()
            return self._tools

    return _LazyTools()
