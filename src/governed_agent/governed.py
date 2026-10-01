"""Self-contained governed query layer.

A compact port of the governed-mcp governance model (row-level security +
masking + withheld disclosure) so this repo builds and tests standalone. To
consume the real governed-mcp MCP server over the wire instead, use
``tools.load_mcp_tools()``.

Synthetic data only — generated from constants.
"""

from __future__ import annotations

from dataclasses import dataclass

REGIONS = ["north", "south", "east", "west", "central", "midwest"]
MANAGERS = ["mgr-01", "mgr-02", "mgr-03"]


@dataclass(frozen=True)
class Customer:
    id: str
    name: str
    region: str
    manager_id: str


@dataclass(frozen=True)
class Account:
    id: str
    customer_id: str
    account_type: str
    balance: float
    account_number: str


def _build() -> tuple[list[Customer], list[Account]]:
    customers: list[Customer] = []
    accounts: list[Account] = []
    ci = 0
    ai = 0
    for ri, region in enumerate(REGIONS):
        manager = MANAGERS[ri % len(MANAGERS)]
        for local in range(2):
            ci += 1
            cid = f"cust-{ci:03d}"
            customers.append(Customer(cid, f"Customer {ci:03d}", region, manager))
            for k in range(2):
                ai += 1
                accounts.append(
                    Account(
                        f"acct-{ai:06d}",
                        cid,
                        "savings" if k == 0 else "checking",
                        float(10000 + (ci * 733) % 90000),
                        f"{ai:010d}",
                    )
                )
    return customers, accounts


CUSTOMERS, ACCOUNTS = _build()
_CUSTOMER_BY_ID = {c.id: c for c in CUSTOMERS}


def resolve_scope(claim: str) -> tuple[str, str | None]:
    """Return (scope_kind, subject) for a caller claim. Fail-closed: unknown
    claims resolve to 'none', which sees zero rows."""
    if not claim or claim == "anonymous":
        return "none", None
    if claim == "admin":
        return "all", None
    if claim.startswith("manager:"):
        return "manager", claim.split(":", 1)[1]
    if claim.startswith("regional:"):
        return "region", claim.split(":", 1)[1]
    return "none", None


def query_customers(claim: str, region: str | None = None) -> dict:
    kind, subject = resolve_scope(claim)
    total = len(CUSTOMERS)
    if kind == "none":
        visible: list[Customer] = []
    elif kind == "all":
        visible = list(CUSTOMERS)
    elif kind == "manager":
        visible = [c for c in CUSTOMERS if c.manager_id == subject]
    elif kind == "region":
        visible = [c for c in CUSTOMERS if c.region == subject]
    else:  # pragma: no cover — resolve_scope is exhaustive
        visible = []
    if region:
        visible = [c for c in visible if c.region == region]
    return {
        "rows": [
            {"id": c.id, "name": c.name, "region": c.region, "manager_id": c.manager_id}
            for c in visible
        ],
        "visible_count": len(visible),
        "withheld_count": total - len(visible),
        "total_count": total,
    }


def query_accounts(claim: str, customer_id: str | None = None) -> dict:
    kind, subject = resolve_scope(claim)
    visible_customers = {
        c.id
        for c in (
            CUSTOMERS
            if kind == "all"
            else [c for c in CUSTOMERS if c.manager_id == subject]
            if kind == "manager"
            else [c for c in CUSTOMERS if c.region == subject]
            if kind == "region"
            else []
        )
    }
    total = len(ACCOUNTS)
    visible = [a for a in ACCOUNTS if a.customer_id in visible_customers]
    if customer_id:
        visible = [a for a in visible if a.customer_id == customer_id]
    return {
        "rows": [
            {
                "id": a.id,
                "customer_id": a.customer_id,
                "account_type": a.account_type,
                "balance": a.balance,
                "account_number": "****-" + a.account_number[-4:],
            }
            for a in visible
        ],
        "visible_count": len(visible),
        "withheld_count": total - len(visible),
        "total_count": total,
    }


def aggregate_balances(claim: str) -> dict:
    kind, subject = resolve_scope(claim)
    by_region: dict[str, dict] = {}
    for a in ACCOUNTS:
        c = _CUSTOMER_BY_ID[a.customer_id]
        if kind == "all" or (kind == "manager" and c.manager_id == subject) or (
            kind == "region" and c.region == subject
        ):
            if kind == "none":
                break
            bucket = by_region.setdefault(
                c.region, {"region": c.region, "total_balance": 0.0, "account_count": 0}
            )
            bucket["total_balance"] += a.balance
            bucket["account_count"] += 1
    rows = sorted(by_region.values(), key=lambda r: r["region"])
    return {
        "rows": rows,
        "visible_count": len(rows),
        "withheld_count": len(REGIONS) - len(rows),
        "total_count": len(REGIONS),
    }
