"""Tests for the governed query layer and tools."""

from governed_agent import governed, tools
from governed_agent.tools import TOOLS


def test_admin_sees_everything():
    res = governed.query_customers("admin")
    assert res["visible_count"] == res["total_count"] == 12
    assert res["withheld_count"] == 0


def test_anonymous_fails_closed():
    res = governed.query_customers("anonymous")
    assert res["visible_count"] == 0
    assert res["withheld_count"] == 12


def test_manager_scope_is_subset():
    res = governed.query_customers("manager:mgr-01")
    assert 0 < res["visible_count"] < 12


def test_region_scope():
    res = governed.query_customers("regional:north")
    assert all(r["region"] == "north" for r in res["rows"])
    assert res["visible_count"] == 2


def test_unknown_claim_fails_closed():
    res = governed.query_customers("superuser")
    assert res["visible_count"] == 0


def test_account_numbers_masked():
    res = governed.query_accounts("admin")
    for row in res["rows"]:
        assert row["account_number"].startswith("****-")
        assert len(row["account_number"]) == 9  # ****- + 4 digits


def test_aggregate_scope_counts():
    res = governed.aggregate_balances("admin")
    assert res["visible_count"] == 6
    assert res["withheld_count"] == 0
    anon = governed.aggregate_balances("anonymous")
    assert anon["visible_count"] == 0
    assert anon["withheld_count"] == 6


def test_tools_honor_configurable_caller():
    res = tools.aggregate_balances.invoke({}, config={"configurable": {"caller": "admin"}})
    assert res["visible_count"] == 6
    res = tools.aggregate_balances.invoke({}, config={"configurable": {"caller": "anonymous"}})
    assert res["visible_count"] == 0


def test_tools_do_not_expose_config_in_schema():
    for t in TOOLS:
        args = t.args
        assert "config" not in args, f"{t.name} leaked config into its schema"
