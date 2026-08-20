"""Offline end-to-end test -- exercises everything except the network.

Runs the full pipeline with a MOCKED Anthropic response, so the plan
extraction, trade simulation, constraint checker, and report renderer are all
verified without an API key or a single token spent.

    python3 -m tests.test_offline
"""
from __future__ import annotations

import json
import sys
import types
from unittest import mock

from src.config import load_config
from src.data.prices import download_prices, latest_closes
from src.features.technicals import compute_metrics, correlation_matrix
from src.features.performance import (annual_returns, current_year_is_partial,
                                      equal_weight_baseline,
                                      per_ticker_consistency,
                                      portfolio_annual_pnl,
                                      summarise_stock_performance)
from src.portfolio import (apply_plan, check_constraints, record_snapshot,
                           value_portfolio)
from src.agent.manager import _extract_plan, request_plan
from src.agent.payload import build_payload, render_user_message
from src.agent.prompts import PLAN_TOOL
from src.report import print_report, write_reports


FAKE_PLAN = {
    "market_assessment": "Breadth is narrow: CAT and GOOGL lead on 12-1 momentum "
                         "while META and HD are in drawdown. Dispersion is wide.",
    "actions": [
        {"ticker": "CAT", "action": "BUY",
         "target_weight_pct": 22.5, "conviction": "high",
         "rationale": "momentum_12_1_pct of 114.6 is the highest in the universe."},
        {"ticker": "JPM", "action": "BUY",
         "target_weight_pct": 21.2, "conviction": "medium",
         "rationale": "vol_1y_ann_pct of 22.3 is among the lowest with positive momentum."},
        {"ticker": "XOM", "action": "BUY",
         "target_weight_pct": 21.7, "conviction": "medium",
         "rationale": "beta_vs_bench of 0.22 diversifies the equity beta."},
        {"ticker": "JNJ", "action": "BUY",
         "target_weight_pct": 24.3, "conviction": "medium",
         "rationale": "corr_vs_bench of -0.02 is the lowest in the watch list."},
    ],
    "expected_cash_pct_after": 10.5,
    "risk_notes": "Four positions near the 25% cap leaves little room to add.",
    "constraint_self_check": "Largest position 24.3% < 25%; cash 10.5% > 10%.",
}


def _all_closed(node) -> bool:
    """Every object node must set additionalProperties: false for strict mode."""
    if isinstance(node, dict):
        if node.get("type") == "object" and node.get("additionalProperties") is not False:
            return False
        return all(_all_closed(v) for v in node.values())
    if isinstance(node, list):
        return all(_all_closed(v) for v in node)
    return True


def _fake_response(mode: str):
    """Minimal stand-in shaped like anthropic.types.Message."""
    usage = types.SimpleNamespace(
        input_tokens=5120, output_tokens=640,
        cache_creation_input_tokens=1180, cache_read_input_tokens=0)
    if mode == "tool":
        block = types.SimpleNamespace(type="tool_use", name=PLAN_TOOL["name"],
                                      input=FAKE_PLAN, id="toolu_fake")
        stop = "tool_use"
    else:
        block = types.SimpleNamespace(type="text", text=json.dumps(FAKE_PLAN))
        stop = "end_turn"
    return types.SimpleNamespace(content=[block], stop_reason=stop,
                                 id="msg_fake123", usage=usage)


def main() -> int:
    cfg = load_config()
    cfg.news["enabled"] = False
    cfg.anthropic_api_key = "sk-ant-fake-for-testing"

    prices = download_prices(cfg)
    closes = latest_closes(prices)
    metrics = compute_metrics(prices, cfg.universe, cfg.benchmark)
    corr = correlation_matrix(prices, cfg.universe)

    annual = annual_returns(prices, cfg.universe)
    bench_annual = annual_returns(prices, [cfg.benchmark])[cfg.benchmark]
    consistency = per_ticker_consistency(annual)
    stock_perf = summarise_stock_performance(annual, bench_annual)

    state = {"cash": 100000.0, "positions": {}, "trades": [], "history": []}
    before = value_portfolio(state, closes)
    pnl_empty = portfolio_annual_pnl(state)
    payload = build_payload(before, metrics, corr, {}, cfg, annual=annual,
                            consistency=consistency, portfolio_pnl=pnl_empty)
    user_message = render_user_message(payload)

    checks: list[tuple[str, bool]] = []
    checks.append(("payload has all 15 tickers",
                   len(payload["watchlist_metrics"]) == len(cfg.universe)))
    checks.append(("every ticker has >=10y history",
                   all(r["history_years"] >= 10 for r in payload["watchlist_metrics"])))
    checks.append(("correlation matrix is 15x15",
                   len(payload["correlation_matrix_1y"]) == len(cfg.universe)))
    checks.append(("prompt is non-trivial", len(user_message) > 5000))

    # --- calendar-year returns ------------------------------------------
    checks.append(("10 calendar years of returns", len(annual) == 10))
    checks.append(("annual returns cover the universe",
                   set(annual.columns) == set(cfg.universe)))
    checks.append(("equal-weight baseline is the cross-sectional mean",
                   abs(equal_weight_baseline(annual).iloc[-1]
                       - annual.iloc[-1].mean()) < 0.01))
    checks.append(("benchmark has its own annual series", len(bench_annual) == 10))
    checks.append(("per-year table carries best and worst",
                   all("best" in r and "worst" in r for r in stock_perf["by_year"])))
    checks.append(("consistency counts positive years",
                   all(0 <= c["positive_years"] <= c["total_years"]
                       for c in consistency.values())))
    checks.append(("annual returns reach the prompt",
                   "annual_returns_pct" in payload["watchlist_metrics"][0]))
    checks.append(("current year flagged partial", current_year_is_partial(prices)))

    # --- portfolio profit-per-year --------------------------------------
    checks.append(("empty history reports unavailable, not zero",
                   pnl_empty["available"] is False))

    # Both output modes must extract the identical plan.
    for mode in ("tool", "json_schema"):
        got = _extract_plan(_fake_response(mode), mode)
        checks.append((f"{mode} mode extracts plan", got == FAKE_PLAN))

    # Full request path, with the SDK client mocked out.
    with mock.patch("anthropic.Anthropic") as Client:
        Client.return_value.messages.create.return_value = _fake_response("tool")
        plan, meta = request_plan(user_message, cfg)
        sent = Client.return_value.messages.create.call_args.kwargs

    checks.append(("request forces the plan tool",
                   sent["tool_choice"] == {"type": "tool", "name": PLAN_TOOL["name"]}))
    checks.append(("tool is marked strict", sent["tools"][0]["strict"] is True))
    # A live 400 came from `minimum`/`maximum` on a number: strict tool use
    # rejects JSON-Schema validation keywords. Guard the whole schema tree.
    unsupported = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
                   "multipleOf", "minLength", "maxLength", "pattern", "format",
                   "minItems", "maxItems", "uniqueItems"}
    offenders: list[str] = []

    def _walk(node, path="schema"):
        if isinstance(node, dict):
            for k, v in node.items():
                if k in unsupported:
                    offenders.append(f"{path}.{k}")
                _walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                _walk(v, f"{path}[{i}]")

    _walk(sent["tools"][0]["input_schema"])
    checks.append(("schema uses no keywords strict tool use rejects", offenders == []))
    checks.append(("every object in the schema forbids extra properties",
                   _all_closed(sent["tools"][0]["input_schema"])))
    checks.append(("system prompt is cached",
                   sent["system"][0]["cache_control"] == {"type": "ephemeral"}))
    checks.append(("model is haiku 4.5", sent["model"] == "claude-haiku-4-5"))
    checks.append(("cost estimated", meta["estimated_cost_usd"] > 0))

    # Simulation + verification.
    new_state, log, sizing = apply_plan(state, plan["actions"], closes)
    after = value_portfolio(new_state, closes)
    violations = check_constraints(after, cfg)
    checks.append(("all 4 buys filled", sum(l.startswith("BUY") for l in log) == 4))
    checks.append(("plan carries no shares_delta from the model",
                   all("shares_delta" not in a for a in plan["actions"])))
    checks.append(("Python derived a share count for every action",
                   len(sizing) == len(plan["actions"])
                   and all(z["shares_delta"] > 0 for z in sizing)))
    # floor() must never overshoot the requested weight
    checks.append(("derived weights never exceed their target",
                   all(z["implied_weight_pct"] <= z["target_weight_pct"] + 1e-9
                       for z in sizing)))
    checks.append(("derived weights are within one share of target",
                   all(z["target_weight_pct"] - z["implied_weight_pct"]
                       <= z["price"] / before["total_value"] * 100 + 1e-9
                       for z in sizing)))
    checks.append(("cash was spent", after["cash"] < before["cash"]))
    checks.append(("compliant plan passes the checker", violations == []))

    # The checker must actually catch violations, not just always pass.
    # Sized to breach BOTH rules at once: a single name far over 25%, which
    # necessarily drags cash under the 10% floor.
    bad_state, _, _ = apply_plan(state, [
        {"ticker": "AAPL", "action": "BUY", "target_weight_pct": 94.0}], closes)
    bad_valuation = value_portfolio(bad_state, closes)
    bad_violations = check_constraints(bad_valuation, cfg)
    checks.append(("checker catches an over-weight position",
                   any("exceeds" in v for v in bad_violations)))
    checks.append(("checker catches a cash breach",
                   any("below" in v for v in bad_violations)))

    result = {
        "as_of": payload["as_of"], "plan": plan, "call_meta": meta,
        "valuation_before": before, "valuation_after": after,
        "execution_log": log, "violations": violations,
        "max_weight_after": max(h["weight_pct"] for h in after["holdings"]),
        "limits": {"max_position_pct": cfg.max_position_pct,
                   "min_cash_pct": cfg.min_cash_pct},
    }
    # Simulate three applied runs a year apart so profit-per-year has real
    # history to fold. Values are hand-set; this tests the ACCOUNTING, not the
    # market.
    hist_state = {"cash": 100000.0, "positions": {}, "trades": [],
                  "history": [
                      {"date": "2024-01-05", "total_value": 100000.0, "cash": 100000.0,
                       "positions_value": 0.0, "realised_pnl_in_run": 0.0},
                      {"date": "2024-12-20", "total_value": 112000.0, "cash": 11000.0,
                       "positions_value": 101000.0, "realised_pnl_in_run": 1500.0},
                      {"date": "2025-12-19", "total_value": 105000.0, "cash": 12000.0,
                       "positions_value": 93000.0, "realised_pnl_in_run": -800.0},
                  ]}
    hist_pnl = portfolio_annual_pnl(hist_state)
    y2024 = next(r for r in hist_pnl["by_year"] if r["year"] == 2024)
    y2025 = next(r for r in hist_pnl["by_year"] if r["year"] == 2025)
    checks.append(("profit-per-year available with history", hist_pnl["available"]))
    checks.append(("2024 profit = 12,000 (+12%)",
                   y2024["profit_usd"] == 12000.0 and y2024["profit_pct"] == 12.0))
    checks.append(("2025 loss carries from 2024 close, not inception",
                   y2025["start_value"] == 112000.0 and y2025["profit_usd"] == -7000.0))
    checks.append(("total profit is inception-to-date",
                   hist_pnl["total_profit_usd"] == 5000.0))
    checks.append(("realised P&L aggregates across years",
                   hist_pnl["total_realised_pnl_usd"] == 700.0))

    # A sell must book realised P&L against average cost.
    seeded = {"cash": 0.0, "positions": {"AAPL": {"shares": 100, "avg_cost": 100.0}},
              "trades": [], "history": []}
    sold, _, _ = apply_plan(seeded, [{"ticker": "AAPL", "action": "SELL",
                                      "target_weight_pct": 0.0}], {"AAPL": 150.0})
    checks.append(("sell books realised P&L against avg cost",
                   sold["last_run_realised_pnl"] == 5000.0))
    checks.append(("trade ledger records the sell",
                   len(sold["trades"]) == 1 and sold["trades"][0]["side"] == "SELL"))
    record_snapshot(sold, value_portfolio(sold, {"AAPL": 150.0}), note="test")
    checks.append(("snapshot captures realised P&L",
                   sold["history"][-1]["realised_pnl_in_run"] == 5000.0))

    # --- regression: replay the two real runs' STATED target weights -----
    # Run 2 (2026-08-20_161017) stated these weights, but its own share counts
    # produced 65.1% invested / 34.9% cash. Deriving shares must land on the
    # weights it actually asked for.
    run2 = [("AAPL", 6.2), ("GOOGL", 6.0), ("CAT", 6.1), ("XOM", 6.0),
            ("JNJ", 6.0), ("KO", 6.0), ("UNH", 6.0)]
    r2_state, _, r2_sizing = apply_plan(
        {"cash": 100000.0, "positions": {}, "trades": [], "history": []},
        [{"ticker": t, "action": "BUY", "target_weight_pct": w} for t, w in run2],
        closes)
    r2_val = value_portfolio(r2_state, closes)
    r2_invested = 100 - r2_val["cash_pct"]
    # Rounding down costs ~1.4pp here, driven by high-priced names (CAT at
    # $802 has coarse share granularity). The model's own share counts were off
    # by +22.8pp in the other direction.
    checks.append(("run 2 replay lands near its stated 42.3% invested (was 65.1%)",
                   abs(r2_invested - 42.3) < 2.0))
    checks.append(("run 2 replay errs toward cash, never toward over-investment",
                   r2_invested <= 42.3))
    checks.append(("run 2 replay now passes constraints",
                   check_constraints(r2_val, cfg) == []))
    r2_aapl = next(z for z in r2_sizing if z["ticker"] == "AAPL")
    checks.append(("run 2 AAPL sized to 6.2%, not 19.9%",
                   abs(r2_aapl["implied_weight_pct"] - 6.2) < 0.4))

    # Run 1 stated weights summing to 90.9% invested -> 9.1% cash, i.e. an
    # allocation 0.9pp too aggressive. Rounding down frees ~1.8pp, which lifts
    # cash to 10.86% and rescues a plan that really did breach at 9.34%. This
    # is the systematic cash bias working as intended, not a fluke.
    run1 = [("CAT", 24.0), ("JNJ", 10.0), ("XOM", 9.9), ("AAPL", 7.9),
            ("GOOGL", 7.8), ("KO", 7.9), ("UNH", 7.7), ("AMZN", 7.9), ("HD", 7.8)]
    r1_state, _, _ = apply_plan(
        {"cash": 100000.0, "positions": {}, "trades": [], "history": []},
        [{"ticker": t, "action": "BUY", "target_weight_pct": w} for t, w in run1],
        closes)
    r1_val = value_portfolio(r1_state, closes)
    r1_violations = check_constraints(r1_val, cfg)
    checks.append(("run 1 replay now passes — rounding down rescued a 9.34% breach",
                   r1_violations == [] and r1_val["cash_pct"] > 10.0))
    checks.append(("run 1 replay still respects the position cap",
                   max(h["weight_pct"] for h in r1_val["holdings"]) <= cfg.max_position_pct))

    # A SELL must exit fully regardless of the weight the model wrote.
    held = {"cash": 0.0, "positions": {"KO": {"shares": 50, "avg_cost": 80.0}},
            "trades": [], "history": []}
    exited, _, _ = apply_plan(held, [{"ticker": "KO", "action": "SELL",
                                      "target_weight_pct": 7.0}], closes)
    checks.append(("SELL exits fully even with a non-zero target weight",
                   "KO" not in exited["positions"]))

    # Over-cap targets must be sized as asked and flagged, never silently
    # clamped -- the constraint checker is what reports them.
    over, over_log, _ = apply_plan(
        {"cash": 100000.0, "positions": {}, "trades": [], "history": []},
        [{"ticker": "KO", "action": "BUY", "target_weight_pct": 40.0}],
        closes, cfg.max_position_pct)
    over_val = value_portfolio(over, closes)
    checks.append(("over-cap target is flagged, not clamped",
                   any("exceeds the 25% cap" in l for l in over_log)
                   and max(h["weight_pct"] for h in over_val["holdings"]) > 25))
    checks.append(("over-cap target is caught by the constraint checker",
                   any("exceeds" in v for v in check_constraints(over_val, cfg))))

    result["sizing"] = sizing
    result["portfolio_pnl"] = hist_pnl
    result["stock_performance"] = stock_perf
    result["current_year_partial"] = current_year_is_partial(prices)
    print_report(result)
    json_path, md_path = write_reports(result)
    md = md_path.read_text()
    checks.append(("markdown report renders the action table", "| **BUY** | CAT |" in md))
    checks.append(("markdown report records the constraint verdict", "**PASSED**" in md))
    checks.append(("json report round-trips", json.loads(json_path.read_text())["plan"] == plan))
    checks.append(("markdown reports profit per year", "## Profit per year" in md))
    checks.append(("markdown reports calendar-year returns",
                   "## Watch-list returns by calendar year" in md))
    checks.append(("markdown disclaims the backtest gap", "not a backtest" in md))

    print("\n--- assertions ---")
    failed = 0
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        failed += not ok
    print(f"\n{len(checks) - failed}/{len(checks)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
