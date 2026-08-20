#!/usr/bin/env python3
"""LLM Portfolio Manager -- notification-only rebalancing agent.

Pipeline:
    yfinance -> pandas technicals -> (Alpha Vantage news) -> Claude -> report

No broker is connected. Nothing is ever traded.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date

from src.config import PORTFOLIO_PATH, REPORT_DIR, load_config
from src.data.news import download_news
from src.data.prices import download_prices, latest_closes
from src.features.technicals import compute_metrics, correlation_matrix
from src.features.performance import (annual_returns, current_year_is_partial,
                                      per_ticker_consistency,
                                      portfolio_annual_pnl,
                                      summarise_stock_performance)
from src.portfolio import (apply_plan, check_constraints, load_portfolio,
                           record_snapshot, save_portfolio, seed_portfolio,
                           value_portfolio)
from src.report import console, print_report, write_reports
from src.agent.manager import ManagerError, request_plan
from src.agent.payload import build_payload, render_user_message


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true",
                   help="build the full prompt and dump it to disk WITHOUT calling the API")
    p.add_argument("--refresh", action="store_true",
                   help="ignore caches and re-download prices and news")
    p.add_argument("--no-news", action="store_true",
                   help="skip Alpha Vantage entirely this run")
    p.add_argument("--apply", action="store_true",
                   help="persist the simulated post-trade portfolio to portfolio.json")
    p.add_argument("--reset", action="store_true",
                   help="reset portfolio.json to all cash, then exit")
    p.add_argument("--model", help="override llm.model from config.yaml")
    p.add_argument("--output-mode", choices=["tool", "json_schema"],
                   help="override llm.output_mode from config.yaml")
    p.add_argument("--thinking", action="store_true",
                   help="enable extended thinking for this run")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config()

    if args.reset:
        seed_portfolio(cfg)
        return 0

    # CLI overrides win over config.yaml.
    if args.model:
        cfg.llm["model"] = args.model
    if args.output_mode:
        cfg.llm["output_mode"] = args.output_mode
    if args.thinking:
        cfg.llm["thinking_enabled"] = True
    if args.no_news:
        cfg.news["enabled"] = False

    # ---- 1. data ---------------------------------------------------------
    prices = download_prices(cfg, force=args.refresh)
    closes = latest_closes(prices)
    news = download_news(cfg, force=args.refresh)

    # ---- 2. features -----------------------------------------------------
    metrics = compute_metrics(prices, cfg.universe, cfg.benchmark)
    corr = correlation_matrix(prices, cfg.universe)
    print(f"[technicals] {metrics.shape[0]} tickers x {metrics.shape[1] - 1} metrics computed")

    # Calendar-year returns: per-stock history (always available) and the
    # equal-weight / benchmark comparison rows.
    annual = annual_returns(prices, cfg.universe)
    bench_annual = annual_returns(prices, [cfg.benchmark])[cfg.benchmark]
    consistency = per_ticker_consistency(annual)
    stock_performance = summarise_stock_performance(annual, bench_annual)
    partial_year = current_year_is_partial(prices)
    print(f"[performance] {len(annual)} calendar years of returns computed "
          f"({'current year partial' if partial_year else 'all years complete'})")

    # ---- 3. current portfolio -------------------------------------------
    state = load_portfolio(cfg)
    # Profit-per-year of THIS portfolio, from snapshots recorded on --apply runs.
    portfolio_pnl = portfolio_annual_pnl(state)
    if portfolio_pnl["available"]:
        print(f"[performance] portfolio tracked since "
              f"{portfolio_pnl['inception_date']}: "
              f"{portfolio_pnl['total_profit_pct']:+.2f}% total")
    else:
        print(f"[performance] portfolio profit-per-year unavailable "
              f"({portfolio_pnl['snapshots']} snapshot(s) recorded)")
    valuation_before = value_portfolio(state, closes)
    pre_violations = check_constraints(valuation_before, cfg)
    if pre_violations:
        console.print("[yellow]Note: the CURRENT portfolio already violates "
                      "constraints:[/yellow]")
        for v in pre_violations:
            console.print(f"  [yellow]• {v}[/yellow]")

    # ---- 4. prompt -------------------------------------------------------
    payload = build_payload(valuation_before, metrics, corr, news, cfg,
                            annual=annual, consistency=consistency,
                            portfolio_pnl=portfolio_pnl)
    user_message = render_user_message(payload)

    if args.dry_run:
        out = REPORT_DIR / f"dry-run_{date.today().isoformat()}.txt"
        out.write_text(user_message)
        console.print(f"[cyan]Dry run.[/cyan] No API call made.")
        console.print(f"  prompt chars : {len(user_message):,} "
                      f"(~{len(user_message) // 4:,} tokens)")
        console.print(f"  written to   : {out}")
        return 0

    # ---- 5. the LLM call -------------------------------------------------
    console.print(f"[cyan]Calling {cfg.llm['model']} "
                  f"(mode={cfg.llm['output_mode']}) ...[/cyan]")
    try:
        plan, call_meta = request_plan(user_message, cfg)
    except ManagerError as exc:
        console.print(f"[bold red]LLM call failed:[/bold red] {exc}")
        return 1

    # ---- 6. simulate + VERIFY -------------------------------------------
    # The model asserts its plan is compliant. We do not take its word for it:
    # we apply the plan locally and re-derive the weights ourselves.
    new_state, execution_log, sizing = apply_plan(
        state, plan.get("actions", []), closes, cfg.max_position_pct)
    valuation_after = value_portfolio(new_state, closes)
    violations = check_constraints(valuation_after, cfg)
    max_weight = max((h["weight_pct"] for h in valuation_after["holdings"]), default=0.0)

    result = {
        "as_of": payload["as_of"],
        "plan": plan,
        "call_meta": call_meta,
        "valuation_before": valuation_before,
        "valuation_after": valuation_after,
        "execution_log": execution_log,
        # Python's weight -> share derivation, kept separate from the model's
        # plan so the provenance of every share count stays unambiguous.
        "sizing": sizing,
        "violations": violations,
        "max_weight_after": max_weight,
        "limits": {"max_position_pct": cfg.max_position_pct,
                   "min_cash_pct": cfg.min_cash_pct},
        "news_used": sorted(news.keys()),
        "universe": cfg.universe,
        "portfolio_pnl": portfolio_pnl,
        "stock_performance": stock_performance,
        "current_year_partial": partial_year,
    }

    # ---- 7. notify -------------------------------------------------------
    print_report(result)
    write_reports(result)

    if args.apply:
        if violations:
            console.print("[red]--apply refused: the plan violates hard constraints. "
                          "portfolio.json is unchanged.[/red]")
        else:
            # Record the post-trade valuation so profit-per-year has a data
            # point for this run. Only applied runs are recorded -- a
            # notification-only run must not mutate the tracked history.
            record_snapshot(new_state, valuation_after,
                            note=f"{len(plan.get('actions', []))} action(s)")
            save_portfolio(new_state)
            console.print(f"[green]--apply: portfolio.json updated "
                          f"({PORTFOLIO_PATH.name}), snapshot recorded "
                          f"({len(new_state['history'])} total).[/green]")

    return 2 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
