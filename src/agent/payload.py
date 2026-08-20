"""Builds the user-message payload the model reasons over.

Kept separate from the API call so the exact JSON sent to Claude can be dumped
to disk and inspected -- `run.py --dry-run` writes it without spending a token.
"""
from __future__ import annotations

import json
from datetime import date

import pandas as pd


def build_payload(valuation: dict, metrics: pd.DataFrame, corr: pd.DataFrame,
                  news: dict, cfg, annual: pd.DataFrame | None = None,
                  consistency: dict | None = None,
                  portfolio_pnl: dict | None = None) -> dict:
    """Assemble every fact the model is allowed to use."""
    # Drop all-null columns so the model is not handed a wall of `null`s.
    records = json.loads(metrics.drop(columns=["ticker"]).reset_index().to_json(
        orient="records"))

    held = {h["ticker"] for h in valuation["holdings"]}
    annual_by_ticker = {}
    if annual is not None:
        annual_by_ticker = {
            t: {int(y): (None if pd.isna(annual.loc[y, t]) else float(annual.loc[y, t]))
                for y in annual.index}
            for t in annual.columns
        }

    for rec in records:
        rec["currently_held"] = rec["ticker"] in held
        # Per-calendar-year returns let the model distinguish a steady
        # compounder from a stock whose average was made in one year.
        if rec["ticker"] in annual_by_ticker:
            rec["annual_returns_pct"] = annual_by_ticker[rec["ticker"]]
        if consistency and rec["ticker"] in consistency:
            rec["annual_consistency"] = consistency[rec["ticker"]]
        if news.get(rec["ticker"]):
            n = news[rec["ticker"]]
            rec["news"] = {
                "article_count": n["article_count"],
                "mean_sentiment": n["mean_sentiment"],
                "headlines": [
                    {"title": h["title"], "sentiment": h["sentiment_label"],
                     "published": h["published"]}
                    for h in n["headlines"]
                ],
            }

    return {
        "as_of": date.today().isoformat(),
        "constraints": {
            "max_single_position_pct": cfg.max_position_pct,
            "min_cash_pct": cfg.min_cash_pct,
            "whole_shares_only": True,
            "tradable_tickers": cfg.universe,
        },
        "current_portfolio": {
            "total_value": valuation["total_value"],
            "cash": valuation["cash"],
            "cash_pct": valuation["cash_pct"],
            "holdings": valuation["holdings"],
        },
        "watchlist_metrics": records,
        "correlation_matrix_1y": json.loads(corr.to_json(orient="index")),
        "news_available": bool(news),
        # The portfolio's own realised track record, when enough runs have been
        # persisted. Absent on a fresh portfolio -- the model is told so.
        "portfolio_performance": portfolio_pnl or {"available": False},
    }


def render_user_message(payload: dict) -> str:
    """The volatile half of the request -- everything after the cache breakpoint."""
    holdings = payload["current_portfolio"]["holdings"]
    position_line = (
        ", ".join(f"{h['ticker']} {h['weight_pct']:.1f}%" for h in holdings)
        if holdings else "none -- the portfolio is 100% cash"
    )
    news_line = ("Recent news sentiment is included per ticker."
                 if payload["news_available"]
                 else "No news data is available this run; decide on price and "
                      "technical metrics alone.")

    perf = payload.get("portfolio_performance", {})
    if perf.get("available"):
        perf_line = (f"This portfolio has been tracked since {perf['inception_date']}: "
                     f"total profit ${perf['total_profit_usd']:,.0f} "
                     f"({perf['total_profit_pct']:+.1f}%) across "
                     f"{perf['snapshots']} recorded runs.")
    else:
        perf_line = ("No realised track record yet for this portfolio -- "
                     "this is an early run.")

    return (
        f"Portfolio review for {payload['as_of']}.\n\n"
        f"Total value ${payload['current_portfolio']['total_value']:,.2f} | "
        f"cash {payload['current_portfolio']['cash_pct']:.1f}% | "
        f"positions: {position_line}\n\n"
        f"{news_line}\n"
        f"{perf_line}\n\n"
        f"Full data follows as JSON.\n\n"
        f"<portfolio_data>\n{json.dumps(payload, indent=1)}\n</portfolio_data>\n\n"
        "Review every watch-list ticker against the current positions, then call "
        "submit_rebalance_plan exactly once with your decision."
    )
