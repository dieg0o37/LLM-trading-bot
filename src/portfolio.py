"""Simulated portfolio state, mark-to-market, and constraint checking.

NOTHING here talks to a broker. The project is notification-only: we compute
what the portfolio is worth, hand that to the LLM, and record what the LLM
proposes. Orders are never placed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path

from .config import PORTFOLIO_PATH, Config


@dataclass
class Position:
    shares: float
    avg_cost: float


@dataclass
class Holding:
    """A position marked to the latest close."""
    ticker: str
    shares: float
    avg_cost: float
    price: float
    market_value: float
    weight_pct: float
    unrealised_pnl: float
    unrealised_pnl_pct: float


def seed_portfolio(cfg: Config, path: Path = PORTFOLIO_PATH) -> dict:
    """Create an all-cash starting portfolio if none exists.

    The inception snapshot is written immediately so that profit-per-year has a
    baseline to measure the first real run against.
    """
    cash = float(cfg.portfolio["starting_cash"])
    state = {
        "cash": cash,
        "positions": {},
        "inception_date": date.today().isoformat(),
        "trades": [],
        "history": [{
            "date": date.today().isoformat(),
            "total_value": cash,
            "cash": cash,
            "positions_value": 0.0,
            "realised_pnl_in_run": 0.0,
            "note": "inception",
        }],
    }
    path.write_text(json.dumps(state, indent=2))
    print(f"[portfolio] seeded new all-cash portfolio "
          f"(${state['cash']:,.0f}) -> {path.name}")
    return state


def load_portfolio(cfg: Config, path: Path = PORTFOLIO_PATH) -> dict:
    if not path.exists():
        return seed_portfolio(cfg, path)
    state = json.loads(path.read_text())
    state.setdefault("cash", 0.0)
    state.setdefault("positions", {})
    state.setdefault("trades", [])
    state.setdefault("history", [])
    state.setdefault("last_run_realised_pnl", 0.0)
    return state


def save_portfolio(state: dict, path: Path = PORTFOLIO_PATH) -> None:
    path.write_text(json.dumps(state, indent=2))


def value_portfolio(state: dict, closes: dict[str, float]) -> dict:
    """Mark the portfolio to market and compute weights.

    Weights are a share of TOTAL portfolio value (positions + cash), which is
    the denominator the 25%/10% constraints are written against.
    """
    holdings: list[Holding] = []
    positions_value = 0.0

    for ticker, pos in state.get("positions", {}).items():
        shares = float(pos["shares"])
        avg_cost = float(pos.get("avg_cost", 0.0))
        price = float(closes.get(ticker, avg_cost))
        mv = shares * price
        positions_value += mv
        cost_basis = shares * avg_cost
        holdings.append(Holding(
            ticker=ticker, shares=shares, avg_cost=round(avg_cost, 2),
            price=round(price, 2), market_value=round(mv, 2), weight_pct=0.0,
            unrealised_pnl=round(mv - cost_basis, 2),
            unrealised_pnl_pct=round((mv / cost_basis - 1) * 100, 2) if cost_basis else 0.0,
        ))

    cash = float(state.get("cash", 0.0))
    total = positions_value + cash
    for h in holdings:
        h.weight_pct = round(h.market_value / total * 100, 2) if total else 0.0

    holdings.sort(key=lambda h: h.market_value, reverse=True)
    return {
        "total_value": round(total, 2),
        "cash": round(cash, 2),
        "cash_pct": round(cash / total * 100, 2) if total else 100.0,
        "positions_value": round(positions_value, 2),
        "holdings": [asdict(h) for h in holdings],
    }


# --------------------------------------------------------------------------
# constraint checking -- run AFTER the model responds
# --------------------------------------------------------------------------
def check_constraints(valuation: dict, cfg: Config) -> list[str]:
    """Deterministic re-check of the two hard rules. Returns violation strings.

    The system prompt states these rules, but an LLM stating "constraints
    satisfied" is not evidence. This function is the actual verifier, and its
    output is what the report trusts.
    """
    violations = []
    for h in valuation["holdings"]:
        if h["weight_pct"] > cfg.max_position_pct + 1e-9:
            violations.append(
                f"{h['ticker']} at {h['weight_pct']:.2f}% exceeds the "
                f"{cfg.max_position_pct:.0f}% single-position limit")
    if valuation["cash_pct"] < cfg.min_cash_pct - 1e-9:
        violations.append(
            f"cash at {valuation['cash_pct']:.2f}% is below the "
            f"{cfg.min_cash_pct:.0f}% minimum")
    return violations


def apply_plan(state: dict, actions: list[dict], closes: dict[str, float]) -> tuple[dict, list[str]]:
    """Simulate the model's plan against the local state.

    Used for two things: the `--apply` flag (persist the new portfolio) and,
    always, to compute the POST-TRADE valuation that the constraint checker
    runs on. Sells are executed before buys so proceeds are available to fund
    purchases -- the same ordering a real rebalance would use.
    """
    new_state = {
        "cash": float(state["cash"]),
        "positions": {k: dict(v) for k, v in state["positions"].items()},
        "inception_date": state.get("inception_date"),
        "trades": list(state.get("trades", [])),
        "history": list(state.get("history", [])),
    }
    log: list[str] = []
    today = date.today().isoformat()
    realised_this_run = 0.0

    def sort_key(a):
        return 0 if float(a.get("shares_delta", 0)) < 0 else 1

    for action in sorted(actions, key=sort_key):
        ticker = action.get("ticker")
        delta = float(action.get("shares_delta", 0) or 0)
        if not ticker or delta == 0:
            continue
        price = closes.get(ticker)
        if price is None:
            log.append(f"SKIP {ticker}: no price available")
            continue

        pos = new_state["positions"].get(ticker, {"shares": 0.0, "avg_cost": 0.0})
        shares, avg_cost = float(pos["shares"]), float(pos["avg_cost"])

        if delta > 0:
            cost = delta * price
            if cost > new_state["cash"] + 1e-6:
                log.append(f"SKIP BUY {ticker}: needs ${cost:,.0f}, "
                           f"only ${new_state['cash']:,.0f} cash")
                continue
            new_avg = (shares * avg_cost + cost) / (shares + delta)
            new_state["cash"] -= cost
            new_state["positions"][ticker] = {"shares": shares + delta,
                                              "avg_cost": round(new_avg, 4)}
            new_state["trades"].append({
                "date": today, "ticker": ticker, "side": "BUY",
                "shares": delta, "price": round(price, 4),
                "value": round(cost, 2), "realised_pnl": 0.0,
            })
            log.append(f"BUY  {ticker} {delta:+,.0f} @ ${price:,.2f} = ${cost:,.0f}")
        else:
            sell = min(-delta, shares)
            if sell <= 0:
                log.append(f"SKIP SELL {ticker}: no shares held")
                continue
            proceeds = sell * price
            # Realised P&L on the shares actually sold, against average cost.
            realised = sell * (price - avg_cost)
            realised_this_run += realised
            new_state["cash"] += proceeds
            remaining = shares - sell
            if remaining <= 1e-9:
                new_state["positions"].pop(ticker, None)
            else:
                new_state["positions"][ticker] = {"shares": remaining, "avg_cost": avg_cost}
            new_state["trades"].append({
                "date": today, "ticker": ticker, "side": "SELL",
                "shares": -sell, "price": round(price, 4),
                "value": round(proceeds, 2), "realised_pnl": round(realised, 2),
            })
            log.append(f"SELL {ticker} {-sell:+,.0f} @ ${price:,.2f} = ${proceeds:,.0f} "
                       f"(realised {realised:+,.0f})")

    new_state["cash"] = round(new_state["cash"], 2)
    new_state["last_run_realised_pnl"] = round(realised_this_run, 2)
    return new_state, log


def record_snapshot(state: dict, valuation: dict, note: str = "") -> dict:
    """Append a valuation snapshot -- the raw material for profit-per-year.

    Called only on `--apply`, because a notification-only run must not mutate
    the tracked history. One snapshot per applied run; `portfolio_annual_pnl()`
    takes the last snapshot of each calendar year as that year's close.
    """
    state.setdefault("history", []).append({
        "date": date.today().isoformat(),
        "total_value": valuation["total_value"],
        "cash": valuation["cash"],
        "positions_value": valuation["positions_value"],
        "realised_pnl_in_run": state.get("last_run_realised_pnl", 0.0),
        "note": note,
    })
    return state
