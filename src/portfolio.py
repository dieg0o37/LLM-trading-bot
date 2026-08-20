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


def apply_plan(state: dict, actions: list[dict], closes: dict[str, float],
               max_position_pct: float | None = None
               ) -> tuple[dict, list[str], list[dict]]:
    """Convert target weights into whole-share trades and simulate them.

    The model supplies `target_weight_pct` only -- never a share count. Live
    runs showed Claude Haiku 4.5's share arithmetic drifting badly (one run
    asked for 63 AAPL shares while calling it a 6.2% position; it was 19.9%),
    so the conversion happens here instead:

        target_shares = floor(target_weight_pct / 100 * total_value / price)

    `total_value` is taken from the PRE-trade portfolio and held fixed. That is
    correct, not an approximation: buying converts cash into stock of equal
    market value, so total value is invariant under trades at the same prices.

    Flooring means a position always lands at or slightly BELOW its target, so
    rounding can only free cash, never consume more than intended. A set of
    weights that respects the cash floor therefore cannot breach it once
    executed -- rounding error can no longer turn a compliant plan into a
    violation.

    Returns (new_state, log, sizing) where `sizing` records the derivation for
    each action so the report can show what Python computed and why.
    """
    new_state = {
        "cash": float(state["cash"]),
        "positions": {k: dict(v) for k, v in state["positions"].items()},
        "inception_date": state.get("inception_date"),
        "trades": list(state.get("trades", [])),
        "history": list(state.get("history", [])),
    }
    log: list[str] = []
    sizing: list[dict] = []
    today = date.today().isoformat()
    realised_this_run = 0.0

    total_value = value_portfolio(state, closes)["total_value"]

    # ---- 1. derive a share delta for every action ------------------------
    planned: list[dict] = []
    for action in actions:
        ticker = action.get("ticker")
        if not ticker:
            continue
        price = closes.get(ticker)
        if price is None or price <= 0:
            log.append(f"SKIP {ticker}: no price available")
            continue

        held = float(new_state["positions"].get(ticker, {}).get("shares", 0.0))
        verb = str(action.get("action", "")).upper()
        # A full exit is an exit regardless of the weight the model wrote.
        target_pct = 0.0 if verb == "SELL" else float(action.get("target_weight_pct", 0) or 0)
        target_pct = max(0.0, target_pct)
        # The 0-25 bound cannot live in the schema -- strict tool use rejects
        # `minimum`/`maximum` on a number. Flag an over-cap request here, but do
        # NOT clamp it: silently correcting the model would hide the error that
        # check_constraints() exists to surface.
        if max_position_pct is not None and target_pct > max_position_pct:
            log.append(f"NOTE {ticker}: target {target_pct:.1f}% exceeds the "
                       f"{max_position_pct:.0f}% cap — sized as asked so the "
                       f"constraint check reports it")

        target_shares = int(total_value * target_pct / 100.0 // price)
        delta = target_shares - held

        entry = {
            "ticker": ticker,
            "action": verb,
            "target_weight_pct": round(target_pct, 2),
            "price": round(price, 2),
            "shares_before": held,
            "target_shares": target_shares,
            "shares_delta": delta,
            "implied_value": round(target_shares * price, 2),
            "implied_weight_pct": round(target_shares * price / total_value * 100, 2)
            if total_value else 0.0,
        }
        sizing.append(entry)

        # Flag a verb that contradicts its own arithmetic rather than silently
        # obeying the number -- it usually means the model misread the position.
        if verb in ("BUY", "ADD") and delta <= 0:
            log.append(f"NOTE {ticker}: {verb} but target {target_pct:.1f}% is at or "
                       f"below the {held:,.0f} shares already held — no trade")
        elif verb == "TRIM" and delta >= 0:
            log.append(f"NOTE {ticker}: TRIM but target {target_pct:.1f}% is at or "
                       f"above the current holding — no trade")
        if delta:
            planned.append(entry)

    # ---- 2. execute, sells first so proceeds fund the buys ---------------
    for entry in sorted(planned, key=lambda e: 0 if e["shares_delta"] < 0 else 1):
        ticker, delta, price = entry["ticker"], entry["shares_delta"], closes[entry["ticker"]]
        pos = new_state["positions"].get(ticker, {"shares": 0.0, "avg_cost": 0.0})
        shares, avg_cost = float(pos["shares"]), float(pos["avg_cost"])

        if delta > 0:
            cost = delta * price
            if cost > new_state["cash"] + 1e-6:
                # Re-size to whatever cash actually remains rather than dropping
                # the trade entirely.
                affordable = int(new_state["cash"] // price)
                if affordable <= 0:
                    log.append(f"SKIP BUY {ticker}: needs ${cost:,.0f}, "
                               f"only ${new_state['cash']:,.0f} cash")
                    entry["shares_delta"] = 0
                    continue
                log.append(f"NOTE {ticker}: buy cut from {delta:,.0f} to "
                           f"{affordable:,.0f} shares — insufficient cash")
                delta = affordable
                entry["shares_delta"] = delta
                cost = delta * price

            new_avg = (shares * avg_cost + cost) / (shares + delta)
            new_state["cash"] -= cost
            new_state["positions"][ticker] = {"shares": shares + delta,
                                              "avg_cost": round(new_avg, 4)}
            new_state["trades"].append({
                "date": today, "ticker": ticker, "side": "BUY",
                "shares": delta, "price": round(price, 4),
                "value": round(cost, 2), "realised_pnl": 0.0,
            })
            log.append(f"BUY  {ticker} {delta:+,.0f} @ ${price:,.2f} = ${cost:,.0f} "
                       f"-> {entry['implied_weight_pct']:.1f}%")
        else:
            sell = min(-delta, shares)
            if sell <= 0:
                log.append(f"SKIP SELL {ticker}: no shares held")
                entry["shares_delta"] = 0
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
                       f"(realised {realised:+,.0f}) -> {entry['implied_weight_pct']:.1f}%")

    new_state["cash"] = round(new_state["cash"], 2)
    new_state["last_run_realised_pnl"] = round(realised_this_run, 2)
    return new_state, log, sizing


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
