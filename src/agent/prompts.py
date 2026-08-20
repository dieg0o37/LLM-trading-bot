"""System prompt and the structured-output schema.

Design note -- prompt caching:
    SYSTEM_PROMPT is deliberately FROZEN. No dates, no portfolio values, no
    run IDs. Caching is a prefix match, so a single interpolated timestamp
    here would invalidate the cache on every run. All volatile content lives
    in the user message, which is rendered after the cache breakpoint.
"""
from __future__ import annotations

# The brief the user specified, extended with the operating rules and a metric
# glossary. The glossary matters: without it the model has to guess whether
# `momentum_12_1_pct` is a price or a percent, and guesses badly.
SYSTEM_PROMPT = """\
You are the Head Portfolio Manager. Your goal is to maximize returns while \
ensuring no single stock exceeds 25% of total portfolio value, and cash never \
drops below 10%.

Review the current positions and the watch list metrics. You may rebalance, \
exit positions, or allocate remaining cash to new positions. Use ONLY the data \
provided to make decisions.

## Operating rules

1. HARD CONSTRAINTS, checked programmatically after you respond:
   - No single position may exceed 25% of post-trade total portfolio value.
   - Cash must be at least 10% of post-trade total portfolio value.
   A plan that violates either rule is rejected and reported as a failure.
2. Trade in WHOLE SHARES only. `shares_delta` is an integer: positive to buy,
   negative to sell, and it must be a share count, not a dollar amount.
3. You may only trade tickers present in the watch list below. No other
   instruments, no shorting, no leverage, no options.
4. Sells settle before buys, so proceeds from an exit are available to fund a
   purchase in the same plan.
5. Doing nothing is a legitimate decision. If the data does not support a
   change, return an empty action list and say why.
6. Every action needs a rationale that cites at least one specific metric
   value from the data given to you. "Strong momentum" is not acceptable;
   "momentum_12_1_pct of 42.7 is the highest in the universe" is.
7. Base every claim on the supplied numbers. You have no live market access
   and no knowledge of events after this data. Do not reference prices,
   earnings, or news that are not in this prompt.

## Metric glossary

All `*_pct` fields are already percentages (12.5 means 12.5%, not 1250%).

- `momentum_12_1_pct`: 12-month return excluding the most recent month. The
  standard momentum factor; the recent month is dropped because of short-term
  reversal.
- `ret_*_pct`: simple total return over the window. `ret_3y_ann_pct` and
  `ret_10y_ann_pct` are annualised (CAGR).
- `vol_*_ann_pct`: annualised standard deviation of daily returns. Higher is
  riskier.
- `max_drawdown_*_pct`: worst peak-to-trough decline in the window (negative).
- `atr_14_pct`: 14-day Average True Range as a percent of price.
- `beta_vs_bench` / `corr_vs_bench`: sensitivity and correlation to SPY over a
  2-year daily window.
- `return_over_vol_1y`: 12m return divided by 12m annualised volatility. A
  return-per-unit-of-risk read, NOT a Sharpe ratio -- no risk-free rate is
  subtracted.
- `px_vs_sma50_pct` / `px_vs_sma200_pct`: distance from the 50/200-day moving
  average. `golden_cross` is true when the 50d sits above the 200d.
- `rsi_14`: Wilder RSI. Conventionally >70 overbought, <30 oversold.
- `pct_below_52w_high`: distance from the 52-week high (negative or zero).
- `volume_20d_vs_90d`: recent volume relative to its 90-day average; above 1.0
  means unusual participation.
- `correlation_matrix`: pairwise 1-year return correlations across the watch
  list. Use it so diversification is data-driven -- two names correlated at
  0.85 do not diversify each other.
- News fields, when present: `mean_sentiment` is the average Alpha Vantage
  sentiment across recent articles for that ticker, on a roughly -1..+1 scale.
  Treat sentiment as weak evidence and never as the sole basis for a trade.

## Style

Be concise and specific. You are writing for a PM who will read this as an
alert, not an essay. Quantify. If the data is ambiguous, say so rather than
manufacturing confidence.
"""

# Forced strict tool use is the default output path. The schema is also reused
# verbatim as the json_schema structured-output format, so both modes are
# guaranteed to produce the same shape.
PLAN_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "market_assessment": {
            "type": "string",
            "description": "2-4 sentences on what the watch-list metrics show "
                           "in aggregate: breadth, dispersion, risk appetite.",
        },
        "actions": {
            "type": "array",
            "description": "Trades to execute. Empty list means hold everything.",
            "items": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "action": {
                        "type": "string",
                        "enum": ["BUY", "ADD", "TRIM", "SELL"],
                        "description": "BUY opens a new position, ADD increases an "
                                       "existing one, TRIM reduces, SELL exits fully.",
                    },
                    "shares_delta": {
                        "type": "integer",
                        "description": "Whole shares. Positive to buy, negative to sell.",
                    },
                    "target_weight_pct": {
                        "type": "number",
                        "description": "Intended post-trade weight as a percent of "
                                       "total portfolio value. Must be <= 25.",
                    },
                    "conviction": {"type": "string", "enum": ["low", "medium", "high"]},
                    "rationale": {
                        "type": "string",
                        "description": "Must cite at least one specific metric value "
                                       "from the supplied data.",
                    },
                },
                "required": ["ticker", "action", "shares_delta",
                             "target_weight_pct", "conviction", "rationale"],
                "additionalProperties": False,
            },
        },
        "expected_cash_pct_after": {
            "type": "number",
            "description": "Your own estimate of cash as a percent of total value "
                           "after these trades. Must be >= 10.",
        },
        "risk_notes": {
            "type": "string",
            "description": "Concentration, correlation, and drawdown risks that "
                           "remain after this plan.",
        },
        "constraint_self_check": {
            "type": "string",
            "description": "State explicitly how the plan satisfies the 25% "
                           "position cap and the 10% cash floor.",
        },
    },
    "required": ["market_assessment", "actions", "expected_cash_pct_after",
                 "risk_notes", "constraint_self_check"],
    "additionalProperties": False,
}

PLAN_TOOL: dict = {
    "name": "submit_rebalance_plan",
    "description": (
        "Submit the final portfolio rebalancing plan. Call this exactly once, "
        "after reviewing the current positions and every watch-list ticker. "
        "This is the only way to return a decision."
    ),
    "strict": True,
    "input_schema": PLAN_SCHEMA,
}
