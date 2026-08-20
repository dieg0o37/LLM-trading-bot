"""Notification output: rich console, plus JSON and Markdown on disk.

No broker, no orders. This module IS the output of the system.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from .config import REPORT_DIR

console = Console()

_ACTION_STYLE = {"BUY": "bold green", "ADD": "green",
                 "TRIM": "yellow", "SELL": "bold red"}


def print_report(result: dict) -> None:
    plan = result["plan"]
    before, after = result["valuation_before"], result["valuation_after"]
    meta = result["call_meta"]

    console.print()
    console.print(Panel(
        f"[bold]LLM Portfolio Manager[/bold] — {result['as_of']}\n"
        f"model [cyan]{meta['model']}[/cyan] · mode [cyan]{meta['output_mode']}[/cyan] "
        f"· NOTIFICATION ONLY — no orders were placed",
        box=box.ROUNDED, border_style="cyan"))

    # --- market view -------------------------------------------------------
    console.print(Panel(plan["market_assessment"], title="Market assessment",
                        border_style="dim", box=box.ROUNDED))

    # --- actions -----------------------------------------------------------
    actions = plan.get("actions", [])
    sizing = {s["ticker"]: s for s in result.get("sizing", [])}
    if actions:
        t = Table(title="Proposed actions  [dim](shares derived by Python from "
                        "target weight)[/dim]",
                  box=box.SIMPLE_HEAVY, title_justify="left")
        for col, just in [("Action", "left"), ("Ticker", "left"),
                          ("Target wt", "right"), ("Shares", "right"),
                          ("Conv.", "left"), ("Rationale", "left")]:
            t.add_column(col, justify=just, overflow="fold")
        for a in actions:
            z = sizing.get(a["ticker"], {})
            delta = z.get("shares_delta", 0)
            t.add_row(
                f"[{_ACTION_STYLE.get(a['action'], 'white')}]{a['action']}[/]",
                a["ticker"],
                f"{a['target_weight_pct']:.1f}%",
                f"{delta:+,.0f}" if delta else "[dim]—[/dim]",
                a["conviction"],
                a["rationale"],
            )
        console.print(t)
    else:
        console.print(Panel("[yellow]No trades proposed — hold current positions.[/yellow]",
                            border_style="yellow", box=box.ROUNDED))

    # --- execution log -----------------------------------------------------
    if result["execution_log"]:
        console.print("[dim]Simulated fills (at last close):[/dim]")
        for line in result["execution_log"]:
            style = "red" if line.startswith("SKIP") else "dim"
            console.print(f"  [{style}]{line}[/{style}]")

    # --- portfolio before/after -------------------------------------------
    t = Table(title="Portfolio", box=box.SIMPLE_HEAVY, title_justify="left")
    t.add_column("Ticker"); t.add_column("Weight before", justify="right")
    t.add_column("Weight after", justify="right"); t.add_column("Value after", justify="right")

    wb = {h["ticker"]: h["weight_pct"] for h in before["holdings"]}
    wa = {h["ticker"]: h for h in after["holdings"]}
    for ticker in sorted(set(wb) | set(wa), key=lambda k: -wa.get(k, {}).get("weight_pct", 0)):
        h = wa.get(ticker)
        t.add_row(ticker,
                  f"{wb.get(ticker, 0):.1f}%",
                  f"{h['weight_pct']:.1f}%" if h else "[red]0.0% (exited)[/red]",
                  f"${h['market_value']:,.0f}" if h else "—")
    t.add_row("[bold]CASH[/bold]", f"{before['cash_pct']:.1f}%",
              f"{after['cash_pct']:.1f}%", f"${after['cash']:,.0f}")
    t.add_row("[bold]TOTAL[/bold]", "", "", f"[bold]${after['total_value']:,.0f}[/bold]")
    console.print(t)

    # --- constraint verdict ------------------------------------------------
    violations = result["violations"]
    if violations:
        body = "\n".join(f"• {v}" for v in violations)
        console.print(Panel(f"[bold red]PLAN VIOLATES HARD CONSTRAINTS[/bold red]\n{body}",
                            border_style="red", box=box.ROUNDED))
    else:
        console.print(Panel(
            f"[bold green]Constraints satisfied[/bold green] — "
            f"max position {result['max_weight_after']:.1f}% (limit "
            f"{result['limits']['max_position_pct']:.0f}%), cash "
            f"{after['cash_pct']:.1f}% (floor {result['limits']['min_cash_pct']:.0f}%)",
            border_style="green", box=box.ROUNDED))

    console.print(Panel(plan["risk_notes"], title="Risk notes",
                        border_style="dim", box=box.ROUNDED))

    _print_portfolio_pnl(result)
    _print_stock_years(result)

    # --- API telemetry -----------------------------------------------------
    u = meta["usage"]
    console.print(
        f"[dim]API · in {u['input_tokens']:,} tok · out {u['output_tokens']:,} tok · "
        f"cache write {u['cache_creation_input_tokens']:,} · "
        f"cache read {u['cache_read_input_tokens']:,} · "
        f"stop_reason {meta['stop_reason']} · "
        f"~${meta['estimated_cost_usd']:.4f}[/dim]")
    console.print()


def write_reports(result: dict) -> tuple[Path, Path]:
    """Persist the run. Timestamped so history accumulates as an audit trail."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    json_path = REPORT_DIR / f"{stamp}.json"
    md_path = REPORT_DIR / f"{stamp}.md"

    json_path.write_text(json.dumps(result, indent=2, default=str))
    md_path.write_text(_render_markdown(result))
    console.print(f"[dim]Reports written: {json_path.name}, {md_path.name}[/dim]")
    return json_path, md_path


def _render_markdown(result: dict) -> str:
    plan, after, meta = result["plan"], result["valuation_after"], result["call_meta"]
    L: list[str] = [
        f"# Portfolio review — {result['as_of']}",
        "",
        "> Notification only. No broker is connected and no orders were placed.",
        "",
        f"**Model:** `{meta['model']}` · **output mode:** `{meta['output_mode']}` · "
        f"**stop_reason:** `{meta['stop_reason']}`",
        "",
        "## Market assessment", "", plan["market_assessment"], "",
        "## Proposed actions", "",
    ]

    if plan.get("actions"):
        sizing = {s["ticker"]: s for s in result.get("sizing", [])}
        L += ["The model supplies `target_weight_pct`; share counts are derived in "
              "Python at the last close.", "",
              "| Action | Ticker | Target wt | Shares | Conviction | Rationale |",
              "|---|---|---:|---:|---|---|"]
        for a in plan["actions"]:
            rationale = a["rationale"].replace("|", "\\|")
            delta = sizing.get(a["ticker"], {}).get("shares_delta", 0)
            L.append(f"| **{a['action']}** | {a['ticker']} | "
                     f"{a['target_weight_pct']:.1f}% | "
                     f"{(f'{delta:+,.0f}' if delta else '—')} | "
                     f"{a['conviction']} | {rationale} |")
    else:
        L.append("_No trades proposed — hold current positions._")

    L += ["", "## Portfolio after plan", "",
          "| Ticker | Shares | Price | Value | Weight |", "|---|---:|---:|---:|---:|"]
    for h in after["holdings"]:
        L.append(f"| {h['ticker']} | {h['shares']:,.0f} | ${h['price']:,.2f} | "
                 f"${h['market_value']:,.0f} | {h['weight_pct']:.1f}% |")
    L.append(f"| **CASH** | | | ${after['cash']:,.0f} | {after['cash_pct']:.1f}% |")
    L.append(f"| **TOTAL** | | | **${after['total_value']:,.0f}** | 100.0% |")

    L += ["", "## Constraint check (verified in Python, not by the model)", ""]
    if result["violations"]:
        L.append("**FAILED**")
        L += [f"- {v}" for v in result["violations"]]
    else:
        L.append(f"**PASSED** — max position {result['max_weight_after']:.1f}% "
                 f"(limit {result['limits']['max_position_pct']:.0f}%), "
                 f"cash {after['cash_pct']:.1f}% "
                 f"(floor {result['limits']['min_cash_pct']:.0f}%).")

    L += ["", "### Model's own self-check", "", f"> {plan['constraint_self_check']}",
          "", "## Risk notes", "", plan["risk_notes"], ""]

    L += _markdown_performance(result)

    if result["execution_log"]:
        L += ["## Simulated fills", "", "```"] + result["execution_log"] + ["```", ""]

    u = meta["usage"]
    L += ["## API call", "",
          f"- input tokens: {u['input_tokens']:,}",
          f"- output tokens: {u['output_tokens']:,}",
          f"- cache creation tokens: {u['cache_creation_input_tokens']:,}",
          f"- cache read tokens: {u['cache_read_input_tokens']:,}",
          f"- estimated cost: ${meta['estimated_cost_usd']:.4f}",
          f"- message id: `{meta['message_id']}`", ""]

    if meta.get("thinking"):
        L += ["<details><summary>Extended thinking</summary>", "",
              "```", meta["thinking"], "```", "", "</details>", ""]

    return "\n".join(L)


def _print_portfolio_pnl(result: dict) -> None:
    """This portfolio's own realised profit per calendar year."""
    pnl = result.get("portfolio_pnl") or {}
    if not pnl.get("available"):
        console.print(f"[dim]Profit per year: not available yet — "
                      f"{pnl.get('reason', 'no history recorded')}[/dim]")
        return

    t = Table(title=f"Portfolio profit per year (since {pnl['inception_date']})",
              box=box.SIMPLE_HEAVY, title_justify="left")
    for col in ["Year", "Start", "End", "Profit $", "Profit %", "Realised $"]:
        t.add_column(col, justify="right" if col != "Year" else "left")
    for row in pnl["by_year"]:
        colour = "green" if row["profit_usd"] >= 0 else "red"
        t.add_row(str(row["year"]),
                  f"${row['start_value']:,.0f}",
                  f"${row['end_value']:,.0f}",
                  f"[{colour}]{row['profit_usd']:+,.0f}[/{colour}]",
                  f"[{colour}]{row['profit_pct']:+.2f}%[/{colour}]",
                  f"{row['realised_pnl_usd']:+,.0f}")
    total_colour = "green" if pnl["total_profit_usd"] >= 0 else "red"
    t.add_row("[bold]TOTAL[/bold]",
              f"${pnl['inception_value']:,.0f}",
              f"${pnl['current_value']:,.0f}",
              f"[bold {total_colour}]{pnl['total_profit_usd']:+,.0f}[/bold {total_colour}]",
              f"[bold {total_colour}]{pnl['total_profit_pct']:+.2f}%[/bold {total_colour}]",
              f"{pnl['total_realised_pnl_usd']:+,.0f}")
    console.print(t)


def _print_stock_years(result: dict) -> None:
    """Calendar-year returns of the universe vs an equal-weight basket."""
    perf = result.get("stock_performance") or {}
    rows = perf.get("by_year", [])
    if not rows:
        return

    t = Table(title="Watch-list returns by calendar year", box=box.SIMPLE_HEAVY,
              title_justify="left")
    t.add_column("Year"); t.add_column("Equal-wt universe", justify="right")
    t.add_column("SPY", justify="right"); t.add_column("Best", justify="right")
    t.add_column("Worst", justify="right")
    for row in rows:
        ew, bm = row.get("equal_weight_universe_pct"), row.get("benchmark_pct")
        label = f"{row['year']}"
        if result.get("current_year_partial") and row["year"] == rows[-1]["year"]:
            label += " *"
        t.add_row(label,
                  _signed(ew), _signed(bm),
                  f"[green]{row['best']}[/green]",
                  f"[red]{row['worst']}[/red]")
    console.print(t)
    if result.get("current_year_partial"):
        console.print("[dim]  * current year is partial (year-to-date)[/dim]")


def _signed(x) -> str:
    if x is None:
        return "—"
    colour = "green" if x >= 0 else "red"
    return f"[{colour}]{x:+.1f}%[/{colour}]"


def _markdown_performance(result: dict) -> list[str]:
    """Profit-per-year sections for the Markdown report."""
    L: list[str] = ["## Profit per year", ""]

    pnl = result.get("portfolio_pnl") or {}
    if pnl.get("available"):
        L += [f"Tracked since **{pnl['inception_date']}** across "
              f"{pnl['snapshots']} recorded runs.", "",
              "| Year | Start | End | Profit $ | Profit % | Realised $ |",
              "|---|---:|---:|---:|---:|---:|"]
        for row in pnl["by_year"]:
            L.append(f"| {row['year']} | ${row['start_value']:,.0f} | "
                     f"${row['end_value']:,.0f} | {row['profit_usd']:+,.0f} | "
                     f"{row['profit_pct']:+.2f}% | {row['realised_pnl_usd']:+,.0f} |")
        L.append(f"| **TOTAL** | ${pnl['inception_value']:,.0f} | "
                 f"${pnl['current_value']:,.0f} | "
                 f"**{pnl['total_profit_usd']:+,.0f}** | "
                 f"**{pnl['total_profit_pct']:+.2f}%** | "
                 f"{pnl['total_realised_pnl_usd']:+,.0f} |")
    else:
        L.append(f"_Not available yet — {pnl.get('reason', 'no history recorded')}_")

    L += ["", "> This is a record of outcomes, not a backtest. It does not show "
          "whether the agent's decisions beat holding the universe — see the "
          "README section *Profit per year is not a backtest*.", ""]

    perf = result.get("stock_performance") or {}
    rows = perf.get("by_year", [])
    if rows:
        L += ["## Watch-list returns by calendar year", "",
              "| Year | Equal-weight universe | SPY | Best | Worst |",
              "|---|---:|---:|---:|---:|"]
        for row in rows:
            label = str(row["year"])
            if result.get("current_year_partial") and row["year"] == rows[-1]["year"]:
                label += " *"
            ew = row.get("equal_weight_universe_pct")
            bm = row.get("benchmark_pct")
            L.append(f"| {label} | {ew:+.1f}% | "
                     f"{(f'{bm:+.1f}%' if bm is not None else '—')} | "
                     f"{row['best']} | {row['worst']} |")
        if result.get("current_year_partial"):
            L += ["", "_* current year is partial (year-to-date)._"]
        L.append("")

    return L
