"""The Anthropic API call -- this is the part the project exists to demonstrate.

Two output modes, both producing the identical schema from prompts.PLAN_SCHEMA:

  "tool"        forced strict tool use. The model must emit a `tool_use` block
                for `submit_rebalance_plan`; `strict: True` guarantees the
                input validates against the schema exactly. This is the more
                illustrative path -- you can see the tool_use block in the raw
                response dump.
  "json_schema" output_config.format. The response body itself is the JSON
                object. Simpler, no tool plumbing.

Model: claude-haiku-4-5. Haiku 4.5 predates adaptive thinking, so extended
thinking uses the older {type: "enabled", budget_tokens: N} form, and
`output_config.effort` is NOT supported on it (it errors).
"""
from __future__ import annotations

import json
from typing import Any

import anthropic

from .prompts import PLAN_SCHEMA, PLAN_TOOL, SYSTEM_PROMPT


class ManagerError(RuntimeError):
    pass


def _build_system(cfg) -> Any:
    """System prompt, optionally marked as a cache breakpoint.

    Caching needs a ~1024-token prefix to engage. The system prompt plus the
    tool definition is near that boundary, so the cache may or may not hit --
    `usage.cache_read_input_tokens` in the returned meta tells you which, and
    the report prints it.
    """
    if not cfg.llm.get("prompt_caching", True):
        return SYSTEM_PROMPT
    return [{"type": "text", "text": SYSTEM_PROMPT,
             "cache_control": {"type": "ephemeral"}}]


def request_plan(user_message: str, cfg) -> tuple[dict, dict]:
    """Send the review to Claude. Returns (plan_dict, call_metadata)."""
    if not cfg.anthropic_api_key:
        raise ManagerError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add "
            "your key from https://console.anthropic.com/settings/keys"
        )

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    llm = cfg.llm
    mode = llm.get("output_mode", "tool")
    thinking_on = bool(llm.get("thinking_enabled", False))

    kwargs: dict[str, Any] = {
        "model": llm["model"],
        "max_tokens": int(llm["max_tokens"]),
        "system": _build_system(cfg),
        "messages": [{"role": "user", "content": user_message}],
    }

    if thinking_on:
        # Haiku 4.5 uses the budget_tokens form. Two API rules apply:
        # temperature must be left at its default, and tool_choice cannot be
        # forced -- so the "tool" mode below falls back to auto.
        kwargs["thinking"] = {"type": "enabled",
                              "budget_tokens": int(llm.get("thinking_budget_tokens", 2000))}
    else:
        kwargs["temperature"] = float(llm.get("temperature", 0.2))

    if mode == "tool":
        kwargs["tools"] = [PLAN_TOOL]
        if not thinking_on:
            kwargs["tool_choice"] = {"type": "tool", "name": PLAN_TOOL["name"]}
    elif mode == "json_schema":
        kwargs["output_config"] = {"format": {"type": "json_schema", "schema": PLAN_SCHEMA}}
    else:
        raise ManagerError(f"unknown llm.output_mode: {mode!r} (expected 'tool' or 'json_schema')")

    try:
        response = client.messages.create(**kwargs)
    except anthropic.AuthenticationError as exc:
        raise ManagerError(f"Authentication failed -- check ANTHROPIC_API_KEY. ({exc})") from exc
    except anthropic.NotFoundError as exc:
        raise ManagerError(f"Model {llm['model']!r} not found for this account. ({exc})") from exc
    except anthropic.RateLimitError as exc:
        raise ManagerError(f"Rate limited by the API; retry shortly. ({exc})") from exc
    except anthropic.APIStatusError as exc:
        raise ManagerError(f"API error {exc.status_code}: {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise ManagerError(f"Could not reach the Anthropic API: {exc}") from exc

    plan = _extract_plan(response, mode)
    return plan, _call_meta(response, llm, mode, thinking_on)


def _extract_plan(response, mode: str) -> dict:
    if mode == "tool":
        for block in response.content:
            if block.type == "tool_use" and block.name == PLAN_TOOL["name"]:
                # block.input is already-parsed JSON. Never string-match on it.
                return dict(block.input)
        text = " ".join(b.text for b in response.content if b.type == "text")
        raise ManagerError(
            f"model returned no tool_use block (stop_reason={response.stop_reason}). "
            f"Text was: {text[:400]}"
        )

    # json_schema mode: output_config.format guarantees the text block is valid JSON.
    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        raise ManagerError(f"no text block in response (stop_reason={response.stop_reason})")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ManagerError(f"structured output was not valid JSON: {exc}") from exc


def _call_meta(response, llm: dict, mode: str, thinking_on: bool) -> dict:
    u = response.usage
    thinking_text = "\n".join(
        b.thinking for b in response.content
        if b.type == "thinking" and getattr(b, "thinking", None)
    )
    # Haiku 4.5 pricing, USD per million tokens.
    in_cost, out_cost = 1.00, 5.00
    return {
        "model": llm["model"],
        "output_mode": mode,
        "thinking_enabled": thinking_on,
        "stop_reason": response.stop_reason,
        "message_id": response.id,
        "usage": {
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
        },
        "estimated_cost_usd": round(
            u.input_tokens / 1e6 * in_cost + u.output_tokens / 1e6 * out_cost, 6),
        "thinking": thinking_text or None,
    }
