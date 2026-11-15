"""Regression checks for the adapter boundary used with acgs-lite 4.x."""

from typing import Any

import pytest

from crewai_acgs import ACGSToolCallHook, Verdict, format_tool_call


@pytest.mark.parametrize("binary_type", [bytes, bytearray])
def test_binary_arguments_remain_scalar_and_match_rules(binary_type: Any) -> None:
    arguments = {"items": [{"body": binary_type(b"password")}]}
    rendered = format_tool_call("record_note", arguments)
    assert "items[0].body: " in rendered
    assert "password" in rendered
    assert "body[0]" not in rendered
    decision = ACGSToolCallHook().evaluate("record_note", arguments)
    assert decision.verdict is Verdict.DENY
    assert decision.error is None
    assert "ACGS-006" in decision.rule_ids


@pytest.mark.parametrize("body", ["Meeting at noon", "password"])
def test_nested_json_arguments_with_default_unsigned_configuration(body: str) -> None:
    arguments: dict[str, Any] = {"items": [{"body": body, "values": [None, True, False, 7, 1.5, {}, []]}]}
    decision = ACGSToolCallHook().evaluate("record_note", arguments)
    assert decision.verdict is (Verdict.DENY if body == "password" else Verdict.ALLOW)
    assert decision.error is None
