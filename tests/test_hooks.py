# SPDX-FileCopyrightText: 2026-present ACGS Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests with real CrewAI crews, agents, tasks, ``@tool`` functions and the real hook registry.

Every crew test runs twice: once with CrewAI's ReAct text loop and once with native tool calling.
"""

from __future__ import annotations

import asyncio
import builtins
import logging
from collections.abc import Iterator, Mapping
from typing import Any

import pytest
from acgs_lite import Constitution, Rule, Severity, ViolationAction
from crewai import Agent, Crew, CrewOutput, Task
from crewai.hooks import (
    HookAborted,
    InterceptionPoint,
    ToolCallHookContext,
    get_hooks,
    register_hook,
)
from crewai.tools import tool

from crewai_acgs import HOOK_SOURCE, ACGSToolCallHook, GovernanceDecision, Verdict
from tests.fakes import ScriptedLLM, native, script

EXECUTED: list[tuple[str, dict[str, Any]]] = []


@tool("send_email")
def send_email(to: str, body: str) -> str:
    """Send an email to a recipient."""
    EXECUTED.append(("send_email", {"to": to, "body": body}))
    return f"Email sent to {to}"


@tool("get_weather")
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    EXECUTED.append(("get_weather", {"city": city}))
    return f"Sunny in {city}"


@tool("issue_refund")
def issue_refund(order_id: str, note: str) -> str:
    """Refund an order."""
    EXECUTED.append(("issue_refund", {"order_id": order_id, "note": note}))
    return f"Refunded {order_id}"


REVIEW_RULE = Rule(
    id="T-REVIEW",
    text="refunds need review",
    severity=Severity.HIGH,
    keywords=["refund"],
    workflow_action=ViolationAction.REQUIRE_HUMAN_REVIEW,
)
BLOCK_RULE = Rule(id="T-BLOCK", text="no deletes", severity=Severity.CRITICAL, keywords=["delete everything"])
RULES = Constitution.from_rules([REVIEW_RULE, BLOCK_RULE], name="hooks-test")

SECRET_EMAIL = {"to": "ops@example.com", "body": "The admin password is hunter2"}
ACGS_006 = (
    "Tool call 'send_email' was not executed. "
    "Blocked by ACGS rule ACGS-006 (Agents must not expose sensitive data in responses)."
)
REVIEW_DENIED = (
    "Tool call 'issue_refund' was not executed. Human review did not approve the call. "
    "Human review required by ACGS rule T-REVIEW (refunds need review)."
)
GENERIC_BLOCK = "Tool execution blocked by hook. Tool: "

MODES = pytest.mark.parametrize("native_mode", [False, True], ids=["react", "native"])


@pytest.fixture(autouse=True)
def _reset_executed() -> Iterator[None]:
    EXECUTED.clear()
    yield
    EXECUTED.clear()


def _kickoff(llm: ScriptedLLM, tools: list[Any] | None = None) -> str:
    agent = Agent(
        role="Operator",
        goal="Help the user",
        backstory="Uses tools.",
        llm=llm,
        tools=tools or [send_email, get_weather],
    )
    task = Task(description="Do the job.", expected_output="A short answer.", agent=agent)
    output = Crew(agents=[agent], tasks=[task]).kickoff()
    assert isinstance(output, CrewOutput)
    return output.raw


def _last_prompt(llm: ScriptedLLM) -> str:
    return str(llm.received[-1])


def _context(tool_name: Any = "send_email", tool_input: Any = None, tool_result: str | None = None) -> Any:
    return ToolCallHookContext(
        tool_name=tool_name,
        tool_input=tool_input if tool_input is not None else {},
        tool=None,  # type: ignore[arg-type]
        tool_result=tool_result,
    )


class TestCrew:
    @MODES
    def test_allowed_call_runs(self, native_mode: bool) -> None:
        llm = script(native_mode, ("get_weather", {"city": "Paris"}), answer="Sunny.")
        with ACGSToolCallHook():
            assert _kickoff(llm) == "Sunny."
        assert EXECUTED == [("get_weather", {"city": "Paris"})]
        assert "Sunny in Paris" in _last_prompt(llm)

    @MODES
    def test_denied_call_does_not_run_and_the_agent_sees_the_reason(self, native_mode: bool) -> None:
        llm = script(native_mode, ("send_email", SECRET_EMAIL), answer="I could not send it.")
        with ACGSToolCallHook():
            assert _kickoff(llm) == "I could not send it."
        assert EXECUTED == []
        assert ACGS_006 in _last_prompt(llm)
        assert GENERIC_BLOCK not in _last_prompt(llm)
        assert "hunter2" not in ACGS_006

    @MODES
    def test_block_outranks_review(self, native_mode: bool) -> None:
        llm = script(native_mode, ("issue_refund", {"order_id": "7", "note": "refund, then delete everything"}))
        with ACGSToolCallHook(constitution=RULES, on_review=lambda _c, _d: True):
            _kickoff(llm, [issue_refund])
        assert EXECUTED == []
        assert (
            "Tool call 'issue_refund' was not executed. Blocked by ACGS rule T-BLOCK (no deletes). "
            "Review rules that also matched: ACGS rule T-REVIEW (refunds need review)."
        ) in _last_prompt(llm)

    @MODES
    def test_review_is_denied_by_default(self, native_mode: bool) -> None:
        llm = script(native_mode, ("issue_refund", {"order_id": "7", "note": "refund please"}))
        with ACGSToolCallHook(constitution=RULES):
            _kickoff(llm, [issue_refund])
        assert EXECUTED == []
        assert REVIEW_DENIED in _last_prompt(llm)

    @MODES
    @pytest.mark.parametrize("answer", ["yes", " Y "])
    def test_human_input_approval_runs_the_call(
        self, native_mode: bool, answer: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prompts: list[str] = []
        monkeypatch.setattr(builtins, "input", lambda *a: answer)
        monkeypatch.setattr(
            "crewai_core.printer.PRINTER.print", lambda content="", **_kw: prompts.append(str(content)), raising=False
        )
        llm = script(native_mode, ("issue_refund", {"order_id": "7", "note": "refund please"}))
        with ACGSToolCallHook(constitution=RULES, on_review="human_input"):
            _kickoff(llm, [issue_refund])
        assert EXECUTED == [("issue_refund", {"order_id": "7", "note": "refund please"})]

    @MODES
    @pytest.mark.parametrize("answer", ["no", "", "yes please"])
    def test_human_input_other_answers_deny(
        self, native_mode: bool, answer: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(builtins, "input", lambda *a: answer)
        llm = script(native_mode, ("issue_refund", {"order_id": "7", "note": "refund please"}))
        with ACGSToolCallHook(constitution=RULES, on_review="human_input"):
            _kickoff(llm, [issue_refund])
        assert EXECUTED == []
        assert REVIEW_DENIED in _last_prompt(llm)

    @MODES
    def test_human_input_without_a_terminal_denies(self, native_mode: bool, monkeypatch: pytest.MonkeyPatch) -> None:
        def no_stdin(*_a: Any) -> str:
            raise EOFError

        monkeypatch.setattr(builtins, "input", no_stdin)
        llm = script(native_mode, ("issue_refund", {"order_id": "7", "note": "refund please"}))
        with ACGSToolCallHook(constitution=RULES, on_review="human_input"):
            _kickoff(llm, [issue_refund])
        assert EXECUTED == []
        assert "ACGS governance could not evaluate this tool call, so it was denied." in _last_prompt(llm)

    @MODES
    def test_review_handler_decides(self, native_mode: bool) -> None:
        seen: list[GovernanceDecision] = []

        def approve(context: ToolCallHookContext, decision: GovernanceDecision) -> bool:
            seen.append(decision)
            return bool(context.tool_input["order_id"] == "7")

        llm = script(
            native_mode,
            ("issue_refund", {"order_id": "7", "note": "refund please"}),
            ("issue_refund", {"order_id": "8", "note": "refund please"}),
        )
        with ACGSToolCallHook(constitution=RULES, on_review=approve):
            _kickoff(llm, [issue_refund])
        assert EXECUTED == [("issue_refund", {"order_id": "7", "note": "refund please"})]
        assert [d.verdict for d in seen] == [Verdict.REVIEW, Verdict.REVIEW]
        assert REVIEW_DENIED in _last_prompt(llm)

    @MODES
    @pytest.mark.parametrize("result", ["yes", 1, None])
    def test_review_handler_must_return_true(self, native_mode: bool, result: Any) -> None:
        llm = script(native_mode, ("issue_refund", {"order_id": "7", "note": "refund please"}))
        with ACGSToolCallHook(constitution=RULES, on_review=lambda _c, _d: result):
            _kickoff(llm, [issue_refund])
        assert EXECUTED == []

    @MODES
    def test_review_handler_error_denies(self, native_mode: bool, caplog: pytest.LogCaptureFixture) -> None:
        def broken(_c: ToolCallHookContext, _d: GovernanceDecision) -> bool:
            raise ConnectionError("approval service down")

        llm = script(native_mode, ("issue_refund", {"order_id": "7", "note": "refund please"}))
        with caplog.at_level(logging.WARNING), ACGSToolCallHook(constitution=RULES, on_review=broken):
            _kickoff(llm, [issue_refund])
        assert EXECUTED == []
        assert "fail-closed" in caplog.text

    @MODES
    def test_formatter_error_denies(self, native_mode: bool) -> None:
        def broken(_name: str, _args: Any) -> str:
            raise RuntimeError("boom")

        llm = script(native_mode, ("get_weather", {"city": "Paris"}))
        with ACGSToolCallHook(tool_call_formatter=broken):
            _kickoff(llm)
        assert EXECUTED == []
        assert (
            "Tool call 'get_weather' was not executed. "
            "ACGS governance could not evaluate this tool call, so it was denied."
        ) in _last_prompt(llm)

    @MODES
    def test_tool_errors_are_reported_by_crewai_not_swallowed(self, native_mode: bool) -> None:
        @tool("flaky")
        def flaky(city: str) -> str:
            """Always fails."""
            raise ValueError("backend unavailable")

        llm = script(native_mode, ("flaky", {"city": "Paris"}))
        with ACGSToolCallHook():
            _kickoff(llm, [flaky])
        assert "backend unavailable" in _last_prompt(llm)

    @MODES
    def test_a_later_hook_that_blocks_keeps_crewais_message(self, native_mode: bool) -> None:
        llm = script(native_mode, ("get_weather", {"city": "Paris"}))
        with ACGSToolCallHook():
            register_hook(InterceptionPoint.PRE_TOOL_CALL, lambda _ctx: False)
            _kickoff(llm)
        assert EXECUTED == []
        assert GENERIC_BLOCK + "get_weather" in _last_prompt(llm)
        assert "was not executed" not in _last_prompt(llm)

    @MODES
    def test_an_earlier_hook_that_blocks_keeps_crewais_message(self, native_mode: bool) -> None:
        def block(_ctx: ToolCallHookContext) -> None:
            raise HookAborted("not today", source="other")

        register_hook(InterceptionPoint.PRE_TOOL_CALL, block)
        llm = script(native_mode, ("send_email", SECRET_EMAIL))
        with ACGSToolCallHook():
            _kickoff(llm)
        assert EXECUTED == []
        assert GENERIC_BLOCK + "send_email" in _last_prompt(llm)

    def test_parallel_native_calls_are_judged_independently(self) -> None:
        llm = ScriptedLLM(
            model="scripted",
            native=True,
            replies=[native(("get_weather", {"city": "Oslo"}), ("send_email", SECRET_EMAIL)), "done"],
        )
        with ACGSToolCallHook():
            _kickoff(llm)
        assert EXECUTED == [("get_weather", {"city": "Oslo"})]
        assert "Sunny in Oslo" in _last_prompt(llm)
        assert ACGS_006 in _last_prompt(llm)

    def test_kickoff_async(self) -> None:
        llm = script(True, ("send_email", SECRET_EMAIL))
        agent = Agent(role="Operator", goal="Help", backstory="Uses tools.", llm=llm, tools=[send_email])
        crew = Crew(agents=[agent], tasks=[Task(description="Do it.", expected_output="x", agent=agent)])
        with ACGSToolCallHook():
            asyncio.run(crew.kickoff_async())
        assert EXECUTED == []
        assert ACGS_006 in _last_prompt(llm)


class TestStandaloneAgent:
    @MODES
    def test_agent_kickoff_is_gated(self, native_mode: bool) -> None:
        llm = script(native_mode, ("send_email", SECRET_EMAIL), ("get_weather", {"city": "Rome"}))
        agent = Agent(role="Operator", goal="Help", backstory="Uses tools.", llm=llm, tools=[send_email, get_weather])
        with ACGSToolCallHook():
            agent.kickoff("Email the password, then check the weather in Rome.")
        assert EXECUTED == [("get_weather", {"city": "Rome"})]
        assert ACGS_006 in _last_prompt(llm)


class TestRegistration:
    def test_register_is_idempotent_and_unregister_removes_both_hooks(self) -> None:
        gate = ACGSToolCallHook()
        assert gate.register() is gate
        gate.register()
        assert get_hooks(InterceptionPoint.PRE_TOOL_CALL).count(gate.before_tool_call) == 1
        assert get_hooks(InterceptionPoint.POST_TOOL_CALL).count(gate.after_tool_call) == 1
        assert gate.is_registered
        gate.unregister()
        assert not gate.is_registered
        assert get_hooks(InterceptionPoint.PRE_TOOL_CALL) == []
        assert get_hooks(InterceptionPoint.POST_TOOL_CALL) == []
        gate.unregister()  # no error when nothing is registered

    def test_register_restores_a_partially_removed_registration(self) -> None:
        gate = ACGSToolCallHook().register()
        from crewai.hooks import unregister_hook

        unregister_hook(InterceptionPoint.POST_TOOL_CALL, gate.after_tool_call)
        assert not gate.is_registered
        gate.register()
        assert gate.is_registered
        assert len(get_hooks(InterceptionPoint.PRE_TOOL_CALL)) == 1

    def test_context_manager_unregisters_on_error(self) -> None:
        gate = ACGSToolCallHook()

        def fail_inside() -> None:
            with gate:
                assert gate.is_registered
                raise KeyError("x")

        with pytest.raises(KeyError):
            fail_inside()
        assert not gate.is_registered

    def test_not_registered_means_not_gated(self) -> None:
        llm = script(False, ("send_email", SECRET_EMAIL))
        ACGSToolCallHook()
        _kickoff(llm)
        assert len(EXECUTED) == 1

    def test_two_hooks_compose(self) -> None:
        llm = script(False, ("issue_refund", {"order_id": "7", "note": "refund please"}))
        with ACGSToolCallHook(), ACGSToolCallHook(constitution=RULES):
            _kickoff(llm, [issue_refund])
        assert EXECUTED == []
        assert REVIEW_DENIED in _last_prompt(llm)


class TestConfiguration:
    def test_defaults(self) -> None:
        gate = ACGSToolCallHook()
        assert gate.on_review == "deny"
        assert gate.constitution.hash == Constitution.default().hash

    def test_custom_constitution(self) -> None:
        assert ACGSToolCallHook(constitution=RULES).constitution is RULES

    @pytest.mark.parametrize("bad", ["confirm", "", None, 3])
    def test_invalid_on_review(self, bad: Any) -> None:
        with pytest.raises(ValueError, match="on_review"):
            ACGSToolCallHook(on_review=bad)

    def test_evaluate(self) -> None:
        decision = ACGSToolCallHook().evaluate("send_email", SECRET_EMAIL)
        assert decision.verdict is Verdict.DENY
        assert decision.rule_ids == ("ACGS-006",)


class TestHookFunctions:
    """The hook methods called directly with real ``ToolCallHookContext`` objects."""

    def test_denial_raises_hook_aborted_with_source_and_reason(self) -> None:
        gate = ACGSToolCallHook()
        with pytest.raises(HookAborted) as raised:
            gate.before_tool_call(_context("send_email", dict(SECRET_EMAIL)))
        assert raised.value.source == HOOK_SOURCE
        assert (
            raised.value.reason == "Blocked by ACGS rule ACGS-006 (Agents must not expose sensitive data in responses)."
        )

    def test_allow_returns_none(self) -> None:
        gate = ACGSToolCallHook()
        context = _context("get_weather", {"city": "Paris"})
        assert gate.before_tool_call(context) is None  # type: ignore[func-returns-value]
        assert gate.after_tool_call(_context("get_weather", {"city": "Paris"}, "Sunny")) is None

    @pytest.mark.parametrize("name", [None, "", 42])
    def test_invalid_tool_name_denies(self, name: Any) -> None:
        gate = ACGSToolCallHook()
        context = _context(name, {"city": "Paris"})
        with pytest.raises(HookAborted, match="could not evaluate"):
            gate.before_tool_call(context)
        assert gate.after_tool_call(_context(str(name), {"city": "Paris"}, GENERIC_BLOCK + "x")) == (
            f"Tool call '{name}' was not executed. ACGS governance could not evaluate this tool call, so it was denied."
        )

    def test_reason_is_only_attached_to_the_matching_blocked_call(self) -> None:
        gate = ACGSToolCallHook()
        with pytest.raises(HookAborted):
            gate.before_tool_call(_context("send_email", dict(SECRET_EMAIL)))
        # A different call (another tool) does not get the explanation, and the pending denial is consumed.
        assert gate.after_tool_call(_context("get_weather", {"city": "x"}, GENERIC_BLOCK + "get_weather")) is None
        assert gate.after_tool_call(_context("send_email", dict(SECRET_EMAIL), GENERIC_BLOCK + "send_email")) is None

    @pytest.mark.parametrize(
        ("tool_input", "result"),
        [({"to": "other@example.com", "body": "x"}, GENERIC_BLOCK + "send_email"), (SECRET_EMAIL, "Email sent")],
    )
    def test_reason_needs_same_input_and_a_blocked_result(self, tool_input: dict[str, Any], result: str) -> None:
        gate = ACGSToolCallHook()
        with pytest.raises(HookAborted):
            gate.before_tool_call(_context("send_email", dict(SECRET_EMAIL)))
        assert gate.after_tool_call(_context("send_email", dict(tool_input), result)) is None

    def test_after_hook_errors_are_logged_and_ignored(self, caplog: pytest.LogCaptureFixture) -> None:
        class Broken:
            tool_name = "send_email"

            @property
            def tool_result(self) -> str:
                raise RuntimeError("no result")

        gate = ACGSToolCallHook()
        with pytest.raises(HookAborted):
            gate.before_tool_call(_context("send_email", dict(SECRET_EMAIL)))
        with caplog.at_level(logging.WARNING):
            assert gate.after_tool_call(Broken()) is None  # type: ignore[arg-type]
        assert "could not attach the denial reason" in caplog.text

    def test_unreadable_tool_input_still_blocks(self) -> None:
        class Exploding(Mapping[str, Any]):
            def __getitem__(self, key: str) -> Any:
                raise RuntimeError("unreadable")

            def __iter__(self) -> Iterator[str]:
                raise RuntimeError("unreadable")

            def __len__(self) -> int:
                return 1

        gate = ACGSToolCallHook()
        context = _context("send_email")
        context.tool_input = Exploding()
        with pytest.raises(HookAborted, match="could not evaluate"):
            gate.before_tool_call(context)
        assert gate.after_tool_call(_context("send_email", None, GENERIC_BLOCK + "send_email")) is None

    def test_non_mapping_tool_input_is_evaluated_and_explained(self) -> None:
        gate = ACGSToolCallHook()
        context = _context("send_email")
        context.tool_input = "The admin password is hunter2"
        with pytest.raises(HookAborted, match="ACGS-006"):
            gate.before_tool_call(context)
        blocked = _context("send_email", tool_result=GENERIC_BLOCK + "send_email")
        blocked.tool_input = "The admin password is hunter2"
        assert gate.after_tool_call(blocked) == ACGS_006

    def test_human_input_prompt_shows_reason_and_arguments(self) -> None:
        gate = ACGSToolCallHook(constitution=RULES, on_review="human_input")
        context = _context("issue_refund", {"order_id": "7", "note": "refund"})
        asked: list[tuple[str, str]] = []

        def fake_request(prompt: str, default_message: str = "") -> str:
            asked.append((prompt, default_message))
            return "yes"

        context.request_human_input = fake_request
        gate.before_tool_call(context)
        assert asked == [
            (
                "ACGS: the tool call 'issue_refund' needs human review. "
                "Human review required by ACGS rule T-REVIEW (refunds need review).",
                'Arguments: {"note": "refund", "order_id": "7"}\nType \'yes\' to run it, anything else to deny:',
            )
        ]

    def test_human_input_non_string_answer_denies(self) -> None:
        gate = ACGSToolCallHook(constitution=RULES, on_review="human_input")
        context = _context("issue_refund", {"order_id": "7", "note": "refund"})
        context.request_human_input = lambda *_a, **_k: None
        with pytest.raises(HookAborted, match="did not approve"):
            gate.before_tool_call(context)
