# SPDX-FileCopyrightText: 2026-present ACGS Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""ACGS governance for CrewAI tool calls, using CrewAI's native tool call hooks."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Literal

from acgs_lite import Constitution
from crewai.hooks import (
    HookAborted,
    InterceptionPoint,
    ToolCallHookContext,
    get_hooks,
    register_hook,
    unregister_hook,
)
from typing_extensions import Self

from crewai_acgs._governance import GovernanceDecision, ToolCallFormatter, ToolCallGovernor, Verdict

logger = logging.getLogger(__name__)

ReviewHandler = Callable[[ToolCallHookContext, GovernanceDecision], bool]
"""A function that decides a review verdict. It must return ``True`` to run the tool call."""

HOOK_SOURCE = "crewai-acgs"
"""The ``source`` set on every ``HookAborted`` raised by this package."""

_BLOCKED_PREFIX = "Tool execution blocked by hook"
_APPROVALS = frozenset({"y", "yes"})
_EVALUATION_FAILED = "ACGS governance could not evaluate this tool call, so it was denied."


@dataclass(frozen=True)
class _Denial:
    """A call this hook blocked, kept until the matching ``POST_TOOL_CALL`` hook runs."""

    tool_name: str
    tool_input: dict[str, Any] | None
    message: str


class ACGSToolCallHook:
    """Check every CrewAI tool call against an ACGS constitution before the tool runs.

    The hook uses CrewAI's tool call hooks (``InterceptionPoint.PRE_TOOL_CALL`` and
    ``InterceptionPoint.POST_TOOL_CALL``):

    - ``before_tool_call`` evaluates the call. Allowed calls run unchanged. Denied calls raise
      ``HookAborted``, so CrewAI does not run the tool and the agent continues.
    - ``after_tool_call`` replaces CrewAI's generic ``"Tool execution blocked by hook"`` result
      with the ACGS reason for calls this hook blocked, so the agent can see why.

    Calls that only match review rules are denied by default. With ``on_review="human_input"``
    the hook asks for approval in the terminal through CrewAI's
    ``ToolCallHookContext.request_human_input``; with a callable, the callable decides.

    Any error while evaluating or reviewing a call denies it. CrewAI ignores exceptions raised
    by hooks other than ``HookAborted``, so the hook never lets one escape.

    CrewAI hooks are global to the Python process: once registered, the hook applies to every
    crew and agent until it is unregistered.

    Example:
        >>> from crewai_acgs import ACGSToolCallHook
        >>> gate = ACGSToolCallHook()
        >>> gate.evaluate("send_email", {"to": "a@example.com", "body": "The password is hunter2"}).verdict
        <Verdict.DENY: 'deny'>
        >>> with gate:  # registered with CrewAI inside the block
        ...     gate.is_registered
        True
        >>> gate.is_registered
        False
    """

    def __init__(
        self,
        *,
        constitution: Constitution | None = None,
        agent_id: str = "crewai-agent",
        on_review: Literal["deny", "human_input"] | ReviewHandler = "deny",
        tool_call_formatter: ToolCallFormatter | None = None,
    ) -> None:
        """Create the hook. Call ``register()`` (or use it as a context manager) to activate it.

        Args:
            constitution: The ACGS constitution to enforce. Defaults to
                ``acgs_lite.Constitution.default()``.
            agent_id: Agent identifier recorded by ACGS for each validation.
            on_review: What to do when only review rules match. ``"deny"`` denies the call.
                ``"human_input"`` asks in the terminal and runs the call only if the answer is
                ``y`` or ``yes``. A callable ``(context, decision) -> bool`` runs the call only if
                it returns ``True``.
            tool_call_formatter: Optional function ``(tool_name, args) -> str`` that replaces
                ``format_tool_call``.

        Raises:
            ValueError: If ``on_review`` is invalid.
        """
        if not callable(on_review) and on_review not in ("deny", "human_input"):
            msg = f"on_review must be 'deny', 'human_input' or a callable, got {on_review!r}"
            raise ValueError(msg)
        self.on_review = on_review
        self._governor = ToolCallGovernor(constitution, agent_id=agent_id, tool_call_formatter=tool_call_formatter)
        self._pending: ContextVar[_Denial | None] = ContextVar(f"crewai_acgs_denial_{id(self)}", default=None)

    @property
    def constitution(self) -> Constitution:
        """The constitution this hook enforces."""
        return self._governor.constitution

    def evaluate(self, tool_name: str, arguments: Any) -> GovernanceDecision:
        """Evaluate a tool call without a running crew. Never raises.

        Args:
            tool_name: Name of the tool.
            arguments: The tool arguments.

        Returns:
            The governance decision.
        """
        return self._governor.evaluate(tool_name, arguments)

    def before_tool_call(self, context: ToolCallHookContext) -> None:
        """``PRE_TOOL_CALL`` hook: return to run the call, raise ``HookAborted`` to block it.

        Args:
            context: The CrewAI tool call context.

        Raises:
            HookAborted: If the call is denied, not approved, or cannot be evaluated.
        """
        self._pending.set(None)
        tool_name = "unknown"
        try:
            tool_name = context.tool_name
            if not isinstance(tool_name, str) or not tool_name:
                msg = f"Tool name must be a non-empty string, got {tool_name!r}."
                raise TypeError(msg)
            decision = self.evaluate(tool_name, context.tool_input)
            if decision.verdict is Verdict.ALLOW:
                return
            if decision.verdict is Verdict.DENY:
                reason = decision.reason
            elif self._review(context, decision):
                logger.info("ACGS review approved tool '%s'.", tool_name)
                return
            else:
                reason = f"Human review did not approve the call. {decision.reason}"
        except Exception:  # fail-closed: CrewAI would otherwise ignore the error and run the tool
            logger.warning("ACGS hook failed for tool %r; denying the call (fail-closed).", tool_name, exc_info=True)
            reason = _EVALUATION_FAILED
        self._block(context, str(tool_name), reason)

    def after_tool_call(self, context: ToolCallHookContext) -> str | None:
        """``POST_TOOL_CALL`` hook: give the agent the ACGS reason for a call this hook blocked.

        Args:
            context: The CrewAI tool call context.

        Returns:
            The replacement tool result, or ``None`` to keep CrewAI's result.
        """
        denial = self._pending.get()
        if denial is None:
            return None
        self._pending.set(None)
        try:
            result = context.tool_result
            if (
                context.tool_name == denial.tool_name
                and isinstance(result, str)
                and result.startswith(_BLOCKED_PREFIX)
                and _snapshot(context.tool_input) == denial.tool_input
            ):
                return denial.message
        except Exception:  # the call is already blocked; only the explanation is lost
            logger.warning("ACGS could not attach the denial reason for %r.", denial.tool_name, exc_info=True)
        return None

    @property
    def is_registered(self) -> bool:
        """Whether both hooks are currently registered with CrewAI."""
        return self.before_tool_call in get_hooks(InterceptionPoint.PRE_TOOL_CALL) and (
            self.after_tool_call in get_hooks(InterceptionPoint.POST_TOOL_CALL)
        )

    def register(self) -> Self:
        """Register the hooks with CrewAI for all crews and agents in this process.

        Registering twice has no further effect. Register this hook after any hook that changes
        ``tool_input``: CrewAI runs hooks in registration order, so ACGS then sees the final input.

        Returns:
            This hook, so ``gate = ACGSToolCallHook().register()`` works.
        """
        if self.before_tool_call not in get_hooks(InterceptionPoint.PRE_TOOL_CALL):
            register_hook(InterceptionPoint.PRE_TOOL_CALL, self.before_tool_call)
        if self.after_tool_call not in get_hooks(InterceptionPoint.POST_TOOL_CALL):
            register_hook(InterceptionPoint.POST_TOOL_CALL, self.after_tool_call)
        return self

    def unregister(self) -> None:
        """Remove the hooks from CrewAI. Does nothing if they are not registered."""
        unregister_hook(InterceptionPoint.PRE_TOOL_CALL, self.before_tool_call)
        unregister_hook(InterceptionPoint.POST_TOOL_CALL, self.after_tool_call)

    def __enter__(self) -> Self:
        """Register the hooks for the duration of a ``with`` block."""
        return self.register()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Unregister the hooks."""
        self.unregister()

    def _review(self, context: ToolCallHookContext, decision: GovernanceDecision) -> bool:
        if self.on_review == "deny":
            return False
        if self.on_review == "human_input":
            arguments = json.dumps(context.tool_input, sort_keys=True, default=str)
            response = context.request_human_input(
                prompt=f"ACGS: the tool call '{decision.tool_name}' needs human review. {decision.reason}",
                default_message=f"Arguments: {arguments}\nType 'yes' to run it, anything else to deny:",
            )
            return isinstance(response, str) and response.strip().lower() in _APPROVALS
        return self.on_review(context, decision) is True

    def _block(self, context: ToolCallHookContext, tool_name: str, reason: str) -> None:
        message = f"Tool call '{tool_name}' was not executed. {reason}"
        try:
            snapshot = _snapshot(context.tool_input)
        except Exception:  # the explanation is optional; blocking is not
            snapshot = None
        self._pending.set(_Denial(tool_name=tool_name, tool_input=snapshot, message=message))
        raise HookAborted(reason=reason, source=HOOK_SOURCE)


def _snapshot(tool_input: object) -> dict[str, Any] | None:
    """Shallow copy of the tool input, used to match a blocked call with its ``POST_TOOL_CALL`` context."""
    return dict(tool_input) if isinstance(tool_input, Mapping) else None
