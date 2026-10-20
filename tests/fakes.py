# SPDX-FileCopyrightText: 2026-present ACGS Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Test doubles. The only double is the model: crews, agents, tasks, tools and hooks are real CrewAI objects."""

from __future__ import annotations

import json
from typing import Any

from crewai.llms.base_llm import BaseLLM
from pydantic import Field


class ScriptedLLM(BaseLLM):
    """Replays scripted replies instead of calling a model, and records the messages it received.

    With ``native=True`` CrewAI uses native tool calling (replies are lists of tool calls);
    otherwise it uses the ReAct text format (replies are ``Action:`` / ``Final Answer:`` text).
    """

    replies: list[Any] = Field(default_factory=list)
    native: bool = False
    received: list[Any] = Field(default_factory=list)

    def call(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        self.received.append(json.dumps(messages, default=str))
        if not self.replies:
            msg = "The scripted model has no replies left."
            raise AssertionError(msg)
        return self.replies.pop(0)

    def supports_function_calling(self) -> bool:
        return self.native

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 8192


def react(tool_name: str, arguments: dict[str, Any]) -> str:
    return f"Thought: I will use a tool.\nAction: {tool_name}\nAction Input: {json.dumps(arguments)}"


def native(*calls: tuple[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"id": f"call-{index}", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
        for index, (name, arguments) in enumerate(calls)
    ]


def final(text: str, *, native_mode: bool) -> str:
    return text if native_mode else f"Thought: I am done.\nFinal Answer: {text}"


def script(native_mode: bool, *calls: tuple[str, dict[str, Any]], answer: str = "done") -> ScriptedLLM:
    """A model that makes ``calls`` one turn at a time, then answers."""
    turns: list[Any] = [native(call) if native_mode else react(*call) for call in calls]
    turns.append(final(answer, native_mode=native_mode))
    return ScriptedLLM(model="scripted", replies=turns, native=native_mode)
