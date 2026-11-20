"""Run one allowed and one denied CrewAI tool call offline, then print a decision audit trail."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any

# Disable CrewAI tracing/telemetry before importing it; no model service is used.
os.environ["CREWAI_TESTING"] = "true"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["CREWAI_DISABLE_TRACKING"] = "true"
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"

from crewai import Agent, Crew, Task
from crewai.llms.base_llm import BaseLLM
from crewai.tools import tool

from crewai_acgs import ACGSToolCallHook, GovernanceDecision


class AuditedHook(ACGSToolCallHook):
    """Collect decisions through the public hook API, without retaining argument values."""

    def __init__(self) -> None:
        super().__init__()
        self.decisions: list[GovernanceDecision] = []

    def evaluate(self, tool_name: str, arguments: Any) -> GovernanceDecision:
        decision = super().evaluate(tool_name, arguments)
        self.decisions.append(decision)
        return decision


class ScriptedLLM(BaseLLM):
    """Return prewritten tool calls without an API key or network request."""

    replies: list[Any]

    def call(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        return self.replies.pop(0)

    def supports_function_calling(self) -> bool:
        return True


@tool("record_note")
def record_note(body: str) -> str:
    """Record a demo note by printing it locally."""
    print(f"Executed record_note: {body}")
    return "Note recorded."


def main() -> None:
    replies: list[Any] = [
        [
            {
                "id": f"note-{index}",
                "function": {"name": "record_note", "arguments": json.dumps({"body": body})},
            }
        ]
        for index, body in enumerate(["The meeting starts at noon.", "The admin password is demo-secret."])
    ]
    replies.append("Done.")
    agent = Agent(
        role="Note taker",
        goal="Record notes safely",
        backstory="Uses a local note tool.",
        llm=ScriptedLLM(model="scripted", replies=replies),
        tools=[record_note],
    )
    task = Task(description="Record the two notes.", expected_output="Done.", agent=agent)
    with AuditedHook() as gate:
        Crew(agents=[agent], tasks=[task]).kickoff()
    print("Audit trail (application decisions):")
    for decision in gate.decisions:
        print(json.dumps(asdict(decision), sort_keys=True))


if __name__ == "__main__":
    main()
