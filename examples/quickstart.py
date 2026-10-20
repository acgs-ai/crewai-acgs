"""Gate CrewAI tool calls with ACGS. Runs offline: a scripted model stands in for the LLM."""

import json
from typing import Any

from acgs_lite import Constitution, Rule, Severity, ViolationAction
from crewai import Agent, Crew, Task
from crewai.hooks import ToolCallHookContext
from crewai.llms.base_llm import BaseLLM
from crewai.tools import tool

from crewai_acgs import ACGSToolCallHook, GovernanceDecision


class ScriptedLLM(BaseLLM):
    """Replays scripted tool calls, then answers. Prints each tool result the agent receives."""

    replies: list[Any]

    def call(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        tool_results = [m for m in messages if isinstance(m, dict) and m.get("role") == "tool"]
        if tool_results:
            print(f"{tool_results[-1].get('name')}: {tool_results[-1]['content']}")
        return self.replies.pop(0)

    def supports_function_calling(self) -> bool:
        return True


def call(name: str, **arguments: Any) -> list[dict[str, Any]]:
    return [{"id": f"call-{name}", "function": {"name": name, "arguments": json.dumps(arguments)}}]


@tool("send_email")
def send_email(to: str, body: str) -> str:
    """Send an email."""
    return f"Email sent to {to}"


@tool("get_weather")
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Sunny in {city}"


@tool("issue_refund")
def issue_refund(order_id: str, amount: float) -> str:
    """Refund an order."""
    return f"Refunded {amount} for {order_id}"


def run(llm: ScriptedLLM, tools: list[Any]) -> None:
    agent = Agent(role="Assistant", goal="Help the user", backstory="Uses tools when needed.", llm=llm, tools=tools)
    task = Task(description="Help the user.", expected_output="A short answer.", agent=agent)
    Crew(agents=[agent], tasks=[task]).kickoff()


# 1. The default constitution blocks the email that leaks a password; the weather lookup runs.
model = ScriptedLLM(
    model="scripted",
    replies=[
        call("send_email", to="ops@example.com", body="The admin password is hunter2"),
        call("get_weather", city="Paris"),
        "Done.",
    ],
)
with ACGSToolCallHook():
    run(model, [send_email, get_weather])


# 2. A review rule: refunds need approval. Here a function approves small refunds only.
refunds = Constitution.from_rules(
    [
        Rule(
            id="REFUND-REVIEW",
            text="Refunds need human approval",
            severity=Severity.HIGH,
            keywords=["refund"],
            workflow_action=ViolationAction.REQUIRE_HUMAN_REVIEW,
        )
    ],
    name="refunds",
)


def approve_small_refunds(context: ToolCallHookContext, decision: GovernanceDecision) -> bool:
    return float(context.tool_input["amount"]) <= 50


model = ScriptedLLM(
    model="scripted",
    replies=[
        call("issue_refund", order_id="A-17", amount=20),
        call("issue_refund", order_id="A-18", amount=500),
        "Done.",
    ],
)
with ACGSToolCallHook(constitution=refunds, on_review=approve_small_refunds):
    run(model, [issue_refund])
