# crewai-acgs

[![CI](https://github.com/acgs-ai/crewai-acgs/actions/workflows/ci.yml/badge.svg)](https://github.com/acgs-ai/crewai-acgs/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/crewai-acgs.svg)](https://pypi.org/project/crewai-acgs/)
[![Python](https://img.shields.io/pypi/pyversions/crewai-acgs.svg)](https://pypi.org/project/crewai-acgs/)

`crewai-acgs` checks every CrewAI tool call against [ACGS](https://pypi.org/project/acgs-lite/) constitution
rules before the tool runs. It uses CrewAI's own
[tool call hooks](https://docs.crewai.com/v1.15.28/en/learn/tool-hooks) (`InterceptionPoint.PRE_TOOL_CALL` and
`POST_TOOL_CALL`), so agents, tasks, tools and crews stay unchanged.

- Calls that match no blocking rule run unchanged.
- Calls that match a blocking rule do not run. The hook raises `HookAborted`, CrewAI skips the tool, and the agent
  receives the ACGS reason as the tool result and continues.
- Calls that only match review rules are denied by default, or sent to a person (`on_review="human_input"`) or to
  your own approval function.
- Any error while evaluating or reviewing a call denies it (fail-closed).

The package is a thin adapter over the public `acgs-lite` API.

## Installation

```bash
pip install crewai-acgs
```

Requires Python 3.10 to 3.13, `crewai>=1.15.3` (the first release with `HookAborted` and the `@on` hook API) and
`acgs-lite>=3.0.1`.

## Usage

```python
from crewai_acgs import ACGSToolCallHook

gate = ACGSToolCallHook().register()  # gates every tool call in this process
# ... crew.kickoff() ...
gate.unregister()
```

`ACGSToolCallHook` also works as a context manager (`with ACGSToolCallHook(): crew.kickoff()`), which registers the
hooks on entry and removes them on exit.

This complete example runs without an API key; a scripted model stands in for the LLM and prints each tool result
the agent receives:

```python
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
```

Output of `python examples/quickstart.py` (on its first run in a project, CrewAI may also print its own tracing
notice):

```text
send_email: Tool call 'send_email' was not executed. Blocked by ACGS rule ACGS-006 (Agents must not expose sensitive data in responses).
get_weather: Sunny in Paris
issue_refund: Refunded 20.0 for A-17
issue_refund: Tool call 'issue_refund' was not executed. Human review did not approve the call. Human review required by ACGS rule REFUND-REVIEW (Refunds need human approval).
```

## How a tool call is evaluated

1. `ToolCallHookContext.tool_name` and `tool_input` are rendered as text by `format_tool_call`:

   ```python
   from crewai_acgs import format_tool_call

   print(format_tool_call("update_config", {"service": "billing", "change": "disable\nlogging", "dryRun": False}))
   ```

   ```text
   tool: update_config
   arguments:
   change: disable logging
   dryRun: false
   service: billing
   normalized: update config | dry run
   ```

   Each argument leaf is one `path: value` line (nested values use `a.b[0].c` paths), so both argument names and
   values are visible to rules. Whitespace inside values is collapsed so phrases split across lines still match.
   Identifiers written in `snake_case`, `kebab-case` or `camelCase` (the tool name, argument names, and
   single-token string values) are repeated as plain words on the `normalized:` line, so the keyword
   `skip audit` also matches a tool called `skip_audit`. Nothing is truncated.
   Pass `tool_call_formatter=` to use a different representation.

2. The text is validated with `acgs_lite.GovernanceEngine.validate` against the constitution
   (`Constitution.default()` unless you pass one).

3. The result decides what happens to the call:

   | ACGS outcome | Result |
   |---|---|
   | No blocking rule matched (warn-only rules may match) | The tool runs |
   | Only `require_human_review` / `escalate_to_senior` rules matched | Review (see below); denied by default |
   | Any `block`, `block_and_notify` or `halt_and_alert` rule matched | Denied |
   | A rule ID not found in the constitution, or an invalid result without a rule | Denied |
   | Any exception while reading, evaluating or reviewing the tool call | Denied (fail-closed) |

   A matched blocking rule is never downgraded to review because a review rule matched as well.
   The denial reason then lists only the blocking rules after "Blocked by" and names the matched review
   rules separately.

A denied call raises `HookAborted(reason, source="crewai-acgs")`, so CrewAI does not run the tool and records the
reason and source on its `HookDispatchedEvent`. CrewAI then gives the agent the generic result
`"Tool execution blocked by hook. Tool: <name>"`; the hook's `POST_TOOL_CALL` half replaces it with
`"Tool call '<name>' was not executed. <reason>"` so the model can adjust. The reason names the matched rules but
never repeats argument values. Exceptions raised by the tool itself are reported by CrewAI as usual.

CrewAI ignores exceptions other than `HookAborted` that a hook raises and then runs the tool anyway. The hook
therefore catches every error itself and turns it into a denial.

`ACGSToolCallHook.evaluate(tool_name, args)` returns the underlying `GovernanceDecision`
(`verdict`, `reason`, `rule_ids`, `warning_rule_ids`, `error`) without running anything.

## Review

CrewAI has no built-in way to pause a run and resume it later after an approval. The `on_review` option decides
what happens to calls that only match review rules:

| `on_review` | Behaviour |
|---|---|
| `"deny"` (default) | The call is denied with the review reason. Works in unattended runs. |
| `"human_input"` | Asks in the terminal with CrewAI's `ToolCallHookContext.request_human_input`, showing the reason and the arguments. Only `y` or `yes` (any case) runs the call; any other answer denies it. If no terminal is available (`EOFError`), the call is denied. |
| a callable `(context, decision) -> bool` | Your function decides, for example by calling an approval service. Only a return value of exactly `True` runs the call; other values and exceptions deny it. |

```python
from crewai_acgs import ACGSToolCallHook

terminal_review = ACGSToolCallHook(on_review="human_input")
own_review = ACGSToolCallHook(on_review=lambda context, decision: context.tool_name == "issue_refund")
```

`"human_input"` blocks the tool call until someone answers, so use it only for interactive runs. Hooks run
synchronously, including inside `kickoff_async`.

## Registration

CrewAI tool call hooks are global to the Python process: once registered, the hook gates every crew and every
agent until it is removed. `register()` adds both hooks (calling it again has no effect), `unregister()` removes
them, and `is_registered` reports whether both are active. `crewai.hooks.clear_all_hooks()` removes them too.

CrewAI runs hooks in registration order and stops at the first one that blocks. Register `ACGSToolCallHook` after
any hook that changes `tool_input`, so that ACGS evaluates the input the tool actually receives. A hook registered
later that changes `tool_input` would run after ACGS has allowed the call.

Several instances can be registered at the same time (for example a general and a domain-specific constitution).
A call runs only if every instance allows it.

## Custom constitutions

```python
from acgs_lite import Constitution, Rule, Severity, ViolationAction

from crewai_acgs import ACGSToolCallHook

constitution = Constitution.from_rules(
    [
        Rule(
            id="NO-DESTRUCTIVE-SQL",
            text="No destructive SQL",
            severity=Severity.CRITICAL,
            keywords=["drop table", "truncate table"],
        ),
        Rule(
            id="REFUND-REVIEW",
            text="Refunds need approval",
            severity=Severity.HIGH,
            keywords=["refund"],
            workflow_action=ViolationAction.REQUIRE_HUMAN_REVIEW,
        ),
    ],
    name="payments",
)
gate = ACGSToolCallHook(constitution=constitution, agent_id="payments-crew")
```

`Constitution.from_yaml("rules.yaml")` works as well.

## Options

| Argument | Default | Meaning |
|---|---|---|
| `constitution` | `Constitution.default()` | Rules to enforce |
| `agent_id` | `"crewai-agent"` | Agent identifier recorded by ACGS |
| `on_review` | `"deny"` | `"deny"`, `"human_input"`, or a callable `(context, decision) -> bool` |
| `tool_call_formatter` | `format_tool_call` | `(tool_name, args) -> str` used as the evaluated text |

## Limitations

- Only tool calls are checked. Model input and output, and tool results, are not.
- Tested with `Crew.kickoff` (ReAct text loop and native tool calling, including parallel native calls),
  `Crew.kickoff_async` and `Agent.kickoff`, against crewai 1.15.3 and the latest release.
- `Constitution.default()` focuses on governance integrity (self-approval, skipping audits, privilege escalation)
  and on secrets and personal data (`password`, `api_key`, `sk-...` and `ghp_...` tokens, US SSN
  patterns). It does **not** cover destructive operations such as `rm -rf` or `DROP TABLE`. Write rules for the
  tools you expose. `tests/test_default_constitution.py` lists what the default rules do and do not catch.
- Keyword rules in acgs-lite 3.0.1 match literal, case-insensitive substrings. They do not understand negation
  ("never share the password" matches `password`) and do not treat `-`, `_` and spaces as equivalent
  (`api-key` does not match the keyword `api_key`).
- Because argument names are part of the evaluated text, a tool with a parameter named `password` is denied under
  the default constitution whenever it is called.

## Development

```bash
uv sync --group dev        # or use hatch: hatch run prepare
hatch run format           # ruff format + ruff check --fix
hatch run lint             # ruff check + ruff format --check
hatch run typecheck        # mypy --strict
hatch run cov              # pytest with a 100% branch-coverage gate, network disabled
hatch run min:cov          # the same tests against crewai 1.15.3 and acgs-lite 3.0.1
pre-commit install         # format, lint, type check and tests on commit; commitizen on commit messages
```

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/); `cz bump` updates the version
and the changelog.

## License

Apache-2.0. See [LICENSE](LICENSE).
