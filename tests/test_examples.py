# SPDX-FileCopyrightText: 2026-present ACGS Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""The README only shows code that runs, and the quickstart output it shows is the real output."""

from __future__ import annotations

import json
import os
import re
import runpy
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
QUICKSTART = ROOT / "examples" / "quickstart.py"


def _blocks(language: str) -> list[str]:
    return [textwrap.dedent(block) for block in re.findall(rf"```{language}\n(.*?)```", README, re.DOTALL)]


def test_quickstart_output_matches_readme() -> None:
    env = {**os.environ, "CREWAI_TESTING": "true", "CREWAI_DISABLE_TELEMETRY": "true", "OTEL_SDK_DISABLED": "true"}
    completed = subprocess.run(  # noqa: S603
        [sys.executable, str(QUICKSTART)], capture_output=True, text=True, check=True, cwd=ROOT, env=env, timeout=300
    )
    [expected] = [block for block in _blocks("text") if block.startswith("send_email:")]
    assert completed.stdout == expected


def test_readme_embeds_the_quickstart_verbatim() -> None:
    assert QUICKSTART.read_text(encoding="utf-8") in _blocks("python")


def test_deny_and_audit_runs_offline_without_api_keys(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for key in list(os.environ):
        if key.endswith("API_KEY"):
            monkeypatch.delenv(key)
    # The example sets these at import time; registering them here makes monkeypatch restore them.
    for key, value in {
        "CREWAI_TESTING": "true",
        "CREWAI_DISABLE_TELEMETRY": "true",
        "CREWAI_DISABLE_TRACKING": "true",
        "CREWAI_TRACING_ENABLED": "false",
        "OTEL_SDK_DISABLED": "true",
    }.items():
        monkeypatch.setenv(key, value)
    # pytest-socket disables network access for this in-process example execution.
    runpy.run_path(str(ROOT / "examples" / "deny_and_audit.py"), run_name="__main__")
    output = capsys.readouterr().out
    execution, audit = output.split("Audit trail (application decisions):\n")
    assert execution == "Executed record_note: The meeting starts at noon.\n"
    decisions = [json.loads(line) for line in audit.splitlines()]
    assert [decision["verdict"] for decision in decisions] == ["allow", "deny"]
    assert [decision["tool_name"] for decision in decisions] == ["record_note", "record_note"]
    assert decisions[0]["rule_ids"] == []
    assert decisions[1]["rule_ids"] == ["ACGS-006"]
    assert "Blocked by" in decisions[1]["reason"]
    assert "demo-secret" not in output


def test_every_readme_python_block_runs(capsys: pytest.CaptureFixture[str]) -> None:
    blocks = _blocks("python")
    assert len(blocks) == 5
    for block in blocks:
        exec(compile(block, "<README>", "exec", dont_inherit=True), {"__name__": "readme"})  # noqa: S102
    output = capsys.readouterr().out
    assert "normalized: update config | dry run" in output
    assert "send_email: Tool call 'send_email' was not executed." in output
