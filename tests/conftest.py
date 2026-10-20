# SPDX-FileCopyrightText: 2026-present ACGS Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Shared test setup: no telemetry, and a clean CrewAI hook registry for every test."""

from __future__ import annotations

import os

# CREWAI_TESTING stops CrewAI from collecting and announcing first-run traces.
os.environ.setdefault("CREWAI_TESTING", "true")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_DISABLE_TRACKING", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from collections.abc import Iterator

import pytest
from crewai.hooks import clear_all_hooks


@pytest.fixture(autouse=True)
def _clean_hooks() -> Iterator[None]:
    clear_all_hooks()
    yield
    clear_all_hooks()
