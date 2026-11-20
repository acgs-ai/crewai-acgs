# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). Entries for new releases are generated with `cz bump`.

## [Unreleased]

## 0.2.0 (unreleased)

### Added

- Offline `examples/deny_and_audit.py`: one allowed and one denied CrewAI tool call, with an application audit trail.
- Regression coverage for JSON-native arguments and bytes/bytearray formatting against acgs-lite 4.0.0.

### Changed

- The locked test environment resolves acgs-lite 4.0.0; the CI `test-min` job keeps testing the 3.0.1 floor.
  Runtime support remains `acgs-lite>=3.0.1,<5` and default compatibility behavior is unchanged.
- Document the compatibility matrix and acgs-lite 4.x production signing and JSON-only payload requirements.

## 0.1.0 (2026-10-20)

### Added

- `ACGSToolCallHook`, which registers CrewAI `PRE_TOOL_CALL` and `POST_TOOL_CALL` hooks that check each tool call
  against an ACGS constitution. Denied calls raise `HookAborted`; the agent receives the ACGS reason as the tool
  result.
- `on_review` (`"deny"`, `"human_input"` or a callable) for calls that only match review rules.
- Fail-closed handling: any error while evaluating or reviewing a call denies it.
- `format_tool_call`, the documented text representation of a tool call that ACGS rules are evaluated against.
