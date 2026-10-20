# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). Entries for new releases are generated with `cz bump`.

## [Unreleased]

## 0.1.0 (unreleased)

### Added

- `ACGSToolCallHook`, which registers CrewAI `PRE_TOOL_CALL` and `POST_TOOL_CALL` hooks that check each tool call
  against an ACGS constitution. Denied calls raise `HookAborted`; the agent receives the ACGS reason as the tool
  result.
- `on_review` (`"deny"`, `"human_input"` or a callable) for calls that only match review rules.
- Fail-closed handling: any error while evaluating or reviewing a call denies it.
- `format_tool_call`, the documented text representation of a tool call that ACGS rules are evaluated against.
