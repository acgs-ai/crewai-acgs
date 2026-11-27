# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). Entries for new releases are generated with `cz bump`.

## [Unreleased]

### Security

- A control character that Python treats as whitespace (vertical tab, form feed, U+001C to U+001F, NEL) inside
  a tool-call argument value was rendered as a space. Inside a word it split the word in two, so
  `del<VT>ete` did not match a `delete` keyword and the call was allowed. A value that contains control
  characters is now also rendered with them removed and with them as spaces, and rules are matched against
  every rendering. This also applies to identifier values such as `skip<NUL>_audit`.
- **Behaviour change:** a value with more than one of these whitespace control characters is denied, because
  each one could be a separator or sit inside a word. Pass your own `tool_call_formatter` if your tools
  legitimately receive such values.
- Checking whether an argument value is a single identifier no longer takes exponential time on crafted
  values.

Not covered:

- A value with several non-whitespace control characters (for example NUL or ESC) where one sits inside a
  word and another separates words, such as `del<NUL>ete<SOH>everything`. Such values are not denied, because
  text with terminal colour codes contains many of these characters. The printable part of such a code, for
  example `[0m`, also stays in the text.
- Other characters that Python treats as whitespace inside a word, such as no-break space or U+2028.
- Invisible format characters such as zero-width space, which acgs-lite 3.x does not remove.
- Control characters in tool names and argument names, and values whose `str()` escapes control characters
  (for example `bytes`).

## 0.2.0 (2026-11-20)

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
