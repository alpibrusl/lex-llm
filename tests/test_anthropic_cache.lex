# Prompt caching (lex-code review, Aug 2026) — the Anthropic adapter must mark
# exactly three cache_control breakpoints per request: the tools array (last
# tool only), the system block, and the last message only. Marking every
# block would blow past Anthropic's 4-breakpoint cap and defeat the point;
# marking none means every agent-loop step re-sends and re-bills the full
# system prompt + tool schemas + history from scratch.

import "../src/providers/anthropic" as anth

import "../src/provider" as prov

import "../src/tool" as t

import "lex-schema/json_value" as jv

import "lex-schema/schema" as s

import "lex-schema/error" as e

import "std.str" as str

import "std.list" as list

fn noop_execute(args :: jv.Json) -> [net, io, proc] Result[jv.Json, e.Errors] {
  Ok(args)
}

fn make_tool(name :: Str) -> t.Tool {
  t.define(name, str.concat("test tool ", name), { title: name, description: "", fields: [s.required_str("x", [])] }, noop_execute)
}

fn assert_true(name :: Str, got :: Bool) -> Result[Unit, Str] {
  if got {
    Ok(())
  } else {
    Err(str.concat(name, " expected true, got false"))
  }
}

# (parts - 1) occurrences of an exact-literal needle in haystack.
fn occurrences(haystack :: Str, needle :: Str) -> Int {
  list.len(str.split(haystack, needle)) - 1
}

fn build_body() -> Str {
  let model := prov.claude_sonnet()
  let tools := [make_tool("read"), make_tool("write"), make_tool("bash")]
  let messages := [UserMsg("first turn"), AssistantMsg("ok, on it", []), ToolMsg("call_1", "tool result"), UserMsg("second turn")]
  anth.build_request(model, "you are a build agent", messages, tools)
}

# Exactly 3 breakpoints total: tools (1) + system (1) + last-message (1).
# Anthropic caps at 4/request -- this proves we aren't marking every block
# (which would both blow the cap on longer histories and pay the ~25%
# cache-write premium on content that never gets reused).
fn test_exactly_three_cache_breakpoints() -> Result[Unit, Str] {
  let body := build_body()
  assert_true("exactly 3 cache_control breakpoints", occurrences(body, "\"cache_control\":") == 3)
}

fn test_system_block_is_cached() -> Result[Unit, Str] {
  let body := build_body()
  assert_true("system block carries cache_control", str.contains(body, "\"system\":[{\"type\":\"text\",\"text\":\"you are a build agent\",\"cache_control\":{\"type\":\"ephemeral\"}}]"))
}

# Only the LAST tool ("bash") is marked -- Anthropic caches the prefix up to
# and including the marked block, so marking the last covers all three.
fn test_only_last_tool_is_cached() -> Result[Unit, Str] {
  let body := build_body()
  let bash_cached := str.contains(body, "\"name\":\"bash\",\"description\":\"test tool bash\",\"input_schema\":{\"$schema\":\"https://json-schema.org/draft/2020-12/schema\",\"title\":\"bash\",\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"string\"}},\"required\":[\"x\"]},\"cache_control\":{\"type\":\"ephemeral\"}}")
  let read_uncached := str.contains(body, "\"name\":\"read\",\"description\":\"test tool read\",\"input_schema\":{\"$schema\":\"https://json-schema.org/draft/2020-12/schema\",\"title\":\"read\",\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"string\"}},\"required\":[\"x\"]}}")
  match assert_true("bash (last tool) carries cache_control", bash_cached) {
    Err(e) => Err(e),
    Ok(_) => assert_true("read (not last) has no cache_control on its own object", read_uncached),
  }
}

# Only the LAST message ("second turn") is marked; the three earlier
# messages (first turn / assistant / tool result) must NOT carry
# cache_control -- only the growing history's current tail should, so
# each successive turn reuses the previous turn's cached prefix.
fn test_only_last_message_is_cached() -> Result[Unit, Str] {
  let body := build_body()
  let last_cached := str.contains(body, "\"text\":\"second turn\",\"cache_control\":{\"type\":\"ephemeral\"}")
  let first_uncached := str.contains(body, "\"text\":\"first turn\"}")
  match assert_true("second turn (last message) carries cache_control", last_cached) {
    Err(e) => Err(e),
    Ok(_) => assert_true("first turn (not last) has no cache_control on its own block", first_uncached),
  }
}

fn run_all() -> Unit {
  let results := [test_exactly_three_cache_breakpoints(), test_system_block_is_cached(), test_only_last_tool_is_cached(), test_only_last_message_is_cached()]
  let failures := list.fold(results, [], fn (acc :: List[Str], r :: Result[Unit, Str]) -> List[Str] {
    match r {
      Ok(_) => acc,
      Err(e) => list.concat(acc, [e]),
    }
  })
  if list.is_empty(failures) {
    ()
  } else {
    let __discard := 1 / 0
    ()
  }
}

