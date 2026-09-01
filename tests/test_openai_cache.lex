# OpenAI's prompt caching is automatic server-side (>=1024 tokens); the one
# client lever is `prompt_cache_key`, which routes repeated same-shape
# traffic to the same cache-warm backend. This adapter also drives OpenCode
# Go, vLLM, MLX, and the LiteLLM proxy (all OpenAI-wire-compatible), so a
# fix here covers "opencode" and "openai" (and the local backends) in one
# place -- same reasoning as the Anthropic breakpoints in
# test_anthropic_cache.lex.

import "../src/providers/openai" as oai

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

fn assert_eq(name :: Str, got :: Str, want :: Str) -> Result[Unit, Str] {
  if got == want {
    Ok(())
  } else {
    Err(str.concat(name, str.concat(" expected ", str.concat(want, str.concat(", got ", got)))))
  }
}

fn build_body(tools :: List[t.Tool]) -> Str {
  let model := prov.gpt4o()
  let messages := [UserMsg("first turn"), AssistantMsg("ok, on it", []), ToolMsg("call_1", "tool result"), UserMsg("second turn")]
  oai.build_request(model, messages, tools)
}

fn test_prompt_cache_key_present_and_deterministic() -> Result[Unit, Str] {
  let tools := [make_tool("read"), make_tool("write")]
  assert_eq("prompt_cache_key value", extract_prompt_cache_key(build_body(tools)), "gpt-4o:read,write")
}

fn test_prompt_cache_key_stable_across_identical_calls() -> Result[Unit, Str] {
  let tools := [make_tool("read"), make_tool("write")]
  let a := extract_prompt_cache_key(build_body(tools))
  let b := extract_prompt_cache_key(build_body(tools))
  assert_eq("same (model, tools) yields the same key on repeat calls -- the whole point, so OpenAI can route them together", a, b)
}

fn test_prompt_cache_key_differs_by_toolset() -> Result[Unit, Str] {
  let build_tools := [make_tool("read"), make_tool("write"), make_tool("bash")]
  let explore_tools := [make_tool("read"), make_tool("grep")]
  let build_key := extract_prompt_cache_key(build_body(build_tools))
  let explore_key := extract_prompt_cache_key(build_body(explore_tools))
  assert_true("build agent and explore agent (different toolsets, same model) get different cache keys, so they don't collide on the same routing shard", not (build_key == explore_key))
}

fn test_prompt_cache_key_present_with_no_tools() -> Result[Unit, Str] {
  assert_eq("no-tool agents (litellm/ollama plan/explore/etc.) still get a stable key from the model alone", extract_prompt_cache_key(build_body([])), "gpt-4o:")
}

fn extract_prompt_cache_key(body :: Str) -> Str {
  match jv.parse_into_errors(body) {
    Err(_) => "<parse error>",
    Ok(j) => match jv.get_field(j, "prompt_cache_key") {
      Some(JStr(s)) => s,
      _ => "<missing>",
    },
  }
}

fn run_all() -> Unit {
  let results := [test_prompt_cache_key_present_and_deterministic(), test_prompt_cache_key_stable_across_identical_calls(), test_prompt_cache_key_differs_by_toolset(), test_prompt_cache_key_present_with_no_tools()]
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

