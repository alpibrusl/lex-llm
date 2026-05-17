# lex-llm — Anthropic Messages API adapter
#
# Implements Provider.chat against POST /v1/messages with stream:true.
# Anthropic's SSE format uses typed event: fields; we track block state
# in a fold to route input_json_delta chunks to the right call_id.
#
# Supported: claude-opus-4-7, claude-sonnet-4-6, claude-haiku-4-5-*

import "../message"  as msg
import "../delta"    as d
import "../tool"     as t
import "../provider" as prov
import "../sse"      as sse

import "lex-schema/json_value" as jv
import "std.list" as list
import "std.str"  as str
import "std.iter" as iter
import "std.map"  as map

let default_base_url  := "https://api.anthropic.com/v1/messages"
let api_version       := "2023-06-01"

type AnthropicConfig = {
  api_key  :: Str,
  base_url :: Str,
}

fn default_config(api_key :: Str) -> AnthropicConfig {
  { api_key: api_key, base_url: default_base_url }
}

fn make_provider(config :: AnthropicConfig) -> prov.Provider {
  {
    name: "anthropic",
    chat: fn (
      model    :: prov.ModelRef,
      messages :: List[msg.Message],
      tools    :: List[t.Tool]
    ) -> [net] Iter[d.Delta] {
      chat(config, model, messages, tools)
    },
  }
}

fn chat(
  config   :: AnthropicConfig,
  model    :: prov.ModelRef,
  messages :: List[msg.Message],
  tools    :: List[t.Tool]
) -> [net] Iter[d.Delta] {
  let (sys, user_msgs) := split_system(messages)
  let body    := build_request(model, sys, user_msgs, tools)
  let headers := build_headers(config.api_key)
  let payloads := sse.data_payloads(
    sse.stream_lines(config.base_url, headers, body))
  parse_stream(payloads)
}

# ---- Request building --------------------------------------------

fn split_system(
  messages :: List[msg.Message]
) -> (Str, List[msg.Message]) {
  let sys := list.fold(messages, "",
    fn (acc :: Str, m :: msg.Message) -> Str {
      match m { msg.SystemMsg(s) => s, _ => acc }
    })
  let rest := list.fold(messages, [],
    fn (acc :: List[msg.Message], m :: msg.Message) -> List[msg.Message] {
      match m { msg.SystemMsg(_) => acc, _ => list.concat(acc, [m]) }
    })
  (sys, rest)
}

fn build_request(
  model    :: prov.ModelRef,
  sys      :: Str,
  messages :: List[msg.Message],
  tools    :: List[t.Tool]
) -> Str {
  let base := [
    ("model",      JStr(model.model)),
    ("max_tokens", JInt(4096)),
    ("stream",     JBool(true)),
    ("messages",   JList(list.map(messages, encode_message))),
  ]
  let with_sys :=
    if str.is_empty(sys) { base }
    else { list.concat(base, [("system", JStr(sys))]) }
  let with_tools :=
    if list.is_empty(tools) { with_sys }
    else { list.concat(with_sys, [("tools", JList(list.map(tools, t.to_anthropic_json)))]) }
  jv.stringify(JObj(with_tools))
}

fn encode_message(m :: msg.Message) -> jv.Json {
  match m {
    msg.UserMsg(text) =>
      JObj([("role", JStr("user")), ("content", JStr(text))]),
    msg.AssistantMsg(text, calls) =>
      if list.is_empty(calls) {
        JObj([("role", JStr("assistant")), ("content", JStr(text))])
      } else {
        JObj([
          ("role", JStr("assistant")),
          ("content", JList(list.map(calls, encode_tool_use_block))),
        ])
      },
    msg.ToolMsg(call_id, content) =>
      JObj([
        ("role", JStr("user")),
        ("content", JList([JObj([
          ("type",        JStr("tool_result")),
          ("tool_use_id", JStr(call_id)),
          ("content",     JStr(content)),
        ])])),
      ]),
    msg.SystemMsg(_) =>
      JObj([("role", JStr("user")), ("content", JStr(""))]),
  }
}

fn encode_tool_use_block(call :: msg.ToolCall) -> jv.Json {
  JObj([
    ("type",  JStr("tool_use")),
    ("id",    JStr(call.id)),
    ("name",  JStr(call.name)),
    ("input", call.args),
  ])
}

fn build_headers(api_key :: Str) -> Map[Str, Str] {
  map.from_list([
    ("x-api-key",         api_key),
    ("anthropic-version", api_version),
    ("content-type",      "application/json"),
    ("accept",            "text/event-stream"),
  ])
}

# ---- SSE stream parsing ------------------------------------------
#
# Relevant Anthropic event types:
#   content_block_start  — starts a text or tool_use block
#   content_block_delta  — text_delta or input_json_delta
#   message_delta        — carries stop_reason at end
#
# We track current block type + tool id in fold state.

type ParseState = {
  block_type :: Str,    # "text" | "tool_use" | ""
  tool_id    :: Str,
  tool_name  :: Str,
}

fn parse_stream(payloads :: List[Str]) -> [net] Iter[d.Delta] {
  let init := { block_type: "", tool_id: "", tool_name: "" }
  let (_, deltas) := list.fold(payloads, (init, []),
    fn (
      acc     :: (ParseState, List[d.Delta]),
      payload :: Str
    ) -> (ParseState, List[d.Delta]) {
      let state  := match acc { (s, _) => s }
      let so_far := match acc { (_, ds) => ds }
      match jv.parse_into_errors(payload) {
        Err(_) => acc,
        Ok(j)  => {
          let (new_state, new_deltas) := handle_event(state, j)
          (new_state, list.concat(so_far, new_deltas))
        },
      }
    })
  iter.from_list(deltas)
}

fn handle_event(
  state :: ParseState,
  j     :: jv.Json
) -> (ParseState, List[d.Delta]) {
  match jv.get_field(j, "type") {
    Some(JStr(t)) =>
      match t {
        "content_block_start" => handle_block_start(state, j),
        "content_block_delta" => handle_block_delta(state, j),
        "message_delta"       => handle_message_delta(state, j),
        _                     => (state, []),
      },
    _ => (state, []),
  }
}

fn handle_block_start(
  state :: ParseState,
  j     :: jv.Json
) -> (ParseState, List[d.Delta]) {
  match jv.get_field(j, "content_block") {
    None => (state, []),
    Some(block) =>
      match jv.get_field(block, "type") {
        Some(JStr("tool_use")) => {
          let id   := str_field(block, "id")
          let name := str_field(block, "name")
          ({ block_type: "tool_use", tool_id: id, tool_name: name },
           [d.ToolCallBegin(id, name)])
        },
        Some(JStr("text")) =>
          ({ block_type: "text", tool_id: state.tool_id, tool_name: state.tool_name },
           []),
        _ => (state, []),
      },
  }
}

fn handle_block_delta(
  state :: ParseState,
  j     :: jv.Json
) -> (ParseState, List[d.Delta]) {
  match jv.get_field(j, "delta") {
    None => (state, []),
    Some(delta) =>
      match jv.get_field(delta, "type") {
        Some(JStr("text_delta")) => {
          let text := str_field(delta, "text")
          if str.is_empty(text) { (state, []) }
          else { (state, [d.TextChunk(text)]) }
        },
        Some(JStr("input_json_delta")) => {
          let chunk := str_field(delta, "partial_json")
          if str.is_empty(chunk) { (state, []) }
          else { (state, [d.ToolArgChunk(state.tool_id, chunk)]) }
        },
        _ => (state, []),
      },
  }
}

fn handle_message_delta(
  state :: ParseState,
  j     :: jv.Json
) -> (ParseState, List[d.Delta]) {
  match jv.get_field(j, "delta") {
    None => (state, []),
    Some(delta) =>
      match jv.get_field(delta, "stop_reason") {
        Some(JStr(reason)) =>
          (state, [d.FinishDelta(normalise_finish(reason))]),
        _ => (state, []),
      },
  }
}

fn normalise_finish(reason :: Str) -> Str
  examples {
    normalise_finish("end_turn")   => "stop",
    normalise_finish("tool_use")   => "tool_calls",
    normalise_finish("max_tokens") => "length",
  }
{
  match reason {
    "end_turn"   => "stop",
    "tool_use"   => "tool_calls",
    "max_tokens" => "length",
    other        => other,
  }
}

fn str_field(j :: jv.Json, key :: Str) -> Str {
  match jv.get_field(j, key) { Some(JStr(s)) => s, _ => "" }
}
