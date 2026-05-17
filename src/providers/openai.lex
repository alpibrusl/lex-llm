# lex-llm — OpenAI Chat Completions adapter
#
# Implements Provider.chat against POST /v1/chat/completions with
# stream:true. SSE line streaming uses http.stream_lines from std.http
# (lex-lang#487), a Rust-backed streaming HTTP POST primitive.
#
# Supports any model accessible via the Chat Completions API:
# GPT-4o, GPT-4o-mini, o1, o3-mini, etc.
# Also works with Azure OpenAI and compatible proxies via base_url.

import "../message"  as msg
import "../delta"    as d
import "../tool"     as t
import "../provider" as prov
import "../sse"      as sse

import "lex-schema/json_value" as jv
import "std.http" as http
import "std.list" as list
import "std.str"  as str
import "std.iter" as iter

let default_base_url := "https://api.openai.com/v1/chat/completions"

type OpenAIConfig = {
  api_key  :: Str,
  base_url :: Str,
}

fn default_config(api_key :: Str) -> OpenAIConfig {
  { api_key: api_key, base_url: default_base_url }
}

fn make_provider(config :: OpenAIConfig) -> prov.Provider {
  {
    name: "openai",
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
  config   :: OpenAIConfig,
  model    :: prov.ModelRef,
  messages :: List[msg.Message],
  tools    :: List[t.Tool]
) -> [net] Iter[d.Delta] {
  let body    := build_request(model, messages, tools)
  let headers := sse.json_post_headers(config.api_key)
  let lines   := match http.stream_lines(config.base_url, headers, body) {
    Err(_)  => iter.from_list([]),
    Ok(it)  => it,
  }
  let payloads := sse.data_payloads(lines)
  let deltas   := list.fold(payloads, [],
    fn (acc :: List[d.Delta], payload :: Str) -> List[d.Delta] {
      match parse_chunk(payload) {
        None    => acc,
        Some(dl) => list.concat(acc, [dl]),
      }
    })
  iter.from_list(deltas)
}

# ---- Request building --------------------------------------------

fn build_request(
  model    :: prov.ModelRef,
  messages :: List[msg.Message],
  tools    :: List[t.Tool]
) -> Str {
  let base := [
    ("model",    JStr(model.model)),
    ("messages", JList(list.map(messages, encode_message))),
    ("stream",   JBool(true)),
  ]
  let with_tools :=
    if list.is_empty(tools) { base }
    else { list.concat(base, [("tools", JList(list.map(tools, t.to_openai_json)))]) }
  jv.stringify(JObj(with_tools))
}

fn encode_message(m :: msg.Message) -> jv.Json {
  match m {
    msg.UserMsg(text) =>
      JObj([("role", JStr("user")), ("content", JStr(text))]),
    msg.SystemMsg(text) =>
      JObj([("role", JStr("system")), ("content", JStr(text))]),
    msg.AssistantMsg(text, calls) =>
      if list.is_empty(calls) {
        JObj([("role", JStr("assistant")), ("content", JStr(text))])
      } else {
        JObj([
          ("role",       JStr("assistant")),
          ("content",    JNull),
          ("tool_calls", JList(list.map(calls, encode_tool_call))),
        ])
      },
    msg.ToolMsg(call_id, content) =>
      JObj([
        ("role",         JStr("tool")),
        ("tool_call_id", JStr(call_id)),
        ("content",      JStr(content)),
      ]),
  }
}

fn encode_tool_call(call :: msg.ToolCall) -> jv.Json {
  JObj([
    ("id",   JStr(call.id)),
    ("type", JStr("function")),
    ("function", JObj([
      ("name",      JStr(call.name)),
      ("arguments", JStr(jv.stringify(call.args))),
    ])),
  ])
}

# ---- SSE chunk parsing -------------------------------------------
#
# OpenAI streaming delta shapes:
#   text:       { "choices": [{ "delta": { "content": "..." } }] }
#   tool call:  { "choices": [{ "delta": { "tool_calls": [{"index":0, "id":"...",
#                  "function": {"name":"...", "arguments":""}}] } }] }
#   finish:     { "choices": [{ "delta": {}, "finish_reason": "stop" }] }

fn parse_chunk(payload :: Str) -> Option[d.Delta] {
  match jv.parse_into_errors(payload) {
    Err(_) => None,
    Ok(j)  => parse_openai_delta(j),
  }
}

fn parse_openai_delta(j :: jv.Json) -> Option[d.Delta] {
  match jv.get_field(j, "choices") {
    None           => None,
    Some(JList(xs)) => match first(xs) {
      None         => None,
      Some(choice) => parse_choice(choice),
    },
    _ => None,
  }
}

fn parse_choice(choice :: jv.Json) -> Option[d.Delta] {
  let finish := match jv.get_field(choice, "finish_reason") {
    Some(JStr(r)) => if r == "null" { None } else { Some(r) },
    _             => None,
  }
  match finish {
    Some(r) => Some(d.FinishDelta(r)),
    None    =>
      match jv.get_field(choice, "delta") {
        None    => None,
        Some(dj) => parse_delta_obj(dj),
      },
  }
}

fn parse_delta_obj(dj :: jv.Json) -> Option[d.Delta] {
  match jv.get_field(dj, "content") {
    Some(JStr(s)) =>
      if str.is_empty(s) { None } else { Some(d.TextChunk(s)) },
    _ =>
      match jv.get_field(dj, "tool_calls") {
        Some(JList(calls)) => parse_tool_call_chunk(calls),
        _                  => None,
      },
  }
}

fn parse_tool_call_chunk(calls :: List[jv.Json]) -> Option[d.Delta] {
  match first(calls) {
    None      => None,
    Some(cj) => {
      let fn_opt   := jv.get_field(cj, "function")
      let id_opt   := jv.get_field(cj, "id")
      match fn_opt {
        None     => None,
        Some(fj) => {
          let name_opt := jv.get_field(fj, "name")
          let args_opt := jv.get_field(fj, "arguments")
          match (id_opt, name_opt) {
            (Some(JStr(id)), Some(JStr(name))) =>
              Some(d.ToolCallBegin(id, name)),
            _ =>
              match (str_field(cj, "id"), args_opt) {
                (id_str, Some(JStr(args)))
                  => if str.is_empty(id_str) { None }
                     else { Some(d.ToolArgChunk(id_str, args)) },
                _ => None,
              },
          }
        },
      }
    },
  }
}

# ---- Helpers -----------------------------------------------------

fn first[T](xs :: List[T]) -> Option[T] {
  list.fold(xs, None, fn (acc :: Option[T], x :: T) -> Option[T] {
    match acc { Some(_) => acc, None => Some(x) }
  })
}

fn str_field(j :: jv.Json, key :: Str) -> Str {
  match jv.get_field(j, key) { Some(JStr(s)) => s, _ => "" }
}
