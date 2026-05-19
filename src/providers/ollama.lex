# lex-llm — Ollama local model adapter
#
# Implements Provider.chat against Ollama's /api/chat endpoint.
# No API key required. Streaming response is NDJSON (one JSON per line)
# with a "done" field on the final line.
#
# Compatible with llama3, mistral, qwen2, phi3, gemma, and any other
# model pulled into a local Ollama instance.
# Ollama ≥0.3 also supports function/tool calling in OpenAI format.

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

fn default_base_url() -> Str { "http://localhost:11434/api/chat" }

type OllamaConfig = { base_url :: Str }

fn default_config() -> OllamaConfig
  examples {
    default_config() => { base_url: "http://localhost:11434/api/chat" },
  }
{ { base_url: default_base_url } }

fn make_provider(config :: OllamaConfig) -> prov.Provider {
  {
    name: "ollama",
    chat: fn (
      model    :: prov.ModelRef,
      messages :: List[msg.Message],
      tools    :: List[t.Tool]
    ) -> [net, llm] Iter[d.Delta] {
      chat(config, model, messages, tools)
    },
  }
}

fn chat(
  config   :: OllamaConfig,
  model    :: prov.ModelRef,
  messages :: List[msg.Message],
  tools    :: List[t.Tool]
) -> [net, llm] Iter[d.Delta] {
  let body    := build_request(model, messages, tools)
  let headers := sse.local_post_headers()
  let raw_lines := match http.stream_lines(config.base_url, headers, body) {
    Err(_)  => iter.from_list([]),
    Ok(it)  => it,
  }
  let lines := iter.to_list(raw_lines)
  parse_stream(lines)
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
          ("content",    JStr("")),
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

# ---- Response parsing --------------------------------------------
#
# Ollama NDJSON shape per line:
#   { "model":"llama3", "message": { "role":"assistant", "content":"..." }, "done":false }
# Final line (done:true) may carry tool_calls:
#   { "message": { "role":"assistant", "content":"",
#                  "tool_calls": [{"function":{"name":"...","arguments":{...}}}] },
#     "done": true }

fn parse_stream(lines :: List[Str]) -> Iter[d.Delta] {
  let deltas := list.fold(lines, [],
    fn (acc :: List[d.Delta], line :: Str) -> List[d.Delta] {
      let t := str.trim(line)
      if str.is_empty(t) { acc }
      else { match jv.parse_into_errors(t) {
        Err(_) => acc,
        Ok(j)  => list.concat(acc, parse_chunk(j)),
      } }
    })
  iter.from_list(deltas)
}

fn parse_chunk(j :: jv.Json) -> List[d.Delta] {
  let done := match jv.get_field(j, "done") {
    Some(JBool(b)) => b, _ => false
  }
  let msg_deltas := match jv.get_field(j, "message") {
    None => [],
    Some(mj) =>
      match jv.get_field(mj, "role") {
        Some(JStr("assistant")) => parse_assistant_message(mj),
        _                       => [],
      },
  }
  let finish_deltas :=
    if done {
      let reason := finish_reason_from_msg(j)
      [d.FinishDelta(reason)]
    } else { [] }
  list.concat(msg_deltas, finish_deltas)
}

fn parse_assistant_message(mj :: jv.Json) -> List[d.Delta] {
  let text_deltas := match jv.get_field(mj, "content") {
    Some(JStr(s)) => if str.is_empty(s) { [] } else { [d.TextChunk(s)] },
    _             => [],
  }
  let call_deltas := match jv.get_field(mj, "tool_calls") {
    Some(JList(calls)) => parse_tool_calls(calls),
    _                  => [],
  }
  list.concat(text_deltas, call_deltas)
}

fn parse_tool_calls(calls :: List[jv.Json]) -> List[d.Delta] {
  list.fold(calls, [],
    fn (acc :: List[d.Delta], cj :: jv.Json) -> List[d.Delta] {
      match jv.get_field(cj, "function") {
        None => acc,
        Some(fj) => {
          let name := match jv.get_field(fj, "name") { Some(JStr(s)) => s, _ => "" }
          let args := match jv.get_field(fj, "arguments") {
            Some(aj) => jv.stringify(aj),
            None     => "{}",
          }
          let id := str.concat("call_", name)
          list.concat(acc, [d.ToolCallBegin(id, name), d.ToolArgChunk(id, args)])
        },
      }
    })
}

fn finish_reason_from_msg(j :: jv.Json) -> Str {
  match jv.get_field(j, "message") {
    Some(mj) =>
      match jv.get_field(mj, "tool_calls") {
        Some(JList(calls)) =>
          if list.is_empty(calls) { "stop" } else { "tool_calls" },
        _ => "stop",
      },
    _ => "stop",
  }
}
