# lex-llm — Anthropic Messages API adapter
#
# Implements Provider.chat against POST /v1/messages with stream:true.
# Anthropic's SSE format uses typed event: fields; we track block state
# in a fold to route input_json_delta chunks to the right call_id.
#
# Supported: claude-opus-5, claude-sonnet-5, claude-haiku-4-5-*

import "../message" as msg

import "../delta" as d

import "../tool" as t

import "../provider" as prov

import "../sse" as sse

import "lex-schema/json_value" as jv

import "std.http" as http

import "std.bytes" as bytes

import "std.list" as list

import "std.str" as str

import "std.iter" as iter

import "std.map" as map

fn default_base_url() -> Str {
  "https://api.anthropic.com/v1/messages"
}

fn api_version() -> Str {
  "2023-06-01"
}

type AnthropicConfig = { api_key :: Str, base_url :: Str }

fn default_config(api_key :: Str) -> AnthropicConfig {
  { api_key: api_key, base_url: default_base_url() }
}

fn make_provider(config :: AnthropicConfig) -> prov.Provider {
  { name: "anthropic", chat: fn (model :: prov.ModelRef, messages :: List[msg.Message], tools :: List[t.Tool]) -> [net, llm] Iter[d.Delta] {
    chat(config, model, messages, tools)
  } }
}

fn chat(config :: AnthropicConfig, model :: prov.ModelRef, messages :: List[msg.Message], tools :: List[t.Tool]) -> [net, llm] Iter[d.Delta] {
  let _sm := split_system(messages)
  let sys := match _sm {
    (s, _) => s,
  }
  let user_msgs := match _sm {
    (_, ms) => ms,
  }
  let body := build_request(model, sys, user_msgs, tools)
  let lines := match http.post(config.base_url, bytes.from_str(body), "application/json") {
    Err(_) => iter.from_list([]),
    Ok(r) => iter.from_list(str.split(match bytes.to_str(r.body) {
      Err(_) => "",
      Ok(s) => s,
    }, "\n")),
  }
  let payloads := sse.data_payloads(lines)
  parse_stream(payloads)
}

# ---- Request building --------------------------------------------
fn split_system(messages :: List[msg.Message]) -> (Str, List[msg.Message]) {
  let sys := list.fold(messages, "", fn (acc :: Str, m :: msg.Message) -> Str {
    match m {
      SystemMsg(s) => s,
      _ => acc,
    }
  })
  let rest := list.fold(messages, [], fn (acc :: List[msg.Message], m :: msg.Message) -> List[msg.Message] {
    match m {
      SystemMsg(_) => acc,
      _ => list.concat(acc, [m]),
    }
  })
  (sys, rest)
}

fn build_request(model :: prov.ModelRef, sys :: Str, messages :: List[msg.Message], tools :: List[t.Tool]) -> Str {
  let base := [("model", JStr(model.model)), ("max_tokens", JInt(4096)), ("stream", JBool(true)), ("messages", JList(encode_messages_cached(messages)))]
  let with_sys := if str.is_empty(sys) {
    base
  } else {
    list.concat(base, [("system", JList([cache_marked(text_block(sys))]))])
  }
  let with_tools := if list.is_empty(tools) {
    with_sys
  } else {
    list.concat(with_sys, [("tools", JList(mark_last_cached(list.map(tools, t.to_anthropic_json))))])
  }
  jv.stringify(JObj(with_tools))
}

# ---- Prompt caching -------------------------------------------------
#
# Three static/growing-prefix breakpoints, well under Anthropic's 4-per-
# request cap: tools (static per agent), system prompt (static per
# agent), and the message history's last entry (grows by append only —
# each `chat()` call resends the full history, so marking the current
# last message caches everything up to it; the next call's longer
# history shares that exact prefix and only pays for the new suffix).
# Tool schemas + system prompts are typically hundreds to low thousands
# of tokens and identical on every step of a `max_steps: 50` agent loop,
# so this is a straight latency/cost win with no behavior change — below
# the provider's minimum cacheable length, cache_control is silently a
# no-op rather than an error.
fn cache_control_ephemeral() -> jv.Json {
  JObj([("type", JStr("ephemeral"))])
}

fn cache_marked(j :: jv.Json) -> jv.Json {
  match j {
    JObj(fields) => JObj(list.concat(fields, [("cache_control", cache_control_ephemeral())])),
    other => other,
  }
}

# Mark the last element of a JSON array as the cache breakpoint —
# Anthropic caches the prefix up to and including the marked block.
fn mark_last_cached(items :: List[jv.Json]) -> List[jv.Json] {
  let n := list.len(items)
  list.map(list.enumerate(items), fn (p :: (Int, jv.Json)) -> jv.Json {
    match p {
      (i, j) => if i == n - 1 {
        cache_marked(j)
      } else {
        j
      },
    }
  })
}

fn text_block(text :: Str) -> jv.Json {
  JObj([("type", JStr("text")), ("text", JStr(text))])
}

fn encode_messages_cached(messages :: List[msg.Message]) -> List[jv.Json] {
  let n := list.len(messages)
  list.map(list.enumerate(messages), fn (p :: (Int, msg.Message)) -> jv.Json {
    match p {
      (i, m) => encode_message(m, i == n - 1),
    }
  })
}

fn encode_message(m :: msg.Message, cached :: Bool) -> jv.Json {
  match m {
    UserMsg(text) => JObj([("role", JStr("user")), ("content", JList([maybe_cached(text_block(text), cached)]))]),
    AssistantMsg(text, calls) => if list.is_empty(calls) {
      JObj([("role", JStr("assistant")), ("content", JList([maybe_cached(text_block(text), cached)]))])
    } else {
      let blocks := list.map(calls, encode_tool_use_block)
      JObj([("role", JStr("assistant")), ("content", JList(if cached {
        mark_last_cached(blocks)
      } else {
        blocks
      }))])
    },
    ToolMsg(call_id, content) => JObj([("role", JStr("user")), ("content", JList([maybe_cached(JObj([("type", JStr("tool_result")), ("tool_use_id", JStr(call_id)), ("content", JStr(content))]), cached)]))]),
    SystemMsg(_) => JObj([("role", JStr("user")), ("content", JStr(""))]),
  }
}

fn maybe_cached(j :: jv.Json, cached :: Bool) -> jv.Json {
  if cached {
    cache_marked(j)
  } else {
    j
  }
}

fn encode_tool_use_block(call :: msg.ToolCall) -> jv.Json {
  JObj([("type", JStr("tool_use")), ("id", JStr(call.id)), ("name", JStr(call.name)), ("input", call.args)])
}

fn build_headers(api_key :: Str) -> Map[Str, Str] {
  map.from_list([("x-api-key", api_key), ("anthropic-version", api_version()), ("content-type", "application/json"), ("accept", "text/event-stream")])
}

# ---- SSE stream parsing ------------------------------------------
#
# Relevant Anthropic event types:
#   content_block_start  — starts a text or tool_use block
#   content_block_delta  — text_delta or input_json_delta
#   message_delta        — carries stop_reason at end
#
# We track current block type + tool id in fold state.
type ParseState = { block_type :: Str, tool_id :: Str, tool_name :: Str }

fn parse_stream(payloads :: List[Str]) -> Iter[d.Delta] {
  let init := { block_type: "", tool_id: "", tool_name: "" }
  let _fd := list.fold(payloads, (init, []), fn (acc :: (ParseState, List[d.Delta]), payload :: Str) -> (ParseState, List[d.Delta]) {
    let state := match acc {
      (s, _) => s,
    }
    let so_far := match acc {
      (_, ds) => ds,
    }
    match jv.parse_into_errors(payload) {
      Err(_) => acc,
      Ok(j) => {
        let _he := handle_event(state, j)
        let new_state := match _he {
          (s, _) => s,
        }
        let new_deltas := match _he {
          (_, ds) => ds,
        }
        (new_state, list.concat(so_far, new_deltas))
      },
    }
  })
  let deltas := match _fd {
    (_, ds) => ds,
  }
  iter.from_list(deltas)
}

fn handle_event(state :: ParseState, j :: jv.Json) -> (ParseState, List[d.Delta]) {
  match jv.get_field(j, "type") {
    Some(JStr(t)) => match t {
      "content_block_start" => handle_block_start(state, j),
      "content_block_delta" => handle_block_delta(state, j),
      "message_delta" => handle_message_delta(state, j),
      _ => (state, []),
    },
    _ => (state, []),
  }
}

fn handle_block_start(state :: ParseState, j :: jv.Json) -> (ParseState, List[d.Delta]) {
  match jv.get_field(j, "content_block") {
    None => (state, []),
    Some(block) => match jv.get_field(block, "type") {
      Some(JStr("tool_use")) => {
        let id := str_field(block, "id")
        let name := str_field(block, "name")
        ({ block_type: "tool_use", tool_id: id, tool_name: name }, [ToolCallBegin(id, name)])
      },
      Some(JStr("text")) => ({ block_type: "text", tool_id: state.tool_id, tool_name: state.tool_name }, []),
      _ => (state, []),
    },
  }
}

fn handle_block_delta(state :: ParseState, j :: jv.Json) -> (ParseState, List[d.Delta]) {
  match jv.get_field(j, "delta") {
    None => (state, []),
    Some(delta) => match jv.get_field(delta, "type") {
      Some(JStr("text_delta")) => {
        let text := str_field(delta, "text")
        if str.is_empty(text) {
          (state, [])
        } else {
          (state, [TextChunk(text)])
        }
      },
      Some(JStr("input_json_delta")) => {
        let chunk := str_field(delta, "partial_json")
        if str.is_empty(chunk) {
          (state, [])
        } else {
          (state, [ToolArgChunk(state.tool_id, chunk)])
        }
      },
      _ => (state, []),
    },
  }
}

fn handle_message_delta(state :: ParseState, j :: jv.Json) -> (ParseState, List[d.Delta]) {
  match jv.get_field(j, "delta") {
    None => (state, []),
    Some(delta) => match jv.get_field(delta, "stop_reason") {
      Some(JStr(reason)) => (state, [FinishDelta(normalise_finish(reason))]),
      _ => (state, []),
    },
  }
}

fn normalise_finish(reason :: Str) -> Str
  examples {
    normalise_finish("end_turn") => "stop",
    normalise_finish("tool_use") => "tool_calls",
    normalise_finish("max_tokens") => "length"
  }
{
  match reason {
    "end_turn" => "stop",
    "tool_use" => "tool_calls",
    "max_tokens" => "length",
    other => other,
  }
}

fn str_field(j :: jv.Json, key :: Str) -> Str {
  match jv.get_field(j, key) {
    Some(JStr(s)) => s,
    _ => "",
  }
}

