# lex-llm — OpenAI Chat Completions adapter
#
# Implements Provider.chat against POST /v1/chat/completions with
# stream:false, and the optional Provider.stream half against the same
# endpoint with stream:true. Uses an Authorization: Bearer header.
#
# This adapter is the one every OpenAI-compatible backend routes through —
# LiteLLM, vLLM, lex-moe, mlx_lm.server, opencode-go — so the streaming half
# added here reaches all of them at once, and any of them that does not in
# fact honour stream:true simply produces no Deltas until the socket closes,
# which is the buffered behaviour it had before.
#
# Supports any model accessible via the Chat Completions API:
# GPT-4o, GPT-4o-mini, o1, o3-mini, etc.
# Also works with Azure OpenAI, Mistral AI, and compatible proxies via base_url.

import "../message" as msg

import "../delta" as d

import "../tool" as t

import "../provider" as prov

import "lex-schema/json_value" as jv

import "std.http" as http

import "std.bytes" as bytes

import "std.list" as list

import "std.str" as str

import "std.iter" as iter

import "std.map" as map

import "std.int" as int

import "../sse" as sse

fn default_base_url() -> Str {
  "https://api.openai.com/v1/chat/completions"
}

type OpenAIConfig = { api_key :: Str, base_url :: Str }

fn default_config(api_key :: Str) -> OpenAIConfig {
  { api_key: api_key, base_url: default_base_url() }
}

fn make_provider(config :: OpenAIConfig) -> prov.Provider {
  { name: "openai", chat: fn (model :: prov.ModelRef, messages :: List[msg.Message], tools :: List[t.Tool]) -> [net, llm] Iter[d.Delta] {
    chat(config, model, messages, tools)
  }, stream: Some({ open: fn (model :: prov.ModelRef, messages :: List[msg.Message], tools :: List[t.Tool]) -> [net, llm] Result[Stream[Str], Str] {
    open_stream(config, model, messages, tools)
  }, init: JList([]), step: stream_step }) }
}

fn stream_headers(api_key :: Str) -> Map[Str, Str] {
  let base := map.set(map.set(map.new(), "content-type", "application/json"), "accept", "text/event-stream")
  if str.is_empty(api_key) {
    base
  } else {
    map.set(base, "authorization", str.concat("Bearer ", api_key))
  }
}

# The api_key is empty for the local backends (vLLM, moe, mlx) that route
# through this adapter; sending `Authorization: Bearer ` to one of those is
# at best ignored and at worst a 401, so the header is omitted rather than
# sent blank -- which is also what the non-streaming path should do, but
# changing that is not this commit's business.
fn open_stream(config :: OpenAIConfig, model :: prov.ModelRef, messages :: List[msg.Message], tools :: List[t.Tool]) -> [net, llm] Result[Stream[Str], Str] {
  http.stream_lines(config.base_url, stream_headers(config.api_key), build_stream_request(model, messages, tools))
}

# chat -- one non-streaming completion, decoded into Deltas.
#
# timeout_ms is 600s, not the 120s this used to carry. A reasoning model
# answering a build prompt routinely spends 4+ minutes and 7k+ completion
# tokens on a single call; at 120s http.send aborted mid-flight while the
# upstream went on to return a perfectly good answer that nothing was left
# listening for.
#
# Every failure path below used to collapse to [] -- an empty delta list,
# which agent.run_steps reads as finish="stop" with empty content. That made
# a timeout, a 500, and a model that genuinely said nothing indistinguishable:
# the caller logged an empty answer either way and reported it as the model's
# output. They now surface as text so the failure reaches the trail and a human.
fn chat(config :: OpenAIConfig, model :: prov.ModelRef, messages :: List[msg.Message], tools :: List[t.Tool]) -> [net, llm] Iter[d.Delta] {
  let body := build_request(model, messages, tools)
  let hdrs := map.set(map.set(map.new(), "content-type", "application/json"), "authorization", str.concat("Bearer ", config.api_key))
  let req := { method: "POST", url: config.base_url, headers: hdrs, body: Some(bytes.from_str(body)), timeout_ms: Some(600000) }
  let deltas := match http.send(req) {
    Err(_) => d.provider_error("request failed or timed out"),
    Ok(r) => if r.status >= 400 {
      d.provider_error(str.concat("HTTP ", int.to_str(r.status)))
    } else {
      match bytes.to_str(r.body) {
        Err(_) => d.provider_error("response body was not valid UTF-8"),
        Ok(s) => match jv.parse_into_errors(s) {
          Err(_) => d.provider_error("response was not valid JSON"),
          Ok(j) => parse_completion(j),
        },
      }
    },
  }
  iter.from_list(deltas)
}

# ---- Request building --------------------------------------------
fn build_request(model :: prov.ModelRef, messages :: List[msg.Message], tools :: List[t.Tool]) -> Str {
  let base := [("model", JStr(model.model)), ("messages", JList(list.map(messages, encode_message))), ("stream", JBool(false)), ("max_tokens", JInt(8192))]
  let with_tools := if list.is_empty(tools) {
    base
  } else {
    list.concat(base, [("tools", JList(list.map(tools, t.to_openai_json))), ("tool_choice", JStr("auto"))])
  }
  jv.stringify(JObj(with_tools))
}

fn encode_message(m :: msg.Message) -> jv.Json {
  match m {
    UserMsg(text) => JObj([("role", JStr("user")), ("content", JStr(text))]),
    SystemMsg(text) => JObj([("role", JStr("system")), ("content", JStr(text))]),
    AssistantMsg(text, calls) => if list.is_empty(calls) {
      JObj([("role", JStr("assistant")), ("content", JStr(text))])
    } else {
      JObj([("role", JStr("assistant")), ("content", JNull), ("tool_calls", JList(list.map(calls, encode_tool_call)))])
    },
    ToolMsg(call_id, content) => JObj([("role", JStr("tool")), ("tool_call_id", JStr(call_id)), ("content", JStr(content))]),
  }
}

fn encode_tool_call(call :: msg.ToolCall) -> jv.Json {
  JObj([("id", JStr(call.id)), ("type", JStr("function")), ("function", JObj([("name", JStr(call.name)), ("arguments", JStr(jv.stringify(call.args)))]))])
}

# ---- Non-streaming response parsing ------------------------------
#
# Non-streaming shape:
#   text:      { "choices": [{ "message": { "role":"assistant", "content":"...", "tool_calls": null }, "finish_reason":"stop" }] }
#   tool call: { "choices": [{ "message": { "role":"assistant", "content": null,
#                  "tool_calls": [{"id":"...","type":"function","function":{"name":"...","arguments":"{}"}}] },
#               "finish_reason":"tool_calls" }] }
fn parse_completion(j :: jv.Json) -> List[d.Delta] {
  let usage_deltas := parse_usage(j)
  match jv.get_field(j, "choices") {
    None => usage_deltas,
    Some(JList(xs)) => match first(xs) {
      None => usage_deltas,
      Some(choice) => {
        let finish := match jv.get_field(choice, "finish_reason") {
          Some(JStr(r)) => r,
          _ => "stop",
        }
        let msg_deltas := match jv.get_field(choice, "message") {
          None => [],
          Some(mj) => parse_message_obj(mj),
        }
        list.concat(usage_deltas, list.concat(msg_deltas, [FinishDelta(finish)]))
      },
    },
    _ => usage_deltas,
  }
}

# Top-level "usage": {"prompt_tokens":N,"completion_tokens":N,"total_tokens":N}
# -- present on OpenAI Chat Completions and every OpenAI-compatible proxy this
# codebase talks to (opencode-go, DeepSeek). Absent entirely => no UsageDelta,
# not a zero -- callers must not conflate "not reported" with "free".
fn parse_usage(j :: jv.Json) -> List[d.Delta] {
  match jv.get_field(j, "usage") {
    None => [],
    Some(uj) => {
      let p := int_field(uj, "prompt_tokens")
      let c := int_field(uj, "completion_tokens")
      let t := int_field(uj, "total_tokens")
      let all_zero := if p == 0 {
        if c == 0 {
          t == 0
        } else {
          false
        }
      } else {
        false
      }
      if all_zero {
        []
      } else {
        [UsageDelta(p, c, t)]
      }
    },
  }
}

fn int_field(j :: jv.Json, key :: Str) -> Int {
  match jv.get_field(j, key) {
    Some(JInt(v)) => v,
    _ => 0,
  }
}

fn parse_message_obj(mj :: jv.Json) -> List[d.Delta] {
  match jv.get_field(mj, "tool_calls") {
    Some(JList(calls)) => if list.is_empty(calls) {
      parse_content_field(mj)
    } else {
      parse_tool_calls_full(calls)
    },
    _ => parse_content_field(mj),
  }
}

# Fall back to the `reasoning` field (reasoning models put their chain-of-thought
# there and leave `content` empty/null) so we surface something rather than an
# empty turn. Bounded by max_tokens in build_request.
fn content_or_reasoning(mj :: jv.Json) -> Str {
  let c := match jv.get_field(mj, "content") {
    Some(JStr(s)) => s,
    _ => "",
  }
  if str.is_empty(str.trim(c)) {
    let r1 := match jv.get_field(mj, "reasoning") {
      Some(JStr(r)) => r,
      _ => "",
    }
    if str.is_empty(str.trim(r1)) {
      match jv.get_field(mj, "reasoning_content") {
        Some(JStr(r)) => r,
        _ => c,
      }
    } else {
      r1
    }
  } else {
    c
  }
}

fn parse_content_field(mj :: jv.Json) -> List[d.Delta] {
  let s := content_or_reasoning(mj)
  if str.is_empty(s) {
    []
  } else {
    match content_tool_call(s) {
      Some(deltas) => deltas,
      None => {
        let cleaned := clean_eos(s)
        if str.is_empty(str.trim(cleaned)) {
          []
        } else {
          [TextChunk(cleaned)]
        }
      },
    }
  }
}

# Try to extract a single {"name":..,"arguments":..} tool call embedded in the
# assistant's text content. Returns None unless BOTH keys are present (so prose
# that merely contains a JSON snippet is not misread as a tool call).
fn content_tool_call(content :: Str) -> Option[List[d.Delta]] {
  let candidate := strip_tool_markers(content)
  if str.starts_with(candidate, "{") {
    match jv.parse_into_errors(candidate) {
      Err(_) => None,
      Ok(j) => match jv.get_field(j, "name") {
        Some(JStr(name)) => if str.is_empty(name) {
          None
        } else {
          match jv.get_field(j, "arguments") {
            None => None,
            Some(aj) => {
              let args := match aj {
                JStr(s) => s,
                _ => jv.stringify(aj),
              }
              let id := str.concat("call_", name)
              Some([ToolCallBegin(id, name), ToolArgChunk(id, args)])
            },
          }
        },
        _ => None,
      },
    }
  } else {
    None
  }
}

# Some OpenAI-compatible servers (mlx_lm.server) leak the chat-template EOS
# token into the visible content. Strip the common ones so they never reach the
# user-facing reply.
fn clean_eos(s :: Str) -> Str {
  let a := str.join(str.split(s, "<|im_end|>"), "")
  let b := str.join(str.split(a, "<|endoftext|>"), "")
  let c := str.join(str.split(b, "<|eot_id|>"), "")
  str.trim(c)
}

# Remove the markdown fence / chat-template tokens that wrap a content-embedded
# tool call, leaving (hopefully) a bare JSON object to parse.
fn strip_tool_markers(s :: Str) -> Str {
  let a := str.join(str.split(s, "```json"), "")
  let b := str.join(str.split(a, "```"), "")
  let c := str.join(str.split(b, "<|im_end|>"), "")
  let dd := str.join(str.split(c, "<tool_call>"), "")
  let ee := str.join(str.split(dd, "</tool_call>"), "")
  str.trim(ee)
}

fn parse_tool_calls_full(calls :: List[jv.Json]) -> List[d.Delta] {
  let folded := list.fold(calls, (0, []), fn (acc :: (Int, List[d.Delta]), cj :: jv.Json) -> (Int, List[d.Delta]) {
    let idx := match acc {
      (i, _) => i,
    }
    let deltas := match acc {
      (_, ds) => ds,
    }
    match jv.get_field(cj, "function") {
      None => (idx + 1, deltas),
      Some(fj) => {
        let name := str_field(fj, "name")
        let id := tool_call_id(cj, name, idx)
        let args := match jv.get_field(fj, "arguments") {
          Some(JStr(s)) => s,
          _ => "{}",
        }
        if str.is_empty(name) {
          (idx + 1, deltas)
        } else {
          (idx + 1, list.concat(deltas, [ToolCallBegin(id, name), ToolArgChunk(id, args)]))
        }
      },
    }
  })
  match folded {
    (_, ds) => ds,
  }
}

# ---- Helpers -----------------------------------------------------
fn tool_call_id(cj :: jv.Json, name :: Str, idx :: Int) -> Str {
  let id := str_field(cj, "id")
  if str.is_empty(id) {
    str.concat(str.concat("call_", name), str.concat("_", int.to_str(idx)))
  } else {
    id
  }
}

fn first[T](xs :: List[T]) -> Option[T] {
  list.fold(xs, None, fn (acc :: Option[T], x :: T) -> Option[T] {
    match acc {
      Some(_) => acc,
      None => Some(x),
    }
  })
}

fn str_field(j :: jv.Json, key :: Str) -> Str {
  match jv.get_field(j, key) {
    Some(JStr(s)) => s,
    _ => "",
  }
}

# ---- Streaming request + step ------------------------------------
#
# Same body as build_request with stream:true, plus stream_options so the
# final chunk carries a usage object — without it a streamed turn reports no
# token counts at all, and delta.UsageDelta's contract is that absence means
# "not reported", so the cost of a streamed turn would silently vanish from
# the trail. Backends that don't know stream_options ignore the field.
fn build_stream_request(model :: prov.ModelRef, messages :: List[msg.Message], tools :: List[t.Tool]) -> Str {
  let base := [("model", JStr(model.model)), ("messages", JList(list.map(messages, encode_message))), ("stream", JBool(true)), ("stream_options", JObj([("include_usage", JBool(true))])), ("max_tokens", JInt(8192))]
  let with_tools := if list.is_empty(tools) {
    base
  } else {
    list.concat(base, [("tools", JList(list.map(tools, t.to_openai_json))), ("tool_choice", JStr("auto"))])
  }
  jv.stringify(JObj(with_tools))
}

# Streaming chunk shape:
#   {"choices":[{"delta":{"content":"Hel"},"finish_reason":null}]}
#   {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",
#       "function":{"name":"read","arguments":"{\"pa"}}]},"finish_reason":null}]}
#   {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}
#   {"choices":[],"usage":{...}}                        <- with stream_options
#
# A tool call arrives split across chunks: the first carries id and name, the
# rest carry only `index` and a fragment of `arguments`. So the parser state
# is the ids seen so far, positionally — index 0 is the first element — which
# is what lets a later ToolArgChunk be routed to the call it belongs to.
# ToolCallBegin is emitted once, on the chunk that first names the call.
fn state_ids(state :: jv.Json) -> List[Str]
  examples {
    state_ids(JList([JStr("a"), JStr("b")])) => ["a", "b"],
    state_ids(JList([])) => [],
    state_ids(JNull) => []
  }
{
  match state {
    JList(xs) => list.fold(xs, [], fn (acc :: List[Str], x :: jv.Json) -> List[Str] {
      match x {
        JStr(s) => list.concat(acc, [s]),
        _ => acc,
      }
    }),
    _ => [],
  }
}

fn encode_ids(ids :: List[Str]) -> jv.Json
  examples {
    encode_ids(["a"]) => JList([JStr("a")]),
    encode_ids([]) => JList([])
  }
{
  JList(list.map(ids, fn (s :: Str) -> jv.Json {
    JStr(s)
  }))
}

# Read the id at `idx`, or "" when the stream has not named that call yet.
fn id_at(ids :: List[Str], idx :: Int) -> Str
  examples {
    id_at(["a", "b"], 1) => "b",
    id_at(["a"], 3) => "",
    id_at([], 0) => ""
  }
{
  let found := list.fold(ids, (0, ""), fn (acc :: (Int, Str), s :: Str) -> (Int, Str) {
    match acc {
      (i, hit) => if i == idx {
        (i + 1, s)
      } else {
        (i + 1, hit)
      },
    }
  })
  match found {
    (_, hit) => hit,
  }
}

# Grow `ids` so position `idx` holds `id`, padding intervening slots.
fn set_id(ids :: List[Str], idx :: Int, id :: Str) -> List[Str]
  examples {
    set_id([], 0, "a") => ["a"],
    set_id(["a"], 1, "b") => ["a", "b"],
    set_id(["a"], 0, "z") => ["z"],
    set_id([], 2, "c") => ["", "", "c"]
  }
{
  let kept := list.fold(ids, (0, []), fn (acc :: (Int, List[Str]), s :: Str) -> (Int, List[Str]) {
    match acc {
      (i, out) => if i == idx {
        (i + 1, list.concat(out, [id]))
      } else {
        (i + 1, list.concat(out, [s]))
      },
    }
  })
  match kept {
    (n, out) => if n > idx {
      out
    } else {
      list.concat(out, list.concat(pad(idx - n), [id]))
    },
  }
}

fn pad(n :: Int) -> List[Str]
  examples {
    pad(0) => [],
    pad(2) => ["", ""]
  }
{
  if n <= 0 {
    []
  } else {
    list.concat([""], pad(n - 1))
  }
}

fn stream_step(state :: jv.Json, line :: Str) -> (jv.Json, List[d.Delta])
  examples {
    stream_step(JList([]), "") => (JList([]), []),
    stream_step(JList([]), ": ping") => (JList([]), []),
    stream_step(JList([]), "data: [DONE]") => (JList([]), []),
    stream_step(JList([]), "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}]}") => (JList([]), [TextChunk("Hi")]),
    stream_step(JList([]), "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}") => (JList([]), [FinishDelta("stop")])
  }
{
  match sse.parse_data_line(sse.strip_cr(line)) {
    None => (state, []),
    Some(payload) => if payload == "[DONE]" {
      (state, [])
    } else {
      match jv.parse_into_errors(payload) {
        Err(_) => (state, []),
        Ok(j) => step_chunk(state, j),
      }
    },
  }
}

fn step_chunk(state :: jv.Json, j :: jv.Json) -> (jv.Json, List[d.Delta]) {
  let usage_deltas := parse_usage(j)
  match jv.get_field(j, "choices") {
    Some(JList(xs)) => match first(xs) {
      Some(choice) => {
        let finish_deltas := match jv.get_field(choice, "finish_reason") {
          Some(JStr(r)) => [FinishDelta(r)],
          _ => [],
        }
        match step_choice_delta(state, choice) {
          (next_state, delta_deltas) => (next_state, list.concat(usage_deltas, list.concat(delta_deltas, finish_deltas))),
        }
      },
      None => (state, usage_deltas),
    },
    _ => (state, usage_deltas),
  }
}

fn step_choice_delta(state :: jv.Json, choice :: jv.Json) -> (jv.Json, List[d.Delta]) {
  match jv.get_field(choice, "delta") {
    None => (state, []),
    Some(dj) => match jv.get_field(dj, "tool_calls") {
      Some(JList(calls)) => step_tool_calls(state, calls),
      _ => (state, step_text(dj)),
    },
  }
}

# `content` on a streamed chunk is the visible text; reasoning models put
# their chain-of-thought in `reasoning` / `reasoning_content` and leave
# content empty, exactly as on the buffered path. Unlike the buffered path
# this does NOT try to recover a tool call embedded in prose (see
# content_tool_call) — that heuristic parses a whole JSON object, and a
# streamed fragment is a prefix of one. A model that answers that way is
# still handled correctly by the buffered path.
fn step_text(dj :: jv.Json) -> List[d.Delta] {
  let text := content_or_reasoning(dj)
  if str.is_empty(text) {
    []
  } else {
    [TextChunk(text)]
  }
}

fn step_tool_calls(state :: jv.Json, calls :: List[jv.Json]) -> (jv.Json, List[d.Delta]) {
  let folded := list.fold(calls, (state_ids(state), []), fn (acc :: (List[Str], List[d.Delta]), cj :: jv.Json) -> (List[Str], List[d.Delta]) {
    match acc {
      (ids, out) => {
        let idx := match jv.get_field(cj, "index") {
          Some(JInt(i)) => i,
          _ => 0,
        }
        let known := id_at(ids, idx)
        let named := str_field(cj, "id")
        let fname := match jv.get_field(cj, "function") {
          Some(fj) => str_field(fj, "name"),
          None => "",
        }
        let id := if str.is_empty(known) {
          if str.is_empty(named) {
            str.concat("call_", int.to_str(idx))
          } else {
            named
          }
        } else {
          known
        }
        let begin := if str.is_empty(known) {
          [ToolCallBegin(id, fname)]
        } else {
          []
        }
        let args := match jv.get_field(cj, "function") {
          Some(fj) => match jv.get_field(fj, "arguments") {
            Some(JStr(a)) => a,
            _ => "",
          },
          None => "",
        }
        let arg_deltas := if str.is_empty(args) {
          []
        } else {
          [ToolArgChunk(id, args)]
        }
        (set_id(ids, idx, id), list.concat(out, list.concat(begin, arg_deltas)))
      },
    }
  })
  match folded {
    (ids, out) => (encode_ids(ids), out),
  }
}

