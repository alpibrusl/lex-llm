# lex-llm — OpenAI Chat Completions adapter
#
# Implements Provider.chat against POST /v1/chat/completions with
# stream:false. Uses http.send with an Authorization: Bearer header.
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
  } }
}

fn chat(config :: OpenAIConfig, model :: prov.ModelRef, messages :: List[msg.Message], tools :: List[t.Tool]) -> [net, llm] Iter[d.Delta] {
  let body := build_request(model, messages, tools)
  let hdrs := map.set(map.set(map.new(), "content-type", "application/json"), "authorization", str.concat("Bearer ", config.api_key))
  let req := { method: "POST", url: config.base_url, headers: hdrs, body: Some(bytes.from_str(body)), timeout_ms: Some(120000) }
  let deltas := match http.send(req) {
    Err(_) => [],
    Ok(r) => if r.status >= 400 {
      []
    } else {
      match bytes.to_str(r.body) {
        Err(_) => [],
        Ok(s) => match jv.parse_into_errors(s) {
          Err(_) => [],
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
    UserPartsMsg(parts) => JObj([("role", JStr("user")), ("content", JList(list.map(parts, encode_part)))]),
  }
}

# OpenAI carries images as a `data:` URI inside an image_url part.
fn encode_part(p :: msg.Part) -> jv.Json {
  match p {
    TextPart(text) => JObj([("type", JStr("text")), ("text", JStr(text))]),
    ImagePart(ImageB64(mime, data)) => JObj([("type", JStr("image_url")), ("image_url", JObj([("url", data_uri(mime, data))]))]),
  }
}

fn data_uri(mime :: Str, data :: Str) -> jv.Json {
  JStr(str.concat("data:", str.concat(mime, str.concat(";base64,", data))))
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

