# lex-llm — Ollama local model adapter
#
# Implements Provider.chat against Ollama's /api/chat endpoint.
# No API key required. Streaming response is NDJSON (one JSON per line)
# with a "done" field on the final line.
#
# Compatible with llama3, mistral, qwen2, phi3, gemma, and any other
# model pulled into a local Ollama instance.
# Ollama ≥0.3 also supports function/tool calling in OpenAI format.

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

fn default_base_url() -> Str {
  "http://localhost:11434/api/chat"
}

type OllamaConfig = { base_url :: Str }

fn default_config() -> OllamaConfig
  examples {
    default_config() => { base_url: "http://localhost:11434/api/chat" }
  }
{
  { base_url: default_base_url() }
}

fn make_provider(config :: OllamaConfig) -> prov.Provider {
  { name: "ollama", chat: fn (model :: prov.ModelRef, messages :: List[msg.Message], tools :: List[t.Tool]) -> [net, llm] Iter[d.Delta] {
    chat(config, model, messages, tools)
  } }
}

fn chat(config :: OllamaConfig, model :: prov.ModelRef, messages :: List[msg.Message], tools :: List[t.Tool]) -> [net, llm] Iter[d.Delta] {
  let body := build_request(model, messages, tools)
  let hdrs := map.set(map.set(map.new(), "content-type", "application/json"), "connection", "close")
  let req := { method: "POST", url: config.base_url, headers: hdrs, body: Some(bytes.from_str(body)), timeout_ms: Some(120000) }
  let lines := match http.send(req) {
    Err(_) => [],
    Ok(r) => if r.status >= 400 {
      []
    } else {
      match bytes.to_str(r.body) {
        Err(_) => [],
        Ok(s) => str.split(s, "\n"),
      }
    },
  }
  parse_stream(lines)
}

# ---- Request building --------------------------------------------
fn build_request(model :: prov.ModelRef, messages :: List[msg.Message], tools :: List[t.Tool]) -> Str {
  let streaming := false
  let base := [("model", JStr(model.model)), ("messages", JList(list.map(messages, encode_message))), ("stream", JBool(streaming))]
  let with_tools := if list.is_empty(tools) {
    base
  } else {
    list.concat(base, [("tools", JList(list.map(tools, t.to_openai_json)))])
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
      JObj([("role", JStr("assistant")), ("content", JStr("")), ("tool_calls", JList(list.map(calls, encode_tool_call)))])
    },
    ToolMsg(call_id, content) => JObj([("role", JStr("tool")), ("tool_call_id", JStr(call_id)), ("content", JStr(content))]),
  }
}

fn encode_tool_call(call :: msg.ToolCall) -> jv.Json {
  JObj([("id", JStr(call.id)), ("type", JStr("function")), ("function", JObj([("name", JStr(call.name)), ("arguments", call.args)]))])
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
  let stream_deltas := list.fold(lines, [], fn (acc :: List[d.Delta], line :: Str) -> List[d.Delta] {
    let t := str.trim(line)
    if str.is_empty(t) {
      acc
    } else {
      match jv.parse_into_errors(t) {
        Err(_) => acc,
        Ok(j) => list.concat(acc, parse_chunk(j)),
      }
    }
  })
  let deltas := if list.is_empty(stream_deltas) {
    let full := str.trim(str.join(lines, ""))
    if str.is_empty(full) {
      []
    } else {
      match jv.parse_into_errors(full) {
        Err(_) => [],
        Ok(j) => parse_chunk(j),
      }
    }
  } else {
    stream_deltas
  }
  iter.from_list(deltas)
}

fn parse_chunk(j :: jv.Json) -> List[d.Delta] {
  let done := match jv.get_field(j, "done") {
    Some(JBool(b)) => b,
    _ => false,
  }
  let msg_deltas := match jv.get_field(j, "message") {
    None => [],
    Some(mj) => match jv.get_field(mj, "role") {
      Some(JStr("assistant")) => parse_assistant_message(mj),
      _ => [],
    },
  }
  let finish_deltas := if done {
    let reason := finish_reason_from_msg(j)
    [FinishDelta(reason)]
  } else {
    []
  }
  list.concat(msg_deltas, finish_deltas)
}

fn parse_assistant_message(mj :: jv.Json) -> List[d.Delta] {
  let raw_content := match jv.get_field(mj, "content") {
    Some(JStr(s)) => s,
    _ => "",
  }
  let call_deltas := match jv.get_field(mj, "tool_calls") {
    Some(JList(calls)) => parse_tool_calls(calls),
    _ => parse_xml_tool_calls(raw_content),
  }
  let thinking := match jv.get_field(mj, "thinking") {
    Some(JStr(s)) => s,
    _ => "",
  }
  let content := if str.is_empty(str.trim(raw_content)) {
    if list.is_empty(call_deltas) {
      thinking
    } else {
      raw_content
    }
  } else {
    raw_content
  }
  let trimmed_content := str.trim(content)
  let is_xml := if str.starts_with(trimmed_content, "<function=") {
    true
  } else {
    str.starts_with(trimmed_content, "<tool_call>")
  }
  let text_deltas := if str.is_empty(content) {
    []
  } else {
    if is_xml {
      []
    } else {
      [TextChunk(content)]
    }
  }
  list.concat(text_deltas, call_deltas)
}

# ---- qwen3 XML tool call parser ------------------------------------
#
# qwen3-coder emits its native agentic format when served locally via
# Ollama (both streaming and non-streaming with a system prompt):
#
#   <tool_call>
#   <function=write>
#   <parameter=path>
#   hello.lex
#   </parameter>
#   <parameter=content>
#   fn hello() -> Str { "hi" }
#   </parameter>
#   </function>
#   </tool_call>
#
# We parse this into the same TCBegin + TCArgs deltas produced by the
# structured tool_calls path, so the rest of the pipeline is unchanged.
type XmlPState = XScan(Str) | XInParam((Str, Str))

fn parse_xml_tool_calls(content :: Str) -> List[d.Delta] {
  let func_parts := str.split(content, "<function=")
  if list.len(func_parts) <= 1 {
    []
  } else {
    match list.fold(func_parts, (true, []), fn (acc :: (Bool, List[d.Delta]), part :: Str) -> (Bool, List[d.Delta]) {
      let skip := match acc {
        (b, _) => b,
      }
      let deltas := match acc {
        (_, ds) => ds,
      }
      if skip {
        (false, deltas)
      } else {
        let name := str.trim(match list.head(str.split(part, ">")) {
          Some(n) => n,
          None => "",
        })
        if str.is_empty(name) {
          (false, deltas)
        } else {
          let id := str.concat("call_", name)
          let args := xml_params_to_json(part)
          (false, list.concat(deltas, [ToolCallBegin(id, name), ToolArgChunk(id, args)]))
        }
      }
    }) {
      (_, ds) => ds,
    }
  }
}

fn xml_params_to_json(func_body :: Str) -> Str {
  let lines := str.split(func_body, "\n")
  let result := list.fold(lines, (XScan(""), []), fn (acc :: (XmlPState, List[(Str, Str)]), line :: Str) -> (XmlPState, List[(Str, Str)]) {
    let state := match acc {
      (s, _) => s,
    }
    let params := match acc {
      (_, p) => p,
    }
    let t := str.trim(line)
    match state {
      XScan(_) => if str.starts_with(t, "<parameter=") {
        let after := str_skip_prefix(t, "<parameter=")
        let name := str.trim(match list.head(str.split(after, ">")) {
          Some(n) => n,
          None => after,
        })
        if str.is_empty(name) {
          acc
        } else {
          (XInParam(name, ""), params)
        }
      } else {
        acc
      },
      XInParam(name, value) => if str.starts_with(t, "</parameter>") {
        (XScan(""), list.concat(params, [(name, str.trim(value))]))
      } else {
        let sep := if str.is_empty(value) {
          ""
        } else {
          "\n"
        }
        let new_val := str.concat(value, str.concat(sep, line))
        (XInParam(name, new_val), params)
      },
    }
  })
  let params := match result {
    (_, p) => p,
  }
  let fields := list.map(params, fn (kv :: (Str, Str)) -> Str {
    let k := match kv {
      (k, _) => k,
    }
    let v := match kv {
      (_, v) => v,
    }
    str.concat(str.concat("\"", str.concat(k, "\": ")), jv.stringify(JStr(v)))
  })
  str.join(["{", str.join(fields, ", "), "}"], "")
}

# Return everything in `s` after the first occurrence of `prefix`.
# Returns `s` unchanged if `prefix` is not found.
fn str_skip_prefix(s :: Str, prefix :: Str) -> Str {
  let parts := str.split(s, prefix)
  match list.fold(parts, (true, ""), fn (acc :: (Bool, Str), part :: Str) -> (Bool, Str) {
    let first := match acc {
      (b, _) => b,
    }
    let so_far := match acc {
      (_, r) => r,
    }
    if first {
      (false, "")
    } else {
      (false, if str.is_empty(so_far) {
        part
      } else {
        str.join([so_far, part], prefix)
      })
    }
  }) {
    (_, r) => r,
  }
}

fn parse_tool_calls(calls :: List[jv.Json]) -> List[d.Delta] {
  list.fold(calls, [], fn (acc :: List[d.Delta], cj :: jv.Json) -> List[d.Delta] {
    match jv.get_field(cj, "function") {
      None => acc,
      Some(fj) => {
        let name := match jv.get_field(fj, "name") {
          Some(JStr(s)) => s,
          _ => "",
        }
        let args := match jv.get_field(fj, "arguments") {
          Some(aj) => jv.stringify(aj),
          None => "{}",
        }
        let id := str.concat("call_", name)
        list.concat(acc, [ToolCallBegin(id, name), ToolArgChunk(id, args)])
      },
    }
  })
}

fn finish_reason_from_msg(j :: jv.Json) -> Str {
  match jv.get_field(j, "message") {
    Some(mj) => match jv.get_field(mj, "tool_calls") {
      Some(JList(calls)) => if list.is_empty(calls) {
        "stop"
      } else {
        "tool_calls"
      },
      _ => "stop",
    },
    _ => "stop",
  }
}

