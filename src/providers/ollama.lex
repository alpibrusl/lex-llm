# lex-llm — Ollama local model adapter
#
# Implements Provider.chat against Ollama's /api/chat endpoint, and the
# optional Provider.stream half against the same endpoint with stream:true.
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

import "std.int" as int

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
  }, stream: Some({ open: fn (model :: prov.ModelRef, messages :: List[msg.Message], tools :: List[t.Tool]) -> [net, llm] Result[Stream[Str], Str] {
    open_stream(config, model, messages, tools)
  }, init: init_state(), step: stream_step }) }
}

fn open_stream(config :: OllamaConfig, model :: prov.ModelRef, messages :: List[msg.Message], tools :: List[t.Tool]) -> [net, llm] Result[Stream[Str], Str] {
  http.stream_lines(config.base_url, map.set(map.new(), "content-type", "application/json"), build_stream_request(model, messages, tools))
}

fn chat(config :: OllamaConfig, model :: prov.ModelRef, messages :: List[msg.Message], tools :: List[t.Tool]) -> [net, llm] Iter[d.Delta] {
  let body := build_request(model, messages, tools)
  let hdrs := map.set(map.set(map.new(), "content-type", "application/json"), "connection", "close")
  let req := { method: "POST", url: config.base_url, headers: hdrs, body: Some(bytes.from_str(body)), timeout_ms: Some(600000) }
  match http.send(req) {
    Err(_) => iter.from_list(d.provider_error("request failed or timed out")),
    Ok(r) => if r.status >= 400 {
      iter.from_list(d.provider_error(str.concat("HTTP ", int.to_str(r.status))))
    } else {
      match bytes.to_str(r.body) {
        Err(_) => iter.from_list(d.provider_error("response body was not valid UTF-8")),
        Ok(s) => parse_stream(str.split(s, "\n")),
      }
    },
  }
}

# ---- Request building --------------------------------------------
fn build_stream_request(model :: prov.ModelRef, messages :: List[msg.Message], tools :: List[t.Tool]) -> Str {
  build_body(model, messages, tools, true)
}

fn build_request(model :: prov.ModelRef, messages :: List[msg.Message], tools :: List[t.Tool]) -> Str {
  build_body(model, messages, tools, false)
}

fn build_body(model :: prov.ModelRef, messages :: List[msg.Message], tools :: List[t.Tool], streaming :: Bool) -> Str {
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
    match list.fold(func_parts, (true, 0, []), fn (acc :: (Bool, Int, List[d.Delta]), part :: Str) -> (Bool, Int, List[d.Delta]) {
      let skip := match acc {
        (b, _, _) => b,
      }
      let idx := match acc {
        (_, i, _) => i,
      }
      let deltas := match acc {
        (_, _, ds) => ds,
      }
      if skip {
        (false, idx, deltas)
      } else {
        let name := str.trim(match list.head(str.split(part, ">")) {
          Some(n) => n,
          None => "",
        })
        if str.is_empty(name) {
          (false, idx + 1, deltas)
        } else {
          let id := synthetic_tool_call_id(name, idx)
          let args := xml_params_to_json(part)
          (false, idx + 1, list.concat(deltas, [ToolCallBegin(id, name), ToolArgChunk(id, args)]))
        }
      }
    }) {
      (_, _, ds) => ds,
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
        let name := match jv.get_field(fj, "name") {
          Some(JStr(s)) => s,
          _ => "",
        }
        let args := match jv.get_field(fj, "arguments") {
          Some(aj) => jv.stringify(aj),
          None => "{}",
        }
        if str.is_empty(name) {
          (idx + 1, deltas)
        } else {
          let id := tool_call_id(cj, name, idx)
          (idx + 1, list.concat(deltas, [ToolCallBegin(id, name), ToolArgChunk(id, args)]))
        }
      },
    }
  })
  match folded {
    (_, ds) => ds,
  }
}

fn tool_call_id(cj :: jv.Json, name :: Str, idx :: Int) -> Str {
  match jv.get_field(cj, "id") {
    Some(JStr(id)) => if str.is_empty(id) {
      synthetic_tool_call_id(name, idx)
    } else {
      id
    },
    _ => synthetic_tool_call_id(name, idx),
  }
}

fn synthetic_tool_call_id(name :: Str, idx :: Int) -> Str {
  str.concat(str.concat("call_", name), str.concat("_", int.to_str(idx)))
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

# ---- Streaming step ----------------------------------------------
#
# Ollama's stream is NDJSON, one complete JSON object per line, so unlike SSE
# there is no framing to strip and each line is independently parseable.
#
# What is NOT independent is the qwen3 XML tool-call format. `<function=…>`
# arrives a token at a time, so a chunk-by-chunk reader would print the
# markup as prose and then never recognise the call. The buffered path
# sidesteps this by having the whole content in hand (parse_assistant_message
# checks the first characters of the completed text); the streaming path has
# to decide before it has seen the end.
#
# So text is held back until the accumulated content is long enough to rule
# XML in or out:
#
#   * while what we have could still grow into "<function=" or "<tool_call>",
#     accumulate and emit nothing;
#   * once it provably cannot, flush the accumulation as one TextChunk and
#     stream freely from there;
#   * if it does turn out to be XML, emit nothing until `done`, then parse
#     the accumulated text as a tool call.
#
# The held-back prefix is at most len("<function=") characters, so a normal
# reply stalls for ten characters and then streams — not a meaningful delay,
# and the alternative is a class of local model whose tool calls break under
# streaming.
#
# Native `tool_calls` on a chunk bypass all of this: they are structured,
# they arrive whole, and they are emitted immediately.
type StreamState = { buf :: Str, flushed :: Bool, saw_native :: Bool }

fn init_state() -> jv.Json {
  encode_stream_state({ buf: "", flushed: false, saw_native: false })
}

fn encode_stream_state(st :: StreamState) -> jv.Json
  examples {
    encode_stream_state({ buf: "hi", flushed: true, saw_native: false }) => JObj([("buf", JStr("hi")), ("flushed", JBool(true)), ("saw_native", JBool(false))])
  }
{
  JObj([("buf", JStr(st.buf)), ("flushed", JBool(st.flushed)), ("saw_native", JBool(st.saw_native))])
}

fn decode_stream_state(j :: jv.Json) -> StreamState
  examples {
    decode_stream_state(JObj([("buf", JStr("hi")), ("flushed", JBool(true)), ("saw_native", JBool(false))])) => { buf: "hi", flushed: true, saw_native: false },
    decode_stream_state(JNull) => { buf: "", flushed: false, saw_native: false }
  }
{
  { buf: match jv.get_field(j, "buf") {
    Some(JStr(s)) => s,
    _ => "",
  }, flushed: bool_field(j, "flushed"), saw_native: bool_field(j, "saw_native") }
}

fn bool_field(j :: jv.Json, key :: Str) -> Bool {
  match jv.get_field(j, key) {
    Some(JBool(b)) => b,
    _ => false,
  }
}

# The two openers a qwen3-style model can be starting to emit.
fn xml_openers() -> List[Str]
  examples {
    xml_openers() => ["<function=", "<tool_call>"]
  }
{
  ["<function=", "<tool_call>"]
}

# Could `s` still grow into one of the XML openers, or already be one?
#
# True for a strict prefix ("<fun"), true once it matches ("<function=read>"),
# false as soon as it diverges ("Here is"). Empty is undecided, so true.
fn maybe_xml(s :: Str) -> Bool
  examples {
    maybe_xml("") => true,
    maybe_xml("<") => true,
    maybe_xml("<fun") => true,
    maybe_xml("<function=read>") => true,
    maybe_xml("<tool_call>\n<function=w>") => true,
    maybe_xml("Here is") => false,
    maybe_xml("x") => false
  }
{
  let t := str.trim(s)
  if str.is_empty(t) {
    true
  } else {
    list.fold(xml_openers(), false, fn (acc :: Bool, opener :: Str) -> Bool {
      if acc {
        true
      } else {
        if str.starts_with(t, opener) {
          true
        } else {
          str.starts_with(opener, t)
        }
      }
    })
  }
}

# One NDJSON line in, (next state, Deltas) out.
fn stream_step(state :: jv.Json, line :: Str) -> (jv.Json, List[d.Delta])
  examples {
    stream_step(JNull, "") => (JNull, []),
    stream_step(JNull, "   ") => (JNull, []),
    stream_step(JNull, "not json") => (JNull, [])
  }
{
  let trimmed := str.trim(line)
  if str.is_empty(trimmed) {
    (state, [])
  } else {
    match jv.parse_into_errors(trimmed) {
      Err(_) => (state, []),
      Ok(j) => step_chunk(decode_stream_state(state), j),
    }
  }
}

fn step_chunk(st :: StreamState, j :: jv.Json) -> (jv.Json, List[d.Delta]) {
  let done := match jv.get_field(j, "done") {
    Some(JBool(b)) => b,
    _ => false,
  }
  let mj := jv.get_field(j, "message")
  let native := match mj {
    None => [],
    Some(m) => match jv.get_field(m, "tool_calls") {
      Some(JList(calls)) => parse_tool_calls(calls),
      _ => [],
    },
  }
  let fragment := match mj {
    None => "",
    Some(m) => chunk_text(m),
  }
  let after_native := { buf: st.buf, flushed: st.flushed, saw_native: if list.is_empty(native) {
    st.saw_native
  } else {
    true
  } }
  match step_text(after_native, fragment) {
    (next_st, text_deltas) => {
      let tail := if done {
        finish_deltas(next_st, j)
      } else {
        []
      }
      (encode_stream_state(next_st), list.concat(native, list.concat(text_deltas, tail)))
    },
  }
}

# `content` is the visible text; a thinking model puts its trace in
# `thinking` and leaves content empty, same as the buffered path.
fn chunk_text(mj :: jv.Json) -> Str {
  let content := match jv.get_field(mj, "content") {
    Some(JStr(s)) => s,
    _ => "",
  }
  if str.is_empty(content) {
    match jv.get_field(mj, "thinking") {
      Some(JStr(s)) => s,
      _ => "",
    }
  } else {
    content
  }
}

# Accumulate, and flush once the buffer proves it is not XML markup.
fn step_text(st :: StreamState, fragment :: Str) -> (StreamState, List[d.Delta]) {
  if st.flushed {
    if str.is_empty(fragment) {
      (st, [])
    } else {
      (st, [TextChunk(fragment)])
    }
  } else {
    let buf := str.concat(st.buf, fragment)
    if maybe_xml(buf) {
      ({ buf: buf, flushed: false, saw_native: st.saw_native }, [])
    } else {
      ({ buf: "", flushed: true, saw_native: st.saw_native }, [TextChunk(buf)])
    }
  }
}

# On the final chunk: whatever is still held back is either an XML tool call
# to parse or trailing text to emit, and then the turn's finish reason.
#
# A model that emitted native tool_calls never has its text reinterpreted as
# XML — saw_native settles it — so a reply that merely mentions
# `<function=` in prose alongside a real tool call is not parsed twice.
fn finish_deltas(st :: StreamState, j :: jv.Json) -> List[d.Delta] {
  let held := str.trim(st.buf)
  let tail := if str.is_empty(held) {
    []
  } else {
    if st.saw_native {
      [TextChunk(st.buf)]
    } else {
      {
        let calls := parse_xml_tool_calls(st.buf)
        if list.is_empty(calls) {
          [TextChunk(st.buf)]
        } else {
          calls
        }
      }
    }
  }
  list.concat(tail, [FinishDelta(finish_reason_from_msg(j))])
}

