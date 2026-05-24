# lex-llm — Google Gemini adapter
#
# Implements Provider.chat against the Gemini streamGenerateContent
# endpoint. Gemini streams NDJSON responses (one JSON object per line)
# rather than text/event-stream, but http.stream_lines handles both since
# both arrive as line-delimited text over HTTP.
#
# Supported: gemini-2.0-flash, gemini-2.5-pro, gemini-1.5-*

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

type GoogleConfig = { api_key :: Str }

fn default_config(api_key :: Str) -> GoogleConfig {
  { api_key: api_key }
}

fn gemini_url(model :: Str, api_key :: Str) -> Str {
  str.concat("https://generativelanguage.googleapis.com/v1beta/models/", str.concat(model, str.concat(":streamGenerateContent?key=", api_key)))
}

fn make_provider(config :: GoogleConfig) -> prov.Provider {
  { name: "google", chat: fn (model :: prov.ModelRef, messages :: List[msg.Message], tools :: List[t.Tool]) -> [net, llm] Iter[d.Delta] {
    chat(config, model, messages, tools)
  } }
}

fn chat(config :: GoogleConfig, model :: prov.ModelRef, messages :: List[msg.Message], tools :: List[t.Tool]) -> [net, llm] Iter[d.Delta] {
  let url := gemini_url(model.model, config.api_key)
  let body := build_request(messages, tools)
  let raw_lines := match http.post(url, bytes.from_str(body), "application/json") {
    Err(_) => iter.from_list([]),
    Ok(r) => iter.from_list(str.split(match bytes.to_str(r.body) {
      Err(_) => "",
      Ok(s) => s,
    }, "\n")),
  }
  let lines := iter.to_list(raw_lines)
  parse_stream(lines)
}

# ---- Request building --------------------------------------------
fn build_request(messages :: List[msg.Message], tools :: List[t.Tool]) -> Str {
  let _em := encode_messages(messages)
  let sys_opt := match _em {
    (s, _) => s,
  }
  let contents := match _em {
    (_, c) => c,
  }
  let base := [("contents", JList(contents))]
  let with_sys := match sys_opt {
    None => base,
    Some(sys) => list.concat(base, [("systemInstruction", JObj([("parts", JList([JObj([("text", JStr(sys))])]))]))]),
  }
  let with_tools := if list.is_empty(tools) {
    with_sys
  } else {
    list.concat(with_sys, [("tools", JList([JObj([("functionDeclarations", JList(list.map(tools, t.to_google_json)))])]))])
  }
  jv.stringify(JObj(with_tools))
}

fn encode_messages(messages :: List[msg.Message]) -> (Option[Str], List[jv.Json]) {
  let sys := list.fold(messages, None, fn (acc :: Option[Str], m :: msg.Message) -> Option[Str] {
    match m {
      SystemMsg(s) => Some(s),
      _ => acc,
    }
  })
  let contents := list.fold(messages, [], fn (acc :: List[jv.Json], m :: msg.Message) -> List[jv.Json] {
    match m {
      SystemMsg(_) => acc,
      _ => list.concat(acc, [encode_content(m)]),
    }
  })
  (sys, contents)
}

fn encode_content(m :: msg.Message) -> jv.Json {
  match m {
    UserMsg(text) => JObj([("role", JStr("user")), ("parts", JList([JObj([("text", JStr(text))])]))]),
    AssistantMsg(text, calls) => if list.is_empty(calls) {
      JObj([("role", JStr("model")), ("parts", JList([JObj([("text", JStr(text))])]))])
    } else {
      JObj([("role", JStr("model")), ("parts", JList(list.map(calls, fn (c :: msg.ToolCall) -> jv.Json {
        JObj([("functionCall", JObj([("name", JStr(c.name)), ("args", c.args)]))])
      })))])
    },
    ToolMsg(call_id, content) => JObj([("role", JStr("user")), ("parts", JList([JObj([("functionResponse", JObj([("name", JStr(call_id)), ("response", JObj([("output", JStr(content))]))]))])]))]),
    SystemMsg(_) => JObj([("role", JStr("user")), ("parts", JList([JObj([("text", JStr(""))])]))]),
  }
}

# ---- Response parsing --------------------------------------------
fn parse_stream(lines :: List[Str]) -> Iter[d.Delta] {
  let deltas := list.fold(lines, [], fn (acc :: List[d.Delta], line :: Str) -> List[d.Delta] {
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
  iter.from_list(deltas)
}

fn parse_chunk(j :: jv.Json) -> List[d.Delta] {
  match jv.get_field(j, "candidates") {
    Some(JList(cands)) => match first(cands) {
      None => [],
      Some(c) => parse_candidate(c),
    },
    _ => [],
  }
}

fn parse_candidate(cand :: jv.Json) -> List[d.Delta] {
  let content_deltas := match jv.get_field(cand, "content") {
    None => [],
    Some(c) => parse_parts(c),
  }
  let finish_deltas := match jv.get_field(cand, "finishReason") {
    Some(JStr(r)) => [d.FinishDelta(normalise_finish(r))],
    _ => [],
  }
  list.concat(content_deltas, finish_deltas)
}

fn parse_parts(content :: jv.Json) -> List[d.Delta] {
  match jv.get_field(content, "parts") {
    Some(JList(parts)) => list.fold(parts, [], fn (acc :: List[d.Delta], part :: jv.Json) -> List[d.Delta] {
      list.concat(acc, parse_part(part))
    }),
    _ => [],
  }
}

fn parse_part(part :: jv.Json) -> List[d.Delta] {
  match jv.get_field(part, "text") {
    Some(JStr(s)) => if str.is_empty(s) {
      []
    } else {
      [d.TextChunk(s)]
    },
    _ => match jv.get_field(part, "functionCall") {
      Some(fc) => {
        let name := str_field(fc, "name")
        let id := str.concat("call_", name)
        let args := match jv.get_field(fc, "args") {
          Some(aj) => jv.stringify(aj),
          None => "{}",
        }
        [d.ToolCallBegin(id, name), d.ToolArgChunk(id, args)]
      },
      None => [],
    },
  }
}

fn normalise_finish(reason :: Str) -> Str
  examples {
    normalise_finish("STOP") => "stop",
    normalise_finish("MAX_TOKENS") => "length"
  }
{
  match reason {
    "STOP" => "stop",
    "MAX_TOKENS" => "length",
    "SAFETY" => "content_filter",
    other => other,
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

