# lex-llm — Google Vertex AI adapter (Gemini via Vertex AI regional endpoint)
#
# Uses the Vertex AI streamGenerateContent REST endpoint with API-key auth.
# The API key must have the Vertex AI API enabled in your Google Cloud project.
#
# EU endpoint: europe-west1-aiplatform.googleapis.com
# Model: gemini-3.5-flash (or any Gemini model available on Vertex AI)
#
# Required env vars (consumed via providers.vertex()):
#   VERTEX_API_KEY   — Google Cloud API key with Vertex AI API enabled
#   VERTEX_PROJECT   — GCP project ID
#   VERTEX_LOCATION  — region (default: europe-west1)
#
# URL shape:
#   https://{location}-aiplatform.googleapis.com/v1/projects/{project}/
#     locations/{location}/publishers/google/models/{model}:streamGenerateContent?key={api_key}
#
# Response format is identical to the Google AI Studio Gemini API,
# so parse_stream / parse_chunk follow the same structure as providers/google.lex.

import "../message" as msg

import "../delta" as d

import "../tool" as t

import "../provider" as prov

import "lex-schema/json_value" as jv

import "std.http" as http

import "std.bytes" as bytes

import "std.map" as map

import "std.list" as list

import "std.str" as str

import "std.iter" as iter

# ── Config ────────────────────────────────────────────────────────────────────
# access_token: OAuth2 Bearer token (from `gcloud auth print-access-token`)
#               or a GCP service account access token.
type VertexConfig = { access_token :: Str, project_id :: Str, location :: Str }

fn default_config(access_token :: Str, project_id :: Str) -> VertexConfig {
  { access_token: access_token, project_id: project_id, location: "eu" }
}

fn config_at(access_token :: Str, project_id :: Str, location :: Str) -> VertexConfig {
  { access_token: access_token, project_id: project_id, location: location }
}

# ── URL builder ───────────────────────────────────────────────────────────────
# The OAuth2 token goes in the `Authorization: Bearer` header (see chat()).
# Google rejects the `?access_token=` query param for streamGenerateContent
# (HTTP 401 ACCESS_TOKEN_TYPE_UNSUPPORTED), so the URL carries no token.
#
# Multi-region codes ("eu", "us", "global") use the .rep.googleapis.com endpoint:
#   https://aiplatform.eu.rep.googleapis.com/v1/projects/.../locations/eu/...
# Regional codes ("europe-west1", etc.) use the legacy regional endpoint:
#   https://europe-west1-aiplatform.googleapis.com/v1/projects/.../locations/europe-west1/...
fn vertex_url(cfg :: VertexConfig, model :: Str) -> Str {
  match cfg.location {
    "eu" => str.join(["https://aiplatform.eu.rep.googleapis.com/v1/projects/", cfg.project_id, "/locations/eu/publishers/google/models/", model, ":streamGenerateContent"], ""),
    "us" => str.join(["https://aiplatform.us.rep.googleapis.com/v1/projects/", cfg.project_id, "/locations/us/publishers/google/models/", model, ":streamGenerateContent"], ""),
    "global" => str.join(["https://aiplatform.googleapis.com/v1/projects/", cfg.project_id, "/locations/global/publishers/google/models/", model, ":streamGenerateContent"], ""),
    loc => str.join(["https://", loc, "-aiplatform.googleapis.com/v1/projects/", cfg.project_id, "/locations/", loc, "/publishers/google/models/", model, ":streamGenerateContent"], ""),
  }
}

# ── Model refs ───────────────────────────────────────────────────────────────
fn gemini_35_flash() -> prov.ModelRef {
  { provider: "vertex", model: "gemini-3.5-flash" }
}

fn gemini_35_pro() -> prov.ModelRef {
  { provider: "vertex", model: "gemini-3.5-pro" }
}

# ── Provider factory ──────────────────────────────────────────────────────────
fn make_provider(config :: VertexConfig) -> prov.Provider {
  { name: "vertex", chat: fn (model :: prov.ModelRef, messages :: List[msg.Message], tools :: List[t.Tool]) -> [net, llm] Iter[d.Delta] {
    chat(config, model, messages, tools)
  } }
}

fn chat(config :: VertexConfig, model :: prov.ModelRef, messages :: List[msg.Message], tools :: List[t.Tool]) -> [net, llm] Iter[d.Delta] {
  let url := vertex_url(config, model.model)
  let body := build_request(messages, tools)
  let req := http.with_header(http.with_header({ method: "POST", url: url, headers: map.new(), body: Some(bytes.from_str(body)), timeout_ms: Some(60000) }, "Content-Type", "application/json"), "Authorization", str.concat("Bearer ", config.access_token))
  let body_str := match http.send(req) {
    Err(_) => "",
    Ok(r) => match bytes.to_str(r.body) {
      Err(_) => "",
      Ok(s) => s,
    },
  }
  parse_stream(body_str)
}

# ── Request building ──────────────────────────────────────────────────────────
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
  let with_config := list.concat(with_tools, [("generationConfig", JObj([("thinkingConfig", JObj([("thinkingBudget", JInt(1024))]))]))])
  jv.stringify(JObj(with_config))
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

# Extract the thoughtSignature stored in call_id after the "|||" separator.
fn thought_sig_from_id(call_id :: Str) -> Str {
  if str.contains(call_id, "|||") {
    let parts := str.split(call_id, "|||")
    let base := match list.head(parts) {
      Some(s) => s,
      None => "",
    }
    match str.strip_prefix(call_id, str.concat(base, "|||")) {
      Some(ts) => ts,
      None => "",
    }
  } else {
    ""
  }
}

# Extract the bare function name from call_id, stripping "call_" prefix and any "|||ts" suffix.
fn fn_name_from_id(call_id :: Str) -> Str {
  let base := if str.contains(call_id, "|||") {
    let parts := str.split(call_id, "|||")
    match list.head(parts) {
      Some(s) => s,
      None => call_id,
    }
  } else {
    call_id
  }
  match str.strip_prefix(base, "call_") {
    Some(name) => name,
    None => base,
  }
}

fn encode_content(m :: msg.Message) -> jv.Json {
  match m {
    UserMsg(text) => JObj([("role", JStr("user")), ("parts", JList([JObj([("text", JStr(text))])]))]),
    AssistantMsg(text, calls) => if list.is_empty(calls) {
      JObj([("role", JStr("model")), ("parts", JList([JObj([("text", JStr(text))])]))])
    } else {
      JObj([("role", JStr("model")), ("parts", JList(list.map(calls, fn (c :: msg.ToolCall) -> jv.Json {
        let fc_obj := JObj([("name", JStr(c.name)), ("args", c.args)])
        let ts := thought_sig_from_id(c.id)
        if str.is_empty(ts) {
          JObj([("functionCall", fc_obj)])
        } else {
          JObj([("functionCall", fc_obj), ("thoughtSignature", JStr(ts))])
        }
      })))])
    },
    ToolMsg(call_id, content) => JObj([("role", JStr("user")), ("parts", JList([JObj([("functionResponse", JObj([("name", JStr(fn_name_from_id(call_id))), ("response", JObj([("output", JStr(content))]))]))])]))]),
    SystemMsg(_) => JObj([("role", JStr("user")), ("parts", JList([JObj([("text", JStr(""))])]))]),
  }
}

# ── Response parsing ──────────────────────────────────────────────────────────
# The multi-region endpoint (aiplatform.eu.rep.googleapis.com) returns a JSON
# array: [{...}, {...}]. The legacy regional endpoint returns NDJSON (one object
# per line). We detect by trying to parse the whole body as JSON first.
#
# Gemini 3.5 Flash on the EU endpoint omits finishReason from all chunks.
# Without a FinishDelta the agent loop's collect_response never flips
# finish_reason to "tool_calls", so tool calls are silently dropped and content
# comes back empty. We append a synthetic FinishDelta("stop") when none was
# emitted; collect_response then detects any accumulated tool calls correctly.
fn parse_stream(body :: Str) -> Iter[d.Delta] {
  let body2 := str.join(str.split(body, "\\u003e"), ">")
  let body3 := str.join(str.split(body2, "\\u003c"), "<")
  let body4 := str.join(str.split(body3, "\\u0026"), "&")
  let raw := match jv.parse_into_errors(body4) {
    Ok(JList(chunks)) => list.fold(chunks, [], fn (acc :: List[d.Delta], chunk :: jv.Json) -> List[d.Delta] {
      list.concat(acc, parse_chunk(chunk))
    }),
    _ => list.fold(str.split(body4, "\n"), [], fn (acc :: List[d.Delta], line :: Str) -> List[d.Delta] {
      let trimmed := str.trim(line)
      if str.is_empty(trimmed) {
        acc
      } else {
        match jv.parse_into_errors(trimmed) {
          Err(_) => acc,
          Ok(j) => list.concat(acc, parse_chunk(j)),
        }
      }
    }),
  }
  let has_finish := list.fold(raw, false, fn (acc :: Bool, dl :: d.Delta) -> Bool {
    match dl {
      FinishDelta(_) => true,
      _ => acc,
    }
  })
  iter.from_list(if has_finish {
    raw
  } else {
    list.concat(raw, [FinishDelta("stop")])
  })
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
    Some(JStr(r)) => [FinishDelta(normalise_finish(r))],
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
  match jv.get_field(part, "functionCall") {
    Some(fc) => {
      let name := str_field(fc, "name")
      let thought_sig := match jv.get_field(part, "thoughtSignature") {
        Some(JStr(ts)) => ts,
        _ => "",
      }
      let id := if str.is_empty(thought_sig) {
        str.concat("call_", name)
      } else {
        str.join(["call_", name, "|||", thought_sig], "")
      }
      let args := match jv.get_field(fc, "args") {
        Some(aj) => jv.stringify(aj),
        None => "{}",
      }
      [ToolCallBegin(id, name), ToolArgChunk(id, args)]
    },
    None => match jv.get_field(part, "text") {
      Some(JStr(s)) => if str.is_empty(s) {
        []
      } else {
        [TextChunk(s)]
      },
      _ => [],
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

