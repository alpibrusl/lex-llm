# lex-llm — SSE parsing helpers
#
# Pure helpers for consuming SSE (text/event-stream) line payloads.
# The transport primitive is http.stream_lines from std.http (lex-lang#487),
# which performs a streaming HTTP POST and yields the response body
# line-by-line as Iter[Str]. Provider adapters call http.stream_lines
# directly and pass the resulting iterator into data_payloads.

import "std.str" as str

import "std.iter" as iter

import "std.list" as list

import "std.map" as map

# Drop a trailing CR.
#
# http.stream_lines splits on LF and leaves the CR of a CRLF pair on the end
# of the line. SSE is specified with CRLF, so a payload that reaches
# jv.parse with a stray CR fails to parse and the whole chunk is silently
# dropped — every adapter's step function needs this before parsing.
fn strip_cr(line :: Str) -> Str
  examples {
    strip_cr("data: {}\r") => "data: {}",
    strip_cr("data: {}") => "data: {}",
    strip_cr("") => ""
  }
{
  match str.strip_suffix(line, "\r") {
    Some(rest) => rest,
    None => line,
  }
}

# Parse one SSE line: strip "data: " prefix, return None for comments/blanks.
fn parse_data_line(line :: Str) -> Option[Str]
  examples {
    parse_data_line("data: {\"x\":1}") => Some("{\"x\":1}"),
    parse_data_line("") => None,
    parse_data_line(": comment") => None
  }
{
  match str.strip_prefix(line, "data: ") {
    Some(data) => Some(data),
    None => None,
  }
}

# Collect data payloads from an SSE line stream, stopping at "[DONE]".
fn data_payloads(lines :: Iter[Str]) -> List[Str] {
  let state := list.fold(iter.to_list(lines), ([], false), fn (acc :: (List[Str], Bool), line :: Str) -> (List[Str], Bool) {
    match acc {
      (lst, done) => if done {
        (lst, true)
      } else {
        match parse_data_line(line) {
          None => (lst, false),
          Some(data) => if data == "[DONE]" {
            (lst, true)
          } else {
            (list.concat(lst, [data]), false)
          },
        }
      },
    }
  })
  match state {
    (lst, _) => lst,
  }
}

# Standard headers for a streaming JSON POST with Bearer auth.
fn json_post_headers(api_key :: Str) -> Map[Str, Str] {
  map.from_list([("authorization", str.concat("Bearer ", api_key)), ("content-type", "application/json"), ("accept", "text/event-stream")])
}

# Headers for local endpoints (Ollama) — no auth.
fn local_post_headers() -> Map[Str, Str] {
  map.from_list([("content-type", "application/json"), ("accept", "text/event-stream")])
}

