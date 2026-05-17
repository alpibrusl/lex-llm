# lex-llm — SSE line consumer abstraction
#
# Prerequisite spike: alpibrusl/lex-lang#487 investigates whether
# std.http already supports client-side SSE streaming. Until that
# spike closes this module uses an EAGER STUB: POST, await full body,
# split on newlines. This works for development and small responses.
#
# UPGRADE PATH once #487 resolves — replace stream_lines with
# whichever option the spike recommends:
#
#   Option A (extend std.http):
#     http.stream_lines(url, headers, body) -> [net] Iter[Str]
#
#   Option B (new std.sse module):
#     sse.consume(url, headers, body) |> iter.map(fn(ev) -> Str { ev.data })
#
#   Option C (no stdlib change — existing primitive confirmed):
#     Replace with the confirmed streaming primitive from the spike report.
#
# All provider adapters call sse.stream_lines — updating this single
# function switches all four providers simultaneously.

import "std.http" as http
import "std.str"  as str
import "std.iter" as iter
import "std.list" as list
import "std.map"  as map

# Stream SSE lines from a POST request.
# EAGER STUB — awaits the full response before yielding any lines.
# Replace body with a true streaming call once lex-lang#487 resolves.
fn stream_lines(
  url     :: Str,
  headers :: Map[Str, Str],
  body    :: Str
) -> [net] Iter[Str] {
  match http.post(url, headers, body) {
    Err(_)   => iter.from_list([]),
    Ok(resp) => iter.from_list(str.lines(resp.body)),
  }
}

# Parse one SSE line: strip "data: " prefix, return None for comments/blanks.
fn parse_data_line(line :: Str) -> Option[Str]
  examples {
    parse_data_line("data: {\"x\":1}") => Some("{\"x\":1}"),
    parse_data_line("")               => None,
    parse_data_line(": comment")      => None,
  }
{
  match str.strip_prefix(line, "data: ") {
    Some(data) => Some(data),
    None       => None,
  }
}

# Collect data payloads from an SSE line stream, stopping at "[DONE]".
fn data_payloads(lines :: Iter[Str]) -> [net] List[Str] {
  list.fold(iter.collect(lines), [],
    fn (acc :: List[Str], line :: Str) -> List[Str] {
      match parse_data_line(line) {
        None       => acc,
        Some(data) =>
          if data == "[DONE]" { acc }
          else { list.concat(acc, [data]) },
      }
    })
}

# Standard headers for a streaming JSON POST with Bearer auth.
fn json_post_headers(api_key :: Str) -> Map[Str, Str] {
  map.from_list([
    ("authorization", str.concat("Bearer ", api_key)),
    ("content-type",  "application/json"),
    ("accept",        "text/event-stream"),
  ])
}

# Headers for local endpoints (Ollama) — no auth.
fn local_post_headers() -> Map[Str, Str] {
  map.from_list([
    ("content-type", "application/json"),
    ("accept",       "text/event-stream"),
  ])
}
