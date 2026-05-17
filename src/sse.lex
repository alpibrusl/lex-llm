# lex-llm — SSE parsing helpers
#
# Pure helpers for consuming SSE (text/event-stream) line payloads.
# The transport primitive is http.stream_lines from std.http (lex-lang#487),
# which performs a streaming HTTP POST and yields the response body
# line-by-line as Iter[Str]. Provider adapters call http.stream_lines
# directly and pass the resulting iterator into data_payloads.

import "std.str"  as str
import "std.iter" as iter
import "std.list" as list
import "std.map"  as map

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
  list.fold(iter.to_list(lines), [],
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
