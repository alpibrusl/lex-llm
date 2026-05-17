# lex-llm — SSE parsing tests
#
# Tests parse_data_line (pure) and data_payloads ([net] due to Iter input).
# run_all uses a permissive policy so [net] is satisfied at runtime even
# though the Iter values here are eager (iter.from_list), not stream handles.

import "../src/sse"  as sse

import "std.str"  as str
import "std.int"  as int
import "std.list" as list
import "std.iter" as iter

# ---- parse_data_line --------------------------------------------

fn test_parse_data_line_data() -> Result[Unit, Str] {
  match sse.parse_data_line("data: {\"x\":1}") {
    Some(s) =>
      if s == "{\"x\":1}" { Ok(()) }
      else { Err(str.concat("wrong payload: ", s)) },
    None => Err("expected Some, got None"),
  }
}

fn test_parse_data_line_empty() -> Result[Unit, Str] {
  match sse.parse_data_line("") {
    None    => Ok(()),
    Some(s) => Err(str.concat("expected None, got Some(", str.concat(s, ")"))),
  }
}

fn test_parse_data_line_comment() -> Result[Unit, Str] {
  match sse.parse_data_line(": keep-alive") {
    None    => Ok(()),
    Some(s) => Err(str.concat("expected None, got Some(", str.concat(s, ")"))),
  }
}

fn test_parse_data_line_no_prefix() -> Result[Unit, Str] {
  match sse.parse_data_line("event: message") {
    None    => Ok(()),
    Some(s) => Err(str.concat("expected None, got Some(", str.concat(s, ")"))),
  }
}

# ---- data_payloads ----------------------------------------------

fn test_data_payloads_basic() -> [net] Result[Unit, Str] {
  let lines    := iter.from_list(["data: a", "data: b", "data: c"])
  let payloads := sse.data_payloads(lines)
  if list.length(payloads) == 3 { Ok(()) }
  else { Err(str.concat("expected 3, got ", int.to_str(list.length(payloads)))) }
}

fn test_data_payloads_stops_at_done() -> [net] Result[Unit, Str] {
  let lines    := iter.from_list(["data: a", "data: [DONE]", "data: b"])
  let payloads := sse.data_payloads(lines)
  if list.length(payloads) == 1 { Ok(()) }
  else { Err(str.concat("expected 1 (stops at [DONE]), got ", int.to_str(list.length(payloads)))) }
}

fn test_data_payloads_skips_comments() -> [net] Result[Unit, Str] {
  let lines    := iter.from_list([": comment", "data: ok", "event: ping"])
  let payloads := sse.data_payloads(lines)
  if list.length(payloads) == 1 { Ok(()) }
  else { Err(str.concat("expected 1, got ", int.to_str(list.length(payloads)))) }
}

fn test_data_payloads_empty() -> [net] Result[Unit, Str] {
  let payloads := sse.data_payloads(iter.from_list([]))
  if list.length(payloads) == 0 { Ok(()) }
  else { Err(str.concat("expected 0, got ", int.to_str(list.length(payloads)))) }
}

# ---- Suite ------------------------------------------------------

fn run_all() -> [net] Unit {
  let results := [
    test_parse_data_line_data(),
    test_parse_data_line_empty(),
    test_parse_data_line_comment(),
    test_parse_data_line_no_prefix(),
    test_data_payloads_basic(),
    test_data_payloads_stops_at_done(),
    test_data_payloads_skips_comments(),
    test_data_payloads_empty(),
  ]
  let failures := list.fold(results, 0, fn (n :: Int, r :: Result[Unit, Str]) -> Int {
    match r { Ok(_) => n, Err(_) => n + 1 }
  })
  if failures == 0 { () }
  else {
    let __discard := 1 / 0
    ()
  }
}
