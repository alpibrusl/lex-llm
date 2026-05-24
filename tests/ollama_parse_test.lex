import "std.iter" as iter

import "std.list" as list

import "std.str" as str

import "std.int" as int

import "lex-schema/json_value" as jv

type Delta = TextChunk(Str) | ToolCallBegin((Str, Str)) | ToolArgChunk((Str, Str)) | FinishDelta(Str)

fn parse_tool_calls(calls :: List[jv.Json]) -> List[Delta] {
  list.fold(calls, [], fn (acc :: List[Delta], cj :: jv.Json) -> List[Delta] {
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

fn parse_assistant_message(mj :: jv.Json) -> List[Delta] {
  let text_deltas := match jv.get_field(mj, "content") {
    Some(JStr(s)) => if str.is_empty(s) {
      []
    } else {
      [TextChunk(s)]
    },
    _ => [],
  }
  let call_deltas := match jv.get_field(mj, "tool_calls") {
    Some(JList(calls)) => parse_tool_calls(calls),
    _ => [],
  }
  list.concat(text_deltas, call_deltas)
}

fn parse_chunk(j :: jv.Json) -> List[Delta] {
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
    [FinishDelta("stop")]
  } else {
    []
  }
  list.concat(msg_deltas, finish_deltas)
}

fn parse_stream(lines :: List[Str]) -> List[Delta] {
  list.fold(lines, [], fn (acc :: List[Delta], line :: Str) -> List[Delta] {
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
}

fn delta_tag(d :: Delta) -> Str {
  match d {
    TextChunk(s) => str.concat("Text:", s),
    ToolCallBegin(i, n) => str.concat("TCBegin:", n),
    ToolArgChunk(i, _) => str.concat("TCArgs:", i),
    FinishDelta(r) => str.concat("Finish:", r),
  }
}

fn main() -> Str {
  let line1 := "{\"model\":\"gemma4:26b\",\"message\":{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"id\":\"call_1\",\"function\":{\"name\":\"write\",\"arguments\":{\"path\":\"hello.lex\",\"content\":\"fn hello() -> Str { \\\"hi\\\" }\"}}}]},\"done\":false}"
  let line2 := "{\"model\":\"gemma4:26b\",\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true,\"done_reason\":\"stop\"}"
  let r1 := {
    let lines := [line1, line2]
    let deltas := parse_stream(lines)
    let tags := list.map(deltas, delta_tag)
    str.concat("orig: ", str.concat(int.to_str(list.len(deltas)), str.concat(": ", str.join(tags, ","))))
  }
  let live1 := "{\"model\":\"gemma4:latest\",\"message\":{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"id\":\"call_eftl5i9l\",\"function\":{\"index\":0,\"name\":\"write\",\"arguments\":{\"content\":\"fn sum() -> Int { 42 }\",\"path\":\"sum.lex\"}}}]},\"done\":false}"
  let live2 := "{\"model\":\"gemma4:latest\",\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true,\"done_reason\":\"stop\"}"
  let r2 := {
    let lines := [live1, live2]
    let deltas := parse_stream(lines)
    let tags := list.map(deltas, delta_tag)
    str.concat("live: ", str.concat(int.to_str(list.len(deltas)), str.concat(": ", str.join(tags, ","))))
  }
  str.concat(r1, str.concat(" | ", r2))
}

