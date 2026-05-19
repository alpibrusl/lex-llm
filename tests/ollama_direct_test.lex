import "../src/providers/ollama" as olla
import "../src/delta" as d
import "std.list" as list
import "std.str"  as str
import "std.int"  as int
import "std.iter" as iter

fn delta_tag(dl :: d.Delta) -> Str {
  match dl {
    d.TextChunk(s)        => str.concat("Text:", s),
    d.ToolCallBegin(i, n) => str.concat("TCBegin:", n),
    d.ToolArgChunk(i, _)  => str.concat("TCArgs:", i),
    d.FinishDelta(r)      => str.concat("Finish:", r),
  }
}

fn main() -> Str {
  # Test with live gemma4:latest tool_calls format (has id + function.index)
  let line1 := "{\"model\":\"gemma4:latest\",\"message\":{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"id\":\"call_eftl5i9l\",\"function\":{\"index\":0,\"name\":\"write\",\"arguments\":{\"content\":\"fn sum() -> Int { 42 }\",\"path\":\"sum.lex\"}}}]},\"done\":false}"
  let line2 := "{\"model\":\"gemma4:latest\",\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true,\"done_reason\":\"stop\"}"
  let lines := [line1, line2]
  let it    := olla.parse_stream(lines)
  let deltas := iter.to_list(it)
  let tags  := list.map(deltas, delta_tag)
  str.concat("direct: ", str.concat(int.to_str(list.len(deltas)), str.concat(": ", str.join(tags, ","))))
}
