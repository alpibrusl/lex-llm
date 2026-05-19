import "std.iter" as iter
import "std.list" as list
import "std.int"  as int

import "../src/delta" as d

type Container = {
  get_deltas :: () -> Iter[d.Delta]
}

fn main() -> Str {
  let c := {
    get_deltas: fn() -> Iter[d.Delta] {
      iter.from_list([
        d.ToolCallBegin("id1", "write"),
        d.ToolArgChunk("id1", "{\"a\":1}"),
        d.FinishDelta("stop")
      ])
    }
  }
  let result := iter.to_list(c.get_deltas())
  int.to_str(list.len(result))
}
