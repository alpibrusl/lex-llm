import "../src/providers/ollama" as olla

import "../src/providers/openai" as oai

import "../src/delta" as d

import "lex-schema/json_value" as jv

import "std.iter" as iter

import "std.list" as list

import "std.str" as str

import "std.int" as int

fn delta_id_sig(dl :: d.Delta) -> Str {
  match dl {
    TextChunk(_) => "Text",
    ToolCallBegin(id, name) => str.concat("Begin:", str.concat(id, str.concat(":", name))),
    ToolArgChunk(id, _) => str.concat("Args:", id),
    FinishDelta(r) => str.concat("Finish:", r),
  }
}

fn id_signature(deltas :: List[d.Delta]) -> Str {
  str.join(list.map(deltas, delta_id_sig), "|")
}

fn assert_eq(name :: Str, got :: Str, want :: Str) -> Result[Unit, Str] {
  if got == want {
    Ok(())
  } else {
    Err(str.concat(name, str.concat(" expected ", str.concat(want, str.concat(", got ", got)))))
  }
}

fn test_ollama_preserves_native_tool_call_ids() -> Result[Unit, Str] {
  let line := "{\"model\":\"llama3\",\"message\":{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"id\":\"call_a\",\"function\":{\"name\":\"write\",\"arguments\":{\"path\":\"a.lex\"}}},{\"id\":\"call_b\",\"function\":{\"name\":\"write\",\"arguments\":{\"path\":\"b.lex\"}}}]},\"done\":true}"
  let got := id_signature(iter.to_list(olla.parse_stream([line])))
  assert_eq("ollama native ids", got, "Begin:call_a:write|Args:call_a|Begin:call_b:write|Args:call_b|Finish:tool_calls")
}

fn test_ollama_xml_synthesizes_unique_tool_call_ids() -> Result[Unit, Str] {
  let content := "<tool_call>\n<function=write>\n<parameter=path>\na.lex\n</parameter>\n</function>\n<function=write>\n<parameter=path>\nb.lex\n</parameter>\n</function>\n</tool_call>"
  let got := id_signature(olla.parse_xml_tool_calls(content))
  assert_eq("ollama xml ids", got, "Begin:call_write_0:write|Args:call_write_0|Begin:call_write_1:write|Args:call_write_1")
}

fn parse_openai_completion(body :: Str) -> List[d.Delta] {
  match jv.parse_into_errors(body) {
    Ok(j) => oai.parse_completion(j),
    Err(_) => [],
  }
}

fn test_openai_synthesizes_unique_missing_tool_call_ids() -> Result[Unit, Str] {
  let body := "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"type\":\"function\",\"function\":{\"name\":\"write\",\"arguments\":\"{\\\"path\\\":\\\"a.lex\\\"}\"}},{\"type\":\"function\",\"function\":{\"name\":\"write\",\"arguments\":\"{\\\"path\\\":\\\"b.lex\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}"
  let got := id_signature(parse_openai_completion(body))
  assert_eq("openai fallback ids", got, "Begin:call_write_0:write|Args:call_write_0|Begin:call_write_1:write|Args:call_write_1|Finish:tool_calls")
}

fn run_all() -> Unit {
  let results := [test_ollama_preserves_native_tool_call_ids(), test_ollama_xml_synthesizes_unique_tool_call_ids(), test_openai_synthesizes_unique_missing_tool_call_ids()]
  let failures := list.fold(results, 0, fn (n :: Int, r :: Result[Unit, Str]) -> Int {
    match r {
      Ok(_) => n,
      Err(_) => n + 1,
    }
  })
  if failures == 0 {
    ()
  } else {
    let __discard := 1 / 0
    ()
  }
}

