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
    UsageDelta(p, c, t) => str.join(["Usage:", int.to_str(p), ",", int.to_str(c), ",", int.to_str(t)], ""),
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

# Real per-call token cost (#94, loom) reads this UsageDelta -- confirm the
# top-level OpenAI-compatible "usage" object survives parse_completion intact
# and lands ahead of the message/finish deltas.
fn test_openai_usage_field_becomes_usage_delta() -> Result[Unit, Str] {
  let body := "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"hi\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":120,\"completion_tokens\":30,\"total_tokens\":150}}"
  let got := id_signature(parse_openai_completion(body))
  assert_eq("openai usage delta", got, "Usage:120,30,150|Text|Finish:stop")
}

# Absent "usage" (a provider/proxy that doesn't report it) must emit NO
# UsageDelta at all -- callers need to distinguish "not reported" from a real
# zero-token response, and a phantom Usage:0,0,0 would erase that distinction.
fn test_openai_missing_usage_emits_no_usage_delta() -> Result[Unit, Str] {
  let body := "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}"
  let got := id_signature(parse_openai_completion(body))
  assert_eq("openai no usage field", got, "Text|Finish:stop")
}

fn run_all() -> Unit {
  let results := [test_ollama_preserves_native_tool_call_ids(), test_ollama_xml_synthesizes_unique_tool_call_ids(), test_openai_synthesizes_unique_missing_tool_call_ids(), test_openai_usage_field_becomes_usage_delta(), test_openai_missing_usage_emits_no_usage_delta()]
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

