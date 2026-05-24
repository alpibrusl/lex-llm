# lex-llm — structured.lex pure-helper tests
#
# Tests collect_text and extract_json_block, which are pure and can
# run without any effect declarations. validate_reply and structured
# require a live model and are not unit-testable here.

import "../src/structured" as st

import "../src/delta" as d

import "std.str" as str

import "std.list" as list

# ---- collect_text -----------------------------------------------
fn test_collect_text_empty() -> Result[Unit, Str] {
  let got := st.collect_text([])
  if got == "" {
    Ok(())
  } else {
    Err(str.concat("expected empty string, got: ", got))
  }
}

fn test_collect_text_single() -> Result[Unit, Str] {
  let got := st.collect_text([d.TextChunk("hello")])
  if got == "hello" {
    Ok(())
  } else {
    Err(str.concat("expected 'hello', got: ", got))
  }
}

fn test_collect_text_multi() -> Result[Unit, Str] {
  let got := st.collect_text([d.TextChunk("foo"), d.TextChunk(" "), d.TextChunk("bar")])
  if got == "foo bar" {
    Ok(())
  } else {
    Err(str.concat("expected 'foo bar', got: ", got))
  }
}

fn test_collect_text_ignores_non_text() -> Result[Unit, Str] {
  let got := st.collect_text([d.TextChunk("a"), d.ToolCallBegin("id1", "fn1"), d.ToolArgChunk("id1", "{"), d.TextChunk("b"), d.FinishDelta("stop")])
  if got == "ab" {
    Ok(())
  } else {
    Err(str.concat("expected 'ab', got: ", got))
  }
}

# ---- extract_json_block -----------------------------------------
fn test_extract_plain_json() -> Result[Unit, Str] {
  let input := "{\"x\":1}"
  let got := st.extract_json_block(input)
  if got == input {
    Ok(())
  } else {
    Err(str.concat("expected passthrough, got: ", got))
  }
}

fn test_extract_json_fence() -> Result[Unit, Str] {
  let got := st.extract_json_block("```json\n{\"x\":1}\n```")
  if got == "{\"x\":1}" {
    Ok(())
  } else {
    Err(str.concat("expected unwrapped JSON, got: ", got))
  }
}

fn test_extract_plain_fence() -> Result[Unit, Str] {
  let got := st.extract_json_block("```\n{\"x\":1}\n```")
  if got == "{\"x\":1}" {
    Ok(())
  } else {
    Err(str.concat("expected unwrapped JSON, got: ", got))
  }
}

fn test_extract_strips_outer_whitespace() -> Result[Unit, Str] {
  let got := st.extract_json_block("  ```json\n{\"x\":1}\n```  ")
  if got == "{\"x\":1}" {
    Ok(())
  } else {
    Err(str.concat("expected trimmed JSON, got: ", got))
  }
}

fn test_extract_empty() -> Result[Unit, Str] {
  let got := st.extract_json_block("")
  if got == "" {
    Ok(())
  } else {
    Err(str.concat("expected empty, got: ", got))
  }
}

# ---- Suite ------------------------------------------------------
fn run_all() -> Unit {
  let results := [test_collect_text_empty(), test_collect_text_single(), test_collect_text_multi(), test_collect_text_ignores_non_text(), test_extract_plain_json(), test_extract_json_fence(), test_extract_plain_fence(), test_extract_strips_outer_whitespace(), test_extract_empty()]
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

