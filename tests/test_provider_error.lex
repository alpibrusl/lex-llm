# lex-llm — provider failure surfacing
#
# Regression cover for the bug that made a transport failure look exactly
# like a model with nothing to say: every provider's chat() collapsed a
# timeout, a 5xx and an unreadable body alike into an empty delta list, and
# agent.run_steps reads an empty delta list as finish="stop" with empty
# content. Callers logged the failed call AS the model's own empty answer.
#
# Found live: a 120s timeout aborted a call the upstream went on to answer
# correctly ~240s later, and nothing anywhere recorded that a call had failed.

import "../src/delta" as d

import "std.str" as str

import "std.list" as list

import "std.io" as io

fn test_provider_error_is_not_empty() -> Result[Unit, Str] {
  if list.is_empty(d.provider_error("timed out")) {
    Err("a failed call must not produce an empty delta list -- that is exactly what run_steps reads as a successful empty answer")
  } else {
    Ok(())
  }
}

fn test_provider_error_carries_the_reason() -> Result[Unit, Str] {
  let text := list.fold(d.provider_error("HTTP 503"), "", fn (acc :: Str, dl :: d.Delta) -> Str {
    match dl {
      TextChunk(s) => str.concat(acc, s),
      _ => acc,
    }
  })
  if str.contains(text, "HTTP 503") {
    Ok(())
  } else {
    Err(str.concat("the reason must reach the caller as text, got: ", text))
  }
}

# The finish reason must not be "stop": run_steps recurses on "tool_calls" and
# treats everything else as a completed answer, so a failure that finished as
# "stop" would be indistinguishable from success at the call site.
fn test_provider_error_finishes_as_error_not_stop() -> Result[Unit, Str] {
  let reasons := list.fold(d.provider_error("boom"), [], fn (acc :: List[Str], dl :: d.Delta) -> List[Str] {
    match d.finish_reason(dl) {
      Some(r) => list.concat(acc, [r]),
      None => acc,
    }
  })
  match list.head(reasons) {
    None => Err("provider_error must emit a FinishDelta"),
    Some(r) => if r == "stop" {
      Err("a provider failure must not finish as 'stop'")
    } else {
      Ok(())
    },
  }
}

fn suite() -> List[Result[Unit, Str]] {
  [test_provider_error_is_not_empty(), test_provider_error_carries_the_reason(), test_provider_error_finishes_as_error_not_stop()]
}

fn run_all() -> [io] Unit {
  let results := suite()
  let __dbg := list.map(results, fn (r :: Result[Unit, Str]) -> [io] Unit {
    match r {
      Ok(_) => (),
      Err(e) => io.print(str.concat("FAIL: ", e)),
    }
  })
  let failures := list.fold(results, 0, fn (n :: Int, r :: Result[Unit, Str]) -> Int {
    match r {
      Ok(_) => n,
      Err(_) => n + 1,
    }
  })
  if failures == 0 {
    ()
  } else {
    let __force_fail := 1 / 0
    ()
  }
}

