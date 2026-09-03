# lex-llm — wrap-up nudge for verification-driven modes (lex-code#97)
#
# spec/test agents were observed live burning their full step budget even
# after finishing the task and verifying it: the loop only stops when the
# model's own response carries no tool calls, and a static system-prompt
# instruction to "stop once verification passes" did not change that
# behaviour. maybe_nudge_wrapup (and the small predicates it composes) is a
# mechanical alternative: inject a fresh, specific stop instruction the
# moment a verification tool call succeeds, rather than relying on prompt
# compliance from turn one.
#
# These test the decision logic directly against the exact Dispatch/conv
# shapes run_steps constructs, rather than simulating a full multi-turn
# provider dialogue — dispatch_calls itself is exercised elsewhere.

import "../src/agent" as ag

import "../src/provider" as prov

import "../src/message" as msg

import "../src/delta" as d

import "../src/tool" as t

import "std.str" as str

import "std.int" as int

import "std.io" as io

import "std.iter" as iter

import "std.list" as list

fn show_bool(b :: Bool) -> Str {
  if b {
    "true"
  } else {
    "false"
  }
}

fn expect_bool(label :: Str, got :: Bool, want :: Bool) -> Result[Unit, Str] {
  if got == want {
    Ok(())
  } else {
    Err(str.join([label, ": expected ", show_bool(want), " got ", show_bool(got)], ""))
  }
}

fn expect_int(label :: Str, got :: Int, want :: Int) -> Result[Unit, Str]
  examples {
    expect_int("n", 3, 3) => Ok(())
  }
{
  if got == want {
    Ok(())
  } else {
    Err(str.join([label, ": expected len ", int.to_str(want), " got ", int.to_str(got)], ""))
  }
}

fn dispatch(name :: Str, success :: Bool) -> ag.Dispatch {
  { call: { id: "c1", name: name, args_raw: "{}" }, success: success, content: "ok" }
}

fn stub_agent_named(name :: Str) -> ag.AgentLoop {
  { name: name, goal: "g", model: prov.make_model_ref("stub", "m"), provider: { name: "stub", chat: fn (_m :: prov.ModelRef, _msgs :: List[msg.Message], _tools :: List[t.Tool]) -> [net, llm] Iter[d.Delta] {
    iter.from_list([])
  }, stream: None }, tools: [], options: { temperature: None, top_p: None, max_steps: Some(5), max_tokens: None }, permission_spec: None }
}

# is_verification_tool: exactly the three tools spec/test modes use to
# confirm their own work, nothing else.
fn test_is_verification_tool() -> Result[Unit, Str] {
  match expect_bool("lex_check", ag.is_verification_tool("lex_check"), true) {
    Err(e) => Err(e),
    Ok(_) => match expect_bool("lex_spec_check", ag.is_verification_tool("lex_spec_check"), true) {
      Err(e) => Err(e),
      Ok(_) => match expect_bool("lex_test", ag.is_verification_tool("lex_test"), true) {
        Err(e) => Err(e),
        Ok(_) => match expect_bool("write is not verification", ag.is_verification_tool("write"), false) {
          Err(e) => Err(e),
          Ok(_) => expect_bool("read is not verification", ag.is_verification_tool("read"), false),
        },
      },
    },
  }
}

# wraps_up_on_verification: scoped to spec/test only. review already
# terminates on its own (concrete report-structure prompt); build/plan/
# explore/refactor/bar legitimately keep working past a lex_check pass, so
# nudging them to stop would be a regression, not a fix.
fn test_wraps_up_on_verification_scope() -> Result[Unit, Str] {
  match expect_bool("spec", ag.wraps_up_on_verification("spec"), true) {
    Err(e) => Err(e),
    Ok(_) => match expect_bool("test", ag.wraps_up_on_verification("test"), true) {
      Err(e) => Err(e),
      Ok(_) => match expect_bool("review excluded", ag.wraps_up_on_verification("review"), false) {
        Err(e) => Err(e),
        Ok(_) => match expect_bool("build excluded", ag.wraps_up_on_verification("build"), false) {
          Err(e) => Err(e),
          Ok(_) => expect_bool("plan excluded", ag.wraps_up_on_verification("plan"), false),
        },
      },
    },
  }
}

# had_successful_verification: true only when a *successful* dispatch of a
# verification tool is present — a failed lex_check, or a successful call
# to some other tool, must not trigger the nudge.
fn test_had_successful_verification() -> Result[Unit, Str] {
  let with_pass := [dispatch("read", true), dispatch("lex_check", true)]
  let with_fail := [dispatch("lex_check", false)]
  let without := [dispatch("read", true), dispatch("write", true)]
  match expect_bool("successful verification present", ag.had_successful_verification(with_pass), true) {
    Err(e) => Err(e),
    Ok(_) => match expect_bool("failed verification does not count", ag.had_successful_verification(with_fail), false) {
      Err(e) => Err(e),
      Ok(_) => expect_bool("no verification tool at all", ag.had_successful_verification(without), false),
    },
  }
}

# already_nudged: only recognises the exact nudge text as a prior nudge —
# an unrelated UserMsg must not suppress a nudge that's actually due.
fn test_already_nudged() -> Result[Unit, Str] {
  let fresh := [UserMsg("do the task")]
  let nudged := [UserMsg("do the task"), UserMsg(ag.wrapup_nudge_text())]
  match expect_bool("fresh conversation, not nudged", ag.already_nudged(fresh), false) {
    Err(e) => Err(e),
    Ok(_) => expect_bool("nudge text present, already nudged", ag.already_nudged(nudged), true),
  }
}

# maybe_nudge_wrapup: the actual call shape run_steps uses. Covers the four
# cases that matter: nudge appended for spec on a fresh pass; not
# duplicated on a second pass; not applied to build even on a real pass;
# untouched when there was nothing to verify.
fn test_maybe_nudge_wrapup_appends_once_for_spec() -> Result[Unit, Str] {
  let agent := stub_agent_named("spec")
  let dispatches := [dispatch("lex_check", true)]
  let conv := [UserMsg("write a spec")]
  let base := [UserMsg("write a spec"), AssistantMsg("done", [])]
  let result := ag.maybe_nudge_wrapup(agent, dispatches, conv, base)
  match expect_int("nudge appended", list.len(result), list.len(base) + 1) {
    Err(e) => Err(e),
    Ok(_) => expect_bool("last message is the nudge", result == list.concat(base, [UserMsg(ag.wrapup_nudge_text())]), true),
  }
}

fn test_maybe_nudge_wrapup_does_not_duplicate() -> Result[Unit, Str] {
  let agent := stub_agent_named("test")
  let dispatches := [dispatch("lex_test", true)]
  let conv := [UserMsg("write tests"), UserMsg(ag.wrapup_nudge_text())]
  let base := list.concat(conv, [AssistantMsg("still going", [])])
  let result := ag.maybe_nudge_wrapup(agent, dispatches, conv, base)
  expect_bool("no second nudge appended", result == base, true)
}

fn test_maybe_nudge_wrapup_skips_build_mode() -> Result[Unit, Str] {
  let agent := stub_agent_named("build")
  let dispatches := [dispatch("lex_check", true)]
  let conv := [UserMsg("implement the feature")]
  let base := [UserMsg("implement the feature"), AssistantMsg("wrote it", [])]
  let result := ag.maybe_nudge_wrapup(agent, dispatches, conv, base)
  expect_bool("build mode untouched by a lex_check pass", result == base, true)
}

fn test_maybe_nudge_wrapup_skips_without_verification() -> Result[Unit, Str] {
  let agent := stub_agent_named("spec")
  let dispatches := [dispatch("read", true), dispatch("grep", true)]
  let conv := [UserMsg("write a spec")]
  let base := [UserMsg("write a spec"), AssistantMsg("still exploring", [])]
  let result := ag.maybe_nudge_wrapup(agent, dispatches, conv, base)
  expect_bool("no verification tool ran, no nudge", result == base, true)
}

fn run_all() -> [io] Int {
  let results := [("is_verification_tool", test_is_verification_tool()), ("wraps_up_on_verification_scope", test_wraps_up_on_verification_scope()), ("had_successful_verification", test_had_successful_verification()), ("already_nudged", test_already_nudged()), ("maybe_nudge_wrapup_appends_once_for_spec", test_maybe_nudge_wrapup_appends_once_for_spec()), ("maybe_nudge_wrapup_does_not_duplicate", test_maybe_nudge_wrapup_does_not_duplicate()), ("maybe_nudge_wrapup_skips_build_mode", test_maybe_nudge_wrapup_skips_build_mode()), ("maybe_nudge_wrapup_skips_without_verification", test_maybe_nudge_wrapup_skips_without_verification())]
  list.fold(results, 0, fn (failures :: Int, entry :: (Str, Result[Unit, Str])) -> [io] Int {
    match entry {
      (name, result) => match result {
        Ok(_) => {
          let __ok := io.print(str.concat("  ok   ", name))
          failures
        },
        Err(e) => {
          let __bad := io.print(str.join(["  FAIL ", name, ": ", e], ""))
          failures + 1
        },
      },
    }
  })
}

