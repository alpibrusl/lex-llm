# lex-llm — approval-gate tests (#41)
#
# Covers the dispatch-layer approval flow: with_approval metadata,
# _approval_answer injection, and exec_with_approval's behavior when no
# ApprovalSink is configured in the runtime (lex-lang's NullApprovalSink
# refuses every request, so the gated path must surface a recoverable
# approval_denied error rather than executing the tool).

import "../src/tool" as t

import "../src/human" as human

import "lex-schema/schema" as s

import "lex-schema/json_value" as jv

import "lex-schema/error" as e

import "std.list" as list

import "std.str" as str

fn echo_params() -> s.ModelSchema {
  { title: "echo_params", description: "echo", fields: [s.required_str("text", [])] }
}

fn echo_tool() -> t.Tool {
  t.define("echo", "Echo the input text.", echo_params(), fn (args :: jv.Json) -> [net, io, proc] Result[jv.Json, e.Errors] {
    match jv.get_field(args, "text") {
      Some(JStr(v)) => Ok(JStr(v)),
      _ => Ok(JStr("")),
    }
  })
}

# ---- metadata ----------------------------------------------------
fn test_define_has_no_approval_scope() -> Result[Unit, Str] {
  match echo_tool().approval_scope {
    None => Ok(()),
    Some(_) => Err("define should not set approval_scope"),
  }
}

fn test_with_approval_sets_scope() -> Result[Unit, Str] {
  let gated := t.with_approval(echo_tool(), "payment")
  match gated.approval_scope {
    Some(scope) => if scope == "payment" {
      Ok(())
    } else {
      Err(str.concat("wrong scope: ", scope))
    },
    None => Err("with_approval did not set approval_scope"),
  }
}

# ---- answer injection --------------------------------------------
fn test_inject_appends_answer_field() -> Result[Unit, Str] {
  let injected := t.inject_approval_answer(JObj([("question", JStr("deploy?"))]), "yes")
  match jv.get_field(injected, "_approval_answer") {
    Some(JStr(answer)) => if answer == "yes" {
      Ok(())
    } else {
      Err(str.concat("wrong answer: ", answer))
    },
    _ => Err("_approval_answer not injected"),
  }
}

fn test_inject_passes_non_object_through() -> Result[Unit, Str] {
  match t.inject_approval_answer(JStr("raw"), "yes") {
    JStr(v) => if v == "raw" {
      Ok(())
    } else {
      Err("non-object args were altered")
    },
    _ => Err("non-object args changed shape"),
  }
}

# ---- exec_with_approval ------------------------------------------
# Ungated tools go straight through validate + execute.
fn test_ungated_tool_executes() -> [net, io, proc, approval] Result[Unit, Str] {
  match t.exec_with_approval(echo_tool(), JObj([("text", JStr("hi"))])) {
    Ok(JStr(v)) => if v == "hi" {
      Ok(())
    } else {
      Err(str.concat("wrong result: ", v))
    },
    Ok(_) => Err("wrong result shape"),
    Err(errs) => Err(str.concat("unexpected error: ", e.format(errs))),
  }
}

# Gated tool with no ApprovalSink configured: the runtime's default sink
# refuses, so exec_with_approval must return approval_denied — and must
# NOT have executed the tool.
fn test_gated_tool_denied_without_sink() -> [net, io, proc, approval] Result[Unit, Str] {
  let gated := t.with_approval(echo_tool(), "payment")
  match t.exec_with_approval(gated, JObj([("text", JStr("hi"))])) {
    Ok(_) => Err("gated tool executed without an ApprovalSink"),
    Err(errs) => {
      let has_denied := list.fold(errs, false, fn (acc :: Bool, err :: e.Error) -> Bool {
        if acc {
          true
        } else {
          err.code == "approval_denied"
        }
      })
      if has_denied {
        Ok(())
      } else {
        Err(str.concat("expected approval_denied, got: ", e.format(errs)))
      }
    },
  }
}

# Invalid args on a gated tool fail validation BEFORE any approval
# request — the operator is never bothered with malformed calls.
fn test_gated_tool_validates_before_approval() -> [net, io, proc, approval] Result[Unit, Str] {
  let gated := t.with_approval(echo_tool(), "payment")
  match t.exec_with_approval(gated, JObj([])) {
    Ok(_) => Err("invalid args should not execute"),
    Err(errs) => {
      let has_denied := list.fold(errs, false, fn (acc :: Bool, err :: e.Error) -> Bool {
        if acc {
          true
        } else {
          err.code == "approval_denied"
        }
      })
      if has_denied {
        Err("approval was requested for args that fail validation")
      } else {
        Ok(())
      }
    },
  }
}

# ---- ask_human ---------------------------------------------------
fn test_ask_human_is_approval_scoped() -> Result[Unit, Str] {
  match human.make_ask_human_tool("human").approval_scope {
    Some(scope) => if scope == "human" {
      Ok(())
    } else {
      Err(str.concat("wrong scope: ", scope))
    },
    None => Err("ask_human should carry an approval_scope"),
  }
}

fn test_ask_human_returns_injected_answer() -> [net, io, proc] Result[Unit, Str] {
  let tool := human.make_ask_human_tool("human")
  let args := JObj([("question", JStr("deploy?")), ("_approval_answer", JStr("ship it"))])
  match tool.execute(args) {
    Ok(JStr(answer)) => if answer == "ship it" {
      Ok(())
    } else {
      Err(str.concat("wrong answer: ", answer))
    },
    _ => Err("ask_human did not return the injected answer"),
  }
}

fn run_all() -> [net, io, proc, approval] Unit {
  let results := [test_define_has_no_approval_scope(), test_with_approval_sets_scope(), test_inject_appends_answer_field(), test_inject_passes_non_object_through(), test_ungated_tool_executes(), test_gated_tool_denied_without_sink(), test_gated_tool_validates_before_approval(), test_ask_human_is_approval_scoped(), test_ask_human_returns_injected_answer()]
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

