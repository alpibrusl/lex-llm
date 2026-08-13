# lex-llm — Generic human-escalation tool
#
# make_ask_human_tool(scope) → Tool   (canonical, #41)
#
# When the LLM calls ask_human { question }, the dispatch layer sees the
# tool's approval_scope and blocks on std.approval.request(scope, question)
# — the [approval] host boundary from lex-lang#737. How a human actually
# gets asked (stdin prompt, dashboard long-poll, Slack ping, ...) is the
# embedding host's choice of ApprovalSink, not this module's concern. The
# operator's answer arrives in the args as `_approval_answer` (injected by
# tool.exec_with_approval) and is returned to the model verbatim. Denial
# or timeout surfaces as a recoverable approval_denied error.
#
# The scope is checked at run time against --allow-approval, so an
# operator can grant e.g. "human" escalations without granting other
# approval channels — or refuse them wholesale by omitting the grant.
#
# make_ask_human_tool_http(dash, customer) is the DEPRECATED predecessor:
# it hand-rolls the same escalate-and-block pattern over raw HTTP against
# a specific dashboard (POST dash/ask-human, long-poll dash/get-answer/
# q-<customer>), declared under generic [net, io, proc] effects. Kept one
# release for existing callers; migrate to make_ask_human_tool + an
# ApprovalSink that talks to your dashboard. Remove in the next minor.

import "std.str" as str

import "std.http" as http

import "std.bytes" as bytes

import "std.map" as map

import "std.io" as io

import "lex-schema/schema" as s

import "lex-schema/json_value" as jv

import "lex-schema/error" as e

import "./tool" as t

import "./message" as msg

fn ask_human_params() -> s.ModelSchema {
  { title: "ask_human_params", description: "Ask the human operator a question and wait for their reply", fields: [s.required_str("question", [])] }
}

# Canonical ask_human (#41): an approval-scoped tool. The execute body is
# pure — the blocking human round-trip happens in the dispatch layer via
# std.approval.request, and the operator's answer arrives pre-injected as
# `_approval_answer`. (The [net, io, proc] row below is the Tool.execute
# contract; a pure body satisfies it structurally.)
fn make_ask_human_tool(scope :: Str) -> t.Tool {
  let base := t.define("ask_human", "Pause and ask the human operator a question. Use when you face a genuine choice you cannot resolve autonomously, need explicit approval, or lack information the operator can supply. Returns the operator's answer as a string. Do not use for routine decisions.", ask_human_params(), fn (args :: jv.Json) -> [net, io, proc] Result[jv.Json, e.Errors] {
    match jv.get_field(args, "_approval_answer") {
      Some(JStr(answer)) => Ok(JStr(answer)),
      _ => Ok(JStr("(no reply)")),
    }
  })
  t.with_approval(base, scope)
}

# DEPRECATED — HTTP-dashboard predecessor of make_ask_human_tool; see the
# module header. Build an ask_human tool bound to a specific dashboard URL
# and customer identity. customer is included in the SSE event so the UI
# can attribute the question.
fn make_ask_human_tool_http(dash :: Str, customer :: Str) -> t.Tool {
  t.define("ask_human", "Pause and ask the human operator a question. Use when you face a genuine choice you cannot resolve autonomously, need explicit approval, or lack information the operator can supply. Returns the operator's answer as a string. Do not use for routine decisions.", ask_human_params(), fn (args :: jv.Json) -> [net, io, proc] Result[jv.Json, e.Errors] {
    let question := match jv.get_field(args, "question") {
      Some(JStr(v)) => v,
      _ => "",
    }
    let qid := str.concat("q-", customer)
    let _pi := io.print(str.join(["  [", customer, "] → ask_human: ", question], ""))
    let escaped := msg.json_escape(question)
    let post_body := str.join(["{\"id\":\"", qid, "\",\"customer\":\"", customer, "\",\"question\":\"", escaped, "\"}"], "")
    let post_req0 := { method: "POST", url: str.concat(dash, "/ask-human"), headers: map.new(), body: Some(bytes.from_str(post_body)), timeout_ms: None }
    let post_req := http.with_header(http.with_timeout_ms(post_req0, 5000), "Content-Type", "application/json")
    let _pr := http.send(post_req)
    let poll_req := http.with_timeout_ms({ method: "GET", url: str.join([dash, "/get-answer/", qid], ""), headers: map.new(), body: None, timeout_ms: None }, 65000)
    match http.send(poll_req) {
      Err(_) => Err([{ path: "", code: "ask_human_timeout", message: "operator response timed out" }]),
      Ok(resp) => match bytes.to_str(resp.body) {
        Err(_) => Ok(JStr("(no reply)")),
        Ok(answer) => if str.is_empty(str.trim(answer)) {
          Ok(JStr("(no reply — operator did not respond in time)"))
        } else {
          let _pa := io.print(str.join(["  [", customer, "] ← operator: ", answer], ""))
          Ok(JStr(answer))
        },
      },
    }
  })
}

