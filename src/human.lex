# lex-llm — Generic human-escalation tool
#
# make_ask_human_tool(dash, customer) → Tool
#
# When the LLM calls ask_human { question }, the tool:
#   1. POSTs the question to dash/ask-human (dashboard broadcasts human_question SSE)
#   2. Issues a single long-poll GET to dash/get-answer/<id>
#      The dashboard server blocks up to ~60 s until the operator submits an answer.
#   3. Returns the operator reply to the LLM so it can continue reasoning.
#
# The question id is deterministic: "q-" + customer, so only one pending
# question per customer is tracked at a time.  A new call replaces the old.
#
# No bazaar-specific knowledge lives here — import this from any agent that
# wants to surface decisions to a human operator.
#
# Effects required: [net, io]  (blocking is server-side; no [time] needed)

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

# Build an ask_human tool bound to a specific dashboard URL and customer identity.
# customer is included in the SSE event so the UI can attribute the question.
fn make_ask_human_tool(dash :: Str, customer :: Str) -> t.Tool {
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

