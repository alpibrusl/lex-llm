# lex-llm — AgentDef + run_loop
#
# run_loop drives the stream → collect → dispatch → continue cycle:
#
#   1. Prepend agent.goal as the system message.
#   2. Derive bindings from the last UserMsg in the conversation.
#   3. Filter agent.tools through spec preconditions (filter_available).
#   4. Call provider.chat with the filtered tool list; collect Deltas.
#   5. Assemble a CollectedResponse from the Deltas.
#   6. If finish_reason = "tool_calls":
#        a. Validate args via lex-schema; invalid → Err → ToolMsg so
#           the model can self-correct on the next turn.
#        b. Check each tool call against agent.permission_spec (Phase 3):
#           denied calls return spec-denied error and emit spec.denied trail
#           event without executing; the model is expected to self-correct.
#        c. Execute allowed tools; append AssistantMsg + ToolMsg; recurse.
#   7. Otherwise emit StepDone.
#
# run_loop_traced: same loop with lex-trail events at each step.
# with_permission_gate: construction-time filter + store spec for runtime check.
# make_agent: canonical constructor; sets permission_spec: None.

import "./message" as msg

import "./delta" as d

import "./tool" as t

import "./provider" as prov

import "lex-schema/json_value" as jv

import "lex-schema/error" as e

import "lex-spec/spec" as sp

import "lex-spec/eval" as ev

import "lex-trail/log" as trail

import "lex-trail/kinds" as kinds

import "std.list" as list

import "std.str" as str

import "std.iter" as iter

import "std.int" as int

type AgentOptions = { temperature :: Option[Float], top_p :: Option[Float], max_steps :: Option[Int], max_tokens :: Option[Int] }

# An agent is a value — compose agents by building AgentDef records.
# Use make_agent() as the canonical constructor; it sets permission_spec: None.
# Use with_permission_gate() to attach a runtime permission spec.
type AgentDef = { name :: Str, goal :: Str, model :: prov.ModelRef, provider :: prov.Provider, tools :: List[t.Tool], options :: AgentOptions, permission_spec :: Option[sp.Spec] }

fn default_options() -> AgentOptions
  examples {
    default_options() => { temperature: Some(0.7), top_p: None, max_steps: Some(20), max_tokens: Some(4096) }
  }
{
  { temperature: Some(0.7), top_p: None, max_steps: Some(20), max_tokens: Some(4096) }
}

fn make_agent(name :: Str, goal :: Str, model :: prov.ModelRef, provider :: prov.Provider, tools :: List[t.Tool], options :: AgentOptions) -> AgentDef {
  { name: name, goal: goal, model: model, provider: provider, tools: tools, options: options, permission_spec: None }
}

# ---- Permission gating -------------------------------------------
#
# Construction-time: filter the tool list the model sees in its prompt.
# Runtime (Phase 3): store the spec on AgentDef so dispatch_one can
# re-evaluate it for every tool call — even if the model somehow tries
# to invoke a tool that was removed at construction time.
#
# Deny or Inconclusive both block dispatch and emit a spec.denied trail event.
fn with_permission_gate(agent :: AgentDef, spec :: sp.Spec) -> AgentDef {
  let allowed := list.filter(agent.tools, fn (tool :: t.Tool) -> Bool {
    let bindings := [("tool", VStr(tool.name))]
    sp.verdict_is_allow(ev.eval(spec, bindings))
  })
  { name: agent.name, goal: agent.goal, model: agent.model, provider: agent.provider, tools: allowed, options: agent.options, permission_spec: Some(spec) }
}

# ---- Internal collected-response type ----------------------------
# Complete response assembled from one turn's Delta stream.
type CollectedResponse = { content :: Str, tool_calls :: List[CollectedCall], finish_reason :: Str }

# Tool call with args fully assembled from streaming ToolArgChunk events.
type CollectedCall = { id :: Str, name :: Str, args_raw :: Str }

# Dispatch result — carries outcome so both Step events and ToolMsg
# body can be built in one pass without re-executing.
type Dispatch = { call :: CollectedCall, success :: Bool, content :: Str }

# ---- Public entry points -----------------------------------------
fn run_loop(agent :: AgentDef, conversation :: List[msg.Message]) -> [net, llm, io, proc] Iter[d.Step] {
  let budget := unwrap_int(agent.options.max_steps, 20)
  iter.from_list(run_steps(agent, conversation, budget))
}

fn run_loop_traced(agent :: AgentDef, conversation :: List[msg.Message], log :: trail.Log, parent :: Option[Str]) -> [net, llm, io, proc, sql, time] Iter[d.Step] {
  let budget := unwrap_int(agent.options.max_steps, 20)
  iter.from_list(run_steps_traced(agent, conversation, budget, log, parent))
}

# ---- Internal recursion ------------------------------------------
fn run_steps(agent :: AgentDef, conv :: List[msg.Message], budget :: Int) -> [net, llm, io, proc] List[d.Step] {
  if budget == 0 {
    [StepDone(AssistantMsg("[max_steps reached]", []))]
  } else {
    let messages := list.concat([SystemMsg(agent.goal)], conv)
    let bindings := bindings_from_conv(conv)
    let avail_tools := t.filter_available(agent.tools, bindings)
    let raw_deltas := iter.to_list(agent.provider.chat(agent.model, messages, avail_tools))
    let delta_steps := list.map(raw_deltas, fn (dl :: d.Delta) -> d.Step {
      StepDelta(dl)
    })
    let response := collect_response(raw_deltas)
    match response.finish_reason {
      "tool_calls" => {
        let dispatches := dispatch_calls(agent.tools, response.tool_calls, agent.permission_spec)
        let exec_steps := dispatches_to_steps(dispatches)
        let tool_messages := dispatches_to_messages(dispatches)
        let assistant_msg := AssistantMsg(response.content, list.map(response.tool_calls, fn (c :: CollectedCall) -> msg.ToolCall {
          { id: c.id, name: c.name, args: parse_args_or_empty(c.args_raw) }
        }))
        let base_conv := list.concat(conv, list.concat([assistant_msg], tool_messages))
        let new_conv := if any_dispatch_failed(dispatches) {
          list.concat(base_conv, [UserMsg("One or more tools returned errors. Read the error messages above, fix the code, and try again.")])
        } else {
          base_conv
        }
        list.concat(delta_steps, list.concat(exec_steps, run_steps(agent, new_conv, budget - 1)))
      },
      _ => list.concat(delta_steps, [StepDone(AssistantMsg(response.content, []))]),
    }
  }
}

fn any_dispatch_failed(dispatches :: List[Dispatch]) -> Bool {
  list.fold(dispatches, false, fn (acc :: Bool, disp :: Dispatch) -> Bool {
    if acc {
      true
    } else {
      if disp.success {
        false
      } else {
        true
      }
    }
  })
}

fn run_steps_traced(agent :: AgentDef, conv :: List[msg.Message], budget :: Int, log :: trail.Log, parent :: Option[Str]) -> [net, llm, io, proc, sql, time] List[d.Step] {
  if budget == 0 {
    [StepDone(AssistantMsg("[max_steps reached]", []))]
  } else {
    let messages := list.concat([SystemMsg(agent.goal)], conv)
    let bindings := bindings_from_conv(conv)
    let avail_tools := t.filter_available(agent.tools, bindings)
    let raw_deltas := iter.to_list(agent.provider.chat(agent.model, messages, avail_tools))
    let delta_steps := list.map(raw_deltas, fn (dl :: d.Delta) -> d.Step {
      StepDelta(dl)
    })
    let response := collect_response(raw_deltas)
    let step_payload := llm_step_json(agent.model, list.len(response.tool_calls))
    let step_evt := trail.append(log, kinds.llm_step(), parent, step_payload)
    let step_id := match step_evt {
      Ok(evt) => Some(evt.id),
      Err(_) => parent,
    }
    match response.finish_reason {
      "tool_calls" => {
        let dispatches := dispatch_calls_traced(agent.tools, response.tool_calls, log, step_id, agent.permission_spec)
        let exec_steps := dispatches_to_steps(dispatches)
        let tool_messages := dispatches_to_messages(dispatches)
        let assistant_msg := AssistantMsg(response.content, list.map(response.tool_calls, fn (c :: CollectedCall) -> msg.ToolCall {
          { id: c.id, name: c.name, args: parse_args_or_empty(c.args_raw) }
        }))
        let base_conv := list.concat(conv, list.concat([assistant_msg], tool_messages))
        let new_conv := if any_dispatch_failed(dispatches) {
          list.concat(base_conv, [UserMsg("One or more tools returned errors. Read the error messages above, fix the code, and try again.")])
        } else {
          base_conv
        }
        list.concat(delta_steps, list.concat(exec_steps, run_steps_traced(agent, new_conv, budget - 1, log, step_id)))
      },
      _ => list.concat(delta_steps, [StepDone(AssistantMsg(response.content, []))]),
    }
  }
}

# ---- Spec bindings -----------------------------------------------
#
# Build the bindings list passed to filter_available on each turn.
# v0.1 exposes a single `user_input` binding with the text of the
# last UserMsg in the conversation. Extend as tools need richer context.
fn last_user_content(conv :: List[msg.Message]) -> Str {
  list.fold(conv, "", fn (acc :: Str, m :: msg.Message) -> Str {
    match m {
      UserMsg(text) => text,
      _ => acc,
    }
  })
}

fn bindings_from_conv(conv :: List[msg.Message]) -> List[(Str, sp.SpecValue)] {
  [("user_input", VStr(last_user_content(conv)))]
}

# ---- Delta → CollectedResponse -----------------------------------
fn collect_response(deltas :: List[d.Delta]) -> CollectedResponse {
  list.fold(deltas, empty_response(), fn (acc :: CollectedResponse, dl :: d.Delta) -> CollectedResponse {
    match dl {
      TextChunk(s) => { content: str.concat(acc.content, s), tool_calls: acc.tool_calls, finish_reason: acc.finish_reason },
      ToolCallBegin(id, name) => { content: acc.content, tool_calls: list.cons({ id: id, name: name, args_raw: "" }, acc.tool_calls), finish_reason: acc.finish_reason },
      ToolArgChunk(id, chunk) => { content: acc.content, tool_calls: append_arg_chunk(acc.tool_calls, id, chunk), finish_reason: acc.finish_reason },
      FinishDelta(reason) => {
        let calls := list.reverse(acc.tool_calls)
        let actual_reason := if reason == "stop" {
          if list.is_empty(calls) {
            "stop"
          } else {
            "tool_calls"
          }
        } else {
          reason
        }
        { content: acc.content, tool_calls: calls, finish_reason: actual_reason }
      },
    }
  })
}

fn empty_response() -> CollectedResponse
  examples {
    empty_response() => { content: "", tool_calls: [], finish_reason: "stop" }
  }
{
  { content: "", tool_calls: [], finish_reason: "stop" }
}

fn append_arg_chunk(calls :: List[CollectedCall], id :: Str, chunk :: Str) -> List[CollectedCall] {
  match list.fold(calls, ([], false), fn (acc :: (List[CollectedCall], Bool), c :: CollectedCall) -> (List[CollectedCall], Bool) {
    match acc {
      (lst, found) => if found {
        (list.concat(lst, [c]), true)
      } else {
        if c.id == id {
          (list.concat(lst, [{ id: c.id, name: c.name, args_raw: str.concat(c.args_raw, chunk) }]), true)
        } else {
          (list.concat(lst, [c]), false)
        }
      },
    }
  }) {
    (result, _) => result,
  }
}

# ---- Tool dispatch -----------------------------------------------
#
# Dispatch always uses the full agent.tools list — not avail_tools —
# so models that reference a tool name from a prior turn (when it was
# available) still get a coherent error rather than "unknown tool".
#
# spec_opt: when Some, each call is checked against the spec before
# execution. Denied calls return a spec-denied error to the model.
fn dispatch_calls(tools :: List[t.Tool], calls :: List[CollectedCall], spec_opt :: Option[sp.Spec]) -> [net, io, proc] List[Dispatch] {
  list.map(calls, fn (call :: CollectedCall) -> [net, io, proc] Dispatch {
    dispatch_one(tools, call, spec_opt)
  })
}

fn dispatch_calls_traced(tools :: List[t.Tool], calls :: List[CollectedCall], log :: trail.Log, parent :: Option[Str], spec_opt :: Option[sp.Spec]) -> [net, io, proc, sql, time] List[Dispatch] {
  list.map(calls, fn (call :: CollectedCall) -> [net, io, proc, sql, time] Dispatch {
    dispatch_one_traced(tools, call, log, parent, spec_opt)
  })
}

fn dispatch_one(tools :: List[t.Tool], call :: CollectedCall, spec_opt :: Option[sp.Spec]) -> [net, io, proc] Dispatch {
  let is_allowed := match spec_opt {
    None => true,
    Some(spec) => {
      let bindings := [("tool", VStr(call.name))]
      sp.verdict_is_allow(ev.eval(spec, bindings))
    },
  }
  if is_allowed {
    let args := parse_args_or_empty(call.args_raw)
    match t.find_by_name(tools, call.name) {
      None => { call: call, success: false, content: str.concat("{\"error\":\"unknown tool: ", str.concat(call.name, "}")) },
      Some(tool) => match t.validate_and_exec(tool, args) {
        Ok(out) => { call: call, success: true, content: jv.stringify(out) },
        Err(errs) => { call: call, success: false, content: str.concat("{\"error\":\"", str.concat(t.format_validation_error(errs), "}")) },
      },
    }
  } else {
    { call: call, success: false, content: str.concat("{\"error\":\"spec-denied: tool '", str.concat(call.name, "' is not permitted by the agent permission policy\"}")) }
  }
}

fn dispatch_one_traced(tools :: List[t.Tool], call :: CollectedCall, log :: trail.Log, parent :: Option[Str], spec_opt :: Option[sp.Spec]) -> [net, io, proc, sql, time] Dispatch {
  let is_allowed := match spec_opt {
    None => true,
    Some(spec) => {
      let bindings := [("tool", VStr(call.name))]
      sp.verdict_is_allow(ev.eval(spec, bindings))
    },
  }
  if is_allowed {
    let inv_payload := cap_invoked_json(call.name, call.args_raw)
    let inv_evt := trail.append(log, kinds.cap_invoked(), parent, inv_payload)
    let inv_id := match inv_evt {
      Ok(evt) => Some(evt.id),
      Err(_) => parent,
    }
    let disp := dispatch_one(tools, call, None)
    let kind := if disp.success {
      kinds.cap_completed()
    } else {
      kinds.cap_failed()
    }
    let out_payload := if disp.success {
      cap_completed_json(call.name, disp.content)
    } else {
      cap_failed_json(call.name, disp.content)
    }
    let _trail_result := trail.append(log, kind, inv_id, out_payload)
    let _verified := match verified_kind_for_tool(call.name, disp.content) {
      None => (),
      Some(vkind) => {
        let vpayload := str.join(["{\"tool\":\"", call.name, "\",\"result\":\"pass\"}"], "")
        let _ve := trail.append(log, vkind, inv_id, vpayload)
        ()
      },
    }
    disp
  } else {
    let denied_payload := str.join(["{\"tool\":\"", call.name, "\",\"reason\":\"spec-denied\"}"], "")
    let _denied_evt := trail.append(log, kinds.spec_denied(), parent, denied_payload)
    { call: call, success: false, content: str.concat("{\"error\":\"spec-denied: tool '", str.concat(call.name, "' is not permitted by the agent permission policy\"}")) }
  }
}

fn dispatches_to_steps(dispatches :: List[Dispatch]) -> List[d.Step] {
  list.fold(dispatches, [], fn (acc :: List[d.Step], disp :: Dispatch) -> List[d.Step] {
    list.concat(acc, [StepToolExec(disp.call.name, disp.call.id), StepToolResult(disp.call.id, disp.success)])
  })
}

fn dispatches_to_messages(dispatches :: List[Dispatch]) -> List[msg.Message] {
  list.map(dispatches, fn (disp :: Dispatch) -> msg.Message {
    ToolMsg(disp.call.id, disp.content)
  })
}

# ---- Trail JSON helpers ------------------------------------------
fn llm_step_json(model :: prov.ModelRef, tool_call_count :: Int) -> Str
  examples {
    llm_step_json(prov.claude_sonnet(), 2) => "{\"model\":\"claude-sonnet-4-6\",\"tokens_in\":0,\"tokens_out\":0,\"tool_calls\":2}"
  }
{
  str.join(["{\"model\":\"", model.model, "\",\"tokens_in\":0,\"tokens_out\":0,\"tool_calls\":", int.to_str(tool_call_count), "}"], "")
}

fn cap_invoked_json(name :: Str, args_raw :: Str) -> Str
  examples {
    cap_invoked_json("search", "{}") => "{\"capability\":\"search\",\"args\":{}}"
  }
{
  str.join(["{\"capability\":\"", name, "\",\"args\":", args_raw, "}"], "")
}

fn cap_completed_json(name :: Str, result :: Str) -> Str
  examples {
    cap_completed_json("search", "\"ok\"") => "{\"capability\":\"search\",\"result\":\"ok\"}"
  }
{
  str.join(["{\"capability\":\"", name, "\",\"result\":", result, "}"], "")
}

fn cap_failed_json(name :: Str, error :: Str) -> Str
  examples {
    cap_failed_json("search", "{\"error\":\"nf\"}") => "{\"capability\":\"search\",\"error\":{\"error\":\"nf\"}}"
  }
{
  str.join(["{\"capability\":\"", name, "\",\"error\":", error, "}"], "")
}

# Return the verified.* kind to emit when a tool call passes verification,
# or None if the tool does not produce a verifiable attestation.
# Content is the stringified JStr result from the tool, so success phrases
# appear inside the JSON string literal.
fn verified_kind_for_tool(tool_name :: Str, content :: Str) -> Option[Str] {
  match tool_name {
    "lex_check" => if str.contains(content, "type check passed") {
      Some(kinds.verified_type_check())
    } else {
      None
    },
    "lex_spec_check" => if str.contains(content, "spec passed") {
      Some(kinds.verified_spec_check())
    } else {
      None
    },
    "lex_test" => if str.contains(content, "tests passed") {
      Some(kinds.verified_test())
    } else {
      None
    },
    _ => None,
  }
}

# ---- Helpers -----------------------------------------------------
fn unwrap_int(opt :: Option[Int], default :: Int) -> Int
  examples {
    unwrap_int(Some(5), 20) => 5,
    unwrap_int(None, 20) => 20
  }
{
  match opt {
    Some(n) => n,
    None => default,
  }
}

# Parse args JSON; fall back to an empty object on failure.
# Invalid args are caught again by tool.validate_and_exec.
fn parse_args_or_empty(raw :: Str) -> jv.Json {
  match jv.parse_into_errors(raw) {
    Ok(j) => j,
    Err(_) => JObj([]),
  }
}

