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
#        b. Execute valid tools against the full agent.tools list so
#           model references to prior-turn tools are not broken.
#        c. Append AssistantMsg + ToolMsg results; recurse within budget.
#   7. Otherwise emit StepDone.
#
# run_loop_traced: same loop with lex-trail events at each step.
# with_permission_gate: filter tools via lex-spec Spec at construction time.

import "./message"  as msg
import "./delta"    as d
import "./tool"     as t
import "./provider" as prov

import "lex-schema/json_value" as jv
import "lex-schema/error"      as e

import "lex-spec/spec" as sp
import "lex-spec/eval" as ev

import "lex-trail/log"   as trail
import "lex-trail/kinds" as kinds

import "std.list" as list
import "std.str"  as str
import "std.iter" as iter
import "std.int"  as int

type AgentOptions = {
  temperature :: Option[Float],
  top_p       :: Option[Float],
  max_steps   :: Option[Int],
  max_tokens  :: Option[Int],
}

# An agent is a value — compose agents by building AgentDef records.
type AgentDef = {
  name     :: Str,
  goal     :: Str,               # injected as SystemMsg before every turn
  model    :: prov.ModelRef,
  provider :: prov.Provider,
  tools    :: List[t.Tool],
  options  :: AgentOptions,
}

fn default_options() -> AgentOptions
  examples {
    default_options() => {
      temperature: Some(0.7),
      top_p:       None,
      max_steps:   Some(20),
      max_tokens:  Some(4096),
    },
  }
{
  { temperature: Some(0.7), top_p: None, max_steps: Some(20), max_tokens: Some(4096) }
}

# ---- Permission gating -------------------------------------------
#
# Filter an AgentDef's tool list to only tools allowed by a lex-spec Spec.
# The spec is evaluated with a single bound variable "tool" (the tool name).
# Deny or Inconclusive both remove the tool from the agent's visible set,
# so the model never sees forbidden tool names in its prompt.
#
# Apply at construction time for maximum isolation:
#
#   let gated := with_permission_gate(base_agent, rules.explore_permission())
#
fn with_permission_gate(agent :: AgentDef, spec :: sp.Spec) -> AgentDef {
  let allowed := list.filter(agent.tools, fn (tool :: t.Tool) -> Bool {
    let bindings := [("tool", sp.VStr(tool.name))]
    sp.verdict_is_allow(ev.eval(spec, bindings))
  })
  { name:     agent.name,
    goal:     agent.goal,
    model:    agent.model,
    provider: agent.provider,
    tools:    allowed,
    options:  agent.options }
}

# ---- Internal collected-response type ----------------------------

# Complete response assembled from one turn's Delta stream.
type CollectedResponse = {
  content       :: Str,
  tool_calls    :: List[CollectedCall],
  finish_reason :: Str,
}

# Tool call with args fully assembled from streaming ToolArgChunk events.
type CollectedCall = {
  id       :: Str,
  name     :: Str,
  args_raw :: Str,
}

# Dispatch result — carries outcome so both Step events and ToolMsg
# body can be built in one pass without re-executing.
type Dispatch = {
  call    :: CollectedCall,
  success :: Bool,
  content :: Str,
}

# ---- Public entry points -----------------------------------------

fn run_loop(
  agent        :: AgentDef,
  conversation :: List[msg.Message]
) -> [net, llm, io, proc] Iter[d.Step] {
  let budget := unwrap_int(agent.options.max_steps, 20)
  iter.from_list(run_steps(agent, conversation, budget))
}

fn run_loop_traced(
  agent        :: AgentDef,
  conversation :: List[msg.Message],
  log          :: trail.Log,
  parent       :: Option[Str],
) -> [net, llm, io, proc, sql, time] Iter[d.Step] {
  let budget := unwrap_int(agent.options.max_steps, 20)
  iter.from_list(run_steps_traced(agent, conversation, budget, log, parent))
}

# ---- Internal recursion ------------------------------------------

fn run_steps(
  agent  :: AgentDef,
  conv   :: List[msg.Message],
  budget :: Int
) -> [net, llm, io, proc] List[d.Step] {
  if budget == 0 {
    [d.StepDone(msg.AssistantMsg("[max_steps reached]", []))]
  } else {
    let messages    := list.concat([msg.SystemMsg(agent.goal)], conv)
    let bindings    := bindings_from_conv(conv)
    let avail_tools := t.filter_available(agent.tools, bindings)
    let raw_deltas  := iter.to_list(agent.provider.chat(agent.model, messages, avail_tools))
    let delta_steps := list.map(raw_deltas, fn (dl :: d.Delta) -> d.Step {
      d.StepDelta(dl)
    })
    let response := collect_response(raw_deltas)
    match response.finish_reason {
      "tool_calls" => {
        let dispatches    := dispatch_calls(agent.tools, response.tool_calls)
        let exec_steps    := dispatches_to_steps(dispatches)
        let tool_messages := dispatches_to_messages(dispatches)
        let assistant_msg := msg.AssistantMsg(
          response.content,
          list.map(response.tool_calls, fn (c :: CollectedCall) -> msg.ToolCall {
            { id: c.id, name: c.name,
              args: parse_args_or_empty(c.args_raw) }
          })
        )
        let new_conv :=
          list.concat(conv, list.concat([assistant_msg], tool_messages))
        list.concat(delta_steps,
          list.concat(exec_steps,
            run_steps(agent, new_conv, budget - 1)))
      },
      _ =>
        list.concat(delta_steps,
          [d.StepDone(msg.AssistantMsg(response.content, []))]),
    }
  }
}

fn run_steps_traced(
  agent  :: AgentDef,
  conv   :: List[msg.Message],
  budget :: Int,
  log    :: trail.Log,
  parent :: Option[Str],
) -> [net, llm, io, proc, sql, time] List[d.Step] {
  if budget == 0 {
    [d.StepDone(msg.AssistantMsg("[max_steps reached]", []))]
  } else {
    let messages    := list.concat([msg.SystemMsg(agent.goal)], conv)
    let bindings    := bindings_from_conv(conv)
    let avail_tools := t.filter_available(agent.tools, bindings)
    let raw_deltas  := iter.to_list(agent.provider.chat(agent.model, messages, avail_tools))
    let delta_steps := list.map(raw_deltas, fn (dl :: d.Delta) -> d.Step { d.StepDelta(dl) })
    let response    := collect_response(raw_deltas)
    let step_payload := llm_step_json(agent.model, list.len(response.tool_calls))
    let step_evt     := trail.append(log, kinds.llm_step(), parent, step_payload)
    let step_id      := match step_evt { Ok(evt) => Some(evt.id), Err(_) => parent }
    match response.finish_reason {
      "tool_calls" => {
        let dispatches    := dispatch_calls_traced(agent.tools, response.tool_calls, log, step_id)
        let exec_steps    := dispatches_to_steps(dispatches)
        let tool_messages := dispatches_to_messages(dispatches)
        let assistant_msg := msg.AssistantMsg(
          response.content,
          list.map(response.tool_calls, fn (c :: CollectedCall) -> msg.ToolCall {
            { id: c.id, name: c.name, args: parse_args_or_empty(c.args_raw) }
          })
        )
        let new_conv := list.concat(conv, list.concat([assistant_msg], tool_messages))
        list.concat(delta_steps,
          list.concat(exec_steps,
            run_steps_traced(agent, new_conv, budget - 1, log, step_id)))
      },
      _ =>
        list.concat(delta_steps,
          [d.StepDone(msg.AssistantMsg(response.content, []))]),
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
      msg.UserMsg(text) => text,
      _                 => acc,
    }
  })
}

fn bindings_from_conv(conv :: List[msg.Message]) -> List[(Str, sp.SpecValue)] {
  [("user_input", sp.VStr(last_user_content(conv)))]
}

# ---- Delta → CollectedResponse -----------------------------------

fn collect_response(deltas :: List[d.Delta]) -> CollectedResponse {
  list.fold(deltas, empty_response(),
    fn (acc :: CollectedResponse, dl :: d.Delta) -> CollectedResponse {
      match dl {
        d.TextChunk(s) => {
          content:       str.concat(acc.content, s),
          tool_calls:    acc.tool_calls,
          finish_reason: acc.finish_reason,
        },
        d.ToolCallBegin(id, name) => {
          content:       acc.content,
          tool_calls:    list.cons({ id: id, name: name, args_raw: "" }, acc.tool_calls),
          finish_reason: acc.finish_reason,
        },
        d.ToolArgChunk(id, chunk) => {
          content:       acc.content,
          tool_calls:    append_arg_chunk(acc.tool_calls, id, chunk),
          finish_reason: acc.finish_reason,
        },
        d.FinishDelta(reason) => {
          let calls := list.reverse(acc.tool_calls)
          # Ollama sends tool_calls in a done=false chunk, then a bare done=true
          # chunk with done_reason="stop". Override so dispatch still fires.
          let actual_reason :=
            if reason == "stop" {
              if list.is_empty(calls) { "stop" } else { "tool_calls" }
            } else { reason }
          {
            content:       acc.content,
            tool_calls:    calls,
            finish_reason: actual_reason,
          }
        },
      }
    })
}

fn empty_response() -> CollectedResponse
  examples {
    empty_response() => { content: "", tool_calls: [], finish_reason: "stop" },
  }
{
  { content: "", tool_calls: [], finish_reason: "stop" }
}

fn append_arg_chunk(
  calls :: List[CollectedCall],
  id    :: Str,
  chunk :: Str
) -> List[CollectedCall] {
  list.map(calls, fn (c :: CollectedCall) -> CollectedCall {
    if c.id == id {
      { id: c.id, name: c.name, args_raw: str.concat(c.args_raw, chunk) }
    } else {
      c
    }
  })
}

# ---- Tool dispatch -----------------------------------------------
#
# Dispatch always uses the full agent.tools list — not avail_tools —
# so models that reference a tool name from a prior turn (when it was
# available) still get a coherent error rather than "unknown tool".

fn dispatch_calls(
  tools :: List[t.Tool],
  calls :: List[CollectedCall]
) -> [net, io, proc] List[Dispatch] {
  list.map(calls, fn (call :: CollectedCall) -> [net, io, proc] Dispatch {
    dispatch_one(tools, call)
  })
}

fn dispatch_calls_traced(
  tools  :: List[t.Tool],
  calls  :: List[CollectedCall],
  log    :: trail.Log,
  parent :: Option[Str],
) -> [net, io, proc, sql, time] List[Dispatch] {
  list.map(calls, fn (call :: CollectedCall) -> [net, io, proc, sql, time] Dispatch {
    dispatch_one_traced(tools, call, log, parent)
  })
}

fn dispatch_one(
  tools :: List[t.Tool],
  call  :: CollectedCall
) -> [net, io, proc] Dispatch {
  let args := parse_args_or_empty(call.args_raw)
  match t.find_by_name(tools, call.name) {
    None =>
      { call: call, success: false,
        content: str.concat("{\"error\":\"unknown tool: ",
          str.concat(call.name, "}")) },
    Some(tool) =>
      match t.validate_and_exec(tool, args) {
        Ok(out)   => { call: call, success: true, content: jv.stringify(out) },
        Err(errs) => { call: call, success: false,
          content: str.concat("{\"error\":\"",
            str.concat(t.format_validation_error(errs), "}")) },
      },
  }
}

fn dispatch_one_traced(
  tools  :: List[t.Tool],
  call   :: CollectedCall,
  log    :: trail.Log,
  parent :: Option[Str],
) -> [net, io, proc, sql, time] Dispatch {
  let inv_payload  := cap_invoked_json(call.name, call.args_raw)
  let inv_evt      := trail.append(log, kinds.cap_invoked(), parent, inv_payload)
  let inv_id       := match inv_evt { Ok(evt) => Some(evt.id), Err(_) => parent }
  let disp         := dispatch_one(tools, call)
  let kind         := if disp.success { kinds.cap_completed() } else { kinds.cap_failed() }
  let out_payload  := if disp.success {
    cap_completed_json(call.name, disp.content)
  } else {
    cap_failed_json(call.name, disp.content)
  }
  let trail_result := trail.append(log, kind, inv_id, out_payload)
  disp
}

fn dispatches_to_steps(dispatches :: List[Dispatch]) -> List[d.Step] {
  list.fold(dispatches, [],
    fn (acc :: List[d.Step], disp :: Dispatch) -> List[d.Step] {
      list.concat(acc, [
        d.StepToolExec(disp.call.name, disp.call.id),
        d.StepToolResult(disp.call.id, disp.success),
      ])
    })
}

fn dispatches_to_messages(dispatches :: List[Dispatch]) -> List[msg.Message] {
  list.map(dispatches, fn (disp :: Dispatch) -> msg.Message {
    msg.ToolMsg(disp.call.id, disp.content)
  })
}

# ---- Trail JSON helpers ------------------------------------------

fn llm_step_json(model :: prov.ModelRef, tool_call_count :: Int) -> Str
  examples {
    llm_step_json(prov.claude_sonnet(), 2) =>
      "{\"model\":\"claude-sonnet-4-6\",\"tokens_in\":0,\"tokens_out\":0,\"tool_calls\":2}",
  }
{
  str.join([
    "{\"model\":\"", model.model,
    "\",\"tokens_in\":0,\"tokens_out\":0,\"tool_calls\":",
    int.to_str(tool_call_count),
    "}"
  ], "")
}

fn cap_invoked_json(name :: Str, args_raw :: Str) -> Str
  examples {
    cap_invoked_json("search", "{}") => "{\"capability\":\"search\",\"args\":{}}",
  }
{
  str.join(["{\"capability\":\"", name, "\",\"args\":", args_raw, "}"], "")
}

fn cap_completed_json(name :: Str, result :: Str) -> Str
  examples {
    cap_completed_json("search", "\"ok\"") => "{\"capability\":\"search\",\"result\":\"ok\"}",
  }
{
  str.join(["{\"capability\":\"", name, "\",\"result\":", result, "}"], "")
}

fn cap_failed_json(name :: Str, error :: Str) -> Str
  examples {
    cap_failed_json("search", "{\"error\":\"nf\"}") =>
      "{\"capability\":\"search\",\"error\":{\"error\":\"nf\"}}",
  }
{
  str.join(["{\"capability\":\"", name, "\",\"error\":", error, "}"], "")
}

# ---- Helpers -----------------------------------------------------

fn unwrap_int(opt :: Option[Int], default :: Int) -> Int
  examples {
    unwrap_int(Some(5), 20) => 5,
    unwrap_int(None,    20) => 20,
  }
{
  match opt { Some(n) => n, None => default }
}

# Parse args JSON; fall back to an empty object on failure.
# Invalid args are caught again by tool.validate_and_exec.
fn parse_args_or_empty(raw :: Str) -> jv.Json {
  match jv.parse_into_errors(raw) {
    Ok(j)  => j,
    Err(_) => JObj([]),
  }
}
