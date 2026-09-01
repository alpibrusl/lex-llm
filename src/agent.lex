# lex-llm — AgentLoop + run_loop
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
#        c. Tools carrying an approval_scope block on a human decision
#           (std.approval.request, #41) before executing; denials return an
#           approval_denied error the model can react to.
#        d. Execute allowed tools; append AssistantMsg + ToolMsg; recurse.
#   7. Otherwise emit StepDone.
#
# run_loop_traced: same loop with lex-trail events at each step.
# run_steps_streamed: same loop as run_loop_traced, but emitting each Step
#   through a caller-supplied callback the moment it happens rather than
#   returning them all at the end. See the note above it.
# with_permission_gate: construction-time filter + store spec for runtime check.
# make_agent: canonical constructor; sets permission_spec: None.

import "./message" as msg

import "./delta" as d

import "./tool" as t

import "./streaming" as streaming

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

# An agent is a value — compose agents by building AgentLoop records.
# Use make_agent() as the canonical constructor; it sets permission_spec: None.
# Use with_permission_gate() to attach a runtime permission spec.
#
# Named AgentLoop (was AgentDef) to disambiguate from lex-agent's AgentDef: this
# is the *brain* (a model + tools + run_loop), not the A2A/MCP *skin*. The two
# compose — a brain runs inside a lex-agent Skill's handle (see lex-agent-llm).
type AgentLoop = { name :: Str, goal :: Str, model :: prov.ModelRef, provider :: prov.Provider, tools :: List[t.Tool], options :: AgentOptions, permission_spec :: Option[sp.Spec] }

# DEPRECATED alias — kept one release so importers (`ag.AgentDef`) keep compiling
# while they migrate to AgentLoop. Remove in the next minor.
type AgentDef = AgentLoop

fn default_options() -> AgentOptions
  examples {
    default_options() => { temperature: Some(0.7), top_p: None, max_steps: Some(20), max_tokens: Some(4096) }
  }
{
  { temperature: Some(0.7), top_p: None, max_steps: Some(20), max_tokens: Some(4096) }
}

fn make_agent(name :: Str, goal :: Str, model :: prov.ModelRef, provider :: prov.Provider, tools :: List[t.Tool], options :: AgentOptions) -> AgentLoop {
  { name: name, goal: goal, model: model, provider: provider, tools: tools, options: options, permission_spec: None }
}

# ---- Permission gating -------------------------------------------
#
# Construction-time: filter the tool list the model sees in its prompt.
# Runtime (Phase 3): store the spec on AgentLoop so dispatch_one can
# re-evaluate it for every tool call — even if the model somehow tries
# to invoke a tool that was removed at construction time.
#
# Deny or Inconclusive both block dispatch and emit a spec.denied trail event.
fn with_permission_gate(agent :: AgentLoop, spec :: sp.Spec) -> AgentLoop {
  let allowed := list.filter(agent.tools, fn (tool :: t.Tool) -> Bool {
    let bindings := [("tool", VStr(tool.name))]
    sp.verdict_is_allow(ev.eval(spec, bindings))
  })
  { name: agent.name, goal: agent.goal, model: agent.model, provider: agent.provider, tools: allowed, options: agent.options, permission_spec: Some(spec) }
}

# ---- Internal collected-response type ----------------------------
# Complete response assembled from one turn's Delta stream.
# prompt_tokens/completion_tokens/total_tokens are 0 iff the provider never
# emitted a UsageDelta this turn (either it doesn't report usage, or genuinely
# used 0) -- callers wanting to distinguish "not reported" from "free" should
# check total_tokens == 0 at the Step level via StepDelta(UsageDelta(...)).
type CollectedResponse = { content :: Str, tool_calls :: List[CollectedCall], finish_reason :: Str, prompt_tokens :: Int, completion_tokens :: Int, total_tokens :: Int }

# Tool call with args fully assembled from streaming ToolArgChunk events.
type CollectedCall = { id :: Str, name :: Str, args_raw :: Str }

# Dispatch result — carries outcome so both Step events and ToolMsg
# body can be built in one pass without re-executing.
type Dispatch = { call :: CollectedCall, success :: Bool, content :: Str }

# ---- Public entry points -----------------------------------------
fn run_loop(agent :: AgentLoop, conversation :: List[msg.Message]) -> [net, llm, io, proc, approval] Iter[d.Step] {
  let budget := unwrap_int(agent.options.max_steps, 20)
  iter.from_list(run_steps(agent, conversation, budget))
}

fn run_loop_traced(agent :: AgentLoop, conversation :: List[msg.Message], log :: trail.Log, parent :: Option[Str]) -> [net, llm, io, proc, sql, time, approval] Iter[d.Step] {
  let budget := unwrap_int(agent.options.max_steps, 20)
  iter.from_list(run_steps_traced(agent, conversation, budget, log, parent))
}

# ---- Internal recursion ------------------------------------------
fn run_steps(agent :: AgentLoop, conv :: List[msg.Message], budget :: Int) -> [net, llm, io, proc, approval] List[d.Step] {
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

# Run the loop, emitting each Step as it happens.
#
# `run_steps_traced` computes the whole multi-turn loop and hands back a
# List[Step], so a caller that walks that list is walking history: every
# token of a two-minute turn arrives at once, two minutes in. That is the
# right shape for a batch caller and the wrong one for anything a person is
# watching.
#
# This is the same loop with one difference: `on_step` fires the moment a
# Step exists — each Delta as it comes off the socket, each tool execution
# as it is dispatched, the final message as it is assembled. The returned
# List[Step] is unchanged, so a caller that wants both the live view and the
# transcript gets them from one call.
#
# **A caller must not also walk the returned list with `on_step`** — every
# Step has already been emitted, and doing both prints the turn twice.
#
# Where the provider declares no streaming half, this still works: the
# buffered `chat` runs, and its Deltas are emitted through the same callback
# in one burst. The callback contract does not change, only the timing, so a
# caller never branches on whether its provider streams.
#
# `on_step` is a plain function parameter rather than a record field, which
# is what lets it carry an effect row at all here. The row is fixed at [io]:
# a printer, a logger, a no-op. A callback that wants to do more than write —
# time each Step, say, or persist it — cannot, and has to work from the
# returned list instead. Widening the row would push that effect onto every
# caller of this function, which is the cost the narrow row is avoiding.
fn run_steps_streamed(agent :: AgentLoop, conv :: List[msg.Message], budget :: Int, log :: trail.Log, parent :: Option[Str], on_step :: (d.Step) -> [io] Unit) -> [net, llm, io, proc, sql, time, approval, stream] List[d.Step] {
  if budget == 0 {
    let done := StepDone(AssistantMsg("[max_steps reached]", []))
    let __e := on_step(done)
    [done]
  } else {
    let messages := list.concat([SystemMsg(agent.goal)], conv)
    let bindings := bindings_from_conv(conv)
    let avail_tools := t.filter_available(agent.tools, bindings)
    let raw_deltas := deltas_streamed(agent, messages, avail_tools, on_step)
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
        let __x := emit_steps(exec_steps, on_step)
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
        list.concat(delta_steps, list.concat(exec_steps, run_steps_streamed(agent, new_conv, budget - 1, log, step_id, on_step)))
      },
      _ => {
        let done := StepDone(AssistantMsg(response.content, []))
        let __d := on_step(done)
        list.concat(delta_steps, [done])
      },
    }
  }
}

# One turn's Deltas, emitted as they arrive when the provider can, in one
# burst when it cannot.
#
# The `None` branch is not a degraded path to apologise for: it is the
# buffered behaviour every caller had before, reached through the same
# callback, so a provider without a streaming half needs no handling at the
# call site.
fn deltas_streamed(agent :: AgentLoop, messages :: List[msg.Message], tools :: List[t.Tool], on_step :: (d.Step) -> [io] Unit) -> [net, llm, io, stream] List[d.Delta] {
  match agent.provider.stream {
    None => {
      let ds := iter.to_list(agent.provider.chat(agent.model, messages, tools))
      let __e := emit_deltas(ds, on_step)
      ds
    },
    Some(sc) => match sc.open(agent.model, messages, tools) {
      Err(e) => {
        let ds := d.provider_error(e)
        let __e := emit_deltas(ds, on_step)
        ds
      },
      Ok(s) => pump_deltas(sc, s, streaming.start(sc), [], on_step, streaming.line_budget()),
    },
  }
}

# Pull one line, emit whatever Deltas it produced, pull again.
#
# Bounded by `budget` for the same reason streaming.drain is: a provider that
# stops sending without closing the socket would otherwise park the turn for
# the transport's full idle timeout.
fn pump_deltas(sc :: prov.StreamChat, s :: Stream[Str], cur :: streaming.Cursor, acc :: List[d.Delta], on_step :: (d.Step) -> [io] Unit, budget :: Int) -> [io, stream] List[d.Delta] {
  if budget == 0 {
    acc
  } else {
    match streaming.pull(sc, s, cur) {
      (next_cur, deltas) => {
        let __e := emit_deltas(deltas, on_step)
        let so_far := list.concat(acc, deltas)
        if streaming.is_done(next_cur) {
          so_far
        } else {
          pump_deltas(sc, s, next_cur, so_far, on_step, budget - 1)
        }
      },
    }
  }
}

fn emit_deltas(deltas :: List[d.Delta], on_step :: (d.Step) -> [io] Unit) -> [io] Unit {
  list.fold(deltas, (), fn (_acc :: Unit, dl :: d.Delta) -> [io] Unit {
    on_step(StepDelta(dl))
  })
}

fn emit_steps(steps :: List[d.Step], on_step :: (d.Step) -> [io] Unit) -> [io] Unit {
  list.fold(steps, (), fn (_acc :: Unit, st :: d.Step) -> [io] Unit {
    on_step(st)
  })
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

fn run_steps_traced(agent :: AgentLoop, conv :: List[msg.Message], budget :: Int, log :: trail.Log, parent :: Option[Str]) -> [net, llm, io, proc, sql, time, approval] List[d.Step] {
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

# Dynamic toolset loading: a tool can emit a "LOADED_TOOLSET:<group>" marker in
# its result (see lex-code's load_toolset tool). We scan the conversation for
# that marker and expose a `loaded_<group>` boolean binding, which gated tools
# reference in their precondition spec to become available once loaded.
fn toolset_loaded(conv :: List[msg.Message], group :: Str) -> Bool {
  let needle := str.concat("LOADED_TOOLSET:", group)
  list.fold(conv, false, fn (acc :: Bool, m :: msg.Message) -> Bool {
    match m {
      ToolMsg(_, content) => if str.contains(content, needle) {
        true
      } else {
        acc
      },
      _ => acc,
    }
  })
}

fn bindings_from_conv(conv :: List[msg.Message]) -> List[(Str, sp.SpecValue)] {
  [("user_input", VStr(last_user_content(conv))), ("loaded_vcs", VBool(toolset_loaded(conv, "vcs"))), ("loaded_spec", VBool(toolset_loaded(conv, "spec"))), ("loaded_store", VBool(toolset_loaded(conv, "store")))]
}

# ---- Delta → CollectedResponse -----------------------------------
fn collect_response(deltas :: List[d.Delta]) -> CollectedResponse {
  list.fold(deltas, empty_response(), fn (acc :: CollectedResponse, dl :: d.Delta) -> CollectedResponse {
    match dl {
      TextChunk(s) => { content: str.concat(acc.content, s), tool_calls: acc.tool_calls, finish_reason: acc.finish_reason, prompt_tokens: acc.prompt_tokens, completion_tokens: acc.completion_tokens, total_tokens: acc.total_tokens },
      ToolCallBegin(id, name) => { content: acc.content, tool_calls: list.cons({ id: id, name: name, args_raw: "" }, acc.tool_calls), finish_reason: acc.finish_reason, prompt_tokens: acc.prompt_tokens, completion_tokens: acc.completion_tokens, total_tokens: acc.total_tokens },
      ToolArgChunk(id, chunk) => { content: acc.content, tool_calls: append_arg_chunk(acc.tool_calls, id, chunk), finish_reason: acc.finish_reason, prompt_tokens: acc.prompt_tokens, completion_tokens: acc.completion_tokens, total_tokens: acc.total_tokens },
      UsageDelta(p, c, t) => { content: acc.content, tool_calls: acc.tool_calls, finish_reason: acc.finish_reason, prompt_tokens: p, completion_tokens: c, total_tokens: t },
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
        { content: acc.content, tool_calls: calls, finish_reason: actual_reason, prompt_tokens: acc.prompt_tokens, completion_tokens: acc.completion_tokens, total_tokens: acc.total_tokens }
      },
    }
  })
}

fn empty_response() -> CollectedResponse
  examples {
    empty_response() => { content: "", tool_calls: [], finish_reason: "stop", prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
  }
{
  { content: "", tool_calls: [], finish_reason: "stop", prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
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
fn dispatch_calls(tools :: List[t.Tool], calls :: List[CollectedCall], spec_opt :: Option[sp.Spec]) -> [net, io, proc, approval] List[Dispatch] {
  list.map(calls, fn (call :: CollectedCall) -> [net, io, proc, approval] Dispatch {
    dispatch_one(tools, call, spec_opt)
  })
}

fn dispatch_calls_traced(tools :: List[t.Tool], calls :: List[CollectedCall], log :: trail.Log, parent :: Option[Str], spec_opt :: Option[sp.Spec]) -> [net, io, proc, sql, time, approval] List[Dispatch] {
  list.map(calls, fn (call :: CollectedCall) -> [net, io, proc, sql, time, approval] Dispatch {
    dispatch_one_traced(tools, call, log, parent, spec_opt)
  })
}

fn dispatch_one(tools :: List[t.Tool], call :: CollectedCall, spec_opt :: Option[sp.Spec]) -> [net, io, proc, approval] Dispatch {
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
      Some(tool) => match t.exec_with_approval(tool, args) {
        Ok(out) => { call: call, success: true, content: jv.stringify(out) },
        Err(errs) => { call: call, success: false, content: str.concat("{\"error\":\"", str.concat(t.format_validation_error(errs), "}")) },
      },
    }
  } else {
    { call: call, success: false, content: str.concat("{\"error\":\"spec-denied: tool '", str.concat(call.name, "' is not permitted by the agent permission policy\"}")) }
  }
}

fn dispatch_one_traced(tools :: List[t.Tool], call :: CollectedCall, log :: trail.Log, parent :: Option[Str], spec_opt :: Option[sp.Spec]) -> [net, io, proc, sql, time, approval] Dispatch {
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
        let _ve := trail.append(log, vkind, inv_id, verified_json(call.name, call.args_raw))
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
    llm_step_json(prov.claude_sonnet(), 2) => "{\"model\":\"claude-sonnet-5\",\"tokens_in\":0,\"tokens_out\":0,\"tool_calls\":2}"
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
# What a verification pass was ABOUT, not just that one happened.
#
# The payload used to carry only the tool name, so a downstream reader
# could say "a type check passed somewhere in this project" and no more —
# lex-code's attestation_query, its .lex/verified.jsonl records and its
# task-spec criteria are all shaped around that limitation, and #32 is
# still open because of it.
#
# The arguments were available the whole time. `call.args_raw` is used two
# lines above this, to build the cap.invoked payload; the verified event
# simply never looked at it. Reading the target out of it costs one field.
#
# `target` is empty when the tool was called with no path — `lex_check`
# defaults to the whole project — and an empty target is honest: the pass
# covers everything and names nothing.
fn verified_json(tool_name :: Str, args_raw :: Str) -> Str
  examples {
    verified_json("lex_check", "{\"path\":\"src/a.lex\"}") => "{\"tool\":\"lex_check\",\"target\":\"src/a.lex\",\"result\":\"pass\"}",
    verified_json("lex_check", "{}") => "{\"tool\":\"lex_check\",\"target\":\"\",\"result\":\"pass\"}",
    verified_json("lex_test", "not json") => "{\"tool\":\"lex_test\",\"target\":\"\",\"result\":\"pass\"}"
  }
{
  jv.stringify(JObj([("tool", JStr(tool_name)), ("target", JStr(target_of(args_raw))), ("result", JStr("pass"))]))
}

# Every verification tool that produces a verified.* event takes its
# subject as `path` — lex_check, lex_spec_check and lex_test all do. A
# tool that does not is not one of them, and gets an empty target rather
# than a guess.
fn target_of(args_raw :: Str) -> Str
  examples {
    target_of("{\"path\":\"src/a.lex\"}") => "src/a.lex",
    target_of("{\"path\":\"src/a.lex\",\"count\":10}") => "src/a.lex",
    target_of("{\"count\":10}") => "",
    target_of("{}") => "",
    target_of("") => ""
  }
{
  match jv.parse_into_errors(args_raw) {
    Err(_) => "",
    Ok(j) => match jv.get_field(j, "path") {
      Some(JStr(p)) => p,
      _ => "",
    },
  }
}

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

