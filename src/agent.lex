# lex-llm — AgentDef + run_loop
#
# run_loop drives the stream → collect → dispatch → continue cycle:
#
#   1. Prepend agent.goal as the system message.
#   2. Call provider.chat; collect Deltas eagerly (one HTTP round-trip).
#   3. Assemble a CollectedResponse from the Deltas.
#   4. If finish_reason = "tool_calls":
#        a. Validate args via lex-schema; invalid → Err → ToolMsg so
#           the model can self-correct on the next turn.
#        b. Execute valid tools; emit StepToolExec + StepToolResult.
#        c. Append AssistantMsg + ToolMsg results; recurse within budget.
#   5. Otherwise emit StepDone.
#
# Inter-turn per-token streaming will improve once lex-lang#487 closes.

import "./message"  as msg
import "./delta"    as d
import "./tool"     as t
import "./provider" as prov

import "lex-schema/json_value" as jv
import "lex-schema/error"      as e

import "std.list" as list
import "std.str"  as str
import "std.iter" as iter

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

# ---- Public entry point ------------------------------------------

fn run_loop(
  agent        :: AgentDef,
  conversation :: List[msg.Message]
) -> [net, llm, io, proc] Iter[d.Step] {
  let budget := unwrap_int(agent.options.max_steps, 20)
  iter.from_list(run_steps(agent, conversation, budget))
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
    let messages   := list.concat([msg.SystemMsg(agent.goal)], conv)
    let raw_deltas := iter.collect(agent.provider.chat(agent.model, messages, agent.tools))
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
          content:       acc.content,
          tool_calls:    list.reverse(acc.tool_calls),
          finish_reason: reason,
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

fn dispatch_calls(
  tools :: List[t.Tool],
  calls :: List[CollectedCall]
) -> [net, io, proc] List[Dispatch] {
  list.map(calls, fn (call :: CollectedCall) -> [net, io, proc] Dispatch {
    dispatch_one(tools, call)
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
