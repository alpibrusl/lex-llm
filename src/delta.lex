# lex-llm — streaming delta and step types
#
# A Delta is one raw chunk from a provider's SSE stream.
# A Step is a higher-level event that run_loop emits to its caller.
# Callers wanting live token streaming consume StepDelta;
# callers only interested in final outputs filter for StepDone.

import "./message" as msg

# Provider-level streaming chunk.
type Delta =
    TextChunk(Str)             # incremental text token
  | ToolCallBegin(Str, Str)    # (call_id, tool_name) — model starts a call
  | ToolArgChunk(Str, Str)     # (call_id, partial_args_json)
  | FinishDelta(Str)           # finish_reason: "stop" | "tool_calls" | "length"

# run_loop-level event visible to callers.
type Step =
    StepDelta(Delta)           # raw provider chunk — stream for live UI
  | StepToolExec(Str, Str)     # (tool_name, call_id) — about to dispatch
  | StepToolResult(Str, Bool)  # (call_id, success)
  | StepDone(msg.Message)      # final assembled assistant message

fn is_finish(delta :: Delta) -> Bool
  examples {
    is_finish(TextChunk("hi"))       => false,
    is_finish(FinishDelta("stop"))   => true,
  }
{
  match delta {
    FinishDelta(_) => true,
    _              => false,
  }
}

fn finish_reason(delta :: Delta) -> Option[Str]
  examples {
    finish_reason(TextChunk("x"))      => None,
    finish_reason(FinishDelta("stop")) => Some("stop"),
  }
{
  match delta {
    FinishDelta(r) => Some(r),
    _              => None,
  }
}
