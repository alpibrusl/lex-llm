# lex-llm — streaming delta and step types
#
# A Delta is one raw chunk from a provider's SSE stream.
# A Step is a higher-level event that run_loop emits to its caller.
# Callers wanting live token streaming consume StepDelta;
# callers only interested in final outputs filter for StepDone.

import "./message" as msg

# Provider-level streaming chunk.
# UsageDelta carries (prompt_tokens, completion_tokens, total_tokens) from the
# provider's own response when it reports them (e.g. OpenAI-compatible chat
# completions' top-level "usage" object) -- providers that don't report usage
# simply never emit this variant, so callers should treat its absence as
# "unknown", not "zero cost".
type Delta = TextChunk(Str) | ToolCallBegin((Str, Str)) | ToolArgChunk((Str, Str)) | FinishDelta(Str) | UsageDelta((Int, Int, Int))

type Step = StepDelta(Delta) | StepToolExec((Str, Str)) | StepToolResult((Str, Bool)) | StepDone(msg.Message)

fn is_finish(delta :: Delta) -> Bool
  examples {
    is_finish(TextChunk("hi")) => false,
    is_finish(FinishDelta("stop")) => true
  }
{
  match delta {
    FinishDelta(_) => true,
    _ => false,
  }
}

fn finish_reason(delta :: Delta) -> Option[Str]
  examples {
    finish_reason(TextChunk("x")) => None,
    finish_reason(FinishDelta("stop")) => Some("stop")
  }
{
  match delta {
    FinishDelta(r) => Some(r),
    _ => None,
  }
}

