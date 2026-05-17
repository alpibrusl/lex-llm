# lex-llm — conversational message ADT
#
# Roles follow the OpenAI/Anthropic wire convention:
#   UserMsg      — human turn
#   SystemMsg    — system prompt injected before the conversation
#   AssistantMsg — model response; carries tool calls when the model
#                  chose to invoke tools rather than produce text
#   ToolMsg      — tool-execution result matched to a call by call_id
#
# The ADT is provider-agnostic; each adapter converts to its wire format.

import "lex-schema/json_value" as jv
import "std.list" as list
import "std.str"  as str

# One tool invocation requested by the model.
type ToolCall = {
  id   :: Str,      # provider-assigned id for result correlation
  name :: Str,      # matches Tool.name in the agent's tool list
  args :: jv.Json,  # raw args JSON — validated by tool.lex before dispatch
}

type Message =
    UserMsg(Str)
  | SystemMsg(Str)
  | AssistantMsg(Str, List[ToolCall])   # (text_content, pending_calls)
  | ToolMsg(Str, Str)                   # (call_id, result_content)

fn content(msg :: Message) -> Str
  examples {
    content(UserMsg("hello"))         => "hello",
    content(SystemMsg("be helpful"))  => "be helpful",
    content(ToolMsg("id1", "42"))     => "42",
  }
{
  match msg {
    UserMsg(s)         => s,
    SystemMsg(s)       => s,
    AssistantMsg(s, _) => s,
    ToolMsg(_, s)      => s,
  }
}

fn role_str(msg :: Message) -> Str
  examples {
    role_str(UserMsg("hi"))    => "user",
    role_str(SystemMsg("sys")) => "system",
  }
{
  match msg {
    UserMsg(_)         => "user",
    SystemMsg(_)       => "system",
    AssistantMsg(_, _) => "assistant",
    ToolMsg(_, _)      => "tool",
  }
}

fn has_tool_calls(msg :: Message) -> Bool
  examples {
    has_tool_calls(UserMsg("hi"))          => false,
    has_tool_calls(AssistantMsg("ok", [])) => false,
  }
{
  match msg {
    AssistantMsg(_, calls) => not list.is_empty(calls),
    _                      => false,
  }
}

fn user(text :: Str)   -> Message { UserMsg(text) }
fn system(text :: Str) -> Message { SystemMsg(text) }

fn tool_result(call_id :: Str, body :: jv.Json) -> Message {
  ToolMsg(call_id, jv.stringify(body))
}

fn tool_error(call_id :: Str, err :: Str) -> Message {
  ToolMsg(call_id,
    str.concat("{\"error\":\"",
      str.concat(json_escape(err), "\"}")))
}

# Minimal JSON string escape for embedding in a string literal.
fn json_escape(s :: Str) -> Str {
  let chars := str.split(s, "")
  list.fold(chars, "", fn (acc :: Str, c :: Str) -> Str {
    str.concat(acc, match c {
      "\"" => "\\\"",
      "\\" => "\\\\",
      "\n" => "\\n",
      "\r" => "\\r",
      _    => c,
    })
  })
}
