# lex-llm — schema-typed completion
#
# Equivalent to pydantic-ai's result_type: ask the model to respond
# with a JSON object conforming to a ModelSchema. The response is
# validated; if invalid the model gets one retry with the validation
# errors as feedback.
#
#   1. Append a JSON-prompt suffix to the user prompt.
#   2. Call provider.chat with no tools — the response IS the result.
#   3. Extract the JSON block; validate via lex-schema.
#   4. On failure: build a retry prompt with error list; call again.
#   5. Return Ok or final Err.

import "./message" as msg

import "./delta" as d

import "./agent" as ag

import "lex-schema/schema" as s

import "lex-schema/json_value" as jv

import "lex-schema/error" as e

import "std.list" as list

import "std.str" as str

import "std.iter" as iter

# Ask the model to respond with a JSON object conforming to schema.
# One attempt + one retry on validation failure.
fn structured(agent :: ag.AgentDef, prompt :: Str, schema :: s.ModelSchema) -> [net, llm] Result[jv.Json, e.Errors] {
  let json_prompt := build_json_prompt(prompt, schema)
  let first_reply := call_once(agent, json_prompt)
  match validate_reply(schema, first_reply) {
    Ok(j) => Ok(j),
    Err(errs) => {
      let retry_prompt := build_retry_prompt(prompt, schema, first_reply, errs)
      let second_reply := call_once(agent, retry_prompt)
      validate_reply(schema, second_reply)
    },
  }
}

fn call_once(agent :: ag.AgentDef, prompt :: Str) -> [net, llm] Str {
  let messages := [SystemMsg(agent.goal), UserMsg(prompt)]
  let deltas := iter.to_list(agent.provider.chat(agent.model, messages, []))
  collect_text(deltas)
}

fn collect_text(deltas :: List[d.Delta]) -> Str
  examples {
    collect_text([TextChunk("hello"), TextChunk(" world")]) => "hello world",
    collect_text([FinishDelta("stop")]) => "",
    collect_text([]) => ""
  }
{
  list.fold(deltas, "", fn (acc :: Str, dl :: d.Delta) -> Str {
    match dl {
      TextChunk(s) => str.concat(acc, s),
      _ => acc,
    }
  })
}

fn validate_reply(schema :: s.ModelSchema, reply :: Str) -> Result[jv.Json, e.Errors] {
  let trimmed := extract_json_block(reply)
  match jv.parse_into_errors(trimmed) {
    Err(es) => Err(es),
    Ok(j) => s.validate(schema, j),
  }
}

fn build_json_prompt(prompt :: Str, schema :: s.ModelSchema) -> Str {
  let schema_str := jv.stringify_pretty(s.to_json_schema(schema))
  str.concat(prompt, str.concat("\n\nRespond with ONLY a JSON object matching this schema (no explanation, no markdown fence):\n", schema_str))
}

fn build_retry_prompt(prompt :: Str, schema :: s.ModelSchema, bad_reply :: Str, errs :: e.Errors) -> Str {
  str.concat(build_json_prompt(prompt, schema), str.concat("\n\nYour previous response was invalid. Validation errors:\n", str.concat(e.format(errs), str.concat("\n\nPrevious response:\n", str.concat(bad_reply, "\n\nPlease output corrected JSON only.")))))
}

# Strip a ```json...``` fence if present; return as-is otherwise.
fn extract_json_block(reply :: Str) -> Str
  examples {
    extract_json_block("{\"x\":1}") => "{\"x\":1}",
    extract_json_block("```json\n{\"x\":1}\n```") => "{\"x\":1}",
    extract_json_block("```\n{\"x\":1}\n```") => "{\"x\":1}",
    extract_json_block("  ```json\n{\"x\":1}\n```  ") => "{\"x\":1}"
  }
{
  let t := str.trim(reply)
  match str.strip_prefix(t, "```json") {
    Some(rest) => match str.strip_suffix(str.trim(rest), "```") {
      Some(inner) => str.trim(inner),
      None => t,
    },
    None => match str.strip_prefix(t, "```") {
      Some(rest) => match str.strip_suffix(str.trim(rest), "```") {
        Some(inner) => str.trim(inner),
        None => t,
      },
      None => t,
    },
  }
}

