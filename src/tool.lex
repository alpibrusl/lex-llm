# lex-llm — Tool ADT + argument validation
#
# A Tool is a value. params is a lex-schema ModelSchema emitted as
# JSON Schema when building provider requests, and used to validate
# args before dispatch (OpenCode pattern: invalid args → recoverable
# error → model retries with corrected input).
#
# Effect note: execute is declared [net] because run_loop is already
# [net]. Pure tools satisfy this constraint structurally. Tools needing
# [io] or [proc] would require those effects added to the agent loop.
# Tracked in lex-llm#3.

import "lex-schema/schema"     as s
import "lex-schema/json_value" as jv
import "lex-schema/error"      as e

import "std.list" as list
import "std.str"  as str

type Tool = {
  name        :: Str,
  description :: Str,
  params      :: s.ModelSchema,
  execute     :: (jv.Json) -> [net] Result[jv.Json, e.Errors],
}

# Canonical constructor.
fn define(
  name        :: Str,
  description :: Str,
  params      :: s.ModelSchema,
  execute     :: (jv.Json) -> [net] Result[jv.Json, e.Errors]
) -> Tool {
  { name: name, description: description, params: params, execute: execute }
}

# Validate args through the tool's param schema then dispatch.
# Invalid args return Err so the caller can feed them back to the model.
fn validate_and_exec(tool :: Tool, args :: jv.Json) -> [net] Result[jv.Json, e.Errors] {
  match s.validate(tool.params, args) {
    Err(errs) => Err(errs),
    Ok(valid) => tool.execute(valid),
  }
}

# Lookup a tool by name — O(n) linear scan.
fn find_by_name(tools :: List[Tool], name :: Str) -> Option[Tool] {
  list.fold(tools, None, fn (acc :: Option[Tool], tool :: Tool) -> Option[Tool] {
    match acc {
      Some(_) => acc,
      None    => if tool.name == name { Some(tool) } else { None },
    }
  })
}

# Format Errors as a short string for a ToolMsg body.
fn format_validation_error(errs :: e.Errors) -> Str {
  str.concat("tool argument validation failed:\n", e.format(errs))
}

# OpenAI function-tool JSON: { "type": "function", "function": { name, description, parameters } }
fn to_openai_json(tool :: Tool) -> jv.Json {
  JObj([
    ("type",     JStr("function")),
    ("function", JObj([
      ("name",        JStr(tool.name)),
      ("description", JStr(tool.description)),
      ("parameters",  s.to_json_schema(tool.params)),
    ])),
  ])
}

# Anthropic tool JSON: { name, description, input_schema }
fn to_anthropic_json(tool :: Tool) -> jv.Json {
  JObj([
    ("name",         JStr(tool.name)),
    ("description",  JStr(tool.description)),
    ("input_schema", s.to_json_schema(tool.params)),
  ])
}

# Google Gemini function declaration: { name, description, parameters }
fn to_google_json(tool :: Tool) -> jv.Json {
  JObj([
    ("name",        JStr(tool.name)),
    ("description", JStr(tool.description)),
    ("parameters",  s.to_json_schema(tool.params)),
  ])
}
