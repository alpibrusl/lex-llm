# lex-llm — Tool ADT + argument validation
#
# A Tool is a value. params is a lex-schema ModelSchema emitted as
# JSON Schema when building provider requests, and used to validate
# args before dispatch (OpenCode pattern: invalid args → recoverable
# error → model retries with corrected input).
#
# Effect note: execute is declared [net, io, proc] so tools may perform
# HTTP calls, filesystem I/O, or shell execution without widening the
# agent-loop signature further. Pure tools satisfy this constraint
# structurally (unused declared effects are ignored by the type checker).
#
# Spec gating: the optional `precondition` field holds a lex-spec `Spec`
# value. `filter_available(tools, bindings)` evaluates each tool's spec
# and drops those that Deny or are Inconclusive, so only contextually
# permitted tools are offered to the model on each turn.

import "lex-schema/schema" as s

import "lex-schema/json_value" as jv

import "lex-schema/error" as e

import "lex-spec/spec" as sp

import "lex-spec/eval" as ev

import "std.list" as list

import "std.str" as str

type Tool = { name :: Str, description :: Str, params :: s.ModelSchema, execute :: (jv.Json) -> [net, io, proc] Result[jv.Json, e.Errors], precondition :: Option[sp.Spec] }

# Canonical constructor — no precondition (always available).
fn define(name :: Str, description :: Str, params :: s.ModelSchema, execute :: (jv.Json) -> [net, io, proc] Result[jv.Json, e.Errors]) -> Tool {
  { name: name, description: description, params: params, execute: execute, precondition: None }
}

# Constructor with a spec precondition attached.
fn define_gated(name :: Str, description :: Str, params :: s.ModelSchema, execute :: (jv.Json) -> [net, io, proc] Result[jv.Json, e.Errors], spec :: sp.Spec) -> Tool {
  { name: name, description: description, params: params, execute: execute, precondition: Some(spec) }
}

# Attach or replace the precondition on an existing tool.
fn with_precondition(tool :: Tool, spec :: sp.Spec) -> Tool {
  { name: tool.name, description: tool.description, params: tool.params, execute: tool.execute, precondition: Some(spec) }
}

# Validate args through the tool's param schema then dispatch.
# Invalid args return Err so the caller can feed them back to the model.
fn validate_and_exec(tool :: Tool, args :: jv.Json) -> [net, io, proc] Result[jv.Json, e.Errors] {
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
      None => if tool.name == name {
        Some(tool)
      } else {
        None
      },
    }
  })
}

# Evaluate a tool's precondition against the supplied bindings.
# A tool with no precondition is always available.
fn is_available(tool :: Tool, bindings :: List[(Str, sp.SpecValue)]) -> Bool {
  match tool.precondition {
    None => true,
    Some(spec) => match ev.eval(spec, bindings) {
      Allow => true,
      _ => false,
    },
  }
}

# Return only the tools whose precondition evaluates to Allow given
# the current bindings. Tools with no precondition are always kept.
fn filter_available(tools :: List[Tool], bindings :: List[(Str, sp.SpecValue)]) -> List[Tool] {
  list.fold(tools, [], fn (acc :: List[Tool], tool :: Tool) -> List[Tool] {
    if is_available(tool, bindings) {
      list.concat(acc, [tool])
    } else {
      acc
    }
  })
}

# Format Errors as a short string for a ToolMsg body.
fn format_validation_error(errs :: e.Errors) -> Str {
  str.concat("tool argument validation failed:\n", e.format(errs))
}

# OpenAI function-tool JSON: { "type": "function", "function": { name, description, parameters } }
fn to_openai_json(tool :: Tool) -> jv.Json {
  JObj([("type", JStr("function")), ("function", JObj([("name", JStr(tool.name)), ("description", JStr(tool.description)), ("parameters", s.to_json_schema(tool.params))]))])
}

# Anthropic tool JSON: { name, description, input_schema }
fn to_anthropic_json(tool :: Tool) -> jv.Json {
  JObj([("name", JStr(tool.name)), ("description", JStr(tool.description)), ("input_schema", s.to_json_schema(tool.params))])
}

# Google Gemini function declaration: { name, description, parameters }
# Vertex AI requires uppercase type names (STRING, INTEGER, OBJECT, ARRAY, BOOLEAN)
# and no $schema key — different from standard JSON Schema used by Anthropic/OpenAI.
fn to_google_json(tool :: Tool) -> jv.Json {
  JObj([("name", JStr(tool.name)), ("description", JStr(tool.description)), ("parameters", schema_to_google(tool.params))])
}

fn schema_to_google(schema :: s.ModelSchema) -> jv.Json {
  let props := list.map(schema.fields, fn (field :: s.Field) -> (Str, jv.Json) {
    (field.name, field_kind_to_google(field))
  })
  let required := list.fold(schema.fields, [], fn (acc :: List[jv.Json], field :: s.Field) -> List[jv.Json] {
    if field.required {
      list.concat(acc, [JStr(field.name)])
    } else {
      acc
    }
  })
  if list.is_empty(required) {
    JObj([("type", JStr("OBJECT")), ("properties", JObj(props))])
  } else {
    JObj([("type", JStr("OBJECT")), ("properties", JObj(props)), ("required", JList(required))])
  }
}

fn field_kind_to_google(field :: s.Field) -> jv.Json {
  let base_entries := kind_entries(field.kind)
  if str.is_empty(field.description) {
    JObj(base_entries)
  } else {
    JObj(list.concat(base_entries, [("description", JStr(field.description))]))
  }
}

fn kind_entries(kind :: s.FieldKind) -> List[(Str, jv.Json)] {
  match kind {
    KStr(_) => [("type", JStr("STRING"))],
    KInt(_) => [("type", JStr("INTEGER"))],
    KFloat(_) => [("type", JStr("NUMBER"))],
    KBool => [("type", JStr("BOOLEAN"))],
    KArray(elem, _) => [("type", JStr("ARRAY")), ("items", JObj(kind_entries(elem)))],
    KObject(sub) => {
      let props := list.map(sub.fields, fn (field :: s.Field) -> (Str, jv.Json) {
        (field.name, field_kind_to_google(field))
      })
      let required := list.fold(sub.fields, [], fn (acc :: List[jv.Json], field :: s.Field) -> List[jv.Json] {
        if field.required {
          list.concat(acc, [JStr(field.name)])
        } else {
          acc
        }
      })
      if list.is_empty(required) {
        [("type", JStr("OBJECT")), ("properties", JObj(props))]
      } else {
        [("type", JStr("OBJECT")), ("properties", JObj(props)), ("required", JList(required))]
      }
    },
  }
}

