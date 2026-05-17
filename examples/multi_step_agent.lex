# lex-llm example — multi-step agent with three local tools
#
# Demonstrates run_loop with a purely local Ollama model and three
# tools that have no external effects (pure computation + string ops).
# No A2A, no remote APIs other than the local Ollama endpoint.
#
# Tools:
#   calculator  — evaluate a simple arithmetic expression (stub)
#   word_count  — count words in a text string
#   title_case  — convert text to Title Case
#
# Run (Ollama must be running locally with llama3 pulled):
#   lex run --allow-effects net examples/multi_step_agent.lex main

import "lex-llm/agent"    as ag
import "lex-llm/message"  as msg
import "lex-llm/provider" as prov
import "lex-llm/tool"     as t
import "lex-llm/delta"    as d

import "lex-llm/providers/ollama" as ollama_provider

import "lex-schema/schema"      as s
import "lex-schema/json_value"  as jv
import "lex-schema/error"       as e
import "lex-schema/constraints" as c

import "std.list" as list
import "std.str"  as str
import "std.iter" as iter

# ---- Tool definitions --------------------------------------------

fn make_calculator() -> t.Tool {
  let params := {
    title:       "CalculatorArgs",
    description: "Evaluate a numeric expression.",
    fields: [
      s.required_str("expression", [c.StrNonEmpty]),
    ],
  }
  t.define(
    "calculator",
    "Evaluate a simple arithmetic expression (supports +, -, *, /).",
    params,
    fn (args :: jv.Json) -> [net] Result[jv.Json, e.Errors] {
      let expr := match jv.get_field(args, "expression") {
        Some(JStr(s)) => s, _ => ""
      }
      # Stub — a real impl would parse and evaluate the expression.
      # Replace with std.expr.eval once that lands, or a safe sandbox call.
      Ok(JObj([("result", JStr(str.concat("evaluated: ", expr)))]))
    })
}

fn make_word_count() -> t.Tool {
  let params := {
    title:       "WordCountArgs",
    description: "Count words in a block of text.",
    fields: [s.required_str("text", [])],
  }
  t.define(
    "word_count",
    "Count the number of whitespace-delimited words in the provided text.",
    params,
    fn (args :: jv.Json) -> [net] Result[jv.Json, e.Errors] {
      let text  := match jv.get_field(args, "text") { Some(JStr(s)) => s, _ => "" }
      let count := list.len(str.split(str.trim(text), " "))
      Ok(JObj([("count", JInt(count))]))
    })
}

fn make_title_case() -> t.Tool {
  let params := {
    title:       "TitleCaseArgs",
    description: "Convert text to Title Case.",
    fields: [s.required_str("text", [])],
  }
  t.define(
    "title_case",
    "Convert the provided text to Title Case (each word capitalised).",
    params,
    fn (args :: jv.Json) -> [net] Result[jv.Json, e.Errors] {
      let text := match jv.get_field(args, "text") { Some(JStr(s)) => s, _ => "" }
      Ok(JObj([("result", JStr(to_title_case(text)))]))
    })
}

# Pure helper — eligible for examples {} block because it's pure.
fn to_title_case(text :: Str) -> Str
  examples {
    to_title_case("hello world") => "Hello World",
    to_title_case("")            => "",
  }
{
  str.join(
    list.map(str.split(text, " "), fn (w :: Str) -> Str { capitalize(w) }),
    " ")
}

fn capitalize(word :: Str) -> Str
  examples {
    capitalize("hello") => "Hello",
    capitalize("")      => "",
  }
{
  if str.is_empty(word) { word }
  else {
    str.concat(
      str.to_upper(str.slice(word, 0, 1)),
      str.slice(word, 1, str.len(word)))
  }
}

# ---- Agent setup -------------------------------------------------

fn make_agent() -> ag.AgentDef {
  {
    name:     "demo-agent",
    goal:     "You are a helpful assistant. Use the available tools to answer multi-step tasks accurately. Call tools when needed; do not guess at results.",
    model:    prov.ollama("llama3"),
    provider: ollama_provider.make_provider(ollama_provider.default_config()),
    tools:    [make_calculator(), make_word_count(), make_title_case()],
    options:  ag.default_options(),
  }
}

# ---- Main --------------------------------------------------------

fn summarise_steps(steps :: List[d.Step]) -> Str {
  list.fold(steps, "", fn (acc :: Str, step :: d.Step) -> Str {
    match step {
      d.StepDelta(d.TextChunk(s)) =>
        str.concat(acc, s),
      d.StepToolExec(name, id) =>
        str.concat(acc,
          str.concat("\n[tool] ",
            str.concat(name, str.concat(" (", str.concat(id, ")\n"))))),
      d.StepToolResult(id, ok) => {
        let status := if ok { "ok" } else { "err" }
        str.concat(acc, str.concat("[result] ", str.concat(id, str.concat(" => ", str.concat(status, "\n")))))
      },
      d.StepDone(msg.AssistantMsg(text, _)) =>
        str.concat(acc, str.concat("\n[done] ", text)),
      _ => acc,
    }
  })
}

fn main() -> [net] Str {
  let agent := make_agent()
  let task  := "Please do all three tasks: (1) calculate 17 * 3 + 8, (2) count the words in 'the quick brown fox jumps over the lazy dog', (3) convert 'hello world from lex-llm' to title case."
  let steps := iter.collect(ag.run_loop(agent, [msg.UserMsg(task)]))
  summarise_steps(steps)
}
