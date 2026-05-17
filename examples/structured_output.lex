# lex-llm example — schema-typed structured output
#
# Demonstrates structured() — the pydantic-ai result_type equivalent.
# The model is asked to extract typed data from freeform text; lex-schema
# validates the output. On validation failure the model gets one retry
# with the error list as feedback.
#
# Two schemas shown:
#   MovieReview  — { title, year, sentiment, score, summary? }
#   ParsedAddress — { street, city, country, postal_code }

import "lex-llm/agent"      as ag
import "lex-llm/provider"   as prov
import "lex-llm/structured" as st

import "lex-llm/providers/anthropic" as anthropic_provider

import "lex-schema/schema"      as s
import "lex-schema/json_value"  as jv
import "lex-schema/error"       as e
import "lex-schema/constraints" as c

import "std.str" as str

# ---- Schemas -----------------------------------------------------

fn movie_review_schema() -> s.ModelSchema {
  {
    title:       "MovieReview",
    description: "Structured extraction of a movie review.",
    fields: [
      s.required_str("title",     [c.StrNonEmpty]),
      s.required_int("year",      [c.IntInRange(1888, 2100)]),
      s.required_str("sentiment", [c.StrOneOf(["positive", "neutral", "negative"])]),
      s.required_float("score",   [c.FloatInRange(0.0, 10.0)]),
      s.with_desc(
        s.optional(s.required_str("summary", [c.StrMaxLen(200)])),
        "One-sentence summary of the review"),
    ],
  }
}

fn address_schema() -> s.ModelSchema {
  {
    title:       "ParsedAddress",
    description: "A parsed postal address.",
    fields: [
      s.required_str("street",      [c.StrNonEmpty]),
      s.required_str("city",        [c.StrNonEmpty]),
      s.required_str("country",     [c.StrNonEmpty]),
      s.required_str("postal_code", [c.StrNonEmpty]),
    ],
  }
}

# ---- Agent setup -------------------------------------------------

fn make_agent(api_key :: Str) -> ag.AgentDef {
  {
    name:     "structured-extractor",
    goal:     "You extract structured data from text. Always respond with valid JSON matching the requested schema exactly.",
    model:    prov.claude_sonnet(),
    provider: anthropic_provider.make_provider(anthropic_provider.default_config(api_key)),
    tools:    [],
    options:  ag.default_options(),
  }
}

# ---- Extraction helpers ------------------------------------------

fn extract_movie_review(
  agent :: ag.AgentDef,
  text  :: Str
) -> [net] Result[jv.Json, e.Errors] {
  st.structured(agent,
    str.concat("Extract a structured movie review from the following text:\n", text),
    movie_review_schema())
}

fn extract_address(
  agent :: ag.AgentDef,
  text  :: Str
) -> [net] Result[jv.Json, e.Errors] {
  st.structured(agent,
    str.concat("Parse the following address into structured fields:\n", text),
    address_schema())
}

# ---- Demo entry points -------------------------------------------

fn demo_movie(agent :: ag.AgentDef) -> [net] Str {
  match extract_movie_review(agent,
    "Oppenheimer (2023) is a visually stunning and intellectually rich biopic. Nolan delivers his most ambitious film yet. Solid 9.2 out of 10.") {
    Ok(j)     => str.concat("[movie_review] ", jv.stringify_pretty(j)),
    Err(errs) => str.concat("[movie_review] FAILED: ", e.format(errs)),
  }
}

fn demo_address(agent :: ag.AgentDef) -> [net] Str {
  match extract_address(agent,
    "221B Baker Street, London, United Kingdom, NW1 6XE") {
    Ok(j)     => str.concat("[address] ", jv.stringify_pretty(j)),
    Err(errs) => str.concat("[address] FAILED: ", e.format(errs)),
  }
}

# main returns the two results as a combined string.
# Replace api_key with your actual key or read from env once
# the [env] effect RFC (lex-lang#XXX) lands.
fn main(api_key :: Str) -> [net] Str {
  let agent := make_agent(api_key)
  str.concat(demo_movie(agent), str.concat("\n", demo_address(agent)))
}
