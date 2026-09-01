# lex-llm

[![CI](https://github.com/alpibrusl/lex-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/alpibrusl/lex-llm/actions/workflows/ci.yml)

**Part of the [Lex](https://lexlang.org) project** — Library · [Manifesto](https://lexlang.org/manifesto) · [All packages](https://lexlang.org)

Pure-Lex LLM-agent runtime. Provider abstraction, multi-step tool-call
loop, and schema-validated structured output — no Rust, no dependencies
beyond [lex-schema](https://github.com/alpibrusl/lex-schema).

---

## Install

```toml
# lex.toml
[dependencies]
"lex-llm" = { git = "https://github.com/alpibrusl/lex-llm" }
```

---

## Demo — spec-gated agent tools

Typed permissions · property-checked · formally verified. Zero LLM API calls needed.

[![asciicast](https://asciinema.org/a/zb1DmHg7vQzhLqGM.svg)](https://asciinema.org/a/zb1DmHg7vQzhLqGM)

```sh
bash examples/spec_gated_agent.sh
```

Define a spec (`IF tool == "submit_order" THEN qty ≤ 1000 AND approved == true`), evaluate it against four tool-call scenarios, property-check it across 100 random inputs, and export a Z3-compatible SMT-LIB script — all from the same typed policy expression.

---

## Quick start

```lex
import "lex-llm/agent"    as ag
import "lex-llm/message"  as msg
import "lex-llm/provider" as prov
import "lex-llm/delta"    as d

import "lex-llm/providers/anthropic" as anthropic

import "std.iter" as iter
import "std.list" as list
import "std.str"  as str

fn main(api_key :: Str) -> [net, llm] Str {
  let provider := anthropic.make_provider(anthropic.default_config(api_key))
  let agent := {
    name:     "assistant",
    goal:     "You are a concise, helpful assistant.",
    model:    prov.claude_sonnet(),
    provider: provider,
    tools:    [],
    options:  ag.default_options(),
  }
  let steps := iter.to_list(ag.run_loop(agent, [msg.UserMsg("What is 2 + 2?")]))
  list.fold(steps, "", fn (acc :: Str, step :: d.Step) -> Str {
    match step {
      d.StepDone(msg.AssistantMsg(text, _)) => str.concat(acc, text),
      d.StepDelta(d.TextChunk(s))           => str.concat(acc, s),
      _                                     => acc,
    }
  })
}
```

Run:

```bash
lex run --allow-effects net,llm main.lex main '"sk-ant-..."'
```

---

## Providers

| Module | Constructor | Models |
|---|---|---|
| `lex-llm/providers/openai` | `make_provider(default_config(api_key))` | GPT-4o, GPT-4o-mini, o1, o3-mini, … |
| `lex-llm/providers/anthropic` | `make_provider(default_config(api_key))` | claude-opus-5, claude-sonnet-5, claude-haiku-4-5, … |
| `lex-llm/providers/google` | `make_provider({ api_key: key })` | gemini-2.0-flash, gemini-2.5-pro, … |
| `lex-llm/providers/ollama` | `make_provider(default_config())` | llama3, mistral, qwen2, … (local) |
| `lex-llm/providers/vertex` | `make_provider(config_at(token, project, location))` | gemini-3.5-flash, … (Vertex AI, EU) |

OpenAI-compatible local servers reuse the `openai` adapter via convenience
factories in `lex-llm/providers`:

| Server | Constructor | Notes |
|---|---|---|
| vLLM | `providers.vllm_at(host)` | host → `host + /v1/chat/completions` |
| MLX (Apple Silicon) | `providers.mlx_at(host)` | `mlx_lm.server`, run with `--host 0.0.0.0`; e.g. `mlx-community/Qwen2.5-7B-Instruct-4bit` |

`providers.select_provider(name, url, key)` builds any of the above from a
`(name, url, key)` triple — `name` ∈ `mlx | ollama | vllm | openai | anthropic |
google | mistral | vertex` (vertex packs `key` as `"<access_token>|||<project>"`).
The OpenAI adapter tolerates content-embedded tool calls (fenced ```json),
leaked EOS tokens, and reasoning-only turns so local servers work out of the box.

Convenience model refs: `prov.gpt4o()`, `prov.claude_sonnet()`,
`prov.gemini_flash()`, `prov.ollama("llama3")`.

---

## Streaming (optional)

`Provider` has two halves. `chat` is required and buffers: the whole turn is
in hand before the first `Delta` is observable. `stream` is optional —
`None` is a complete adapter — and yields `Delta`s as they come off the
socket.

| adapter | `stream` |
|---|---|
| `openai` (and everything routed through it: Mistral, LiteLLM, vLLM, lex-moe, MLX, opencode-go) | ✅ |
| `anthropic` | ✅ |
| `ollama` | ✅ |
| `google`, `vertex` | `None` — Gemini returns a JSON *array*, not SSE, so there is no line-at-a-time parse without `?alt=sse` |

Nothing existing changes: a caller that only uses `chat` sees identical
behaviour, and `streaming.collect_via` falls back to `chat` for an adapter
that declares `None`.

To paint tokens as they arrive, drive the cursor yourself — one pull, print
what it produced, pull again:

```lex
fn paint(sc :: prov.StreamChat, s :: Stream[Str], cur :: streaming.Cursor) -> [io, stream] Unit {
  match streaming.pull(sc, s, cur) {
    (next, deltas) => {
      let _e := emit(deltas)
      if streaming.is_done(next) { () } else { paint(sc, s, next) }
    },
  }
}
```

`examples/streaming_tokens.lex` is that loop as a runnable program.

Or, for the whole turn without a per-token loop,
`streaming.collect_via(provider, model, messages, tools)`.

Consuming a stream carries `[stream]` — add it to `--allow-effects`.

---

## Tools

```lex
import "lex-llm/tool"          as t
import "lex-schema/schema"      as s
import "lex-schema/json_value"  as jv
import "lex-schema/error"       as e
import "lex-schema/constraints" as c

fn search_tool() -> t.Tool {
  t.define(
    "search",
    "Search the web for a query and return a summary.",
    {
      title:       "SearchArgs",
      description: "Web search parameters.",
      fields:      [s.required_str("query", [c.StrNonEmpty])],
    },
    fn (args :: jv.Json) -> [net, io, proc] Result[jv.Json, e.Errors] {
      let q := match jv.get_field(args, "query") { Some(JStr(s)) => s, _ => "" }
      # ... call your search API here ...
      Ok(JStr(str.concat("results for: ", q)))
    })
}
```

Pass tools into the agent and run with the matching effects:

```bash
lex run --allow-effects net,llm,io,proc agent.lex main '"api-key"'
```

The loop validates tool arguments against the schema before calling
`execute`. Invalid args are returned to the model as an error so it can
self-correct on the next turn.

### Human approval gating

Mark any tool with `t.with_approval(tool, scope)` and the dispatch layer
blocks on a human decision — lex-lang's `[approval]` host boundary —
before every execution:

```lex
let dangerous := t.with_approval(submit_order_tool(), "payment")
```

Approved → the tool runs (the operator's answer is available to the tool
as the `_approval_answer` arg). Denied, or no `ApprovalSink` configured
in the host → the model gets a recoverable `approval_denied` error. The
scope is checked against `--allow-approval` at run time, so an operator
can grant `payment` approvals without opening other channels.

`human.make_ask_human_tool(scope)` is the canonical answer-carrying
example: the whole tool is the question→answer round-trip, so its body
just returns the injected answer. How a human is actually reached
(stdin, dashboard, chat) is the embedding host's `ApprovalSink`, not
Lex-side code. Requires the `[approval]` effect (lex-lang > v0.10.9).

---

## Structured output

Ask the model to return a typed JSON object conforming to a schema:

```lex
import "lex-llm/structured"     as st
import "lex-schema/schema"       as s
import "lex-schema/constraints"  as c
import "lex-schema/json_value"   as jv
import "lex-schema/error"        as e

fn sentiment_schema() -> s.ModelSchema {
  {
    title:       "Sentiment",
    description: "Sentiment classification result.",
    fields: [
      s.required_str("label",  [c.StrOneOf(["positive", "neutral", "negative"])]),
      s.required_float("score", [c.FloatInRange(0.0, 1.0)]),
    ],
  }
}

fn classify(agent :: ag.AgentDef, text :: Str) -> [net, llm] Result[jv.Json, e.Errors] {
  st.structured(agent, str.concat("Classify the sentiment of: ", text), sentiment_schema())
}
```

One attempt + one automatic retry on validation failure.

---

## Effects

| Effect | Why |
|---|---|
| `[net]` | HTTP calls — `http.send` on the buffered path, `http.stream_lines` on the streaming one |
| `[stream]` | Pulling a streamed response line-by-line — only for callers of `src/streaming.lex` |
| `[llm]` | Semantic annotation for LLM-inference; enables independent policy gating |
| `[io]`, `[proc]` | Available to tools that need filesystem or shell access |
| `[approval]` | Human decision points — dispatch blocks on `std.approval.request` for approval-scoped tools |

`ag.run_loop` declares `[net, llm, io, proc, approval]`. Pass
`--allow-effects net,llm,approval` for tool-free agents; add `io,proc` if
any tool needs them, and `--allow-approval scope1,scope2` to restrict
which approval channels tools may open.

Pure tools (no actual I/O) satisfy `[net, io, proc]` structurally — the
type checker accepts a closure that never uses those effects.

---

## Running the examples

```bash
# Requires a local Ollama instance with llama3 pulled
lex run --allow-effects net,llm,io,proc examples/multi_step_agent.lex main

# Requires an Anthropic API key
lex run --allow-effects net,llm examples/structured_output.lex main '"sk-ant-..."'
```

---

## Tests

```bash
lex test --allow-effects io,time,crypto,random,sql,fs_read,fs_write,net,concurrent,llm,proc,approval,stream tests/
```

---

## Known limitations

- **The agent loop is still buffered.** `Provider.stream` makes one turn's
  `Delta`s arrive incrementally, and that is what a UI needs. `ag.run_loop`
  does not use it: it returns `List[Step]` for the whole multi-turn loop, so
  a caller wanting live tokens drives `streaming.pull` directly rather than
  going through `run_loop`.
- **`google` and `vertex` do not stream** — see the table above.
- **No built-in retry** — wrap `run_loop` in `flow.retry` for transient
  HTTP error handling.

---

Built under the principles of [Trust Without Comprehension](https://lexlang.org/manifesto).

## License

Copyright (c) 2026 lex-llm contributors.

Licensed under the [EUPL-1.2](LICENSE) — the European Union Public Licence, as used across the `lex-*` ecosystem.
