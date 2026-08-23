# lex-llm — Agent Guidelines

Pure-Lex LLM-agent runtime. Provides provider abstraction, a tool-call
loop, and schema-validated structured output. No Rust — only Lex source
and `lex-schema` as a dependency.

---

## Core loop

```
AgentDef
  → run_loop(agent, messages) -> [net, llm, io, proc] Iter[Step]
      1. Prepend agent.goal as SystemMsg
      2. provider.chat → Iter[Delta]   (one HTTP round-trip per turn)
      3. collect Deltas → CollectedResponse
      4. finish_reason = "tool_calls"?
           a. validate args via lex-schema → Err feeds back to model
           b. execute tools → StepToolExec + StepToolResult
           c. append AssistantMsg + ToolMsg; recurse within budget
      5. otherwise → StepDone
```

`run_loop` returns a lazy `Iter[Step]` backed by eager collection; call
`iter.to_list` or `iter.fold` to drain it.

---

## Key types

| type | file | purpose |
|---|---|---|
| `AgentDef` | `src/agent.lex` | agent value — model, provider, tools, goal, options |
| `Provider` | `src/provider.lex` | `chat :: (ModelRef, List[Message], List[Tool]) -> [net, llm] Iter[Delta]` |
| `Tool` | `src/tool.lex` | `execute :: (Json) -> [net, io, proc] Result[Json, Errors]` |
| `Delta` | `src/delta.lex` | streaming event: `TextChunk`, `ToolCallBegin`, `ToolArgChunk`, `FinishDelta` |
| `Step` | `src/delta.lex` | agent step: `StepDelta`, `StepToolExec`, `StepToolResult`, `StepDone` |
| `Message` | `src/message.lex` | `UserMsg`, `SystemMsg`, `AssistantMsg`, `ToolMsg`, `UserPartsMsg` |
| `Part` | `src/message.lex` | multimodal turn piece: `TextPart`, `ImagePart` |
| `Image` | `src/message.lex` | `ImageB64(mime, base64)` — inline bytes; no URL variant (see note) |

### Images

Build a multimodal turn with `msg.user_with_images(text, images)`, or
`msg.user_with_jpeg(text, jpeg_b64)` for the single-frame case. Images are
inline base64 with an explicit media type — deliberately no URL variant,
because Gemini's `inline_data` has no URL form and a url case would
type-check everywhere then fail at runtime on some providers. Encode bytes
with `crypto.base64_encode` (standard alphabet — NOT `base64url_encode`,
whose output providers reject).

Each adapter maps `UserPartsMsg` to its own wire shape: OpenAI a `data:` URI
in an `image_url` part, Anthropic a typed base64 `source` block, Gemini
(google/vertex) an `inline_data` part, Ollama a sibling `images` array of
bare base64 with `content` left a plain string. `tests/test_image_parts.lex`
pins all five.

**Adding a provider, or a Message variant?** Lex does not check match
exhaustiveness: a `match` missing an arm type-checks clean, passes
`lex check --strict`, and panics at runtime with `non-exhaustive match` the
first time that variant arrives. The test file is the only thing standing in
for the compiler here — extend it.

---

## Providers

| import | constructor | notes |
|---|---|---|
| `lex-llm/providers/openai` | `make_provider(config)` | OpenAI Chat Completions; also Azure + compatible proxies |
| `lex-llm/providers/anthropic` | `make_provider(config)` | Anthropic Messages API |
| `lex-llm/providers/google` | `make_provider(config)` | Google Gemini generateContent |
| `lex-llm/providers/ollama` | `make_provider(config)` | Ollama local inference |

Convenience constructors: `prov.gpt4o()`, `prov.claude_sonnet()`,
`prov.gemini_flash()`, `prov.ollama("llama3")`.

---

## Effects

| declared | why |
|---|---|
| `[net]` | `http.stream_lines` in every provider |
| `[llm]` | semantic annotation — LLM-inference calls; lets policies gate separately |
| `[io]`, `[proc]` | `Tool.execute` may do filesystem or shell I/O |

`run_loop` / `run_steps` declare `[net, llm, io, proc]`. Callers must
allow the same set (or a superset) in their policy.

Pure tools satisfy `[net, io, proc]` structurally — unused declared
effects are ignored at the call site.

---

## Defining a tool

```lex
import "lex-llm/tool"    as t
import "lex-schema/schema"     as s
import "lex-schema/json_value" as jv
import "lex-schema/error"      as e
import "lex-schema/constraints" as c

fn my_tool() -> t.Tool {
  t.define(
    "my_tool",
    "One-line description for the model.",
    {
      title:       "MyToolArgs",
      description: "Argument schema.",
      fields: [s.required_str("query", [c.StrNonEmpty])],
    },
    fn (args :: jv.Json) -> [net, io, proc] Result[jv.Json, e.Errors] {
      let q := match jv.get_field(args, "query") { Some(JStr(s)) => s, _ => "" }
      Ok(JStr(str.concat("result for: ", q)))
    })
}
```

---

## Structured output

`structured.structured(agent, prompt, schema)` asks the model to respond
with a JSON object conforming to a `ModelSchema`. One attempt + one retry
on validation failure. Returns `[net, llm] Result[Json, Errors]`.

---

## Running tests

```bash
lex test           # runs tests/test_*.lex
```

Tests cover pure helpers (`sse`, `structured`). Effects-heavy paths
(provider HTTP, agent loop) are covered by the example programs under
`examples/` — run them against a local Ollama instance or with a live API key.

---

## Known limitations

- `http.stream_lines` (the SSE transport) buffers the full response body
  before yielding lines. LLM provider APIs close the connection after all
  events, so this works in practice. Generic persistent SSE feeds would
  hang — see lex-lang docs for the ureq upgrade roadmap.
- No built-in retry / back-off on transient HTTP errors; wrap
  `run_loop` in a `flow.retry` if needed.
