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
| `Provider` | `src/provider.lex` | `chat` (required, buffered) + `stream :: Option[StreamChat]` (optional, incremental) |
| `StreamChat` | `src/provider.lex` | `open` `[net, llm]` → `Result[Stream[Str], Str]`, plus a pure `init` / `step` parser |
| `Cursor` | `src/streaming.lex` | a partly-consumed stream: parser state + end-of-stream flag |

Three entry points to the loop, differing only in when the caller learns
things: `run_loop` (buffered, untraced), `run_loop_traced` (buffered, with
trail events), `run_steps_streamed` (trail events, and `on_step` fires the
moment each Step exists). The streamed one returns the same `List[Step]` the
others do — **do not also walk that list with `on_step`**, or the turn prints
twice. It works with a `stream: None` provider too: the Deltas arrive in one
burst through the same callback, so a caller never branches on whether its
provider streams.
| `Tool` | `src/tool.lex` | `execute :: (Json) -> [net, io, proc] Result[Json, Errors]` |
| `Delta` | `src/delta.lex` | streaming event: `TextChunk`, `ToolCallBegin`, `ToolArgChunk`, `FinishDelta` |
| `Step` | `src/delta.lex` | agent step: `StepDelta`, `StepToolExec`, `StepToolResult`, `StepDone` |
| `Message` | `src/message.lex` | `UserMsg`, `SystemMsg`, `AssistantMsg`, `ToolMsg` |

---

## Providers

| import | constructor | notes |
|---|---|---|
| `lex-llm/providers/openai` | `make_provider(config)` | OpenAI Chat Completions; also Azure + compatible proxies |
| `lex-llm/providers/anthropic` | `make_provider(config)` | Anthropic Messages API |
| `lex-llm/providers/google` | `make_provider(config)` | Google Gemini generateContent |
| `lex-llm/providers/ollama` | `make_provider(config)` | Ollama local inference |

`stream` is implemented for `openai` (and therefore Mistral, LiteLLM, vLLM,
lex-moe, MLX, opencode-go), `anthropic` and `ollama`; `google` and `vertex`
declare `None` because Gemini answers with a JSON array rather than SSE.

**Adding a streaming half to an adapter.** Three parts, split on the effect
line: `open` does the request with `http.stream_lines` and returns the raw
lines; `init` and `step` are pure, and `step` takes one line and the state
so far and returns the next state and any `Delta`s. The state is `jv.Json`
because the record field's type is fixed across every adapter. Do not put
the pull loop in the adapter — it belongs to the caller, which is what lets
a TUI paint between lines.

Test a new `step` with `streaming.replay(sc, lines)`: it folds `step` over
recorded lines with no network and no `[stream]` grant, so an adapter's
parser is testable from a captured response. See `tests/test_streaming.lex`.

Convenience constructors: `prov.gpt4o()`, `prov.claude_sonnet()`,
`prov.gemini_flash()`, `prov.ollama("llama3")`.

---

## Effects

| declared | why |
|---|---|
| `[net]` | `http.send` (buffered) / `http.stream_lines` (streaming) in every provider |
| `[stream]` | pulling a streamed response — `src/streaming.lex` and its callers only |
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
lex test --allow-effects io,time,crypto,random,sql,fs_read,fs_write,net,concurrent,llm,proc,approval,stream tests/
```

Tests cover pure helpers (`sse`, `structured`) and the streaming parsers
(`test_streaming.lex`, which replays recorded provider responses through
each adapter's `step` — no network). Effects-heavy paths (provider HTTP,
agent loop) are covered by the example programs under `examples/` — run them
against a local Ollama instance or with a live API key.

---

## Known limitations

- `run_loop` / `run_loop_traced` are still buffered — they return
  `List[Step]` for the whole multi-turn loop. `run_steps_streamed` is the
  live variant: same loop, but `on_step` fires as each Step happens. Its
  callback row is fixed at `[io]`, so a callback can print or log but not
  time or persist; that work goes on the returned list.
- `google` and `vertex` have no streaming half.
- No built-in retry / back-off on transient HTTP errors; wrap
  `run_loop` in a `flow.retry` if needed.
