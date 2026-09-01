# lex-llm — Provider interface
#
# A Provider is a value carrying a concrete chat implementation and
# any auth/config state captured in a closure. The interface is minimal:
# give me a model, messages, and available tools; return a lazy Iter[Delta].
#
# Effect note: all adapters declare [net, llm]. [net] is required by
# http.stream_lines; [llm] is a semantic annotation that identifies
# LLM-inference calls in the effect graph so policies can gate them
# independently of plain HTTP.

import "./message" as msg

import "./delta" as d

import "./tool" as t

import "lex-schema/json_value" as jv

# Identifies a model at a specific provider.
type ModelRef = { provider :: Str, model :: Str }

# OpenAI
fn gpt4o() -> ModelRef {
  { provider: "openai", model: "gpt-4o" }
}

fn gpt4o_mini() -> ModelRef {
  { provider: "openai", model: "gpt-4o-mini" }
}

# Anthropic
fn claude_opus() -> ModelRef {
  { provider: "anthropic", model: "claude-opus-5" }
}

fn claude_sonnet() -> ModelRef {
  { provider: "anthropic", model: "claude-sonnet-5" }
}

fn claude_haiku() -> ModelRef {
  { provider: "anthropic", model: "claude-haiku-4-5-20251001" }
}

# Google
fn gemini_flash() -> ModelRef {
  { provider: "google", model: "gemini-2.0-flash" }
}

fn gemini_pro() -> ModelRef {
  { provider: "google", model: "gemini-2.5-pro" }
}

# Mistral
fn mistral_large() -> ModelRef {
  { provider: "mistral", model: "mistral-large-latest" }
}

fn mistral_small() -> ModelRef {
  { provider: "mistral", model: "mistral-small-latest" }
}

fn codestral() -> ModelRef {
  { provider: "mistral", model: "codestral-latest" }
}

fn mistral_nemo() -> ModelRef {
  { provider: "mistral", model: "open-mistral-nemo" }
}

# Ollama (local)
fn ollama(m :: Str) -> ModelRef {
  { provider: "ollama", model: m }
}

# lex-moe (local, OpenAI-compatible `moe serve`; see providers.moe_local/moe_at)
fn moe(m :: Str) -> ModelRef {
  { provider: "moe", model: m }
}

# vLLM (local or remote, OpenAI-compatible)
fn vllm(m :: Str) -> ModelRef {
  { provider: "vllm", model: m }
}

fn make_model_ref(provider_name :: Str, model_name :: Str) -> ModelRef {
  { provider: provider_name, model: model_name }
}

type StreamChat = { open :: (ModelRef, List[msg.Message], List[t.Tool]) -> [net, llm] Result[Stream[Str], Str], init :: jv.Json, step :: (jv.Json, Str) -> (jv.Json, List[d.Delta]) }

type Provider = { name :: Str, chat :: (ModelRef, List[msg.Message], List[t.Tool]) -> [net, llm] Iter[d.Delta], stream :: Option[StreamChat] }

# The provider interface, in two halves.
#
# `chat` is the required half: give it a model, messages and tools, get back
# every Delta of one completed turn. It buffers — the whole response is in
# hand before the first Delta is observable — and every adapter implements it.
#
# `stream` is the optional half (`None` is a valid, complete adapter). It
# exists because the buffered shape cannot be made incremental without
# changing `chat`'s type: `Iter` has no producer that yields as bytes arrive,
# and `chat`'s effect row `[net, llm]` is fixed by the record field, so an
# adapter cannot add the `[stream]` that pulling a live socket requires. So
# streaming is a second field rather than a change to the first, and callers
# that never look at it are unaffected.
#
# The three parts of a StreamChat split cleanly along the effect line:
#
#   open  [net, llm]  starts the request, hands back the raw response lines
#                     as they arrive. `Err` is a connect-time failure.
#   init  pure        the parser's starting state.
#   step  pure        one line in, (next state, Deltas) out.
#
# The pull loop itself is deliberately NOT here. Consuming a `Stream` carries
# `[stream]`, and what to do with each Delta — print it, count it, drop it —
# belongs to the caller, not the adapter. `streaming.lex` supplies a driver
# for callers that just want the whole turn; a TUI wanting live tokens calls
# `streaming.pull` in its own loop and paints between pulls.
#
# The parser state is `jv.Json` rather than a per-provider record because the
# record field's type is fixed across all adapters: Anthropic tracks the
# current content block, OpenAI tracks tool-call ids by index, Ollama needs no
# state at all. Json is the one shape all three can carry.
#
# Effect note: both halves declare [net, llm]. [net] is required by the
# transport; [llm] is a semantic annotation that identifies LLM-inference
# calls in the effect graph so policies can gate them independently of plain
# HTTP.
#
# Deliberately placed after the `type` declarations: a comment directly above
# a `type` is silently deleted by lex fmt (lex-lang#755).
fn has_streaming(p :: Provider) -> Bool {
  match p.stream {
    Some(_) => true,
    None => false,
  }
}

