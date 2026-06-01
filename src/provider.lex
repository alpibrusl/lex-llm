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
  { provider: "anthropic", model: "claude-opus-4-7" }
}

fn claude_sonnet() -> ModelRef {
  { provider: "anthropic", model: "claude-sonnet-4-6" }
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

# vLLM (local or remote, OpenAI-compatible)
fn vllm(m :: Str) -> ModelRef {
  { provider: "vllm", model: m }
}

fn make_model_ref(provider_name :: Str, model_name :: Str) -> ModelRef {
  { provider: provider_name, model: model_name }
}

# The provider interface.
# chat returns a lazy Iter[Delta]; the caller collects it or re-emits for streaming.
type Provider = { name :: Str, chat :: (ModelRef, List[msg.Message], List[t.Tool]) -> [net, llm] Iter[d.Delta] }

