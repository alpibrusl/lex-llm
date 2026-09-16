# lex-llm — Mistral AI Chat Completions adapter
#
# Mistral's /v1/chat/completions endpoint is OpenAI-compatible:
# same SSE format, same tool-call schema, same auth header.
# We reuse the OpenAI chat implementation verbatim and only
# override the provider name and default base URL.

import "../provider" as prov

import "./openai" as openai

import "std.str" as str

fn default_base_url() -> Str {
  "https://api.mistral.ai/v1/chat/completions"
}

type MistralConfig = { api_key :: Str, base_url :: Str, timeout_ms :: Option[Int] }

fn default_config(api_key :: Str) -> MistralConfig {
  { api_key: api_key, base_url: default_base_url(), timeout_ms: None }
}

# The same, against an endpoint other than the default — a proxy, a gateway,
# or a host on the LAN. Mirrors `vertex.config_at` and `openai.config_at`.
#
# Every provider config here needs one: without a constructor taking the
# endpoint, callers hand-write the record, and then adding a field to the
# config breaks every caller at once. That is not hypothetical — it is how
# lex-conquest, lex-robot and lex-werewolf all broke (see openai.config_at).
fn config_at(api_key :: Str, base_url :: Str) -> MistralConfig {
  { api_key: api_key, base_url: base_url, timeout_ms: None }
}

# The same config with an explicit client timeout, for a caller that knows its
# model is slower than the default (a local 27B answering a build prompt) or
# faster.
fn with_timeout(c :: MistralConfig, ms :: Int) -> MistralConfig {
  { api_key: c.api_key, base_url: c.base_url, timeout_ms: Some(ms) }
}

fn make_provider(config :: MistralConfig) -> prov.Provider {
  let inner := openai.make_provider({ api_key: config.api_key, base_url: config.base_url, extra_header: None, timeout_ms: config.timeout_ms })
  { name: "mistral", chat: inner.chat, stream: inner.stream }
}

