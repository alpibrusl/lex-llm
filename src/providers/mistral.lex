# lex-llm — Mistral AI Chat Completions adapter
#
# Mistral's /v1/chat/completions endpoint is OpenAI-compatible:
# same SSE format, same tool-call schema, same auth header.
# We reuse the OpenAI chat implementation verbatim and only
# override the provider name and default base URL.

import "../provider" as prov
import "./openai"    as openai

import "std.str" as str

let default_base_url := "https://api.mistral.ai/v1/chat/completions"

type MistralConfig = {
  api_key  :: Str,
  base_url :: Str,
}

fn default_config(api_key :: Str) -> MistralConfig {
  { api_key: api_key, base_url: default_base_url }
}

fn make_provider(config :: MistralConfig) -> prov.Provider {
  let inner := openai.make_provider({ api_key: config.api_key, base_url: config.base_url })
  { name: "mistral", chat: inner.chat }
}
