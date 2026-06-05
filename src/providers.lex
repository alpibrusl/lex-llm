# lex-llm — Convenience provider factory functions
#
# Reads API keys from environment variables so callers only need:
#   import "lex-llm/providers" as providers
#   let p := providers.anthropic()
#
# Each function reads the canonical env var for that provider.
# If the key is absent the provider is still constructed; the first
# chat call will fail with an auth error rather than a startup panic.

import "./provider" as prov

import "std.str" as str

import "./providers/anthropic" as anth

import "./providers/openai" as oai

import "./providers/google" as goog

import "./providers/ollama" as olla

import "./providers/mistral" as mist

import "./providers/vertex" as vtx

import "std.env" as env

fn get_key(var_name :: Str) -> [env] Str {
  match env.get(var_name) {
    None => "",
    Some(k) => k,
  }
}

fn anthropic() -> [env] prov.Provider {
  anth.make_provider(anth.default_config(get_key("ANTHROPIC_API_KEY")))
}

fn openai() -> [env] prov.Provider {
  oai.make_provider(oai.default_config(get_key("OPENAI_API_KEY")))
}

fn google() -> [env] prov.Provider {
  goog.make_provider(goog.default_config(get_key("GOOGLE_API_KEY")))
}

fn mistral() -> [env] prov.Provider {
  mist.make_provider(mist.default_config(get_key("MISTRAL_API_KEY")))
}

fn mistral_with_key(api_key :: Str) -> prov.Provider {
  mist.make_provider(mist.default_config(api_key))
}

fn anthropic_with_key(api_key :: Str) -> prov.Provider {
  anth.make_provider(anth.default_config(api_key))
}

fn openai_with_key(api_key :: Str) -> prov.Provider {
  oai.make_provider(oai.default_config(api_key))
}

fn google_with_key(api_key :: Str) -> prov.Provider {
  goog.make_provider(goog.default_config(api_key))
}

fn ollama_local() -> prov.Provider {
  olla.make_provider(olla.default_config())
}

fn ollama_model() -> [env] Str {
  match env.get("OLLAMA_MODEL") {
    None => "gemma4:latest",
    Some(m) => m,
  }
}

fn ollama_at(host :: Str) -> prov.Provider {
  olla.make_provider({ base_url: str.concat(host, "/api/chat") })
}

# vLLM — OpenAI-compatible, no key required by default.
# Model name must match the model loaded in the vLLM server.
# Override via VLLM_BASE_URL env var for remote deployments.
fn vllm_model() -> [env] Str {
  match env.get("VLLM_MODEL") {
    None => "mistralai/Mistral-7B-Instruct-v0.3",
    Some(m) => m,
  }
}

fn vllm_local() -> [env] prov.Provider {
  let base_url := match env.get("VLLM_BASE_URL") {
    None => "http://localhost:8000/v1/chat/completions",
    Some(u) => u,
  }
  oai.make_provider({ api_key: "", base_url: base_url })
}

fn vllm_at(host :: Str) -> prov.Provider {
  oai.make_provider({ api_key: "", base_url: str.concat(host, "/v1/chat/completions") })
}

# ── Vertex AI (Gemini via Google Cloud multi-region endpoint) ─────────────────
# Reads VERTEX_ACCESS_TOKEN, VERTEX_PROJECT, VERTEX_LOCATION from environment.
# VERTEX_ACCESS_TOKEN = output of `gcloud auth print-access-token`.
# Default location: eu (aiplatform.eu.rep.googleapis.com).
fn vertex() -> [env] prov.Provider {
  let token := get_key("VERTEX_ACCESS_TOKEN")
  let project := get_key("VERTEX_PROJECT")
  let location := match env.get("VERTEX_LOCATION") {
    None => "eu",
    Some(l) => if str.is_empty(l) {
      "eu"
    } else {
      l
    },
  }
  vtx.make_provider(vtx.config_at(token, project, location))
}

fn vertex_with_config(access_token :: Str, project_id :: Str, location :: Str) -> prov.Provider {
  vtx.make_provider(vtx.config_at(access_token, project_id, location))
}

