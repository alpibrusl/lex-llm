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

import "std.list" as list

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

# OpenCode Go plan — https://opencode.ai/docs/zen
# Set OPENCODE_API_KEY to the key in ~/.credentials/opencode/key
fn opencode_go() -> [env] prov.Provider {
  oai.make_provider({ api_key: get_key("OPENCODE_API_KEY"), base_url: "https://opencode.ai/zen/go/v1/chat/completions" })
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

# ── MLX (Apple Silicon) ───────────────────────────────────────────────────────
# mlx_lm.server exposes an OpenAI-compatible POST /v1/chat/completions endpoint
# with tool-calling support, so the OpenAI adapter drives it unchanged. Runs on
# the host (needs Metal — cannot run inside a Linux container); reach it via
# host.docker.internal. Model name must match what mlx_lm.server was started with
# (e.g. mlx-community/Qwen2.5-Coder-7B-Instruct-4bit).
fn mlx_model() -> [env] Str {
  match env.get("MLX_MODEL") {
    None => "mlx-community/Qwen2.5-7B-Instruct-4bit",
    Some(m) => m,
  }
}

fn mlx_at(host :: Str) -> prov.Provider {
  oai.make_provider({ api_key: "", base_url: str.concat(host, "/v1/chat/completions") })
}

# ── LiteLLM proxy (OpenAI-compatible, routes to any backend) ─────────────────
# Run LiteLLM with: docker run -p 4000:4000 ghcr.io/berriai/litellm --config config.yaml
# Set LITELLM_BASE_URL to override the default localhost:4000 endpoint.
# Set LITELLM_API_KEY if your LiteLLM deployment requires a master key.
fn litellm() -> [env] prov.Provider {
  let base := match env.get("LITELLM_BASE_URL") {
    None => "http://localhost:4000",
    Some(u) => if str.is_empty(u) {
      "http://localhost:4000"
    } else {
      u
    },
  }
  let url := if str.contains(base, "/v1") {
    base
  } else {
    str.concat(base, "/v1/chat/completions")
  }
  let api_key := match env.get("LITELLM_API_KEY") {
    None => "",
    Some(k) => k,
  }
  oai.make_provider({ api_key: api_key, base_url: url })
}

fn litellm_at(base_url :: Str) -> prov.Provider {
  let url := if str.contains(base_url, "/v1") {
    base_url
  } else {
    str.concat(base_url, "/v1/chat/completions")
  }
  oai.make_provider({ api_key: "", base_url: url })
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

# ── Canonical provider selector ───────────────────────────────────────────────
# Build a Provider from a (name, url, key) triple. This is the SHARED selector so
# every agent runner / app (lex-soft consumers, lex-ev-fleet, …) wires providers
# the same way instead of copy-pasting the mapping. Agent runners that carry
# provider_name/provider_url/provider_key (e.g. lex-soft AgentConfig) pass them
# straight through.
#
#   name  : "mlx" | "ollama" | "vllm" | "openai" | "anthropic" | "google" |
#           "mistral" | "vertex" (any other value → vertex).
#   url   : base URL/host for local & OpenAI-compatible servers (mlx/ollama/vllm);
#           the Vertex location for "vertex"; ignored by cloud-key providers.
#   key   : API key for cloud providers; for "vertex" it is the packed
#           "<access_token>|||<project_id>"; ignored for local providers.
fn select_provider(name :: Str, url :: Str, key :: Str) -> prov.Provider {
  if name == "mlx" {
    mlx_at(url)
  } else {
    if name == "ollama" {
      ollama_at(url)
    } else {
      if name == "vllm" {
        vllm_at(url)
      } else {
        if name == "openai" {
          openai_with_key(key)
        } else {
          if name == "anthropic" {
            anthropic_with_key(key)
          } else {
            if name == "google" {
              google_with_key(key)
            } else {
              if name == "mistral" {
                mistral_with_key(key)
              } else {
                let parts := str.split(key, "|||")
                let token := match list.head(parts) {
                  Some(s) => s,
                  None => "",
                }
                let project := match str.strip_prefix(key, str.concat(token, "|||")) {
                  Some(s) => s,
                  None => "",
                }
                let location := if str.is_empty(url) {
                  "eu"
                } else {
                  url
                }
                vertex_with_config(token, project, location)
              }
            }
          }
        }
      }
    }
  }
}

