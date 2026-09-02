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
# Set OPENCODE_BASE_URL to route through a local proxy (e.g. bench/reasoning-proxy.py)
fn opencode_go() -> [env] prov.Provider {
  let url := match env.get("OPENCODE_BASE_URL") {
    None => "",
    Some(u) => if str.is_empty(u) {
      ""
    } else {
      str.concat(u, "/chat/completions")
    },
  }
  opencode_go_at(url, get_key("OPENCODE_API_KEY"))
}

# OpenCode Go plan with an explicit key and optional base-url override.
# Used by select_provider so the key can arrive via the agent request JSON
# (provider_key) instead of the environment. Empty url → the Go endpoint.
fn opencode_go_at(url :: Str, key :: Str) -> prov.Provider {
  let base := if str.is_empty(url) {
    "https://opencode.ai/zen/go/v1/chat/completions"
  } else {
    url
  }
  oai.make_provider({ api_key: key, base_url: base })
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

# Every other local provider here reads its endpoint from the environment
# — LITELLM_BASE_URL, VLLM_BASE_URL — and this one did not, so an Ollama
# anywhere but localhost:11434 was simply unreachable: `ollama_at` existed
# but nothing on the `ollama` provider tag called it.
#
# OLLAMA_BASE_URL is the name lex-loom's LiteLLM config already uses for
# the same host, so one variable points both at the same daemon.
fn ollama_default_url() -> Str
  examples {
    ollama_default_url() => "http://localhost:11434"
  }
{
  "http://localhost:11434"
}

# A trailing slash would make `ollama_at` build `.../api/chat` off
# `host//api/chat`. Cheap to tolerate, confusing to debug.
fn normalize_host(raw :: Str) -> Str
  examples {
    normalize_host("http://box:11434") => "http://box:11434",
    normalize_host("http://box:11434/") => "http://box:11434",
    normalize_host("  http://box:11434/  ") => "http://box:11434",
    normalize_host("") => "http://localhost:11434",
    normalize_host("   ") => "http://localhost:11434"
  }
{
  let t := str.trim(raw)
  if str.is_empty(t) {
    ollama_default_url()
  } else {
    match str.strip_suffix(t, "/") {
      None => t,
      Some(without) => without,
    }
  }
}

fn ollama_local() -> [env] prov.Provider {
  ollama_at(normalize_host(match env.get("OLLAMA_BASE_URL") {
    None => "",
    Some(u) => u,
  }))
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

# ── lex-moe (self-hosted, streamed-from-NVMe MoE inference) ──────────────────
# `moe serve --store DIR --budget-mb N` exposes the same OpenAI-shaped
# POST /v1/chat/completions this adapter already drives (default 127.0.0.1:8080,
# no key). Today it is plain-chat only: the server accepts and ignores unknown
# request fields, so a `tools` list sent here is silently dropped and replies
# never carry `tool_calls` — the oai.lex parser just sees an assistant text
# reply. Tool-calling and a real (non-placeholder) chat template are tracked
# upstream (alpibrusl/lex-moe#44, #45); once those land this adapter starts
# working for agent loops with no lex-llm-side change, since the wire shape
# is unchanged.
fn moe_model() -> [env] Str {
  match env.get("MOE_MODEL") {
    None => "default",
    Some(m) => m,
  }
}

fn moe_local() -> [env] prov.Provider {
  let base_url := match env.get("MOE_BASE_URL") {
    None => "http://127.0.0.1:8080/v1/chat/completions",
    Some(u) => u,
  }
  oai.make_provider({ api_key: "", base_url: base_url })
}

fn moe_at(host :: Str) -> prov.Provider {
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
#   name  : "opencode-go" | "mlx" | "ollama" | "vllm" | "moe" | "openai" |
#           "anthropic" | "google" | "mistral" | "vertex" (any other value →
#           vertex).
#   url   : base URL/host for local & OpenAI-compatible servers
#           (mlx/ollama/vllm/moe); the OpenCode Go endpoint override for
#           "opencode-go" (empty → default); the Vertex location for
#           "vertex"; ignored by cloud-key providers.
#   key   : API key for cloud providers; the OpenCode key for "opencode-go"; for
#           "vertex" it is the packed "<access_token>|||<project_id>"; ignored for
#           local providers.
fn select_provider(name :: Str, url :: Str, key :: Str) -> prov.Provider {
  if name == "opencode-go" {
    opencode_go_at(url, key)
  } else {
    if name == "mlx" {
      mlx_at(url)
    } else {
      if name == "ollama" {
        ollama_at(url)
      } else {
        if name == "vllm" {
          vllm_at(url)
        } else {
          if name == "moe" {
            moe_at(url)
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
  }
}

