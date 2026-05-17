# lex-llm — Provider interface
#
# A Provider is a value carrying a concrete chat implementation and
# any auth/config state captured in a closure. The interface is minimal:
# give me a model, messages, and available tools; return a lazy Iter[Delta].
#
# Effect note: all adapters use [net] (HTTP). The narrower [llm]
# named sub-capability is tracked in alpibrusl/lex-lang#483; once that
# lands signatures will narrow from [net] to [llm].

import "./message" as msg
import "./delta"   as d
import "./tool"    as t

# Identifies a model at a specific provider.
type ModelRef = {
  provider :: Str,   # "openai" | "anthropic" | "google" | "ollama"
  model    :: Str,   # "gpt-4o" | "claude-opus-4-7" | ...
}

fn gpt4o()          -> ModelRef { { provider: "openai",    model: "gpt-4o" } }
fn gpt4o_mini()     -> ModelRef { { provider: "openai",    model: "gpt-4o-mini" } }
fn claude_opus()    -> ModelRef { { provider: "anthropic", model: "claude-opus-4-7" } }
fn claude_sonnet()  -> ModelRef { { provider: "anthropic", model: "claude-sonnet-4-6" } }
fn gemini_flash()   -> ModelRef { { provider: "google",    model: "gemini-2.0-flash" } }
fn gemini_pro()     -> ModelRef { { provider: "google",    model: "gemini-2.5-pro" } }
fn ollama(m :: Str) -> ModelRef { { provider: "ollama",    model: m } }

# The provider interface.
# chat returns a lazy Iter[Delta]; the caller collects it or re-emits for streaming.
type Provider = {
  name :: Str,
  chat :: (ModelRef, List[msg.Message], List[t.Tool]) -> [net] Iter[d.Delta],
}
