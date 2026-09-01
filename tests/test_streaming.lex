# lex-llm — streaming adapter tests
#
# Every test here replays a recorded response — the SSE or NDJSON lines a
# real provider actually sends — through the adapter's `step` function via
# streaming.replay. No network, no Stream handle, no [stream] grant: the
# transport is not under test, the parsers are.
#
# What is worth asserting is the part the buffered path never had to do:
# carrying state across lines. A tool call arrives split over several
# chunks, and getting its id onto the fragments that follow is the whole
# difficulty of the streaming shape.

import "../src/streaming" as streaming

import "../src/provider" as prov

import "../src/delta" as d

import "../src/sse" as sse

import "../src/providers/openai" as oai

import "../src/providers/anthropic" as anth

import "../src/providers/ollama" as olla

import "../src/providers/mistral" as mist

import "../src/providers/google" as goog

import "std.str" as str

import "std.int" as int

import "std.list" as list

import "std.io" as io

# ---- Delta rendering, so a failure says what it got ---------------
fn show(dl :: d.Delta) -> Str {
  match dl {
    TextChunk(s) => str.join(["Text(", s, ")"], ""),
    ToolCallBegin(id, name) => str.join(["Begin(", id, ",", name, ")"], ""),
    ToolArgChunk(id, chunk) => str.join(["Arg(", id, ",", chunk, ")"], ""),
    FinishDelta(r) => str.join(["Finish(", r, ")"], ""),
    UsageDelta(p, c, t) => str.join(["Usage(", int.to_str(p), ",", int.to_str(c), ",", int.to_str(t), ")"], ""),
  }
}

fn render(ds :: List[d.Delta]) -> Str {
  str.join(list.map(ds, show), " ")
}

fn expect(label :: Str, got :: Str, want :: Str) -> Result[Unit, Str] {
  if got == want {
    Ok(())
  } else {
    Err(str.join([label, ": expected <", want, "> got <", got, ">"], ""))
  }
}

fn stream_of(p :: prov.Provider) -> Result[prov.StreamChat, Str] {
  match p.stream {
    Some(sc) => Ok(sc),
    None => Err(str.concat("provider declares no streaming half: ", p.name)),
  }
}

fn replay_of(p :: prov.Provider, lines :: List[Str]) -> Result[Str, Str] {
  match stream_of(p) {
    Err(e) => Err(e),
    Ok(sc) => Ok(render(streaming.replay(sc, lines))),
  }
}

fn openai_provider() -> prov.Provider {
  oai.make_provider({ api_key: "k", base_url: "http://127.0.0.1:1/v1/chat/completions" })
}

fn anthropic_provider() -> prov.Provider {
  anth.make_provider(anth.default_config("k"))
}

fn ollama_provider() -> prov.Provider {
  olla.make_provider(olla.default_config())
}

# ---- Which adapters advertise the optional half -------------------
fn test_streaming_declared() -> Result[Unit, Str] {
  if prov.has_streaming(openai_provider()) {
    if prov.has_streaming(anthropic_provider()) {
      if prov.has_streaming(ollama_provider()) {
        Ok(())
      } else {
        Err("ollama should declare a streaming half")
      }
    } else {
      Err("anthropic should declare a streaming half")
    }
  } else {
    Err("openai should declare a streaming half")
  }
}

# google's JSON-array framing has no line-at-a-time parse; None is the
# honest answer and callers fall back to the buffered chat.
fn test_google_declares_none() -> Result[Unit, Str] {
  if prov.has_streaming(goog.make_provider(goog.default_config("k"))) {
    Err("google should declare no streaming half")
  } else {
    Ok(())
  }
}

# mistral delegates to the openai adapter, so it inherits the half rather
# than silently dropping it — the bug this test exists to catch is a
# `{ name: ..., chat: inner.chat }` that forgets `stream: inner.stream`.
fn test_mistral_inherits_streaming() -> Result[Unit, Str] {
  if prov.has_streaming(mist_provider()) {
    Ok(())
  } else {
    Err("mistral should inherit openai's streaming half")
  }
}

fn mist_provider() -> prov.Provider {
  mist.make_provider(mist.default_config("k"))
}

# ---- OpenAI -------------------------------------------------------
fn test_openai_text_chunks() -> Result[Unit, Str] {
  match replay_of(openai_provider(), ["data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}", "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"},\"finish_reason\":null}]}", "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":null}]}", "", "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}", "data: [DONE]"]) {
    Err(e) => Err(e),
    Ok(got) => expect("openai text", got, "Text(Hel) Text(lo) Finish(stop)"),
  }
}

# The point of the parser state: chunks 2 and 3 carry only `index`, so the
# id has to come from the chunk that named it.
fn test_openai_split_tool_call() -> Result[Unit, Str] {
  match replay_of(openai_provider(), ["data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_a\",\"function\":{\"name\":\"read\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}", "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"pa\"}}]},\"finish_reason\":null}]}", "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"th\\\":1}\"}}]},\"finish_reason\":null}]}", "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}"]) {
    Err(e) => Err(e),
    Ok(got) => expect("openai tool call", got, "Begin(call_a,read) Arg(call_a,{\"pa) Arg(call_a,th\":1}) Finish(tool_calls)"),
  }
}

# Two calls in one turn must not be merged onto one id.
fn test_openai_two_tool_calls() -> Result[Unit, Str] {
  match replay_of(openai_provider(), ["data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"a\",\"function\":{\"name\":\"read\",\"arguments\":\"{}\"}}]},\"finish_reason\":null}]}", "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"b\",\"function\":{\"name\":\"grep\",\"arguments\":\"{}\"}}]},\"finish_reason\":null}]}", "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"!\"}}]},\"finish_reason\":null}]}"]) {
    Err(e) => Err(e),
    Ok(got) => expect("openai two calls", got, "Begin(a,read) Arg(a,{}) Begin(b,grep) Arg(b,{}) Arg(a,!)"),
  }
}

# stream_options: include_usage puts the cost on a final choices-empty chunk.
fn test_openai_usage_chunk() -> Result[Unit, Str] {
  match replay_of(openai_provider(), ["data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":4,\"total_tokens\":14}}"]) {
    Err(e) => Err(e),
    Ok(got) => expect("openai usage", got, "Usage(10,4,14)"),
  }
}

# Blank lines, SSE comments and the terminator are framing, not content.
fn test_openai_ignores_framing() -> Result[Unit, Str] {
  match replay_of(openai_provider(), ["", ": ping", "data: [DONE]", "data: not json"]) {
    Err(e) => Err(e),
    Ok(got) => expect("openai framing", got, ""),
  }
}

# ---- Anthropic ----------------------------------------------------
fn test_anthropic_text_and_finish() -> Result[Unit, Str] {
  match replay_of(anthropic_provider(), ["event: content_block_start", "data: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"text\"}}", "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}", "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}"]) {
    Err(e) => Err(e),
    Ok(got) => expect("anthropic text", got, "Text(Hi) Finish(stop)"),
  }
}

# input_json_delta carries no id of its own — it is routed by the block that
# started, which is exactly the state the streaming shape has to carry.
fn test_anthropic_tool_use_routing() -> Result[Unit, Str] {
  match replay_of(anthropic_provider(), ["data: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"write\"}}", "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"p\"}}", "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\":1}\"}}", "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"}}"]) {
    Err(e) => Err(e),
    Ok(got) => expect("anthropic tool_use", got, "Begin(toolu_1,write) Arg(toolu_1,{\"p) Arg(toolu_1,\":1}) Finish(tool_calls)"),
  }
}

# CRLF is what the SSE spec actually says; stream_lines splits on LF and
# leaves the CR, so a parser that does not strip it drops every chunk.
fn test_anthropic_crlf() -> Result[Unit, Str] {
  match replay_of(anthropic_provider(), ["data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\r"]) {
    Err(e) => Err(e),
    Ok(got) => expect("anthropic crlf", got, "Text(ok)"),
  }
}

# ---- Ollama -------------------------------------------------------
fn test_ollama_text_flushes() -> Result[Unit, Str] {
  match replay_of(ollama_provider(), ["{\"message\":{\"role\":\"assistant\",\"content\":\"Hello \"},\"done\":false}", "{\"message\":{\"role\":\"assistant\",\"content\":\"there\"},\"done\":false}", "{\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true}"]) {
    Err(e) => Err(e),
    Ok(got) => expect("ollama text", got, "Text(Hello ) Text(there) Finish(stop)"),
  }
}

# A qwen3-style XML call arrives a token at a time. Nothing may be printed
# while it could still turn out to be markup, and the call has to be
# recovered at done — the buffered path gets the whole string at once and
# never faces this.
fn test_ollama_xml_tool_call_held_back() -> Result[Unit, Str] {
  match replay_of(ollama_provider(), ["{\"message\":{\"role\":\"assistant\",\"content\":\"<tool\"},\"done\":false}", "{\"message\":{\"role\":\"assistant\",\"content\":\"_call>\\n<function=read>\\n<parameter=path>\\nx.lex\\n</parameter>\\n</function>\\n</tool_call>\"},\"done\":false}", "{\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true}"]) {
    Err(e) => Err(e),
    Ok(got) => expect("ollama xml", got, "Begin(call_read_0,read) Arg(call_read_0,{\"path\": \"x.lex\"}) Finish(stop)"),
  }
}

# The buffer must not swallow ordinary prose that merely starts with "<".
fn test_ollama_angle_bracket_prose() -> Result[Unit, Str] {
  match replay_of(ollama_provider(), ["{\"message\":{\"role\":\"assistant\",\"content\":\"<b>bold\"},\"done\":false}", "{\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true}"]) {
    Err(e) => Err(e),
    Ok(got) => expect("ollama prose", got, "Text(<b>bold) Finish(stop)"),
  }
}

# Native tool_calls are structured and arrive whole — no buffering.
fn test_ollama_native_tool_call() -> Result[Unit, Str] {
  match replay_of(ollama_provider(), ["{\"message\":{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"function\":{\"name\":\"read\",\"arguments\":{\"path\":\"a\"}}}]},\"done\":true}"]) {
    Err(e) => Err(e),
    Ok(got) => expect("ollama native", got, "Begin(call_read_0,read) Arg(call_read_0,{\"path\":\"a\"}) Finish(tool_calls)"),
  }
}

# ---- sse.strip_cr -------------------------------------------------
fn test_strip_cr() -> Result[Unit, Str] {
  if sse.strip_cr("data: {}\r") == "data: {}" {
    if sse.strip_cr("data: {}") == "data: {}" {
      Ok(())
    } else {
      Err("strip_cr changed a line with no CR")
    }
  } else {
    Err("strip_cr did not remove a trailing CR")
  }
}

# ---- Cursor -------------------------------------------------------
fn test_cursor_starts_open() -> Result[Unit, Str] {
  match stream_of(openai_provider()) {
    Err(e) => Err(e),
    Ok(sc) => if streaming.is_done(streaming.start(sc)) {
      Err("a fresh cursor should not be done")
    } else {
      Ok(())
    },
  }
}

fn run_all() -> [io] Int {
  let results := [("streaming_declared", test_streaming_declared()), ("google_declares_none", test_google_declares_none()), ("mistral_inherits_streaming", test_mistral_inherits_streaming()), ("openai_text_chunks", test_openai_text_chunks()), ("openai_split_tool_call", test_openai_split_tool_call()), ("openai_two_tool_calls", test_openai_two_tool_calls()), ("openai_usage_chunk", test_openai_usage_chunk()), ("openai_ignores_framing", test_openai_ignores_framing()), ("anthropic_text_and_finish", test_anthropic_text_and_finish()), ("anthropic_tool_use_routing", test_anthropic_tool_use_routing()), ("anthropic_crlf", test_anthropic_crlf()), ("ollama_text_flushes", test_ollama_text_flushes()), ("ollama_xml_tool_call_held_back", test_ollama_xml_tool_call_held_back()), ("ollama_angle_bracket_prose", test_ollama_angle_bracket_prose()), ("ollama_native_tool_call", test_ollama_native_tool_call()), ("strip_cr", test_strip_cr()), ("cursor_starts_open", test_cursor_starts_open())]
  list.fold(results, 0, fn (failures :: Int, entry :: (Str, Result[Unit, Str])) -> [io] Int {
    match entry {
      (name, result) => match result {
        Ok(_) => {
          let _ok := io.print(str.concat("  ok   ", name))
          failures
        },
        Err(e) => {
          let _bad := io.print(str.join(["  FAIL ", name, ": ", e], ""))
          failures + 1
        },
      },
    }
  })
}

