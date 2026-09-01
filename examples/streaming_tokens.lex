# lex-llm example — printing tokens as the model produces them
#
# The buffered path (`provider.chat`) hands you the whole turn at once, so a
# UI built on it sits blank for however long the model takes and then prints
# everything. This example uses the optional streaming half instead: it pulls
# one line off the wire at a time and prints each text fragment as it lands.
#
# Run (Ollama must be running locally with the model pulled):
#   lex run --allow-effects net,llm,io,stream examples/streaming_tokens.lex main
#
# Swap `ollama_provider` for `anthropic_provider` / `openai_provider` and the
# rest is unchanged — the loop below has no provider-specific knowledge in it.

import "lex-llm/provider" as prov

import "lex-llm/message" as msg

import "lex-llm/delta" as d

import "lex-llm/streaming" as streaming

import "lex-llm/providers/ollama" as ollama_provider

import "std.io" as io

import "std.str" as str

import "std.iter" as iter

import "std.list" as list

fn main() -> [net, llm, io, stream] Unit {
  let p := ollama_provider.make_provider(ollama_provider.default_config())
  let model := prov.ollama("llama3")
  let messages := [SystemMsg("You are terse."), UserMsg("Name three primes and say why each is prime.")]
  say(p, model, messages)
}

# The whole point: `p.stream` is an Option, so a caller decides once whether
# it is streaming and both branches produce the same Deltas. An adapter that
# does not stream is not a special case to handle — it is the else branch.
fn say(p :: prov.Provider, model :: prov.ModelRef, messages :: List[msg.Message]) -> [net, llm, io, stream] Unit {
  match p.stream {
    None => {
      let _b := io.print("(this provider has no streaming half — buffering)\n")
      emit(iter.to_list(p.chat(model, messages, [])))
    },
    Some(sc) => match sc.open(model, messages, []) {
      Err(e) => io.print(str.join(["stream failed to open: ", e, "\n"], "")),
      Ok(s) => {
        let _done := paint(sc, s, streaming.start(sc))
        io.print("\n")
      },
    },
  }
}

# One pull, print what it produced, pull again. Everything a UI wants to do
# between tokens — repaint, check for a keypress, update a spinner — goes
# where `emit` is called.
fn paint(sc :: prov.StreamChat, s :: Stream[Str], cur :: streaming.Cursor) -> [io, stream] Unit {
  match streaming.pull(sc, s, cur) {
    (next, deltas) => {
      let _e := emit(deltas)
      if streaming.is_done(next) {
        ()
      } else {
        paint(sc, s, next)
      }
    },
  }
}

# Text goes out as it arrives; a tool call is announced so it is not mistaken
# for the model's prose.
#
# One fragment per line, because std.io has only `print`, which appends a
# newline — there is no newline-free write to build a flowing line of text
# from. That is a rendering limit, not a streaming one: the Deltas arrive
# incrementally either way, and a real TUI (lex-code's, say) paints them
# through its own renderer instead of std.io.
fn emit(deltas :: List[d.Delta]) -> [io] Unit {
  list.fold(deltas, (), fn (_acc :: Unit, dl :: d.Delta) -> [io] Unit {
    match dl {
      TextChunk(text) => io.print(text),
      ToolCallBegin(_id, name) => io.print(str.join(["\n[calling ", name, "]\n"], "")),
      ToolArgChunk(_id, _chunk) => (),
      FinishDelta(reason) => io.print(str.join(["\n[", reason, "]\n"], "")),
      UsageDelta(_p, _c, _t) => (),
    }
  })
}
