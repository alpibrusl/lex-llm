# lex-llm — driving a provider's optional streaming half
#
# `provider.StreamChat` deliberately stops at the transport and the parser:
# it opens the response and turns one line into Deltas, and nothing more.
# This module supplies the loop that sits between them.
#
# Two entry points, because there are two kinds of caller:
#
#   pull     one line, one step. The caller keeps the cursor, decides what to
#            do with each Delta, and calls again. This is what a TUI wants —
#            it paints tokens between pulls, which is the entire point of
#            streaming.
#
#   collect  drain the turn into a List[Delta]. Same result as calling the
#            buffered `chat`, so it is not a way to make a turn faster; it is
#            for callers that want the streaming transport (a live socket
#            with a long idle tolerance) without a per-token loop, and for
#            testing an adapter's `step` against a real response.
#
# Consuming a Stream carries [stream], so every function here that touches
# one declares it. That effect stops at this module's callers: an adapter's
# `open` and `step` stay [net, llm] and pure respectively.

import "./provider" as prov

import "./delta" as d

import "./message" as msg

import "./tool" as t

import "lex-schema/json_value" as jv

import "std.stream" as stream

import "std.iter" as iter

import "std.list" as list

# Where a partly-consumed stream has got to: the parser state that `step`
# threads, plus whether the producer has signalled end-of-stream.
#
# `done` is not redundant with an empty Delta list — a heartbeat line, an SSE
# comment, or a chunk carrying only a role marker all parse to no Deltas
# while the turn is very much still running. Only `done` means stop.
type Cursor = { state :: jv.Json, done :: Bool }

fn start(sc :: prov.StreamChat) -> Cursor {
  { state: sc.init, done: false }
}

fn is_done(cur :: Cursor) -> Bool
  examples {
    is_done({ state: JNull, done: false }) => false,
    is_done({ state: JNull, done: true }) => true
  }
{
  cur.done
}

# One pull: take the next line off the wire and feed it to the parser.
#
# Returns the advanced cursor and whatever Deltas that line produced — often
# none. A caller loops until `is_done`, painting between calls.
#
# Pulling past end-of-stream is safe and idempotent: it returns the same
# done cursor and no Deltas, so a caller that overshoots by a pull does not
# get a spurious Delta or a hang.
fn pull(sc :: prov.StreamChat, s :: Stream[Str], cur :: Cursor) -> [stream] (Cursor, List[d.Delta]) {
  if cur.done {
    (cur, [])
  } else {
    match stream.next(s) {
      None => ({ state: cur.state, done: true }, []),
      Some(line) => match sc.step(cur.state, line) {
        (next_state, deltas) => ({ state: next_state, done: false }, deltas),
      },
    }
  }
}

# Drain an already-opened stream to the end.
#
# Bounded by `budget` rather than looping until end-of-stream, because a
# provider that stops sending without closing the socket would otherwise
# park this call for the transport's full idle timeout with no way for a
# caller to intervene. 200_000 lines is far past any real turn; see
# `line_budget`.
fn drain(sc :: prov.StreamChat, s :: Stream[Str], cur :: Cursor, acc :: List[d.Delta], budget :: Int) -> [stream] List[d.Delta] {
  if budget == 0 {
    acc
  } else {
    match pull(sc, s, cur) {
      (next_cur, deltas) => {
        let so_far := list.concat(acc, deltas)
        if next_cur.done {
          so_far
        } else {
          drain(sc, s, next_cur, so_far, budget - 1)
        }
      },
    }
  }
}

fn line_budget() -> Int
  examples {
    line_budget() => 200000
  }
{
  200000
}

# Open a turn on the streaming half and drain it.
#
# A connect-time failure comes back as the same provider_error Deltas the
# buffered path emits, so a caller reading Deltas cannot tell the two paths
# apart and a transport failure never reads as the model's own empty answer
# (the bug #45 fixed on the buffered path).
fn collect(sc :: prov.StreamChat, model :: prov.ModelRef, messages :: List[msg.Message], tools :: List[t.Tool]) -> [net, llm, stream] List[d.Delta] {
  match sc.open(model, messages, tools) {
    Err(e) => d.provider_error(e),
    Ok(s) => drain(sc, s, start(sc), [], line_budget()),
  }
}

# Fold `step` over lines already in hand, without a Stream.
#
# This is the pull loop with the transport removed, and it exists so an
# adapter's parser can be tested against a recorded response — the SSE or
# NDJSON a real provider actually sent — with no network and no [stream]
# grant. `drain` and `replay` must agree on any given input, since both are
# the same fold over the same `step`; a test that passes here is a test of
# the parser, not of the transport.
fn replay(sc :: prov.StreamChat, lines :: List[Str]) -> List[d.Delta] {
  let folded := list.fold(lines, (sc.init, []), fn (acc :: (jv.Json, List[d.Delta]), line :: Str) -> (jv.Json, List[d.Delta]) {
    match acc {
      (st, out) => match sc.step(st, line) {
        (next_st, deltas) => (next_st, list.concat(out, deltas)),
      },
    }
  })
  match folded {
    (_, out) => out,
  }
}

# Stream the turn when the provider offers it, fall back to the buffered
# `chat` when it does not. The Deltas are the same either way, so this is the
# safe default for a caller that wants streaming where available and does not
# want to branch on `provider.has_streaming`.
fn collect_via(p :: prov.Provider, model :: prov.ModelRef, messages :: List[msg.Message], tools :: List[t.Tool]) -> [net, llm, stream] List[d.Delta] {
  match p.stream {
    None => iter.to_list(p.chat(model, messages, tools)),
    Some(sc) => collect(sc, model, messages, tools),
  }
}

