# lex-llm — run_steps_streamed emits as it goes
#
# The difference between run_steps_traced and run_steps_streamed is entirely
# a matter of *when* the caller learns things, and "when" is the one property
# a returned List[Step] cannot express. So these tests record the callback
# into a file as it fires and assert on the resulting trace: what was emitted,
# in what order, and that it matches what the call finally returns.
#
# The provider here is a stub with `stream: None`. That is deliberate — it is
# the branch where the loop has no streaming half to lean on, and it is the
# branch that must still emit through the callback rather than silently going
# back to buffered behaviour. The streaming branch's parsers are covered by
# test_streaming.lex; its transport needs a socket and is verified by running
# examples/streaming_tokens.lex against a real endpoint.

import "../src/agent" as ag

import "../src/provider" as prov

import "../src/message" as msg

import "../src/delta" as d

import "../src/tool" as t

import "lex-trail/log" as log

import "std.io" as io

import "std.str" as str

import "std.int" as int

import "std.list" as list

import "std.iter" as iter

fn trace_path() -> Str {
  "/tmp/lex-llm-streamed-trace.txt"
}

# The recorder: append what fired, so the trace reads in emission order.
# A callback cannot accumulate in a pure value, so the file is the accumulator.
fn record(s :: Str) -> [io] Unit {
  let prev := match io.read(trace_path()) {
    Ok(x) => x,
    Err(_) => "",
  }
  let __w := io.write(trace_path(), str.concat(prev, s))
  ()
}

fn reset_trace() -> [io] Unit {
  let __w := io.write(trace_path(), "")
  ()
}

fn read_trace() -> [io] Str {
  match io.read(trace_path()) {
    Ok(x) => x,
    Err(_) => "<unreadable>",
  }
}

fn on_step(st :: d.Step) -> [io] Unit {
  record(show_step(st))
}

fn show_step(st :: d.Step) -> Str {
  match st {
    StepDelta(dl) => show_delta(dl),
    StepToolExec(name, args) => str.join(["Exec(", name, ",", args, ")"], ""),
    StepToolResult(name, ok) => str.join(["Result(", name, ",", if ok {
      "ok"
    } else {
      "err"
    }, ")"], ""),
    StepDone(m) => str.join(["Done(", show_msg(m), ")"], ""),
  }
}

fn show_msg(m :: msg.Message) -> Str {
  match m {
    AssistantMsg(text, _calls) => text,
    UserMsg(text) => text,
    SystemMsg(text) => text,
    ToolMsg(_id, content) => content,
  }
}

fn show_delta(dl :: d.Delta) -> Str {
  match dl {
    TextChunk(s) => str.join(["Text(", s, ")"], ""),
    ToolCallBegin(id, name) => str.join(["Begin(", id, ",", name, ")"], ""),
    ToolArgChunk(id, c) => str.join(["Arg(", id, ",", c, ")"], ""),
    FinishDelta(r) => str.join(["Finish(", r, ")"], ""),
    UsageDelta(p, c, tt) => str.join(["Usage(", int.to_str(p), ",", int.to_str(c), ",", int.to_str(tt), ")"], ""),
  }
}

# A provider whose chat is a pure closure. The declared [net, llm] row is
# satisfied structurally by a body that uses neither, which is what makes a
# provider stub possible at all without a socket.
fn stub_provider(deltas :: List[d.Delta]) -> prov.Provider {
  { name: "stub", chat: fn (_m :: prov.ModelRef, _msgs :: List[msg.Message], _tools :: List[t.Tool]) -> [net, llm] Iter[d.Delta] {
    iter.from_list(deltas)
  }, stream: None }
}

fn stub_agent(deltas :: List[d.Delta]) -> ag.AgentLoop {
  { name: "stub", goal: "be terse", model: prov.make_model_ref("stub", "m"), provider: stub_provider(deltas), tools: [], options: { temperature: None, top_p: None, max_steps: Some(5), max_tokens: None }, permission_spec: None }
}

fn expect(label :: Str, got :: Str, want :: Str) -> Result[Unit, Str] {
  if got == want {
    Ok(())
  } else {
    Err(str.join([label, ": expected <", want, "> got <", got, ">"], ""))
  }
}

# Every Delta reaches the callback, in order, followed by the done Step —
# and the same Steps come back from the call, so a caller gets both views
# from one turn rather than paying for two.
fn test_emits_every_step() -> [net, llm, io, proc, sql, time, approval, stream, fs_write] Result[Unit, Str] {
  match log.open_memory() {
    Err(e) => Err(str.concat("log.open_memory failed: ", e)),
    Ok(lg) => {
      let __r := reset_trace()
      let steps := ag.run_steps_streamed(stub_agent([TextChunk("Hel"), TextChunk("lo"), FinishDelta("stop")]), [UserMsg("hi")], 5, lg, None, on_step)
      let emitted := read_trace()
      let returned := str.join(list.map(steps, show_step), "")
      match expect("emission order", emitted, "Text(Hel)Text(lo)Finish(stop)Done(Hello)") {
        Err(e) => Err(e),
        Ok(_) => expect("returned steps match emitted", returned, emitted),
      }
    },
  }
}

# A provider with no streaming half must still emit through the callback.
# The bug this guards against is a `None` branch that quietly returns the
# buffered Deltas without emitting them, which looks fine in the returned
# list and leaves the caller's UI blank.
fn test_none_provider_still_emits() -> [net, llm, io, proc, sql, time, approval, stream, fs_write] Result[Unit, Str] {
  match log.open_memory() {
    Err(e) => Err(str.concat("log.open_memory failed: ", e)),
    Ok(lg) => {
      let __r := reset_trace()
      let __s := ag.run_steps_streamed(stub_agent([TextChunk("x"), FinishDelta("stop")]), [UserMsg("hi")], 5, lg, None, on_step)
      let emitted := read_trace()
      if str.contains(emitted, "Text(x)") {
        Ok(())
      } else {
        Err(str.concat("callback saw no Deltas from a stream:None provider: ", emitted))
      }
    },
  }
}

# An exhausted budget is still an emission, not a silent empty turn.
fn test_budget_exhausted_emits() -> [net, llm, io, proc, sql, time, approval, stream, fs_write] Result[Unit, Str] {
  match log.open_memory() {
    Err(e) => Err(str.concat("log.open_memory failed: ", e)),
    Ok(lg) => {
      let __r := reset_trace()
      let __s := ag.run_steps_streamed(stub_agent([TextChunk("x"), FinishDelta("stop")]), [UserMsg("hi")], 0, lg, None, on_step)
      expect("budget exhausted", read_trace(), "Done([max_steps reached])")
    },
  }
}

# The streamed loop and the traced loop must agree on the transcript; only
# the timing differs. If they diverge, one of them has a bug in the part
# that is supposed to be shared.
fn test_agrees_with_traced() -> [net, llm, io, proc, sql, time, approval, stream, fs_write] Result[Unit, Str] {
  match log.open_memory() {
    Err(e) => Err(str.concat("log.open_memory failed: ", e)),
    Ok(lg) => {
      let deltas := [TextChunk("a"), TextChunk("b"), FinishDelta("stop")]
      let __r := reset_trace()
      let streamed := ag.run_steps_streamed(stub_agent(deltas), [UserMsg("hi")], 5, lg, None, on_step)
      let traced := ag.run_steps_traced(stub_agent(deltas), [UserMsg("hi")], 5, lg, None)
      expect("streamed == traced", str.join(list.map(streamed, show_step), ""), str.join(list.map(traced, show_step), ""))
    },
  }
}

fn run_all() -> [net, llm, io, proc, sql, time, approval, stream, fs_write] Int {
  let results := [("emits_every_step", test_emits_every_step()), ("none_provider_still_emits", test_none_provider_still_emits()), ("budget_exhausted_emits", test_budget_exhausted_emits()), ("agrees_with_traced", test_agrees_with_traced())]
  list.fold(results, 0, fn (failures :: Int, entry :: (Str, Result[Unit, Str])) -> [io] Int {
    match entry {
      (name, result) => match result {
        Ok(_) => {
          let __ok := io.print(str.concat("  ok   ", name))
          failures
        },
        Err(e) => {
          let __bad := io.print(str.join(["  FAIL ", name, ": ", e], ""))
          failures + 1
        },
      },
    }
  })
}

