# lex-llm — OTel-instrumented agent entry point
#
# Wraps run_loop in an OTLP span and exports it via lex-log. Kept
# separate from agent.lex so that test files that import agent.lex
# (or its transitive deps) are not pulled into lex-log's effect scope.

import "./agent" as agent

import "./message" as msg

import "./delta" as d

import "lex-log/context" as log_ctx

import "lex-log/span" as lspan

import "lex-log/exporter" as exp

import "std.list" as list

import "std.str" as str

import "std.iter" as iter

import "std.int" as int

# Instrumented run_loop. Wraps the full agent run in an OTLP span and
# exports it via `cfg`. Materialises all steps before returning so the
# span end_ms is recorded after the last tool call, not at first yield.
# Pass `parent_ctx = Some(tc)` to nest inside an existing trace.
fn run_loop_otel(ag :: agent.AgentDef, conversation :: List[msg.Message], cfg :: exp.Config, parent_ctx :: Option[log_ctx.TraceCtx]) -> [net, llm, io, proc, time, random] Iter[d.Step] {
  let trace_ctx := match parent_ctx {
    Some(p) => log_ctx.child(p),
    None => log_ctx.new_root(),
  }
  let sp := lspan.start(trace_ctx, str.concat("llm.run ", ag.name), cfg.service)
  let steps := iter.to_list(agent.run_loop(ag, conversation))
  let sp2 := lspan.set_attr(lspan.finish(sp), "llm.model", ag.model.model)
  let sp3 := lspan.set_attr(sp2, "llm.steps", int.to_str(list.len(steps)))
  let __lex_discard_1 := exp.export_spans(cfg, [sp3])
  iter.from_list(steps)
}

