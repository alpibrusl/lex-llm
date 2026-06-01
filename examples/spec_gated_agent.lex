# lex-llm example — Spec-gated agent tool permissions
#
# Demonstrates the three-layer permission stack built into lex-llm:
#
#   Layer 1: eval     — evaluate a typed spec against concrete bindings → Verdict
#   Layer 2: check    — property-check the spec over 100 random inputs
#   Layer 3: SMT-LIB  — export the spec as a Z3 script for formal verification
#
# Zero LLM API calls. The spec evaluator, property checker, and SMT
# exporter are all pure functions — no effects beyond io.print.
#
# Run:
#   lex run --allow-effects io examples/spec_gated_agent.lex main

import "std.io"   as io
import "std.str"  as str
import "std.list" as list
import "std.int"  as int

import "lex-schema/schema"     as s
import "lex-schema/json_value" as jv
import "lex-schema/error"      as e

import "lex-spec/spec"  as sp
import "lex-spec/eval"  as ev
import "lex-spec/check" as check
import "lex-spec/smt"   as smt

import "../src/tool" as t

# ── Spec definitions ─────────────────────────────────────────────────────────

# order_submit_policy: submit_order requires qty ≤ 1000 AND approved == true.
# For all other tools the implication antecedent is false → vacuously true.
fn order_submit_policy() -> sp.Spec {
  {
    name: "order_submit_policy",
    quantifiers: [sp.QStr("tool"), sp.QInt("qty"), sp.QBool("approved")],
    predicate: sp.EImplies(
      sp.EBinop({ op: "==", lhs: sp.EVar("tool"), rhs: sp.EConst(sp.VStr("submit_order")) }),
      sp.EAnd(
        sp.EBinop({ op: "<=", lhs: sp.EVar("qty"), rhs: sp.EConst(sp.VInt(1000)) }),
        sp.EBinop({ op: "==", lhs: sp.EVar("approved"), rhs: sp.EConst(sp.VBool(true)) })
      )
    ),
  }
}

# ── Tool schema stubs ─────────────────────────────────────────────────────────

fn empty_schema(title :: Str) -> s.ModelSchema {
  { title: title, description: "", fields: [] }
}

fn stub_exec(args :: jv.Json) -> [net, io, proc] Result[jv.Json, e.Errors] {
  Ok(JNull)
}

# ── Tool definitions ──────────────────────────────────────────────────────────

# Tool 1 — get_quote: no precondition, always available.
fn tool_get_quote() -> t.Tool {
  t.define(
    "get_quote",
    "Get the current market quote for a symbol.",
    empty_schema("GetQuoteArgs"),
    stub_exec)
}

# Tool 2 — read_positions: no precondition, always available.
fn tool_read_positions() -> t.Tool {
  t.define(
    "read_positions",
    "Read the current portfolio positions.",
    empty_schema("ReadPositionsArgs"),
    stub_exec)
}

# Tool 3 — submit_order: gated by order_submit_policy.
fn tool_submit_order() -> t.Tool {
  t.define_gated(
    "submit_order",
    "Submit a trading order (qty ≤ 1000, requires approval).",
    empty_schema("SubmitOrderArgs"),
    stub_exec,
    order_submit_policy())
}

# ── Display helpers ───────────────────────────────────────────────────────────

fn hr() -> [io] Unit {
  io.print("  ──────────────────────────────────────────────────────────────")
}

fn section(title :: Str) -> [io] Unit {
  let __1 := io.print("")
  let __2 := hr()
  let __3 := io.print("  " + title)
  hr()
}

fn show_bindings(tool_name :: Str, qty :: Int, approved :: Bool) -> [io] Unit {
  let app_str := if approved { "true" } else { "false" }
  io.print("    bindings: tool=" + tool_name + "  qty=" + int.to_str(qty) + "  approved=" + app_str)
}

fn show_verdict(v :: sp.Verdict) -> [io] Unit {
  match v {
    sp.Allow =>
      io.print("    verdict:  Allow ✓"),
    sp.Deny(reason) =>
      io.print("    verdict:  Deny ✗  — " + reason),
    sp.Inconclusive(reason) =>
      io.print("    verdict:  Inconclusive — " + reason),
  }
}

fn eval_and_show(label :: Str, bindings :: List[(Str, sp.SpecValue)]) -> [io] Unit {
  let verdict := ev.eval(order_submit_policy(), bindings)
  let __1 := io.print("")
  let __2 := io.print("  " + label)
  let __3 := show_verdict(verdict)
  ()
}

# ── Section 1: The spec ───────────────────────────────────────────────────────

fn show_spec_definition() -> [io] Unit {
  let __s := section("1 — The spec: order_submit_policy")
  let __1 := io.print("  IF tool == \"submit_order\"")
  let __2 := io.print("  THEN qty <= 1000 AND approved == true")
  io.print("  (vacuously true for all other tool names)")
}

# ── Section 2: Four evaluations ───────────────────────────────────────────────

fn show_evaluations() -> [io] Unit {
  let __s := section("2 — Spec evaluation: four tool-call scenarios")

  # Case A: submit_order, qty=100, approved=true → Allow
  let b1 := [("tool", sp.VStr("submit_order")), ("qty", sp.VInt(100)), ("approved", sp.VBool(true))]
  let v1 := ev.eval(order_submit_policy(), b1)
  let __1 := io.print("")
  let __2 := io.print("  A  tool=submit_order  qty=100   approved=true")
  let __3 := show_verdict(v1)

  # Case B: submit_order, qty=5000, approved=true → Deny (qty exceeds 1000)
  let b2 := [("tool", sp.VStr("submit_order")), ("qty", sp.VInt(5000)), ("approved", sp.VBool(true))]
  let v2 := ev.eval(order_submit_policy(), b2)
  let __4 := io.print("")
  let __5 := io.print("  B  tool=submit_order  qty=5000  approved=true")
  let __6 := show_verdict(v2)

  # Case C: submit_order, qty=100, approved=false → Deny (not approved)
  let b3 := [("tool", sp.VStr("submit_order")), ("qty", sp.VInt(100)), ("approved", sp.VBool(false))]
  let v3 := ev.eval(order_submit_policy(), b3)
  let __7 := io.print("")
  let __8 := io.print("  C  tool=submit_order  qty=100   approved=false")
  let __9 := show_verdict(v3)

  # Case D: get_quote, qty=0, approved=false → Allow (antecedent false → vacuously true)
  let b4 := [("tool", sp.VStr("get_quote")), ("qty", sp.VInt(0)), ("approved", sp.VBool(false))]
  let v4 := ev.eval(order_submit_policy(), b4)
  let __10 := io.print("")
  let __11 := io.print("  D  tool=get_quote     qty=0     approved=false")
  show_verdict(v4)
}

# ── Section 3: Property check ────────────────────────────────────────────────

fn show_property_check() -> [io] Unit {
  let __s := section("3 — Property check: check_random(spec, 100, seed=42)")
  let result := check.check_random(order_submit_policy(), 100, 42)
  match result {
    check.Holds(n) =>
      io.print("  ✓  spec holds for " + int.to_str(n) + " random inputs"),
    check.Falsified(counter) => {
      let __1 := io.print("  FALSIFIED — counter-example found:")
      io.print("  " + check.render_counter(counter))
    },
    check.Indeterminate(counter) => {
      let __1 := io.print("  INDETERMINATE — could not decide:")
      io.print("  " + check.render_counter(counter))
    },
  }
}

# ── Section 4: SMT-LIB export ────────────────────────────────────────────────

fn show_smt() -> [io] Unit {
  let __s := section("4 — SMT-LIB export (paste into z3 to formally verify)")
  let script := smt.to_smt_lib(order_submit_policy())
  io.print(script)
}

# ── Section 5: Tool availability filtering ───────────────────────────────────
#
# filter_available evaluates each tool's precondition against the supplied
# bindings. The bindings must include "tool" set to each tool's name — which
# is exactly what agent.with_permission_gate does at construction time.
# Here we replicate that: build per-tool bindings so the demo shows the
# correct gate behaviour.

fn count_tools(tools :: List[t.Tool]) -> Str {
  int.to_str(list.len(tools))
}

fn show_tool_names(tools :: List[t.Tool]) -> [io] Unit {
  list.fold(tools, (), fn (acc :: Unit, tool :: t.Tool) -> [io] Unit {
    io.print("    • " + tool.name)
  })
}

# Build bindings with the tool's own name injected as "tool",
# then evaluate is_available. This mirrors with_permission_gate.
fn is_permitted(tool :: t.Tool, context_qty :: Int, context_approved :: Bool) -> Bool {
  let bindings := [
    ("tool",     sp.VStr(tool.name)),
    ("qty",      sp.VInt(context_qty)),
    ("approved", sp.VBool(context_approved))
  ]
  t.is_available(tool, bindings)
}

fn filter_with_context(tools :: List[t.Tool], context_qty :: Int, context_approved :: Bool) -> List[t.Tool] {
  list.filter(tools, fn (tool :: t.Tool) -> Bool {
    is_permitted(tool, context_qty, context_approved)
  })
}

fn show_filter() -> [io] Unit {
  let __s := section("5 — Tool availability filtering")
  let tools := [tool_get_quote(), tool_read_positions(), tool_submit_order()]

  # Trading-desk context: approved=true, qty=100 — all 3 tools permitted
  let available_trading := filter_with_context(tools, 100, true)
  let __1 := io.print("")
  let __2 := io.print("  Permission level: trading-desk  (approved=true, qty-context=100)")
  let __3 := io.print("  Available: " + count_tools(available_trading) + "/3 tools")
  let __4 := show_tool_names(available_trading)

  # Read-only context: approved=false — submit_order filtered out (Deny)
  let available_readonly := filter_with_context(tools, 0, false)
  let __5 := io.print("")
  let __6 := io.print("  Permission level: read-only     (approved=false)")
  let __7 := io.print("  Available: " + count_tools(available_readonly) + "/3 tools")
  show_tool_names(available_readonly)
}

# ── main ──────────────────────────────────────────────────────────────────────

fn main() -> [io] Unit {
  let __h1 := io.print("")
  let __h2 := io.print("  lex-llm  ·  Spec-gated agent tools")
  let __h3 := io.print("  Typed permissions · property-checked · formally verified")
  let __h4 := io.print("  (zero LLM API calls — spec evaluator and tool filter are pure functions)")
  let __s1 := show_spec_definition()
  let __s2 := show_evaluations()
  let __s3 := show_property_check()
  let __s4 := show_smt()
  let __s5 := show_filter()
  let __f1 := io.print("")
  let __f2 := hr()
  io.print("")
}
