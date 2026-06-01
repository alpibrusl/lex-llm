#!/usr/bin/env bash
# Theatrical demo — lex-llm: spec-gated agent tools
# Typed permissions · property-checked · formally verified
#
# Usage:   bash examples/spec_gated_agent.sh
#          asciinema rec examples/spec_gated_agent.cast \
#            -c "bash examples/spec_gated_agent.sh" --overwrite
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
LEX="${LEX:-lex}"

BOLD=$'\033[1m'; DIM=$'\033[2m'; CYAN=$'\033[36m'
GREEN=$'\033[32m'; BLUE=$'\033[34m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; RESET=$'\033[0m'

slow() { echo "$@" | pv -qL 55; }
pause() { sleep "${1:-1.2}"; }
hr()  { printf '%s' "$DIM"; printf '─%.0s' {1..72}; printf '%s\n' "$RESET"; }
hdr() { echo; hr; echo "  ${BOLD}${CYAN}$*${RESET}"; hr; echo; }
cmd() { echo "${BOLD}${BLUE}\$${RESET}  $*"; pause 0.6; }

# Format the demo output: newlines around HRs, before ✓/✗ lines,
# and pretty-print JSON blocks.
fmt() {
  python3 -c "
import sys, re, json
raw = sys.stdin.read().replace('null', '').strip()
# Newline around HR sequences
text = re.sub(r'(─{10,})', lambda m: '\n' + m.group(0) + '\n', raw)
# Newline before section headings (digit space dash)
text = re.sub(r'  (\d+ —)', r'\n  \1', text)
# Newline before verdict lines
text = re.sub(r'(    verdict:)', r'\n\1', text)
# Newline before ✓ / ✗ lines
text = re.sub(r'(  [✓✗])', r'\n\1', text)
# Newline before Permission level lines
text = re.sub(r'(  Permission level:)', r'\n\1', text)
# Newline before Available: lines
text = re.sub(r'(  Available:)', r'\n\1', text)
# Newline before bullet lines
text = re.sub(r'(    •)', r'\n\1', text)
# Pretty-print standalone JSON objects and SMT-LIB blocks
out = []
for line in text.split('\n'):
    s = line.strip()
    if s.startswith('{') and s.endswith('}'):
        try:
            out.append(json.dumps(json.loads(s), indent=2))
            continue
        except Exception:
            pass
    out.append(line)
print('\n'.join(out))
"
}

clear
echo
echo "  ${BOLD}lex-llm${RESET}  ·  Spec-gated agent tools"
echo "  ${DIM}Typed permissions · property-checked · formally verified${RESET}"
echo
sleep 2

# ── What this demo shows ─────────────────────────────────────────────────────
hdr "Three layers — same spec, three enforcement points"
slow "  Layer 1 — eval:    Spec is a typed value evaluated against concrete"
slow "            bindings at every tool call. Allow / Deny / Inconclusive."
slow "  Layer 2 — check:   check_random() runs 100 seeded random inputs and"
slow "            proves the spec holds over the entire bounded domain."
slow "  Layer 3 — SMT-LIB: to_smt_lib() emits a Z3 script that encodes the"
slow "            spec's negation. 'unsat' is a formal proof of correctness."
echo
slow "  Zero LLM API calls — the spec evaluator, property checker, and SMT"
slow "  exporter are pure functions in lex-spec."
echo
pause 1.5

# ── Type check ───────────────────────────────────────────────────────────────
hdr "Type check — all effects declared before a byte runs"
cmd "lex check examples/spec_gated_agent.lex"
pause 0.4
"$LEX" check examples/spec_gated_agent.lex
echo "${GREEN}${BOLD}✓  ok${RESET}"
echo
pause 1.2

# ── Run the demo ─────────────────────────────────────────────────────────────
hdr "End to end — spec definition, evaluation, property check, SMT, filtering"
slow "  The policy: submit_order requires qty ≤ 1000 AND approved == true."
slow "  Four bindings evaluated: two Allow, two Deny, each with the reason."
slow "  Property check: 100 random inputs, seed 42 — spec holds."
slow "  SMT-LIB: Z3 script asserting the negation — paste and run z3 to verify."
slow "  Tool filter: trading-desk sees all 3 tools; read-only sees 2."
echo
pause 0.8

cmd "lex run --allow-effects fs_write,io,net,proc,sql,time \\"
echo "        examples/spec_gated_agent.lex main 2>&1 | fmt"
pause 0.5
"$LEX" run --allow-effects fs_write,io,net,proc,sql,time \
  examples/spec_gated_agent.lex main 2>&1 | fmt
echo
pause 1.5

# ── Summary ──────────────────────────────────────────────────────────────────
hr
echo
echo "  ${BOLD}${GREEN}DONE${RESET}"
echo
echo "  The same spec that gates tool calls at runtime can be property-checked"
echo "  and formally verified — without touching an LLM or a database."
echo
echo "  ${CYAN}eval${RESET}      three-valued verdict (Allow / Deny / Inconclusive) per call"
echo "  ${CYAN}check${RESET}     100 seeded random inputs, deterministic, no flakiness"
echo "  ${CYAN}smt${RESET}       Z3-ready negation — paste into z3, get 'unsat' = proof"
echo "  ${CYAN}filter${RESET}    trading-desk: 3/3 tools · read-only: 2/3 (submit_order gated)"
echo
hr
echo
