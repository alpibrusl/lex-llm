# timeout.lex -- how long a provider call may take before the client gives up.
#
# Every provider hardcoded 600s. That is generous for a hosted model and too
# short for a local one: a 27B model on a laptop, answering a build prompt with
# a long tool transcript, routinely runs past ten minutes. When it does, the
# client aborts mid-generation, the server logs a 500 ("context canceled"), and
# the caller sees a bare "provider error" -- the work is thrown away seconds
# before it would have finished. Nine of those in one afternoon on one machine
# (alpibrusl/lex-loom, 2026-09-13) cost that run its whole build phase.
#
# So it is a knob, in milliseconds:
#
#   LLM_TIMEOUT_MS=1800000   # 30 minutes, for a slow local model
#
# and providers take it from their config, which is where the caller can also
# set it explicitly. Nothing infers it from the model or the host: a proxy can
# front anything, and guessing would trade one silent failure for another.

import "std.env" as env

import "std.str" as str

fn default_timeout_ms() -> Int {
  600000
}

fn resolved_timeout_ms() -> [env] Int {
  match env.get("LLM_TIMEOUT_MS") {
    None => default_timeout_ms(),
    Some(v) => match str.to_int(str.trim(v)) {
      Some(n) => if n > 0 {
        n
      } else {
        default_timeout_ms()
      },
      None => default_timeout_ms(),
    },
  }
}

# What a provider uses when its config carries no explicit value.
fn or_default(configured :: Option[Int]) -> Int
  examples {
    or_default(Some(1000)) => 1000,
    or_default(None) => 600000,
    or_default(Some(0)) => 600000
  }
{
  match configured {
    Some(n) => if n > 0 {
      n
    } else {
      default_timeout_ms()
    },
    None => default_timeout_ms(),
  }
}

