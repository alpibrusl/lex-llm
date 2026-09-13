# test_timeout.lex -- the client timeout is a knob, and a wrong value is not
# a silent 0-millisecond timeout.
#
# Why this matters: every provider hardcoded 600s. A local 27B model answering
# a build prompt runs past that, the client aborts mid-generation, the server
# logs a 500 ("context canceled") and the caller sees "provider error" -- work
# thrown away seconds before it would have finished.

import "std.str" as str

import "std.list" as list

import "std.io" as io

import "../src/timeout" as tmo

import "../src/providers/ollama" as olla

fn check(name :: Str, cond :: Bool) -> Result[Unit, Str] {
  if cond {
    Ok(())
  } else {
    Err(name)
  }
}

fn test_default_is_ten_minutes() -> Result[Unit, Str] {
  check("the default is 600s", tmo.default_timeout_ms() == 600000)
}

fn test_or_default_takes_a_real_value() -> Result[Unit, Str] {
  check("an explicit value wins", tmo.or_default(Some(1800000)) == 1800000)
}

fn test_or_default_rejects_nonsense() -> Result[Unit, Str] {
  match check("None falls back", tmo.or_default(None) == 600000) {
    Err(e) => Err(e),
    Ok(_) => check("zero falls back rather than timing out instantly", tmo.or_default(Some(0)) == 600000),
  }
}

fn test_with_timeout_keeps_the_rest_of_the_config() -> Result[Unit, Str] {
  let c := olla.with_timeout({ base_url: "http://h/api/chat", think: Some("false"), timeout_ms: None }, 900000)
  match check("the timeout is set", c.timeout_ms == Some(900000)) {
    Err(e) => Err(e),
    Ok(_) => match check("base_url survives", c.base_url == "http://h/api/chat") {
      Err(e) => Err(e),
      Ok(_) => check("think survives", c.think == Some("false")),
    },
  }
}

fn suite() -> List[Result[Unit, Str]] {
  [test_default_is_ten_minutes(), test_or_default_takes_a_real_value(), test_or_default_rejects_nonsense(), test_with_timeout_keeps_the_rest_of_the_config()]
}

fn run_all() -> [io] Unit {
  let results := suite()
  let __dbg := list.map(results, fn (r :: Result[Unit, Str]) -> [io] Unit {
    match r {
      Ok(_) => (),
      Err(e) => io.print(str.concat("FAIL: ", e)),
    }
  })
  let failures := list.fold(results, 0, fn (n :: Int, r :: Result[Unit, Str]) -> Int {
    match r {
      Ok(_) => n,
      Err(_) => n + 1,
    }
  })
  if failures == 0 {
    io.print("ok   4 timeout tests")
  } else {
    let __force_fail := 1 / 0
    ()
  }
}

