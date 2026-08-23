# Multimodal (image) turns must reach EVERY provider in that provider's own
# wire shape.
#
# These tests are load-bearing in a way most are not. Lex does not check match
# exhaustiveness — a `match` missing an arm for a new Message variant type-checks
# clean, passes `lex check --strict`, and then panics at RUNTIME with
# "non-exhaustive match" the first time that variant reaches it. Adding
# UserPartsMsg to the Message ADT therefore could not be validated by the
# compiler; only by encoding an image through each adapter, which is what this
# file does. If you add a provider, add it here too.

import "../src/message" as msg

import "../src/providers/openai" as oai

import "../src/providers/anthropic" as ant

import "../src/providers/ollama" as olla

import "../src/providers/google" as goo

import "../src/providers/vertex" as vx

import "lex-schema/json_value" as jv

import "std.str" as str

import "std.int" as int

import "std.list" as list

fn assert_eq(name :: Str, got :: Str, want :: Str) -> Result[Unit, Str] {
  if got == want {
    Ok(())
  } else {
    Err(str.concat(name, str.concat("\n  expected: ", str.concat(want, str.concat("\n  got:      ", got)))))
  }
}

# One text part then one JPEG — the camera case.
fn frame_msg() -> msg.Message {
  msg.user_with_jpeg("what do you see?", "QUJD")
}

fn test_openai_uses_a_data_uri_image_url_part() -> Result[Unit, Str] {
  assert_eq("openai", jv.stringify(oai.encode_message(frame_msg())), "{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"what do you see?\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/jpeg;base64,QUJD\"}}]}")
}

fn test_anthropic_uses_a_typed_base64_source_block() -> Result[Unit, Str] {
  assert_eq("anthropic", jv.stringify(ant.encode_message(frame_msg())), "{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"what do you see?\"},{\"type\":\"image\",\"source\":{\"type\":\"base64\",\"media_type\":\"image/jpeg\",\"data\":\"QUJD\"}}]}")
}

fn test_ollama_puts_images_in_a_sibling_array_not_in_content() -> Result[Unit, Str] {
  assert_eq("ollama", jv.stringify(olla.encode_message(frame_msg())), "{\"role\":\"user\",\"content\":\"what do you see?\",\"images\":[\"QUJD\"]}")
}

fn test_google_uses_gemini_inline_data_parts() -> Result[Unit, Str] {
  assert_eq("google", jv.stringify(goo.encode_content(frame_msg())), "{\"role\":\"user\",\"parts\":[{\"text\":\"what do you see?\"},{\"inline_data\":{\"mime_type\":\"image/jpeg\",\"data\":\"QUJD\"}}]}")
}

fn test_vertex_matches_the_google_part_shape() -> Result[Unit, Str] {
  assert_eq("vertex", jv.stringify(vx.encode_content(frame_msg())), "{\"role\":\"user\",\"parts\":[{\"text\":\"what do you see?\"},{\"inline_data\":{\"mime_type\":\"image/jpeg\",\"data\":\"QUJD\"}}]}")
}

fn test_text_only_turns_are_byte_for_byte_unchanged() -> Result[Unit, Str] {
  match assert_eq("openai text", jv.stringify(oai.encode_message(msg.user("hi"))), "{\"role\":\"user\",\"content\":\"hi\"}") {
    Err(e) => Err(e),
    Ok(_) => assert_eq("ollama text", jv.stringify(olla.encode_message(msg.user("hi"))), "{\"role\":\"user\",\"content\":\"hi\"}"),
  }
}

fn test_content_returns_text_and_omits_images() -> Result[Unit, Str] {
  assert_eq("content", msg.content(frame_msg()), "what do you see?")
}

fn test_image_count_sees_what_content_cannot() -> Result[Unit, Str] {
  let two := msg.user_with_images("pair", [msg.image_b64("image/png", "AA"), msg.image_b64("image/jpeg", "BB")])
  match assert_eq("two", int.to_str(msg.image_count(two)), "2") {
    Err(e) => Err(e),
    Ok(_) => assert_eq("none", int.to_str(msg.image_count(msg.user("plain"))), "0"),
  }
}

fn test_role_of_a_multimodal_turn_is_user() -> Result[Unit, Str] {
  assert_eq("role", msg.role_str(frame_msg()), "user")
}

fn test_text_is_placed_before_the_image() -> Result[Unit, Str] {
  let j := jv.stringify(goo.encode_content(frame_msg()))
  if str.contains(j, "{\"text\":\"what do you see?\"},{\"inline_data\"") {
    Ok(())
  } else {
    Err(str.concat("text part should precede the image part, got ", j))
  }
}

fn run_all() -> Unit {
  let results := [test_openai_uses_a_data_uri_image_url_part(), test_anthropic_uses_a_typed_base64_source_block(), test_ollama_puts_images_in_a_sibling_array_not_in_content(), test_google_uses_gemini_inline_data_parts(), test_vertex_matches_the_google_part_shape(), test_text_only_turns_are_byte_for_byte_unchanged(), test_content_returns_text_and_omits_images(), test_image_count_sees_what_content_cannot(), test_role_of_a_multimodal_turn_is_user(), test_text_is_placed_before_the_image()]
  let failures := list.fold(results, 0, fn (n :: Int, r :: Result[Unit, Str]) -> Int {
    match r {
      Ok(_) => n,
      Err(_) => n + 1,
    }
  })
  if failures == 0 {
    ()
  } else {
    let __discard := 1 / 0
    ()
  }
}

