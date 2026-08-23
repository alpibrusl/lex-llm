# lex-llm — conversational message ADT
#
# Roles follow the OpenAI/Anthropic wire convention:
#   UserMsg      — human turn
#   SystemMsg    — system prompt injected before the conversation
#   AssistantMsg — model response; carries tool calls when the model
#                  chose to invoke tools rather than produce text
#   ToolMsg      — tool-execution result matched to a call by call_id
#
# The ADT is provider-agnostic; each adapter converts to its wire format.

import "lex-schema/json_value" as jv

import "std.list" as list

import "std.str" as str

# One tool invocation requested by the model.
type ToolCall = { id :: Str, name :: Str, args :: jv.Json }

# An image the model can look at.
#
# Base64 inline bytes only, deliberately — NOT a URL. Every provider adapter
# here accepts inline image data, but URL support is uneven: Gemini's
# `inline_data` has no URL form at all (its `file_data` wants a Files-API
# URI, not an arbitrary link). A url variant would therefore typecheck
# everywhere and then fail at runtime on some providers, which is exactly the
# kind of gap this ADT exists to prevent. Callers holding a URL should fetch
# it and pass the bytes.
#
# `mime` is the image media type ("image/jpeg", "image/png"); `data` is the
# payload base64-encoded WITHOUT a `data:` prefix — adapters add whatever
# framing their wire format wants.
type Image = ImageB64((Str, Str))

# One piece of a multimodal turn. A turn is an ordered list of these, because
# position carries meaning: "what is wrong in this photo?" reads differently
# before and after the image.
type Part = TextPart(Str) | ImagePart(Image)

type Message = UserMsg(Str) | SystemMsg(Str) | AssistantMsg((Str, List[ToolCall])) | ToolMsg((Str, Str)) | UserPartsMsg(List[Part])

fn content(msg :: Message) -> Str
  examples {
    content(UserMsg("hello")) => "hello",
    content(SystemMsg("be helpful")) => "be helpful",
    content(ToolMsg("id1", "42")) => "42",
    content(UserPartsMsg([TextPart("look:"), ImagePart(ImageB64("image/jpeg", "AAA"))])) => "look:"
  }
{
  match msg {
    UserMsg(s) => s,
    SystemMsg(s) => s,
    AssistantMsg(s, _) => s,
    ToolMsg(_, s) => s,
    UserPartsMsg(parts) => list.fold(parts, "", fn (acc :: Str, p :: Part) -> Str {
      match p {
        TextPart(t) => if str.is_empty(acc) {
          t
        } else {
          str.concat(acc, str.concat("\n", t))
        },
        ImagePart(_) => acc,
      }
    }),
  }
}

# How many images a turn carries. Useful for budget/telemetry: image tokens
# dominate cost, and `content` above deliberately cannot see them.
fn image_count(msg :: Message) -> Int
  examples {
    image_count(UserMsg("hi")) => 0,
    image_count(UserPartsMsg([TextPart("a"), ImagePart(ImageB64("image/png", "AA"))])) => 1
  }
{
  match msg {
    UserPartsMsg(parts) => list.fold(parts, 0, fn (acc :: Int, p :: Part) -> Int {
      match p {
        ImagePart(_) => acc + 1,
        TextPart(_) => acc,
      }
    }),
    _ => 0,
  }
}

fn role_str(msg :: Message) -> Str
  examples {
    role_str(UserMsg("hi")) => "user",
    role_str(SystemMsg("sys")) => "system"
  }
{
  match msg {
    UserMsg(_) => "user",
    SystemMsg(_) => "system",
    AssistantMsg(_, _) => "assistant",
    ToolMsg(_, _) => "tool",
    UserPartsMsg(_) => "user",
  }
}

fn has_tool_calls(msg :: Message) -> Bool
  examples {
    has_tool_calls(UserMsg("hi")) => false,
    has_tool_calls(AssistantMsg("ok", [])) => false
  }
{
  match msg {
    AssistantMsg(_, calls) => not list.is_empty(calls),
    _ => false,
  }
}

fn user(text :: Str) -> Message {
  UserMsg(text)
}

# A base64 image. `mime` is the media type; `data` is raw base64 with no
# `data:` prefix (adapters add their own framing).
fn image_b64(mime :: Str, data :: Str) -> Image {
  ImageB64(mime, data)
}

# The common multimodal turn: some text, then one or more images.
#
# Text first is not arbitrary. Providers differ on whether a leading image
# with no context is even accepted, and models reliably do better when the
# question precedes the picture; putting the caller's text first makes the
# default the well-behaved one. Callers needing another order build
# `UserPartsMsg` themselves.
fn user_with_images(text :: Str, images :: List[Image]) -> Message
  examples {
    user_with_images("what is this?", []) => UserPartsMsg([TextPart("what is this?")])
  }
{
  UserPartsMsg(list.concat([TextPart(text)], list.map(images, fn (i :: Image) -> Part {
    ImagePart(i)
  })))
}

# A single JPEG frame with a question — the camera case, which is the reason
# this exists.
fn user_with_jpeg(text :: Str, jpeg_b64 :: Str) -> Message {
  user_with_images(text, [ImageB64("image/jpeg", jpeg_b64)])
}

fn system(text :: Str) -> Message {
  SystemMsg(text)
}

fn tool_result(call_id :: Str, body :: jv.Json) -> Message {
  ToolMsg(call_id, jv.stringify(body))
}

fn tool_error(call_id :: Str, err :: Str) -> Message {
  ToolMsg(call_id, str.concat("{\"error\":\"", str.concat(json_escape(err), "\"}")))
}

# Minimal JSON string escape for embedding in a string literal.
fn json_escape(s :: Str) -> Str {
  let chars := str.split(s, "")
  list.fold(chars, "", fn (acc :: Str, c :: Str) -> Str {
    str.concat(acc, match c {
      "\"" => "\\\"",
      "\\" => "\\\\",
      "\n" => "\\n",
      "\r" => "\\r",
      _ => c,
    })
  })
}

