---
title: Decision requests
description: Typed d1 inference through SystemOne and the native Rust and Python SDKs.
---

Decision models answer named questions about an input, such as a message, an image, or an audio clip, without generating text. Each answer is a typed value, such as a probability, a chosen option, a score, extracted text, or a bounding box. mistral.rs supports the d1 models `LiquidAI/d1-2B` and `LiquidAI/d1-Omni-2B`, and serves them at `POST /v1/systemone`. For a walkthrough, see [Use decision models](/guides/models/use-decision-models/).

```bash
mistralrs serve -p 1234 -m LiquidAI/d1-Omni-2B
curl http://localhost:1234/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "default",
    "state": {"message": "Please cancel my subscription today."},
    "questions": {
      "cancel": {"type": "noul", "instructions": "Does the customer want to cancel?"},
      "intent": {
        "type": "choice",
        "instructions": "What is the customer requesting?",
        "criteria": {"cancel": "Cancel service", "billing": "Explain a payment", "other": null}
      }
    }
  }'
```

## Request fields

| Field | Description |
| --- | --- |
| `questions` | Required. Named questions; each name becomes a key in the response's `answers`. See [question types](#question-types). |
| `state` | What the questions are about: a string or any JSON value. Use `null` when the questions are only about media. |
| `model` | Which loaded model to use. Defaults to the default model. |
| `images` | A list of image URL or data URL strings. |
| `media` | Images, audio, and video, in order. See [media](#media-and-spoken-instructions). |
| `set_size` | How many questions the model answers in one pass. Defaults to the model's own setting. |

Any other field is rejected, including generation settings such as `temperature`, `max_tokens`, and `stream`.

mistral.rs splits the questions into groups of `set_size` and runs one pass per group, repeating the state and media in each. `set_size` doesn't limit how many questions you can ask.

The response's `usage.input_tokens` counts everything the model read, including media tokens and the state once per group. `usage.output_tokens` is always 0.

## GGUF

GGUF models take the same requests:

```bash
mistralrs serve -p 1234 -m LiquidAI/d1-Omni-2B-GGUF --quant 8
```

The CLI downloads the matching projector file for you. The GGUF files alone answer `noul`, single-answer `choice`, and `score` questions. For the other types, and for `choice` with `multiple: true`, also pass the original model so mistral.rs can read its config and calibration:

```bash
mistralrs serve -p 1234 -m LiquidAI/d1-Omni-2B-GGUF --quant 8 --tok-model-id LiquidAI/d1-Omni-2B
```

See the [d1 model notes](/guides/models/model-family-notes/#d1-and-d1-omni) for d1-2B and local loading.

## Question types

Every question has a `type` and `instructions`. Instructions are a string or, with d1-Omni, an audio content part.

| Type | Additional fields | Answer |
| --- | --- | --- |
| `noul` (yes/no) | Optional `criteria`: `{"true": "...", "false": "..."}`, describing what counts as yes and no | `noul`: probability of yes, from 0 to 1. |
| `choice` | `criteria`: option names mapped to descriptions, or to `null`. Optional `multiple: true` lets several options apply. | One option: `choice`, `confidence`, and `probabilities` that sum to 1. With `multiple`: `choices`, and an independent probability per option. |
| `score` | `criteria`: 2 to 10 level descriptions, in order | `score`: the expected level on a 0 to N-1 scale, plus `confidence`, `probabilities`, and `legend`. |
| `text` | None | `text` copied from the state, its `span`, `found` probability, and optional `path`. |
| `number` | Optional `range: [min, max]`, `step`, and `unit` | Without `range`: a `number` read from the state. With `range`: an estimated `number` and `interval` within it. |
| `date` | None | Normalized `date`, source `text` and `span`, `found`, and optional `path`. |
| `box` | None | One `box` with `top`, `bottom`, `left`, and `right`, plus `found` and `confidence`. |
| `boxes` | Optional `max` (1 to 16 for the shipped checkpoints) | `boxes`, `count`, `at_least`, and a `presence` probability per slot. |

How to read extraction and box answers:

- Extracted values are copied from the state, never invented.
- `span` is a pair of Unicode character offsets into the state as rendered, with an exclusive end. JSON state is rendered with two-space indentation.
- When the value comes from a single JSON string, `path` is a JSON Pointer to it.
- Box questions need exactly one image. Coordinates run from 0 to 1, relative to the original image.

## Media and spoken instructions

`images` is a shortcut that takes a plain list of image URL or data URL strings. To mix images, audio, and video in a set order, use `media` with the same content parts as chat requests:

```json
{
  "state": null,
  "media": [
    {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
    {"type": "audio_url", "audio_url": {"url": "https://example.com/question.wav"}},
    {"type": "video_url", "video_url": {"url": "https://example.com/clip.mp4"}}
  ],
  "questions": {
    "visible": {"type": "noul", "instructions": "Is the described object visible?"}
  }
}
```

- WAV and MP3 clips can also be sent inline: `{"type":"input_audio","input_audio":{"data":"BASE64","format":"wav"}}`.
- To ask a question out loud, put an audio part where the `instructions` string would go.
- `images` come before `media` in the input.
- Over HTTP, media must be a remote URL or an inline data URL. The native SDKs also accept local files.
- Video frames are sampled using the model's settings. The soundtrack isn't extracted, so send audio separately.

## Native SDKs

Rust uses typed `DecisionRequest`, `DecisionQuestion`, `DecisionAnswer`, and `DecisionMedia` values. Call `Model::send_decision_request(request)`, or `send_decision_request_with_model` to pick a loaded model. See the [Rust reference](/reference/rust-sdk/#decision-requests).

Python:

```python
from mistralrs import DecisionQuestion, DecisionRequest, Runner, Which

runner = Runner(which=Which.MultimodalPlain(model_id="LiquidAI/d1-Omni-2B"))
response = runner.send_decision_request(DecisionRequest(
    state={"message": "Please cancel my subscription."},
    questions={"cancel": DecisionQuestion("noul", "Does the customer want to cancel?")},
))
print(response.answers["cancel"].noul)
print(response.to_dict())
```

In Python:

- Pass `model_id=` to pick another loaded model.
- Questions can also be dictionaries in the HTTP shape.
- An answer's attributes depend on its `type`. Optional attributes return `None` when absent.
- `to_dict()` returns the HTTP JSON shape, so fields the HTTP response leaves out are left out there too.
- `images` and `media` accept URLs, data URLs, and local `file://` URLs.

To load d1 from GGUF, pass the projector file with Rust's `GgufModelBuilder::with_mmproj_files` or Python's `Which.GGUF(mmproj_filename=...)`. For every question type, also set `with_tok_model_id` or `tok_model_id` to the original model or a local folder with its `config.json` and `calibration.json`. Requests work the same. See [GGUF SDK loading](/guides/models/run-gguf/#multimodal-gguf).

Runnable examples: [HTTP](/examples/server/lfm3-d1/), [Python](/examples/python/lfm3-d1/), and [Rust](/examples/rust/models/lfm3-d1/).
