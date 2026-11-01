---
title: Decision requests
description: Typed d1 inference through SystemOne and the native Rust and Python SDKs.
---

`LiquidAI/d1-2B` and `LiquidAI/d1-Omni-2B` answer named questions with probabilities and structured values. They score prepared answer positions directly, without generating an answer token by token. The canonical HTTP endpoint is `POST /v1/systemone`.

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

The request accepts `model`, `state`, `questions`, `images`, `media`, and `set_size`. `state` can be any JSON value, including `null`; question names become keys in the response's `answers` object. The response also contains `model` and `usage.input_tokens` / `usage.output_tokens`. `input_tokens` counts the expanded input across question groups, including repeated state. `output_tokens` is always zero because decision inference generates no tokens. Generation controls such as `temperature`, `max_tokens`, and `stream` are not accepted.

GGUF uses the same request API:

```bash
mistralrs serve -p 1234 -m LiquidAI/d1-Omni-2B-GGUF --quant 8
```

The CLI selects the companion projector automatically. Standalone GGUF supports `noul`, single-selection `choice`, and `score`. The remaining question types and multiple-selection choice require the matching original configuration and calibration, supplied with `--tok-model-id LiquidAI/d1-Omni-2B` for this model. See the [d1 model notes](/guides/models/model-family-notes/#d1-and-d1-omni) for d1-2B and local loading.

## Question types

Every question has `type` and `instructions`. Instructions are a string, or an audio content part when using d1-Omni.

| Type | Additional fields | Answer |
| --- | --- | --- |
| `noul` | Optional `criteria` object | `noul`: probability of yes, from 0 to 1. |
| `choice` | `criteria`: object mapping option names to descriptions or `null`; optional `multiple` | Single choice: `choice`, `confidence`, and normalized `probabilities`. Multiple choice: `choices` and independent option `probabilities`. |
| `score` | `criteria`: 2 to 10 ordered descriptions | `score`: expected ordinal level on the 0 to N-1 scale, plus `confidence`, `probabilities`, and `legend`. |
| `text` | None | Extracted `text`, `span`, `found` probability, and optional `path`. |
| `number` | Optional `range: [min, max]`, `step`, and `unit` | Without a range, extracts and parses a number from state. With a range, estimates `number` and `interval` within that range. |
| `date` | None | Normalized `date`, source `text` / `span`, `found`, and optional `path`. |
| `box` | None | One normalized `box` with `top`, `bottom`, `left`, `right`, plus `found` and `confidence`. |
| `boxes` | Optional `max` (1 to 16 for the shipped checkpoints) | `boxes`, `count`, `at_least`, and per-slot `presence` probabilities. |

Extraction copies from the state; it does not invent text. `span` contains Unicode character offsets in the rendered state, with an exclusive end. JSON state is rendered with two-space indentation. When an extracted value lies in a single JSON string, `path` identifies that value using a JSON Pointer. Box questions require exactly one image. Coordinates range from 0 to 1 in the original image.

`set_size` limits the number of questions grouped into each evaluation set; leave it omitted to use the model default. It does not limit the number of named questions in the request.

## Media and spoken instructions

For SystemOne image compatibility, `images` accepts image URL or data URL strings. To combine modalities in order, use `media` with the existing content-part shapes:

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

WAV and MP3 clips can also use `{"type":"input_audio","input_audio":{"data":"BASE64","format":"wav"}}`. The same audio part can replace a question's `instructions` string for spoken instructions. The top-level `images` are placed before `media`. HTTP media uses remote URLs or inline data URLs; local files are accepted by the native SDK. Video sampling follows the selected model's settings. Video soundtrack extraction is not automatic: submit audio separately.

## Native SDKs

Rust exposes `DecisionRequest`, `DecisionQuestion`, `DecisionAnswer`, and `DecisionResponse`, with typed `DecisionMedia::{Image, Audio, Video}` values and `DecisionInstructions::{Text, Audio}`. Call `Model::send_decision_request(request)` or `send_decision_request_with_model(request, Some(model_id))`.

Python exposes the same named request/response concepts:

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

Pass `model_id=` to select another loaded model. Question dictionaries with the HTTP shape are also accepted. Answer attributes depend on their `type`; optional attributes for that type return `None` when absent. `to_dict()` preserves the wire representation, including omitted fields. Python `images` and `media` accept URLs, data URLs, or local `file://` URLs.

For GGUF, use Rust's `GgufModelBuilder` with `with_mmproj_files`, or Python's `Which.GGUF` with `mmproj_filename`, to select the companion projector. Optionally set `with_tok_model_id` or `tok_model_id` to the matching original model or a local config/calibration directory to enable the full decision API. `send_decision_request` is unchanged. See [GGUF SDK loading](/guides/models/run-gguf/#multimodal-gguf).

Runnable examples: [HTTP](/examples/server/lfm3-d1/), [Python](/examples/python/lfm3-d1/), and [Rust](/examples/rust/models/lfm3-d1/).
