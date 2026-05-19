---
title: Vision model walkthroughs
description: Model-family notes for Qwen-VL, Gemma, Llama, Mistral, Phi, Idefics, LLaVA, and MiniCPM-O.
sidebar:
  order: 7
---

All supported vision families use the same OpenAI-style multimodal message shape. The differences are model IDs, supported modalities, and a few model-specific flags.

## Common pattern

CLI:

```bash
mistralrs run -m Qwen/Qwen3-VL-4B-Instruct --isq 4 --image photo.jpg -i "Describe this image"
mistralrs serve -m Qwen/Qwen3-VL-4B-Instruct --isq 4 -p 1234
```

HTTP:

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:1234/v1/")

completion = client.chat.completions.create(
    model="default",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "file:///absolute/path/photo.jpg"}},
                {"type": "text", "text": "What is this?"},
            ],
        }
    ],
    max_tokens=256,
)
print(completion.choices[0].message.content)
```

Python SDK:

```python
from mistralrs import ChatCompletionRequest, MultimodalArchitecture, Runner, Which

runner = Runner(
    which=Which.MultimodalPlain(
        model_id="Qwen/Qwen3-VL-4B-Instruct",
        arch=MultimodalArchitecture.Qwen3VL,
    ),
    in_situ_quant="4",
)

response = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "file:///absolute/path/photo.jpg"}},
                    {"type": "text", "text": "What is this?"},
                ],
            }
        ],
        max_tokens=256,
    )
)
print(response.choices[0].message.content)
```

Use `file://` URLs for local files, `https://` for remote files, and `data:image/...;base64,...` for inline images.

## Family quick starts

| Family | Example model | Python architecture | Modalities | Notes |
|---|---|---|---|---|
| Gemma 3 | `google/gemma-3-12b-it` | `MultimodalArchitecture.Gemma3` | image | 128k context vision-language family. |
| Gemma 3n | `google/gemma-3n-E4B-it` | `MultimodalArchitecture.Gemma3n` | image, audio, video | MatFormer slices can trade quality for memory. |
| Gemma 4 | `google/gemma-4-E4B-it` | `MultimodalArchitecture.Gemma4` | image, audio, video | Supports strict tool grammar and mixed media in one message. |
| Idefics 2 | `HuggingFaceM4/idefics2-8b` | `MultimodalArchitecture.Idefics2` | image | Older but useful image-text family. |
| Idefics 3 / SmolVLM | `HuggingFaceM4/Idefics3-8B-Llama3` | `MultimodalArchitecture.Idefics3` | image | SmolVLM follows the same loader path. |
| LLaVA / LLaVA Next | `llava-hf/llava-v1.6-mistral-7b-hf` | `MultimodalArchitecture.LLaVANext` | image | Vicuna-backed checkpoints need the Vicuna chat template. |
| Llama 3.2 Vision | `meta-llama/Llama-3.2-11B-Vision-Instruct` | `MultimodalArchitecture.VLlama` | image | Device mapping applies to the text backbone. |
| Llama 4 | `meta-llama/Llama-4-Scout-17B-16E-Instruct` | `MultimodalArchitecture.Llama4` | image | Sparse multimodal model with tool calling and web-search support. |
| MiniCPM-O 2.6 | `openbmb/MiniCPM-o-2_6` | `MultimodalArchitecture.MiniCpmO` | image, audio | Check the [supported models reference](/mistral.rs/reference/supported-models/) when modality support matters. |
| Mistral Small 3 | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | `MultimodalArchitecture.Mistral3` | image | Tool calling requires the provided Mistral Small tool-call template. |
| Phi 3.5 Vision | `microsoft/Phi-3.5-vision-instruct` | `MultimodalArchitecture.Phi3V` | image | Best with one image; multiple images are resized together. |
| Phi 4 Multimodal | `microsoft/Phi-4-multimodal-instruct` | `MultimodalArchitecture.Phi4MM` | image, audio | Audio and image can be sent in the same message. |
| Qwen2-VL / Qwen2.5-VL | `Qwen/Qwen2-VL-7B-Instruct` | `MultimodalArchitecture.Qwen2VL` / `MultimodalArchitecture.Qwen2_5VL` | image, video | Good baseline Qwen vision family. |
| Qwen3-VL | `Qwen/Qwen3-VL-4B-Instruct` | `MultimodalArchitecture.Qwen3VL` / `MultimodalArchitecture.Qwen3VLMoE` | image, video | Dense and MoE variants. MoE variants support MoQE. |
| Qwen3.5 | `Qwen/Qwen3.5-27B` | `MultimodalArchitecture.Qwen3_5` / `MultimodalArchitecture.Qwen3_5Moe` | image | Dense and MoE variants. MoE variants support MoQE. |

## Video

Use `--video` on the CLI or a `video_url` content part over HTTP:

```bash
mistralrs run -m google/gemma-4-E4B-it --isq 8 --video clip.mp4 -i "Summarize this clip."
```

```json
{
  "role": "user",
  "content": [
    {"type": "video_url", "video_url": {"url": "file:///absolute/path/clip.mp4"}},
    {"type": "text", "text": "What happens in this video?"}
  ]
}
```

FFmpeg requirements, supported containers, and platform install commands are centralized in [Set up video input](/mistral.rs/guides/models/video-setup/). Per-request frame-sampling controls are not currently exposed.

## Audio inside multimodal models

Gemma 4, Gemma 3n, Phi 4 Multimodal, MiniCPM-O, and Voxtral can accept audio content parts when supported by the selected checkpoint:

```json
{
  "role": "user",
  "content": [
    {"type": "audio_url", "audio_url": {"url": "file:///absolute/path/audio.wav"}},
    {"type": "image_url", "image_url": {"url": "file:///absolute/path/photo.jpg"}},
    {"type": "text", "text": "Describe what you hear and see."}
  ]
}
```

WAV, MP3, FLAC, and OGG are decoded natively. Other formats require FFmpeg conversion; see [Set up video input](/mistral.rs/guides/models/video-setup/) for FFmpeg installation.

## Gemma 3n MatFormer

Gemma 3n supports dynamic model slicing. Use this when you want one checkpoint to cover several memory and latency budgets:

```bash
mistralrs run -m google/gemma-3n-E4B-it \
  --matformer-config-path matformer_configs/gemma3n.csv \
  --matformer-slice-name "Config for E2.49B (block-level)"
```

The same fields exist in the Python selector as `matformer_config_path` and `matformer_slice_name`. Without a slice, the default configuration loads.

The bundled `matformer_configs/gemma3n.csv` includes the full E4B configuration, the official E2B slice, and intermediate E1.96B-E3.79B slices. Use the full configuration for quality and smaller slices for constrained devices.

## Mistral Small tool calling

Mistral Small 3 checkpoints can do tool calling, but some model repos do not ship the correct chat template. Use the bundled template when you need tools:

```bash
mistralrs serve -p 1234 --isq 4 \
  --jinja-explicit chat_templates/mistral_small_tool_call.jinja \
  -m mistralai/Mistral-Small-3.2-24B-Instruct-2506
```

## LLaVA chat templates

Mistral-backed LLaVA checkpoints usually work with the default template. Vicuna-backed checkpoints need the Vicuna template:

```bash
mistralrs run -m llava-hf/llava-v1.6-vicuna-7b-hf \
  --isq 4 \
  -c ./chat_templates/vicuna.json \
  --image photo.jpg \
  -i "Describe this image"
```

## Device mapping and topology

For most multimodal models, the text backbone contains most parameters. Device mapping and topology mainly apply to that text portion; the vision, audio, or video encoder stays on its supported device path.

For MoE Qwen3-VL and Qwen3.5 variants, combine ISQ with MoQE when expert memory dominates:

```bash
mistralrs run -m Qwen/Qwen3-VL-235B-A22B-Instruct \
  --isq 4 \
  --isq-organization moqe \
  --image photo.jpg \
  -i "Describe this image"
```

The same setting is `organization=IsqOrganization.MoQE` in `Which.MultimodalPlain(...)`.

## Example files

Long-form SDK examples live in the repository so they can stay checked against the current APIs:

- Python: [`examples/python/`](https://github.com/EricLBuehler/mistral.rs/tree/master/examples/python)
- HTTP/OpenAI clients: [`examples/server/`](https://github.com/EricLBuehler/mistral.rs/tree/master/examples/server)
- Rust multimodal models: [`mistralrs/examples/models/multimodal_models/main.rs`](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/models/multimodal_models/main.rs)
