---
title: Chat templates
description: When the auto-detected template is wrong or missing, and how to override it.
sidebar:
  order: 4
---

A chat template tells the engine how to format messages into the string the model actually sees. Different models use different formats, and getting the format wrong produces output that is coherent but subtly worse than it should be. mistral.rs auto-detects the right template for almost every supported model; this guide covers what to do when auto-detection fails or when you want a different one.

## How auto-detection works

mistralrs looks at several places, in order:

1. The `chat_template` field in the model's `tokenizer_config.json` on Hugging Face. Most modern models include this.
2. A bundled template in `chat_templates/` that ships with mistralrs, keyed by architecture.
3. A generic fallback that works for some older models.

If none of those match what the model expects, auto-detection has failed and you need to override.

Symptoms of a wrong template:

- Output quality is noticeably worse than you expect for the model.
- The model sometimes leaks special tokens (`<|im_start|>`, `<bos>`, etc.) into its output.
- Multi-turn conversations degrade faster than single-turn.
- The model ignores system prompts, or treats them as user input.

## Overriding the template

Two ways to supply a custom template.

### A file

Pass a Jinja template file with `--chat-template`:

```bash
mistralrs run -m <model> --chat-template my-template.jinja
```

The template is standard Jinja2 with the conventions HuggingFace uses (variables for messages, bos_token, eos_token, and so on). If you are not sure what the template should look like, copy the original from the model's tokenizer_config.json and modify from there.

### Inline

`--jinja-explicit` takes a template as a string on the command line. Useful for one-off tests:

```bash
mistralrs run -m <model> --jinja-explicit "{% for msg in messages %}..."
```

The `--jinja-explicit` form overrides `--chat-template` if both are set.

## Picking a bundled template

mistralrs ships with templates for common architectures in the `chat_templates/` directory of the source repository. These are tested and known-correct for their respective models. If a new model of a known architecture is not auto-detected, point at the bundled template:

```bash
mistralrs run -m <new-model> --chat-template chat_templates/llama3.jinja
```

The bundled templates available ship with the source; they are included in the binary distribution too and are looked up by name relative to the binary's install location when the argument is not a full path.

## Writing a template from scratch

If you are running a model for which no template exists (a custom fine-tune, an experimental architecture), you can write one. The general pattern:

```jinja
{% if messages[0]['role'] == 'system' %}
{{ bos_token }}<|system|>
{{ messages[0]['content'] }}<|end|>
{% endif %}
{% for msg in messages[(1 if messages[0]['role'] == 'system' else 0):] %}
{% if msg['role'] == 'user' %}
<|user|>
{{ msg['content'] }}<|end|>
{% elif msg['role'] == 'assistant' %}
<|assistant|>
{{ msg['content'] }}<|end|>
{% endif %}
{% endfor %}
{% if add_generation_prompt %}<|assistant|>
{% endif %}
```

The variables available:

- `messages`: the list of chat messages.
- `bos_token`, `eos_token`: the model's special tokens.
- `add_generation_prompt`: a flag that is true when building a prompt for generation (as opposed to tokenizing a complete conversation).

For model-specific details (which special tokens, which role markers), the model's Hugging Face page is the authoritative source.

## Multimodal templates

When a multimodal model is loaded, the template needs to handle the `content` parts that are images, audio, or video. Most models use placeholder tokens like `<|image|>` or `<|audio|>` in specific positions. The bundled templates handle this for the architectures we support; custom templates need to as well.

If you are writing a multimodal template, the existing templates for Gemma 4 and Qwen3-VL in `chat_templates/` are reasonable starting points.
