---
title: Chat templates
description: When the auto-detected template is wrong or missing, and how to override it.
sidebar:
  order: 4
---

A chat template formats messages into the string the model receives. Different models use different formats, and the wrong format produces output that is coherent but degraded. mistral.rs auto-detects the template for almost every supported model. This guide covers manual override.

## Auto-detection

mistral.rs checks, in order:

1. The `chat_template` field in the model's `tokenizer_config.json` on Hugging Face. Most modern models include this.
2. A bundled template in `chat_templates/` keyed by architecture.
3. A generic fallback for some older models.

If none match, auto-detection has failed and an override is required.

Symptoms of a wrong template:

- Output quality below expectations.
- Special tokens (`<|im_start|>`, `<bos>`, etc.) leaking into output.
- Multi-turn degrading faster than single-turn.
- System prompts ignored or treated as user input.

## Overriding the template

### A file

Pass a Jinja template file with `--chat-template`:

```bash
mistralrs run -m <model> --chat-template my-template.jinja
```

The template uses standard Jinja2 with HuggingFace conventions (variables for messages, bos_token, eos_token). Copy the original from `tokenizer_config.json` and modify as a starting point.

### Inline

`--jinja-explicit` accepts an inline template string for one-off tests:

```bash
mistralrs run -m <model> --jinja-explicit "{% for msg in messages %}..."
```

`--jinja-explicit` overrides `--chat-template` when both are set.

## Picking a bundled template

mistral.rs ships templates for common architectures in `chat_templates/`. For new models of a known architecture not auto-detected, point at the bundled template:

```bash
mistralrs run -m <new-model> --chat-template chat_templates/llama3.jinja
```

Bundled templates are looked up by name relative to the binary install location when not given as a full path.

## Writing a template from scratch

For models without an existing template, write one. General pattern:

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

Available variables:

- `messages`: the chat message list.
- `bos_token`, `eos_token`: model special tokens.
- `add_generation_prompt`: true when building a prompt for generation.

For model-specific tokens and role markers, the model's Hugging Face page is authoritative.

## Multimodal templates

Multimodal models need templates that handle non-text content parts. Most models use placeholder tokens like `<|image|>` or `<|audio|>`. Bundled templates handle this for supported architectures; custom multimodal templates must do so too.

The `chat_templates/` directory contains templates for Gemma 4 and Qwen3-VL.
