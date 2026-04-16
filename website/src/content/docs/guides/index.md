---
title: Guides
description: Task-oriented recipes for things you want to do.
---

Guides answer specific "how do I…" questions. They assume you already have mistral.rs installed and know roughly what it does. If you do not, start with the [Tutorials](/mistral.rs/tutorials/) instead.

Each guide is self-contained and short. It picks a problem, walks through a solution, and ends. Guides are organized by what you are trying to accomplish rather than by which component of mistral.rs is involved.

## Install and deploy

Platform-specific install steps, Docker images, and things to check before putting a server in production.

## Serve

How to configure the HTTP server, run multiple models behind one process, expose the web UI, and work with the OpenAI Responses API surface.

## Performance

Picking a quantization method for the hardware you have, using the `tune` command to auto-benchmark, enabling Flash and Paged attention, and splitting a model across multiple GPUs or machines.

## Agents

Turning on tool calling, code execution, and web search. Connecting an MCP server, or exposing your mistral.rs instance as one. Persisting agent sessions across requests.

## Python SDK

Streaming completions, passing images and video, and using the Python session API for multi-turn agents.

## Rust SDK

Streaming from the Rust SDK, and embedding mistral.rs inside an existing Axum application.

## Models

Using vision input, image generation, speech, and embedding models. Working with chat templates when auto-detection is not enough.

## Customize

Attaching LoRA adapters, configuring AnyMoE or MatFormer, controlling sampling parameters, and pointing the CLI at a TOML config instead of flags.
