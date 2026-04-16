---
title: Tutorials
description: Step-by-step lessons that get you to a working result.
---

These are lessons, not recipes. Follow them in order and you should end up with something running by the time you reach the bottom of each page. There are no "you could also" branches to worry about.

If you already know what you want to do, the [Guides](/mistral.rs/guides/) section is probably a better starting point. If you just want to look up a flag or an API method, try [Reference](/mistral.rs/reference/). For the reasoning behind how mistral.rs is put together, see [Explanation](/mistral.rs/explanation/).

## Available tutorials

1. [Install and run your first model](/mistral.rs/tutorials/01-install-and-run/). Install the binary and chat with a model in your terminal.
2. Serve a model as an API. Take the same model and put it behind an OpenAI-compatible HTTP endpoint.
3. Use the Python SDK. Call models from a Python program without running a separate server.
4. Use the Rust SDK. Embed mistral.rs directly into a Rust application.
5. Build an agent. Turn on tool calling, web search, and code execution so the model can take actions, not just reply.
6. Quantize a model. Shrink a larger model so it fits on the GPU you actually have.

Later tutorials assume you have done the earlier ones, so they skip over things like install steps and how interactive mode works.
