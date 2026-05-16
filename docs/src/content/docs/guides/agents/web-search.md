---
title: Web search
description: Built-in web search tool.
sidebar:
  order: 4
---

`--enable-search` exposes a `web_search` tool to the model.

The built-in search and extraction tools use [strict tool calling](/mistral.rs/guides/agents/strict-tool-calling/) by default, so generated queries and URLs are constrained to the declared JSON Schema.

## Turning it on

```bash
mistralrs serve --enable-search -m <model>
```

The built-in backend uses DuckDuckGo (`https://html.duckduckgo.com/html/?q=...`). Up to 10 results are returned per query. Results pass through a readability-style extractor.

## Reranking

Retrieved results pass through an embedding-based reranker before reaching the model. To enable a reranker:

```bash
mistralrs serve --enable-search \
  --search-embedding-model embedding-gemma \
  -m <model>
```

`--search-embedding-model` accepts `embedding-gemma`. `--search-embedding-model` requires `--enable-search`.

## Per-request options

The OpenAI `web_search_options` field controls per-request behavior:

```json
{
  "model": "default",
  "messages": [{"role": "user", "content": "What happened at CES this year?"}],
  "web_search_options": {
    "search_context_size": "medium"
  }
}
```

Fields on `WebSearchOptions`:

- `search_context_size`: `low`, `medium` (default), `high`.
- `user_location`: optional location hint.
- `search_description`: optional description shown to the model.
- `extract_description`: optional description for content extraction.

## Custom search backends

The Python and Rust SDKs accept a `search_callback`. The callback receives a query string and returns a list of result dicts. Used for searching internal corpora.

Python:

```python
def my_search(query: str) -> list[dict]:
    return [
        {"title": "...", "description": "...", "url": "internal://...", "content": "..."},
        ...
    ]

runner = Runner(
    which=Which.Plain(model_id="Qwen/Qwen3-4B"),
    enable_search=True,
    search_callback=my_search,
)
```
