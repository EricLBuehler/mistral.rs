---
title: Web search
description: Turn on the built-in web search tool and configure how it retrieves and ranks results.
sidebar:
  order: 3
---

When web search is enabled, mistral.rs exposes a `web_search` tool that retrieves pages from the internet and returns their text content. The model decides when to use it.

Search applies when the model needs current information (news, releases, prices), references to documents that do not fit in the prompt, or citations.

## Turning it on

```bash
mistralrs serve --enable-search -m <model>
```

The default backend is a no-key DuckDuckGo-compatible search. Result pages are fetched and parsed with a readability-style extractor.

## Reranking retrieved results

Retrieved results pass through an embedding-based reranker before reaching the model. Default reranker: `google/embeddinggemma-300m`, downloaded on first run and cached alongside other models.

To use a different reranker:

```bash
mistralrs serve --enable-search \
  --search-embedding-model <embedding-repo> \
  -m <model>
```

Models under 500M are the practical sweet spot for reranker latency without quality loss.

## Controlling context size

Web search is configured per-request via the OpenAI `web_search_options` field, which mistral.rs implements compatibly. Per-request options override server defaults:

```json
{
  "model": "default",
  "messages": [{"role": "user", "content": "What happened at CES this year?"}],
  "web_search_options": {
    "search_context_size": "medium"
  }
}
```

`search_context_size` is `low`, `medium`, or `high`:

- `low` — fewer results, shorter snippets. Fast.
- `medium` — default. Balanced.
- `high` — more results, longer snippets. Expensive but thorough.

## User location

For location-sensitive queries (local restaurants, timezone-correct event times):

```json
{
  "web_search_options": {
    "user_location": {
      "type": "approximate",
      "approximate": {
        "city": "San Francisco",
        "country": "US",
        "region": "California",
        "timezone": "America/Los_Angeles"
      }
    }
  }
}
```

The location is included in prompt context so the model can factor it into queries and answers. No user-identifying data is sent to the search backend.

## How the model decides when to search

The model receives the `web_search` tool schema alongside other enabled tools. The decision to search is the model's; there is no hardcoded rule.

In practice:

- Current-events questions reliably trigger searches.
- Well-known facts often do not, since the model has them in training data.
- Ambiguous cases depend on model calibration. Qwen3 and Gemma 4 use search reasonably; smaller or older models may over- or under-search.

To force or suppress search per request, use `tool_choice` (see [tool calling basics](/mistral.rs/guides/agents/tool-calling-basics/)).

## Custom search backends

To replace the built-in search, use the Python or Rust SDK's `search_callback`. The callback takes a query string and returns a list of results. Useful for searching internal documents or a customer knowledge base.

Python example:

```python
def my_search(query: str) -> list[dict]:
    # Custom retrieval logic
    return [
        {"url": "internal://...", "snippet": "...", "title": "..."},
        ...
    ]

runner = Runner(
    which=Which.Plain(model_id="Qwen/Qwen3-4B"),
    enable_search=True,
    search_callback=my_search,
)
```

The engine uses the callback in place of built-in search whenever the model requests web search.
