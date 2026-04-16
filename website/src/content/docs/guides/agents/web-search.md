---
title: Web search
description: Turn on the built-in web search tool and configure how it retrieves and ranks results.
sidebar:
  order: 3
---

With web search enabled, mistral.rs gives the model a `web_search` tool that retrieves pages from the internet and returns their text content. The model decides when to use it based on the prompt; you do not need to prompt for search behavior specifically.

Search is useful when the model needs current information (news, releases, prices), when answering a question requires reference to a specific document you cannot fit in the prompt, or when you want citations backing an answer.

## Turning it on

```bash
mistralrs serve --enable-search -m <model>
```

The default search backend is a no-key DuckDuckGo-compatible search. Result pages are fetched and their text extracted using a readability-style parser.

## Reranking retrieved results

Retrieved results go through an embedding-based reranker before being sent to the model. The default reranker is `google/embeddinggemma-300m`, which is downloaded on first run and cached alongside your other models.

To pick a different reranker:

```bash
mistralrs serve --enable-search \
  --search-embedding-model <embedding-repo> \
  -m <model>
```

A few embedding models make sense here. Small models (under 500M) are the sweet spot for reranking latency without sacrificing quality.

## Controlling context size

Web search is included in the request via the OpenAI `web_search_options` field. This is an extension OpenAI defined for their own API, and we implement it compatibly. Per-request options override the server defaults:

```json
{
  "model": "default",
  "messages": [{"role": "user", "content": "What happened at CES this year?"}],
  "web_search_options": {
    "search_context_size": "medium"
  }
}
```

`search_context_size` is one of `low`, `medium`, or `high`, trading off cost and completeness:

- `low`: Fewer results, shorter snippets. Fast.
- `medium`: The default. A good middle ground.
- `high`: More results, longer snippets. Expensive but thorough.

## User location

Some queries benefit from knowing where the user is (local restaurants, event times in the right timezone). Pass an approximate location in the request:

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

The location is included in the prompt context, so the model can factor it into both its queries and its final answer. Nothing is sent to the search backend as user-identifying data.

## How the model decides when to search

The model sees a tool schema for `web_search` along with any other tools enabled. Whether it chooses to use search is up to the model; there is no hardcoded rule.

In practice:

- Questions about current events or recent information reliably trigger searches.
- Questions about well-known facts often do not, because the model already has the answer in its training data.
- Ambiguous cases depend on the model's calibration. Qwen3 and Gemma 4 both use search reasonably when it is warranted; smaller or older models may over-search or under-search.

If you want to force or suppress search for a specific request, the `tool_choice` field (see [tool calling basics](/mistral.rs/guides/agents/tool-calling-basics/)) is the right mechanism.

## Custom search backends

To replace the built-in search entirely, use the Python or Rust SDK's `search_callback`. The callback is a function you define that takes a query string and returns a list of results. This is the path for organizations that want to search their own corpus (internal docs, a customer knowledge base) rather than the open web.

Example from Python:

```python
def my_search(query: str) -> list[dict]:
    # Your own retrieval logic
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

The engine will use your callback instead of its built-in search whenever the model requests a web search.
