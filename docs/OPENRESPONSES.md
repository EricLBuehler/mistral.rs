# OpenResponses API

mistral.rs supports the [OpenResponses API specification](https://www.openresponses.org/specification).

## Endpoints

- `POST /v1/responses` - Create a response
- `GET /v1/responses/{id}` - Retrieve a response
- `DELETE /v1/responses/{id}` - Delete a response
- `POST /v1/responses/{id}/cancel` - Cancel a background response

## Unsupported Parameters

The following parameters are accepted for API compatibility but will return errors if set to non-default values:

| Parameter | Behavior |
|-----------|----------|
| `parallel_tool_calls` | Only `true` or omitted is supported; `false` returns an error |
| `max_tool_calls` | Not supported; setting any value returns an error |

## mistral.rs Extensions

These additional parameters are available beyond the spec:

- `stop` - Stop sequences
- `repetition_penalty` - Token repetition penalty
- `top_k` - Top-k sampling
- `grammar` - Constrained generation grammar
- `min_p` - Min-p sampling
- `dry_multiplier`, `dry_base`, `dry_allowed_length`, `dry_sequence_breakers` - DRY sampling
- `web_search_options` - Web search integration

See [HTTP.md](HTTP.md) for usage examples.
