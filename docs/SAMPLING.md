# Sampling and penalty techniques in mistral.rs

mistral.rs supports a comprehensive set of sampling and penalty techniques to control text generation. These can be configured via the HTTP API, Python SDK, or Rust SDK.

## Temperature

Controls the randomness of token selection. Lower values make output more deterministic, higher values increase creativity and randomness.

- **Range**: 0.0 to 2.0 (typically 0.0 to 1.0)
- **Default**: Model-dependent, usually around 0.7
- **Effect**: At 0.0, always selects the most likely token (greedy). At higher values, sampling becomes more diverse.

## Top K

Limits token selection to the K most likely tokens.

- **Range**: 1 to vocabulary size
- **Effect**: Lower values restrict choices to only the most probable tokens, reducing randomness.

## Top P (Nucleus Sampling)

Limits token selection to the smallest set of tokens whose cumulative probability exceeds P.

- **Range**: 0.0 to 1.0
- **Effect**: At 0.1, only tokens comprising the top 10% probability mass are considered. More adaptive than Top K as it adjusts based on the probability distribution.

## Min P

Filters out tokens with probability less than `min_p * max_probability`.

- **Range**: 0.0 to 1.0
- **Effect**: Removes low-probability tokens relative to the most likely token. Useful for preventing unlikely tokens from being selected.

## Stop Sequences

Strings that, when generated, cause generation to stop immediately.

- **Type**: Array of strings
- **Effect**: Generation terminates as soon as any stop sequence is produced. Useful for controlling output boundaries.

## Repetition Penalty

Applies a multiplicative penalty to tokens that have already appeared in the context.

- **Range**: Typically 1.0 to 2.0
- **Effect**: Values > 1.0 make repeated tokens less likely. This is distinct from frequency and presence penalties.

## Frequency Penalty

Penalizes tokens based on how many times they've appeared in the generated text so far.

- **Range**: -2.0 to 2.0
- **Effect**: Positive values reduce repetition proportionally to token frequency. Negative values encourage repetition.

## Presence Penalty

Penalizes tokens that have appeared at least once in the generated text.

- **Range**: -2.0 to 2.0
- **Effect**: Positive values discourage any repetition (binary penalty). Negative values encourage reusing tokens.

## DRY (Don't Repeat Yourself) Penalty

An advanced anti-repetition technique that detects and penalizes repeated sequences of tokens, not just individual tokens. See the [original implementation](https://github.com/oobabooga/text-generation-webui/pull/5677) for details.

### DRY Parameters

- **`dry_multiplier`**: Controls the strength of the penalty. Higher values more strongly discourage repetition.
- **`dry_base`**: Base value for the exponential penalty calculation.
- **`dry_allowed_length`**: Minimum sequence length before the penalty applies. Sequences shorter than this are not penalized.
- **`dry_sequence_breakers`**: Array of tokens (like newlines, punctuation) that reset the sequence tracking. When these tokens appear, the DRY penalty starts fresh.

### Example DRY Configuration (HTTP API)

```json
{
  "dry_multiplier": 0.8,
  "dry_base": 1.75,
  "dry_allowed_length": 2,
  "dry_sequence_breakers": ["\n", ".", "!", "?", ";"]
}
```

## API Usage

All sampling parameters can be set in API requests:

### HTTP API
```json
{
  "model": "default",
  "messages": [...],
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 40,
  "min_p": 0.05,
  "repetition_penalty": 1.1,
  "frequency_penalty": 0.5,
  "presence_penalty": 0.5,
  "stop": ["END", "\n\n"],
  "dry_multiplier": 0.8,
  "dry_base": 1.75,
  "dry_allowed_length": 2,
  "dry_sequence_breakers": ["\n"]
}
```

### Python SDK
```python
response = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[...],
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        min_p=0.05,
        repetition_penalty=1.1,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        stop_seqs=["END", "\n\n"],
        dry_multiplier=0.8,
        dry_base=1.75,
        dry_allowed_length=2,
        dry_sequence_breakers=["\n"],
    )
)
```

Please suggest more sampling techniques by [raising an issue](https://github.com/EricLBuehler/mistral.rs/issues)!
