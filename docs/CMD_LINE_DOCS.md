# Mistral.rs Command line docs
## `./mistralrs_server --help`

```bash
Fast and easy LLM serving.

Usage: mistralrs-server [OPTIONS] <COMMAND>

Commands:
  plain        Select a plain model
  x-lora       Select an X-LoRA architecture
  lora         Select a LoRA architecture
  gguf         Select a GGUF model
  x-lora-gguf  Select a GGUF model with X-LoRA
  lora-gguf    Select a GGUF model with LoRA
  ggml         Select a GGML model
  x-lora-ggml  Select a GGML model with X-LoRA
  lora-ggml    Select a GGML model with LoRA
  help         Print this message or the help of the given subcommand(s)

Options:
      --serve-ip <SERVE_IP>
          IP to serve on. Defaults to "0.0.0.0"
  -p, --port <PORT>
          Port to serve on
  -l, --log <LOG>
          Log all responses and requests to this file
  -t, --truncate-sequence
          If a sequence is larger than the maximum model length, truncate the number of tokens such that the sequence will fit at most the maximum length. If `max_tokens` is not specified in the request, space for 10 tokens will be reserved instead
      --max-seqs <MAX_SEQS>
          Maximum running sequences at any time. If the `tgt_non_granular_index` flag is set for X-LoRA models, this will be set to 1 [default: 16]
      --no-kv-cache
          Use no KV cache
  -c, --chat-template <CHAT_TEMPLATE>
          JINJA chat template with `messages`, `add_generation_prompt`, `bos_token`, `eos_token`, and `unk_token` as inputs. Used if the automatic deserialization fails. If this ends with `.json` (ie., it is a file) then that template is loaded
      --token-source <TOKEN_SOURCE>
          Source of the token for authentication. Can be in the formats: "literal:<value>", "env:<value>", "path:<value>", "cache" to use a cached token or "none" to use no token. Defaults to using a cached token [default: cache]
  -i, --interactive-mode
          Enter interactive mode instead of serving a chat server
      --prefix-cache-n <PREFIX_CACHE_N>
          Number of prefix caches to hold on the device. Other caches are evicted to the CPU based on a LRU strategy [default: 16]
      --prompt <PROMPT>
          Run a single prompt. This cannot be used with interactive mode
      --prompt-concurrency <PROMPT_CONCURRENCY>
          Requires --prompt. Number of prompt completions to run concurrently in prompt mode [default: 1]
      --prompt-max-tokens <PROMPT_MAX_TOKENS>
          Requires --prompt. Number of prompt tokens to generate [default: 128]
  -h, --help
          Print help
  -V, --version
          Print version
```

## For plain models
This is an example for a plain model. This can be seen by: `./mistralrs plain --help`

```bash
Select a plain model

Usage: mistralrs-server plain [OPTIONS] --model-id <MODEL_ID> --arch <ARCH>

Options:
  -m, --model-id <MODEL_ID>
          Model ID to load from
  -t, --tokenizer-json <TOKENIZER_JSON>
          Path to local tokenizer.json file. If this is specified it is used over any remote file
      --repeat-last-n <REPEAT_LAST_N>
          Control the application of repeat penalty for the last n tokens [default: 64]
  -a, --arch <ARCH>
          
  -h, --help
          Print help
```

## For quantized models
This is an example for quantized models. This is specifically for: `./mistralrs gguf --help`

```bash
Select a GGUF model

Usage: mistralrs-server gguf [OPTIONS] --tok-model-id <TOK_MODEL_ID> --quantized-model-id <QUANTIZED_MODEL_ID> --quantized-filename <QUANTIZED_FILENAME>

Options:
  -t, --tok-model-id <TOK_MODEL_ID>
          Model ID to load the tokenizer from
      --tokenizer-json <TOKENIZER_JSON>
          Path to local tokenizer.json file. If this is specified it is used over any remote file
  -m, --quantized-model-id <QUANTIZED_MODEL_ID>
          Quantized model ID to find the `quantized_filename`, only applicable if `quantized` is set. If it is set to an empty string then the quantized filename will be used as a path to the GGUF file
  -f, --quantized-filename <QUANTIZED_FILENAME>
          Quantized filename, only applicable if `quantized` is set
      --repeat-last-n <REPEAT_LAST_N>
          Control the application of repeat penalty for the last n tokens [default: 64]
  -h, --help
          Print help
```

## For X-LoRA and quantized models

This is an example which is roughly the same for all adapter + quantized models. This is specifically for: `./mistralrs_server x-lora-gguf --help`

```bash
Select a GGUF model with X-LoRA

Usage: mistralrs-server x-lora-gguf [OPTIONS] --quantized-model-id <QUANTIZED_MODEL_ID> --quantized-filename <QUANTIZED_FILENAME> --xlora-model-id <XLORA_MODEL_ID> --order <ORDER>

Options:
  -t, --tok-model-id <TOK_MODEL_ID>
          Model ID to load the tokenizer from
      --tokenizer-json <TOKENIZER_JSON>
          Path to local tokenizer.json file. If this is specified it is used over any remote file
  -m, --quantized-model-id <QUANTIZED_MODEL_ID>
          Quantized model ID to find the `quantized_filename`, only applicable if `quantized` is set
  -f, --quantized-filename <QUANTIZED_FILENAME>
          Quantized filename, only applicable if `quantized` is set
      --repeat-last-n <REPEAT_LAST_N>
          Control the application of repeat penalty for the last n tokens [default: 64]
  -x, --xlora-model-id <XLORA_MODEL_ID>
          Model ID to load X-LoRA from
  -o, --order <ORDER>
          Ordering JSON file
      --tgt-non-granular-index <TGT_NON_GRANULAR_INDEX>
          Index of completion tokens to generate scalings up until. If this is 1, then there will be one completion token generated before it is cached. This makes the maximum running sequences 1
  -h, --help
          Print help
```