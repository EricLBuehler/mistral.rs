# Mistral.rs Web Chat App

> **Deprecated:** The standalone `mistralrs-web-chat` binary is deprecated. Use `mistralrs serve --ui` instead for the same functionality.
>
> **Migration:**
> ```bash
> # Old
> cargo run --release --features cuda --bin mistralrs-web-chat -- --text-model Qwen/Qwen3-4B
>
> # New
> mistralrs serve --ui -m Qwen/Qwen3-4B
> ```
>
> The new built-in UI provides the same features and is accessible at `/ui` when running the server.

A minimal, fast, and modern web chat interface for [mistral.rs](https://github.com/EricBuehler/mistral.rs), supporting text, vision, and speech models with drag-and-drop image and file upload, markdown rendering, and multi-model selection.

<img src="../res/chat.gif" alt="Demonstration" />

---

## Features

- **Multi-model Support:** Choose from multiple loaded models. See [multi-model docs](../docs/multi_model/README.md) for advanced configuration.
- **Text-to-speech Model Support:** Generate speech from text prompts and download as WAV files.
- **Speech Input Model Support:** Upload audio inputs for multimodal models.
- **File Upload:** Upload and work with files.
- **Drag & Drop Image Upload:** Instantly preview and send images to vision models.
- **Markdown Output:** Responses are rendered in markdown, including code blocks.
- **Copy Button:** One-click copying for all code snippets.
- **Responsive UI:** Clean layout for desktop and mobile.
- **Generation Settings:** Configure temperature, top_p, top_k, max_tokens, and repetition penalty directly in the UI.
- **System Prompt:** Set a custom system prompt for all conversations.
- **Settings Persistence:** User settings are saved to localStorage and persist across sessions.
- **Web Search:** Optional web search integration (when enabled on server).
- **Keyboard Shortcuts:**
  - `Ctrl+Enter` (or `Cmd+Enter` on Mac) to send messages.
  - Button click also sends.

---

## Quickstart

1) Build the app.

> Note: choose the features based on [this guide](../README.md#supported-accelerators).

```bash
cargo run --release --features <specify feature(s) here> --bin mistralrs-web-chat -- \
  --text-model Qwen/Qwen3-4B \
  --vision-model google/gemma-3-4b-it \
  --speech-model nari-labs/Dia-1.6B
```

- At least one model is required (text, vision, or speech).
- Multiple `--text-model`, `--vision-model`, or `--speech-model` can be specified.
- `--port` is optional (defaults to 1234).

---

2) Access the app!

- Open http://localhost:1234 (or your chosen port).

---

## CLI Options

```
Options:
  --isq <ISQ>                    In-situ quantization to apply. Defaults to Q6K on CPU, AFQ6 on Metal.
                                 Options: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2K, Q3K, Q4K, Q5K, Q6K, Q8K,
                                          HQQ1, HQQ2, HQQ3, HQQ4, HQQ8, AFQ2, AFQ4, AFQ6, AFQ8
  --text-model <MODEL>           Text-only models (HuggingFace ID or local path). Can be repeated.
  --vision-model <MODEL>         Vision models (HuggingFace ID or local path). Can be repeated.
  --speech-model <MODEL>         Speech/TTS models (HuggingFace ID or local path). Can be repeated.
  --enable-search                Enable web search tool (requires embedding model)
  --search-embedding-model <M>   Built-in search embedding model (e.g., embedding_gemma)
  -p, --port <PORT>              Port to listen on (default: 1234)
  --host <HOST>                  IP address to serve on (default: 0.0.0.0)
  --cpu                          Use CPU only (disable GPU acceleration)
  --temperature <TEMP>           Default temperature for generation (0.0-2.0). Default: 0.7
  --top-p <TOP_P>                Default top_p for generation (0.0-1.0). Default: 0.9
  --top-k <TOP_K>                Default top_k for generation. Default: 40
  --max-tokens <MAX>             Default max tokens to generate. Default: 2048
  --repetition-penalty <PEN>     Default repetition penalty (1.0 = no penalty). Default: 1.1
  --system-prompt <PROMPT>       Default system prompt for all chats
  -h, --help                     Print help
  -V, --version                  Print version
```

### Examples

Basic usage with a text model:
```bash
cargo run --release --features cuda --bin mistralrs-web-chat -- \
  --text-model meta-llama/Llama-3.2-3B-Instruct
```

With custom generation defaults:
```bash
cargo run --release --features cuda --bin mistralrs-web-chat -- \
  --text-model Qwen/Qwen3-4B \
  --temperature 0.8 \
  --max-tokens 4096 \
  --system-prompt "You are a helpful coding assistant."
```

Multiple models with web search:
```bash
cargo run --release --features cuda --bin mistralrs-web-chat -- \
  --text-model Qwen/Qwen3-4B \
  --vision-model google/gemma-3-4b-it \
  --enable-search \
  --search-embedding-model embedding_gemma
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/ws` | WebSocket connection for streaming chat |
| GET | `/api/settings` | Get server default settings |
| GET | `/api/list_models` | List available models |
| POST | `/api/select_model` | Switch active model |
| GET | `/api/list_chats` | List saved chats |
| POST | `/api/new_chat` | Create new chat |
| POST | `/api/load_chat` | Load chat history |
| POST | `/api/delete_chat` | Delete chat |
| POST | `/api/rename_chat` | Rename chat |
| POST | `/api/upload_image` | Upload image (vision models) |
| POST | `/api/upload_text` | Upload text/code file |
| POST | `/api/upload_audio` | Upload audio file |
| POST | `/api/generate_speech` | Generate speech (TTS models) |

---

## WebSocket Message Format

Messages sent to the WebSocket can include generation parameters:

```json
{
  "content": "Your message here",
  "generation_params": {
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 50,
    "max_tokens": 2048,
    "repetition_penalty": 1.1
  },
  "web_search_options": {
    "search_context_size": "medium"
  }
}
```

To update the system prompt:
```json
{
  "set_system_prompt": "You are a helpful assistant."
}
```

---

## Development

### Frontend
- Edit static/index.html and supporting JS/CSS.
- Uses marked.js for markdown rendering.
- Settings stored in localStorage under `mistralrs_settings`.

### Backend
- See main.rs for server and model loading logic.
- Models are managed via CLI arguments; hot reloading is not supported.
- Generation parameters can be configured via CLI and overridden per-message.
