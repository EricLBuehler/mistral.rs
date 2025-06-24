# Mistral.rs Web Chat App

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
  --speech-model nari-labs/Dia-1.6B \
```

- At least one model is required (text, vision, or speech).
- Multiple `--text-model`, `--vision-model`, or `--speech-model` can be specified.
- `--port` is optional (defaults to 8080).

---

2) Access the app!

- Open http://localhost:8080 (or your chosen port).

---

## Development

### Frontend
- Edit static/index.html and supporting JS/CSS.
- Uses marked.js for markdown rendering.

### Backend
- See main.rs for server and model loading logic.
- Models are managed via CLI arguments; hot reloading is not supported.
