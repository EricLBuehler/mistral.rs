# LFM3 native SDK check

Install the current mistralrs native binding in the Python environment you use for this check. It loads `LiquidAI/LFM3-Audio-2B` directly with Q8_0 and paged attention; no HTTP server is needed. Use a build with paged attention support, such as Metal, or add `--no-paged-attn` for a CPU build without it.

```bash
python scripts/lfm3/check_native_sdk.py
python scripts/lfm3/check_native_sdk.py --compressed
python scripts/lfm3/check_native_sdk.py --model LiquidAI/LFM3-Omni-2B --compressed
python scripts/lfm3/check_native_sdk.py --no-paged-attn
```

The script checks `Runner.generate_audio` PCM output, single-part standard `input_audio` transcription, and completion of both choices in `n_choices=2` PCM streaming through the regular and per-model request methods. `--compressed` also checks native MP3 output and requires FFmpeg on PATH. `--model` selects another compatible checkpoint, such as `LiquidAI/LFM3-Omni-2B`.

On macOS, an older SDK used to build the Python executable can cause Metal compilation to fail with `undeclared identifier bfloat`. Use Python built against a recent macOS SDK; the Python version itself is not the requirement. Alternatively, use a CPU build of the binding with `--no-paged-attn`. This is a host Metal toolchain issue.
