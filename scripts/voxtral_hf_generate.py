#!/usr/bin/env python3
"""Run HF Transformers Voxtral generate() to verify expected token output pattern.

Counts how many STREAMING_PAD tokens appear before real text to validate
whether the Rust implementation's behavior is correct.
"""

import sys
import torch
import numpy as np
import soundfile as sf

AUDIO_PATH = "audio.wav"
MODEL_ID = "mistralai/Voxtral-Mini-4B-Realtime-2602"
MAX_NEW_TOKENS = 300
STREAMING_PAD_ID = 32


def main():
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    print(f"Loading processor from {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    print(f"Loading model from {MODEL_ID} (float32, CPU)...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32
    )
    model.eval()

    print(f"Reading audio from {AUDIO_PATH}...")
    audio_data, sr = sf.read(AUDIO_PATH)
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    print(f"  samples={len(audio_data)}, sr={sr}")

    # Process audio through HF processor
    print("Processing audio through HF processor...")
    inputs = processor(
        audio=[audio_data],
        sampling_rate=sr,
        return_tensors="pt",
    )
    print(f"  input_features shape: {inputs['input_features'].shape}")
    print(f"  input_ids shape: {inputs['input_ids'].shape}")
    print(f"  input_ids: {inputs['input_ids'][0].tolist()}")

    # Run generate
    print(f"\nRunning generate(max_new_tokens={MAX_NEW_TOKENS})...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    # Analyze output
    prompt_len = inputs['input_ids'].shape[1]
    generated_ids = output_ids[0, prompt_len:].tolist()
    all_ids = output_ids[0].tolist()

    print(f"\nPrompt tokens ({prompt_len}): {all_ids[:prompt_len]}")
    print(f"Generated tokens ({len(generated_ids)}): {generated_ids}")

    # Count leading STREAMING_PAD tokens
    n_pad = 0
    for tid in generated_ids:
        if tid == STREAMING_PAD_ID:
            n_pad += 1
        else:
            break

    print(f"\n=== RESULTS ===")
    print(f"Leading STREAMING_PAD tokens: {n_pad}")
    print(f"Total generated tokens: {len(generated_ids)}")

    # Decode the non-pad part
    text_ids = [t for t in generated_ids if t != STREAMING_PAD_ID]
    if text_ids:
        decoded = processor.tokenizer.decode(text_ids, skip_special_tokens=True)
        print(f"Decoded text: {decoded!r}")
    else:
        print("No text tokens generated (all STREAMING_PAD)")

    # Show token-by-token with decoding
    print(f"\n=== TOKEN-BY-TOKEN ===")
    for i, tid in enumerate(generated_ids):
        tok_str = processor.tokenizer.decode([tid])
        marker = " <-- first non-PAD" if i == n_pad and tid != STREAMING_PAD_ID else ""
        print(f"  step {i:3d}: id={tid:5d} {tok_str!r}{marker}")
        if i > n_pad + 20:
            print(f"  ... ({len(generated_ids) - i - 1} more tokens)")
            break


if __name__ == "__main__":
    main()
