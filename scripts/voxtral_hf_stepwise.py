#!/usr/bin/env python3
"""Step-by-step HF Voxtral forward to see token predictions at each step.

Instead of full generate() (slow on CPU), manually process each step.
Demonstrates how the model handles the encoder chunking + token prediction.
"""

import torch
import soundfile as sf
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

AUDIO_PATH = "audio.wav"
MODEL_ID = "mistralai/Voxtral-Mini-4B-Realtime-2602"
STREAMING_PAD_ID = 32
NUM_STEPS = 300

def main():
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID, dtype="float32")
    model.eval()

    audio_data, sr = sf.read(AUDIO_PATH)
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    inputs = processor(audio=[audio_data], sampling_rate=sr, return_tensors="pt")
    num_delay_tokens = inputs["num_delay_tokens"]
    input_features = inputs["input_features"]
    input_ids = inputs["input_ids"]

    print(f"input_ids: {input_ids[0].tolist()}")
    print(f"input_features shape: {input_features.shape}")
    print(f"num_delay_tokens: {num_delay_tokens}")

    # Pre-compute conv output for chunked encoder
    with torch.no_grad():
        at = model.audio_tower
        # Run conv layers on full mel
        mel = input_features  # [1, 128, 592]
        emb = at.embedder

        # Conv layers (replicate the embedder forward but capture conv output)
        x = mel
        for conv_layer in emb.conv_layers:
            x = conv_layer(x)
        conv_output = x.transpose(1, 2)  # [1, T_conv, dim]
        print(f"conv_output shape: {conv_output.shape}")

    # Method: use model.generate() but just capture the output
    # Actually, let's just loop manually
    generated_ids = []
    n_pad = 0
    first_text_step = -1

    with torch.no_grad():
        # Use generate with max_new_tokens=NUM_STEPS
        print(f"\nRunning generate(max_new_tokens={NUM_STEPS})...")
        output = model.generate(
            input_ids=input_ids,
            input_features=input_features,
            num_delay_tokens=num_delay_tokens,
            max_new_tokens=NUM_STEPS,
            do_sample=False,
        )

    prompt_len = input_ids.shape[1]
    gen_ids = output[0, prompt_len:].tolist()

    print(f"\nTotal generated tokens: {len(gen_ids)}")

    for i, tid in enumerate(gen_ids):
        tok_str = processor.tokenizer.decode([tid])
        if tid == STREAMING_PAD_ID:
            n_pad += 1
        elif first_text_step < 0:
            first_text_step = i

        if i < 10 or (first_text_step >= 0 and i >= first_text_step - 2 and i <= first_text_step + 20) or i >= len(gen_ids) - 5:
            marker = ""
            if i == first_text_step:
                marker = " <--- FIRST TEXT TOKEN"
            print(f"  step {i:3d}: id={tid:5d} {tok_str!r}{marker}")

    print(f"\n=== SUMMARY ===")
    print(f"Leading STREAMING_PAD tokens: {n_pad if first_text_step >= 0 else len(gen_ids)}")
    print(f"First text token at step: {first_text_step}")
    print(f"Total tokens: {len(gen_ids)}")

    text_ids = [t for t in gen_ids if t not in (STREAMING_PAD_ID, 1, 2)]
    if text_ids:
        decoded = processor.tokenizer.decode(text_ids, skip_special_tokens=True)
        print(f"Decoded text: {decoded!r}")
    else:
        print("No text tokens generated")


if __name__ == "__main__":
    main()
