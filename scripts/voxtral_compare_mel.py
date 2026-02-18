#!/usr/bin/env python3
"""Compare mel spectrogram from HF vs Rust (saved as binary)."""

import torch
import numpy as np
import soundfile as sf
from transformers import AutoProcessor

MODEL_ID = "mistralai/Voxtral-Mini-4B-Realtime-2602"

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

audio_data, sr = sf.read("bcn_weather.wav")
if audio_data.ndim > 1:
    audio_data = audio_data.mean(axis=1)

inputs = processor(audio=[audio_data], sampling_rate=sr, return_tensors="pt")
hf_mel = inputs["input_features"]
print(f"HF mel shape: {hf_mel.shape}")  # [1, 128, T]
print(f"HF mel dtype: {hf_mel.dtype}")
print(f"HF mel range: [{hf_mel.min():.6f}, {hf_mel.max():.6f}]")
print(f"HF mel mean: {hf_mel.mean():.6f}, std: {hf_mel.std():.6f}")

# Save HF mel as binary for Rust comparison
hf_mel_np = hf_mel.numpy().flatten()
hf_mel_np.astype(np.float32).tofile("hf_mel.bin")
print(f"Saved hf_mel.bin ({len(hf_mel_np)} floats)")

# Also save transposed version [1, T, 128] (how Rust processes it)
hf_mel_t = hf_mel.permute(0, 2, 1)  # [1, T, 128]
print(f"HF mel transposed shape: {hf_mel_t.shape}")
hf_mel_t_np = hf_mel_t.numpy().flatten()
hf_mel_t_np.astype(np.float32).tofile("hf_mel_transposed.bin")
print(f"Saved hf_mel_transposed.bin ({len(hf_mel_t_np)} floats)")

# Show first few values
print(f"\nHF mel[0, :5, :5] (first 5 mel bins, first 5 frames):")
print(hf_mel[0, :5, :5])
print(f"\nHF mel transposed[0, :5, :5] (first 5 frames, first 5 mel bins):")
print(hf_mel_t[0, :5, :5])
