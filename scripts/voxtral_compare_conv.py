#!/usr/bin/env python3
"""Compare just the conv embedder output between HF and Rust."""

import torch
import numpy as np
import soundfile as sf
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

MODEL_ID = "mistralai/Voxtral-Mini-4B-Realtime-2602"

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID, dtype="float32")
model.eval()

audio_data, sr = sf.read("bcn_weather.wav")
if audio_data.ndim > 1:
    audio_data = audio_data.mean(axis=1)

inputs = processor(audio=[audio_data], sampling_rate=sr, return_tensors="pt")
hf_mel = inputs["input_features"]  # [1, 128, 1496]

# Explore model structure
print("Audio tower type:", type(model.audio_tower))
print("Audio tower children:")
for name, child in model.audio_tower.named_children():
    print(f"  {name}: {type(child)}")
    if hasattr(child, 'named_children'):
        for n2, c2 in child.named_children():
            print(f"    {n2}: {type(c2)}")
            if hasattr(c2, 'named_children'):
                for n3, c3 in c2.named_children():
                    print(f"      {n3}: {type(c3)}")

# Get embedder
embedder = model.audio_tower.embedder
print(f"\nEmbedder type: {type(embedder)}")
print(f"Embedder children:")
for name, child in embedder.named_children():
    print(f"  {name}: {type(child)}")
    if hasattr(child, 'named_children'):
        for n2, c2 in child.named_children():
            print(f"    {n2}: {type(c2)}")

with torch.no_grad():
    conv_out = embedder(hf_mel)
    print(f"\nHF conv output shape: {conv_out.shape}")
    print(f"HF conv mean: {conv_out.mean():.6f}, std: {conv_out.std():.6f}")

    # Save for comparison
    conv_np = conv_out.numpy().flatten()
    conv_np.astype(np.float32).tofile("hf_conv_out.bin")
    print(f"Saved hf_conv_out.bin ({conv_np.size} floats)")

    # Show first few values for comparison
    print(f"\nHF conv_out[0, :5, :5]:")
    print(conv_out[0, :5, :5])
