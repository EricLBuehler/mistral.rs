#!/usr/bin/env python3
"""Save HF adapter output for injection into Rust decoder."""

import torch
import numpy as np
import soundfile as sf
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

MODEL_ID = "mistralai/Voxtral-Mini-4B-Realtime-2602"

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID, dtype="float16")
model.eval()

audio_data, sr = sf.read("bcn_weather.wav")
if audio_data.ndim > 1:
    audio_data = audio_data.mean(axis=1)

inputs = processor(audio=[audio_data], sampling_rate=sr, return_tensors="pt")
hf_mel = inputs["input_features"].half()

with torch.no_grad():
    hf_adapter = model.get_audio_features(
        input_features=hf_mel,
        return_dict=True
    ).pooler_output  # [1, 187, 3072] in f16

print(f"HF adapter shape: {hf_adapter.shape}, dtype: {hf_adapter.dtype}")
print(f"HF adapter mean: {hf_adapter.float().mean():.6f}")

# Save as f16 binary (matching our model's dtype)
adapter_np = hf_adapter.cpu().numpy()
# Convert f16 to raw bytes
adapter_bytes = adapter_np.tobytes()
with open("hf_adapter_f16.bin", "wb") as f:
    f.write(adapter_bytes)
print(f"Saved hf_adapter_f16.bin ({len(adapter_bytes)} bytes)")

# Also save shape info
print(f"Shape: {list(hf_adapter.shape)}")
