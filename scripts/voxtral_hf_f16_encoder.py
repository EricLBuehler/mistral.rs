#!/usr/bin/env python3
"""Compare HF encoder output in F32 vs F16."""

import torch
import numpy as np
import soundfile as sf
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

MODEL_ID = "mistralai/Voxtral-Mini-4B-Realtime-2602"

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

audio_data, sr = sf.read("bcn_weather.wav")
if audio_data.ndim > 1:
    audio_data = audio_data.mean(axis=1)
inputs = processor(audio=[audio_data], sampling_rate=sr, return_tensors="pt")

# F32 encoder
model_f32 = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID, dtype="float32")
model_f32.eval()
with torch.no_grad():
    enc_f32 = model_f32.audio_tower(inputs["input_features"])
    if hasattr(enc_f32, 'last_hidden_state'):
        enc_f32 = enc_f32.last_hidden_state
    print(f"F32 encoder shape: {enc_f32.shape}, mean: {enc_f32.mean():.6f}, std: {enc_f32.std():.6f}")
del model_f32

# F16 encoder
model_f16 = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID, dtype="float16")
model_f16.eval()
with torch.no_grad():
    enc_f16 = model_f16.audio_tower(inputs["input_features"].half())
    if hasattr(enc_f16, 'last_hidden_state'):
        enc_f16 = enc_f16.last_hidden_state
    enc_f16 = enc_f16.float()  # cast back for comparison
    print(f"F16 encoder shape: {enc_f16.shape}, mean: {enc_f16.mean():.6f}, std: {enc_f16.std():.6f}")

# Compare
a = enc_f32.numpy().flatten()
b = enc_f16.numpy().flatten()
diff = np.abs(a - b)
cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
print(f"\nF32 vs F16 encoder:")
print(f"  Max abs diff: {diff.max():.6f}")
print(f"  Mean abs diff: {diff.mean():.6f}")
print(f"  Cosine similarity: {cos_sim:.8f}")

# Load Rust encoder output
rust_enc = np.fromfile("rust_encoder_out.bin", dtype=np.float32).reshape(1, 748, 1280)
c = rust_enc.flatten()
cs_f32_rust = np.dot(a, c) / (np.linalg.norm(a) * np.linalg.norm(c))
cs_f16_rust = np.dot(b, c) / (np.linalg.norm(b) * np.linalg.norm(c))
print(f"\nRust vs HF F32: cosine = {cs_f32_rust:.8f}")
print(f"Rust vs HF F16: cosine = {cs_f16_rust:.8f}")

# Also compare adapter outputs
with torch.no_grad():
    # Full audio features from F32
    model_f32_2 = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID, dtype="float32")
    model_f32_2.eval()
    adapt_f32 = model_f32_2.get_audio_features(input_features=inputs["input_features"], return_dict=True).pooler_output
    print(f"\nF32 adapter shape: {adapt_f32.shape}")

    model_f16_2 = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID, dtype="float16")
    model_f16_2.eval()
    adapt_f16 = model_f16_2.get_audio_features(input_features=inputs["input_features"].half(), return_dict=True).pooler_output.float()
    print(f"F16 adapter shape: {adapt_f16.shape}")

    a2 = adapt_f32.numpy().flatten()
    b2 = adapt_f16.numpy().flatten()
    cs_adapt = np.dot(a2, b2) / (np.linalg.norm(a2) * np.linalg.norm(b2))
    print(f"F32 vs F16 adapter cosine: {cs_adapt:.8f}")

    rust_adapt = np.fromfile("rust_adapter_out.bin", dtype=np.float32).reshape(adapt_f32.shape)
    c2 = rust_adapt.flatten()
    cs_f32_rust = np.dot(a2, c2) / (np.linalg.norm(a2) * np.linalg.norm(c2))
    cs_f16_rust = np.dot(b2, c2) / (np.linalg.norm(b2) * np.linalg.norm(c2))
    print(f"Rust adapter vs HF F32: cosine = {cs_f32_rust:.8f}")
    print(f"Rust adapter vs HF F16: cosine = {cs_f16_rust:.8f}")
