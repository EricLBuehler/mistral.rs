#!/usr/bin/env python3
"""Compare encoder/adapter outputs between HF and Rust."""

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
hf_mel = inputs["input_features"]
print(f"HF mel shape: {hf_mel.shape}")

# Run HF encoder
with torch.no_grad():
    # The encoder is model.audio_tower
    # audio_tower.embedder processes mel -> conv features
    # audio_tower.encoder is the transformer

    # Step 1: Conv embedder
    mel_input = hf_mel  # [1, 128, T]
    conv_out = model.audio_tower.embedder(mel_input)
    print(f"HF conv output shape: {conv_out.shape}")  # Should be [1, T', 1280]

    # Step 2: Full encoder with transformer layers
    encoder_out = model.audio_tower(mel_input)
    hf_encoder = encoder_out.last_hidden_state
    print(f"HF encoder output shape: {hf_encoder.shape}")
    print(f"HF encoder mean: {hf_encoder.mean():.6f}, std: {hf_encoder.std():.6f}")

    # Step 3: Adapter (projector)
    # The projector is model.multi_modal_projector
    hf_adapter = model.get_audio_features(
        input_features=hf_mel,
        return_dict=True
    ).pooler_output
    print(f"HF adapter (pooler) output shape: {hf_adapter.shape}")
    print(f"HF adapter mean: {hf_adapter.mean():.6f}, std: {hf_adapter.std():.6f}")

# Load Rust outputs
try:
    rust_mel = np.fromfile("rust_mel.bin", dtype=np.float32).reshape(1, 1496, 128)
    print(f"\nRust mel shape: {rust_mel.shape}")

    # Compare mel (Rust is [1,T,128], HF is [1,128,T])
    hf_mel_t = hf_mel.numpy().transpose(0, 2, 1)  # [1, T, 128]
    mel_diff = np.abs(rust_mel - hf_mel_t)
    print(f"Mel max abs diff: {mel_diff.max():.8f}")
    print(f"Mel mean abs diff: {mel_diff.mean():.8f}")

    # Cosine similarity
    a = rust_mel.flatten()
    b = hf_mel_t.flatten()
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    print(f"Mel cosine similarity: {cos_sim:.8f}")
except Exception as e:
    print(f"Could not load Rust mel: {e}")

try:
    rust_encoder = np.fromfile("rust_encoder_out.bin", dtype=np.float32)
    expected_size = hf_encoder.shape[0] * hf_encoder.shape[1] * hf_encoder.shape[2]
    print(f"\nRust encoder output size: {rust_encoder.size}, expected: {expected_size}")
    rust_encoder = rust_encoder.reshape(hf_encoder.shape)

    hf_enc_np = hf_encoder.numpy()
    enc_diff = np.abs(rust_encoder - hf_enc_np)
    print(f"Encoder max abs diff: {enc_diff.max():.6f}")
    print(f"Encoder mean abs diff: {enc_diff.mean():.6f}")

    a = rust_encoder.flatten()
    b = hf_enc_np.flatten()
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    print(f"Encoder cosine similarity: {cos_sim:.8f}")

    # Per-frame cosine similarity
    for i in [0, 100, 200, 400, 600, 747]:
        a = rust_encoder[0, i]
        b = hf_enc_np[0, i]
        cs = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        print(f"  Frame {i} cosine: {cs:.6f}, max diff: {np.abs(a-b).max():.6f}")
except Exception as e:
    print(f"Could not load Rust encoder output: {e}")

try:
    rust_adapter = np.fromfile("rust_adapter_out.bin", dtype=np.float32)
    expected_size = hf_adapter.shape[0] * hf_adapter.shape[1] * hf_adapter.shape[2]
    print(f"\nRust adapter output size: {rust_adapter.size}, expected: {expected_size}")
    rust_adapter = rust_adapter.reshape(hf_adapter.shape)

    hf_adp_np = hf_adapter.numpy()
    adp_diff = np.abs(rust_adapter - hf_adp_np)
    print(f"Adapter max abs diff: {adp_diff.max():.6f}")
    print(f"Adapter mean abs diff: {adp_diff.mean():.6f}")

    a = rust_adapter.flatten()
    b = hf_adp_np.flatten()
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    print(f"Adapter cosine similarity: {cos_sim:.8f}")

    # Per-position cosine similarity
    for i in [0, 10, 20, 50, 100, 150, 186]:
        a = rust_adapter[0, i]
        b = hf_adp_np[0, i]
        cs = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        print(f"  Position {i} cosine: {cs:.6f}, max diff: {np.abs(a-b).max():.6f}")
except Exception as e:
    print(f"Could not load Rust adapter output: {e}")
