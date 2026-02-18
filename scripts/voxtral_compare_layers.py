#!/usr/bin/env python3
"""Compare encoder transformer layers between HF and Rust layer by layer.
Feeds HF conv output through HF encoder layers and saves each intermediate."""

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

with torch.no_grad():
    enc = model.audio_tower

    # Conv output
    conv_out = enc.embedder(hf_mel)  # [1, 748, 1280]
    print(f"Conv output: {conv_out.shape}")

    # Run each transformer layer
    hidden = conv_out
    for i, layer in enumerate(enc.layers):
        # Prepare inputs for the layer
        # We need position_ids and attention_mask
        # Let's check what the encoder forward does
        pass

    # Actually, let's hook into the actual encoder forward
    # to capture per-layer outputs
    layer_outputs = []

    def make_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            layer_outputs.append((layer_idx, h.detach().cpu().clone()))
        return hook

    hooks = []
    for i, layer in enumerate(enc.layers):
        h = layer.register_forward_hook(make_hook(i))
        hooks.append(h)

    # Run full encoder
    encoder_out = enc(hf_mel)
    if hasattr(encoder_out, 'last_hidden_state'):
        encoder_out = encoder_out.last_hidden_state

    # Remove hooks
    for h in hooks:
        h.remove()

    print(f"\nCaptured {len(layer_outputs)} layer outputs")

    # Load Rust encoder output
    rust_enc = np.fromfile("rust_encoder_out.bin", dtype=np.float32).reshape(1, 748, 1280)

    # Compare each layer output against Rust final output
    # (We only have Rust final output, not per-layer)
    # But we can see how the HF per-layer outputs evolve
    for idx, h in layer_outputs:
        h_np = h.numpy()
        r_diff = np.abs(h_np - rust_enc)
        a = h_np.flatten()
        b = rust_enc.flatten()
        cs = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        print(f"  Layer {idx:2d}: mean={h_np.mean():.6f}, std={h_np.std():.6f}, "
              f"vs_rust_final cosine={cs:.6f}")

    # Now compare HF final (after norm) vs Rust final
    hf_final = encoder_out.detach().cpu().numpy()
    a = hf_final.flatten()
    b = rust_enc.flatten()
    cs = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    print(f"\nHF final (after norm): cosine vs Rust final = {cs:.6f}")

    # Save per-layer outputs from HF so Rust can compare against them
    for idx, h in layer_outputs:
        h_np = h.numpy().flatten()
        h_np.astype(np.float32).tofile(f"hf_enc_layer_{idx}.bin")

    print("\nSaved HF layer outputs as hf_enc_layer_*.bin")
    print("To compare, add Rust per-layer debug output")
