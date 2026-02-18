#!/usr/bin/env python3
"""Compare intermediate activations between HF Transformers Voxtral and Rust implementation."""

import numpy as np
import torch

def load_bin(path, shape):
    data = np.frombuffer(open(path, 'rb').read(), dtype=np.float32)
    return data.reshape(shape)

def compare(name, ref, rust):
    if ref.shape != rust.shape:
        print(f"  {name}: SHAPE MISMATCH ref={ref.shape} rust={rust.shape}")
        return float('inf')
    diff = np.abs(ref.flatten() - rust.flatten())
    cos = np.dot(ref.flatten(), rust.flatten()) / (np.linalg.norm(ref.flatten()) * np.linalg.norm(rust.flatten()) + 1e-10)
    print(f"  {name}: shape={ref.shape}")
    print(f"    ref  mean={ref.mean():.6f} std={ref.std():.6f}")
    print(f"    rust mean={rust.mean():.6f} std={rust.std():.6f}")
    print(f"    diff max={diff.max():.6f} mean={diff.mean():.6f} | cosine={cos:.6f}")

def save_bin(arr, path):
    with open(path, 'wb') as f:
        f.write(arr.astype(np.float32).tobytes())

def main():
    print("Loading HF model...")
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        'mistralai/Voxtral-Mini-4B-Realtime-2602', dtype='float32'
    )
    model.eval()

    # Use the HF processor mel (592 frames, [1, 128, 592])
    hf_mel = np.load('/tmp/voxtral_hf_proc_mel.npy')
    print(f"HF mel shape: {hf_mel.shape}")  # [128, 592]
    mel_tensor = torch.from_numpy(hf_mel).unsqueeze(0).float()  # [1, 128, 592]

    at = model.audio_tower
    emb = at.embedder

    print("\n--- HF Encoder step-by-step ---")
    with torch.no_grad():
        # Conv1 (causal: left_pad=2)
        x = mel_tensor
        left_pad1 = 2  # kernel=3, stride=1
        x_padded = torch.nn.functional.pad(x, (left_pad1, 0))
        x_conv1 = torch.nn.Conv1d.forward(emb.conv1, x_padded)
        x_conv1_gelu = torch.nn.functional.gelu(x_conv1)
        print(f"  conv1+gelu: {x_conv1_gelu.shape} mean={x_conv1_gelu.mean():.6f} std={x_conv1_gelu.std():.6f}")
        save_bin(x_conv1_gelu.numpy(), "/tmp/voxtral_hf_post_conv1.bin")

        # Conv2 (causal: left_pad=1)
        left_pad2 = 1  # kernel=3, stride=2
        x_padded2 = torch.nn.functional.pad(x_conv1_gelu, (left_pad2, 0))
        x_conv2 = torch.nn.Conv1d.forward(emb.conv2, x_padded2)
        x_conv2_gelu = torch.nn.functional.gelu(x_conv2)
        print(f"  conv2+gelu: {x_conv2_gelu.shape} mean={x_conv2_gelu.mean():.6f} std={x_conv2_gelu.std():.6f}")
        x_post_conv = x_conv2_gelu.transpose(1, 2)  # [B, T, dim]
        print(f"  post_conv: {x_post_conv.shape}")
        save_bin(x_post_conv.numpy(), "/tmp/voxtral_hf_post_conv.bin")

        # Full encoder
        encoder_out = at(mel_tensor)
        hidden = encoder_out.last_hidden_state
        print(f"  encoder_out: {hidden.shape} mean={hidden.mean():.6f} std={hidden.std():.6f}")
        save_bin(hidden.numpy(), "/tmp/voxtral_hf_encoder_out.bin")

        # Projector with reshape
        ds = model.config.downsample_factor
        enc_dim = model.config.audio_config.hidden_size
        hidden_rs = hidden.reshape(hidden.shape[0], -1, enc_dim * ds)
        proj_out = model.multi_modal_projector(hidden_rs)
        print(f"  projector: {proj_out.shape} mean={proj_out.mean():.6f} std={proj_out.std():.6f}")
        save_bin(proj_out.numpy(), "/tmp/voxtral_hf_audio_embeds.bin")

        # Text embeddings for [BOS=1, PAD=32*38]
        prompt_ids = torch.tensor([[1] + [32] * 38], dtype=torch.long)
        text_embeds = model.get_input_embeddings()(prompt_ids)
        print(f"  text_embeds: {text_embeds.shape} mean={text_embeds.mean():.6f} std={text_embeds.std():.6f}")
        save_bin(text_embeds.numpy(), "/tmp/voxtral_hf_text_embeds.bin")

        # Time embedding
        time_tensor = torch.full((1,), 6.0)
        t_cond = model.time_embedding(time_tensor)
        t_cond_expanded = t_cond[None, ...]
        print(f"  t_cond: {t_cond_expanded.shape} mean={t_cond_expanded.mean():.6f} std={t_cond_expanded.std():.6f}")
        save_bin(t_cond_expanded.numpy(), "/tmp/voxtral_hf_t_cond.bin")

    # Compare with Rust
    print("\n=== COMPARING WITH RUST ===")

    # Mel
    rust_mel = load_bin("/tmp/voxtral_rust_mel.bin", (592, 128))
    compare("mel", hf_mel.T, rust_mel)  # HF is [128, 592], Rust is [592, 128]

    # Encoder output
    compare("encoder_out",
        load_bin("/tmp/voxtral_hf_encoder_out.bin", (1, 296, 1280)),
        load_bin("/tmp/voxtral_rust_encoder_out.bin", (1, 296, 1280)))

    # Audio embeddings
    compare("audio_embeds",
        load_bin("/tmp/voxtral_hf_audio_embeds.bin", (1, 74, 3072)),
        load_bin("/tmp/voxtral_rust_audio_embeds.bin", (1, 74, 3072)))

    # Text embeddings
    compare("text_embeds",
        load_bin("/tmp/voxtral_hf_text_embeds.bin", (1, 39, 3072)),
        load_bin("/tmp/voxtral_rust_text_embeds.bin", (1, 39, 3072)))

    print("\nDone!")


if __name__ == "__main__":
    main()
