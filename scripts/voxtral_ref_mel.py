#!/usr/bin/env python3
"""Generate reference mel spectrogram and filterbank using mistral_common for comparison."""

import numpy as np
import scipy.io.wavfile as wavfile
import mistral_common.audio as audio_mod

# Audio params from Voxtral config
SAMPLING_RATE = 16000
NUM_MEL_BINS = 128
HOP_LENGTH = 160
WINDOW_SIZE = 400  # n_fft
N_FREQS = WINDOW_SIZE // 2 + 1  # 201
GLOBAL_LOG_MEL_MAX = 1.5
FRAME_RATE = 12.5
N_LEFT_PAD_TOKENS = 32
N_RIGHT_PAD_TOKENS = 17

def samples_per_token():
    return int(SAMPLING_RATE / FRAME_RATE)  # 1280

def compute_mel_spectrogram(samples):
    """Compute mel spectrogram matching the reference implementation."""
    n_fft = WINDOW_SIZE
    hop = HOP_LENGTH
    n_freqs = n_fft // 2 + 1

    # Reference mel filterbank (Slaney scale + Slaney normalization)
    mel_filters = audio_mod.mel_filter_bank(
        num_frequency_bins=n_freqs,
        num_mel_bins=NUM_MEL_BINS,
        min_frequency=0.0,
        max_frequency=SAMPLING_RATE / 2.0,
        sampling_rate=SAMPLING_RATE,
    )
    # mel_filters shape: (n_freqs, num_mel_bins) = (201, 128)
    print(f"mel_filters shape: {mel_filters.shape}")
    np.save("/tmp/voxtral_ref_mel_filters.npy", mel_filters)

    # Hann window
    window = np.hanning(n_fft + 1)[:n_fft]  # periodic Hann

    num_frames = (len(samples) - n_fft) // hop + 1
    print(f"num_frames: {num_frames}")

    log_mel_floor = GLOBAL_LOG_MEL_MAX - 8.0

    mel_features = np.zeros((num_frames, NUM_MEL_BINS), dtype=np.float32)
    for frame_idx in range(num_frames):
        start = frame_idx * hop
        frame = samples[start:start + n_fft] * window

        # FFT
        fft_result = np.fft.rfft(frame, n=n_fft)
        power = np.abs(fft_result) ** 2  # shape: (n_freqs,)

        # Mel filterbank projection: power @ mel_filters
        mel_energies = power @ mel_filters  # (n_freqs,) @ (n_freqs, num_mel_bins) = (num_mel_bins,)

        # Log normalization
        log_val = np.log10(np.maximum(mel_energies, 1e-10))
        clamped = np.maximum(log_val, log_mel_floor)
        mel_features[frame_idx] = (clamped + 4.0) / 4.0

    return mel_features


def main():
    # Load audio
    sr, data = wavfile.read("/home/ericbuehler/mistral.rs/audio.wav")
    print(f"Audio: sr={sr}, shape={data.shape}, dtype={data.dtype}")

    # Convert to mono float32
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype == np.int16:
        samples = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        samples = data.astype(np.float32) / 2147483648.0
    else:
        samples = data.astype(np.float32)

    print(f"Mono samples: {len(samples)}, range=[{samples.min():.4f}, {samples.max():.4f}]")

    # Resample if needed
    if sr != SAMPLING_RATE:
        import soxr
        samples = soxr.resample(samples, sr, SAMPLING_RATE)
        print(f"Resampled to {SAMPLING_RATE}: {len(samples)} samples")

    # Save raw samples for Rust comparison
    np.save("/tmp/voxtral_ref_raw_samples.npy", samples)

    # Pad with silence
    spt = samples_per_token()
    left_pad = N_LEFT_PAD_TOKENS * spt
    right_pad = N_RIGHT_PAD_TOKENS * spt
    padded = np.zeros(left_pad + len(samples) + right_pad, dtype=np.float32)
    padded[left_pad:left_pad + len(samples)] = samples
    print(f"Padded: {len(padded)} samples (left={left_pad}, audio={len(samples)}, right={right_pad})")

    # Save padded samples
    np.save("/tmp/voxtral_ref_padded_samples.npy", padded)

    # Compute mel
    mel = compute_mel_spectrogram(padded)
    print(f"Mel shape: {mel.shape}")
    print(f"Mel stats: min={mel.min():.4f}, max={mel.max():.4f}, mean={mel.mean():.4f}")

    # Save mel
    np.save("/tmp/voxtral_ref_mel.npy", mel)

    # Also save just the filterbank for comparison
    mel_filters = audio_mod.mel_filter_bank(
        num_frequency_bins=WINDOW_SIZE // 2 + 1,
        num_mel_bins=NUM_MEL_BINS,
        min_frequency=0.0,
        max_frequency=SAMPLING_RATE / 2.0,
        sampling_rate=SAMPLING_RATE,
    )

    # Show first few filter center frequencies for sanity
    mel_min = audio_mod.hertz_to_mel(0.0)
    mel_max = audio_mod.hertz_to_mel(SAMPLING_RATE / 2.0)
    mel_freqs = np.linspace(mel_min, mel_max, NUM_MEL_BINS + 2)
    filter_freqs = audio_mod.mel_to_hertz(mel_freqs)
    print(f"Filter center freqs (first 10): {filter_freqs[:10]}")
    print(f"Filter center freqs (last 5): {filter_freqs[-5:]}")

    print("\nDone! Files saved:")
    print("  /tmp/voxtral_ref_raw_samples.npy")
    print("  /tmp/voxtral_ref_padded_samples.npy")
    print("  /tmp/voxtral_ref_mel_filters.npy")
    print("  /tmp/voxtral_ref_mel.npy")


if __name__ == "__main__":
    main()
