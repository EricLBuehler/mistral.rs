#!/usr/bin/env python3
"""Run HF Voxtral on bcn_weather.wav to see if the model produces text."""

import torch
import soundfile as sf
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

MODEL_ID = "mistralai/Voxtral-Mini-4B-Realtime-2602"

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID, dtype="float32")
model.eval()

audio_data, sr = sf.read("bcn_weather.wav")
if audio_data.ndim > 1:
    audio_data = audio_data.mean(axis=1)
print(f"Audio: {len(audio_data)} samples, {sr}Hz, {len(audio_data)/sr:.1f}s")

inputs = processor(audio=[audio_data], sampling_rate=sr, return_tensors="pt")
print(f"input_ids shape: {inputs['input_ids'].shape}")
print(f"input_features shape: {inputs['input_features'].shape}")
print(f"num_delay_tokens: {inputs.get('num_delay_tokens', 'N/A')}")

with torch.no_grad():
    output = model.generate(
        input_ids=inputs["input_ids"],
        input_features=inputs["input_features"],
        num_delay_tokens=inputs["num_delay_tokens"],
        max_new_tokens=2000,
        do_sample=False,
    )

prompt_len = inputs["input_ids"].shape[1]
gen_ids = output[0, prompt_len:].tolist()
print(f"Prompt len: {prompt_len}, Generated: {len(gen_ids)}")

# Count token types
pad_count = sum(1 for t in gen_ids if t == 32)
eos_count = sum(1 for t in gen_ids if t == 2)
text_tokens = [t for t in gen_ids if t not in (32, 1, 2)]
print(f"PAD tokens: {pad_count}, EOS tokens: {eos_count}, Text tokens: {len(text_tokens)}")

# Show non-PAD tokens with positions
non_pad = []
for i, tid in enumerate(gen_ids):
    if tid != 32:
        tok_str = processor.tokenizer.decode([tid])
        non_pad.append(f"pos={i+prompt_len}:{repr(tok_str)}(id={tid})")
print(f"Non-PAD tokens: {non_pad[:80]}")

# Decode text
decoded = processor.batch_decode(output, skip_special_tokens=True)
print(f"Decoded: {repr(decoded)}")
