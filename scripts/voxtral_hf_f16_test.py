#!/usr/bin/env python3
"""Test if HF Voxtral in F16 also produces garbage."""

import torch
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
# Convert inputs to f16
inputs["input_features"] = inputs["input_features"].half()

print(f"Running HF in F16...")
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
pad_count = sum(1 for t in gen_ids if t == 32)
text_tokens = [t for t in gen_ids if t not in (32, 1, 2)]
print(f"Prompt: {prompt_len}, Generated: {len(gen_ids)}, PAD: {pad_count}, Text: {len(text_tokens)}")

decoded = processor.batch_decode(output, skip_special_tokens=True)
print(f"Decoded: {repr(decoded)}")

# Show non-PAD tokens
non_pad = []
for i, tid in enumerate(gen_ids):
    if tid != 32:
        tok_str = processor.tokenizer.decode([tid])
        non_pad.append(f"pos={i+prompt_len}:{repr(tok_str)}(id={tid})")
print(f"Non-PAD tokens: {non_pad[:30]}")
