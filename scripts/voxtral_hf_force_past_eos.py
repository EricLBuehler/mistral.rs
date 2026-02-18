#!/usr/bin/env python3
"""Run HF Voxtral with EOS disabled to see what tokens appear after audio boundary."""

import torch
import soundfile as sf
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

MODEL_ID = "mistralai/Voxtral-Mini-4B-Realtime-2602"

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID, dtype="float32")
model.eval()

audio_data, sr = sf.read("audio.wav")
if audio_data.ndim > 1:
    audio_data = audio_data.mean(axis=1)
inputs = processor(audio=[audio_data], sampling_rate=sr, return_tensors="pt")

with torch.no_grad():
    output = model.generate(
        input_ids=inputs["input_ids"],
        input_features=inputs["input_features"],
        num_delay_tokens=inputs["num_delay_tokens"],
        max_new_tokens=300,
        do_sample=False,
        eos_token_id=999999,
    )

prompt_len = inputs["input_ids"].shape[1]
gen_ids = output[0, prompt_len:].tolist()
print(f"Total generated: {len(gen_ids)}")

text_parts = []
for i, tid in enumerate(gen_ids):
    tok_str = processor.tokenizer.decode([tid])
    if tid != 32:
        text_parts.append(f"pos={i+prompt_len}:{tok_str}(id={tid})")
print(f"Non-PAD tokens: {text_parts[:50]}")

text_ids = [t for t in gen_ids if t not in (32, 1, 2)]
if text_ids:
    print(f"Decoded (filtered): {processor.tokenizer.decode(text_ids)}")
