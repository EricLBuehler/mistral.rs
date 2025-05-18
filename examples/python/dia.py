from mistralrs import (
    Runner,
    Which,
    SpeechLoaderType,
)
from pathlib import Path
import wave, struct

# text_to_speak = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face."
text_to_speak = "[S1] mistral r s is a local LLM inference engine. [S2] You can run text and vision models, and also image generation and speech generation. [S1] There is agentic web search, tool calling, and a convenient Python API. [S2] Check it out on github."

runner = Runner(
    which=Which.Speech(
        model_id="nari-labs/Dia-1.6B",
        arch=SpeechLoaderType.Dia,
    ),
)

res = runner.generate_speech(text_to_speak)
print(res.choices[0].url)

pcm_data = res.pcm  # list of floats between -1.0 and 1.0
output_path = Path("output.wav")

# convert floats to 16-bit PCM ints
pcm_ints = [int(max(-32768, min(32767, int(sample * 32767)))) for sample in pcm_data]
with wave.open(output_path, "wb") as wf:
    wf.setnchannels(res.channels)  # mono
    wf.setsampwidth(2)  # 2 bytes per sample (16-bit)
    wf.setframerate(res.rate)  # set sample rate (adjust if needed)
    wf.writeframes(b"".join(struct.pack("<h", s) for s in pcm_ints))

print(f"WAV audio written to {output_path.resolve()}")
