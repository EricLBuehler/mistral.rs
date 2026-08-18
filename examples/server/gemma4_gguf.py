"""Send image and audio input to a Gemma 4 GGUF model.

Start the server with:

```bash
mistralrs serve -m unsloth/gemma-4-E4B-it-GGUF --quant 4
```
"""

from openai import OpenAI


client = OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

IMAGE_URL = (
    "https://raw.githubusercontent.com/EricLBuehler/mistral.rs/master/res/banner.png"
)
AUDIO_URL = "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/apps/sample-data/journal1.wav"

completion = client.chat.completions.create(
    model="default",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": IMAGE_URL}},
                {
                    "type": "text",
                    "text": "Describe the image, then transcribe the audio.",
                },
                {"type": "audio_url", "audio_url": {"url": AUDIO_URL}},
            ],
        }
    ],
    max_tokens=256,
)

print(completion.choices[0].message.content)
