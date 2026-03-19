from openai import OpenAI


# Point the client to the locally running server
client = OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

# Remote assets – feel free to swap for anything else
IMAGE_URL = "https://www.allaboutbirds.org/guide/assets/og/528129121-1200px.jpg"
AUDIO_URL = "https://upload.wikimedia.org/wikipedia/commons/4/42/Bird_singing.ogg"

completion = client.chat.completions.create(
    model="default",
    messages=[
        {
            "role": "user",
            "content": [
                {  # Audio clip
                    "type": "audio_url",
                    "audio_url": {"url": AUDIO_URL},
                },
                {  # Image
                    "type": "image_url",
                    "image_url": {"url": IMAGE_URL},
                },
                {
                    "type": "text",
                    "text": "Describe in detail what is happening, referencing both what you hear and what you see.",
                },
            ],
        }
    ],
    max_tokens=256,
    temperature=0.2,
    top_p=0.9,
)

print(completion.choices[0].message.content)
