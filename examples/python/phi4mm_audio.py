from mistralrs import Runner, Which, ChatCompletionRequest, VisionArchitecture


# Choose a Vision model that supports both modalities
runner = Runner(
    which=Which.VisionPlain(
        model_id="microsoft/Phi-4-multimodal-instruct",
        arch=VisionArchitecture.Phi4MM,
    ),
)

# Remote media assets (swap for anything you like)
IMAGE_URL = "https://www.allaboutbirds.org/guide/assets/og/528129121-1200px.jpg"
AUDIO_URL = "https://upload.wikimedia.org/wikipedia/commons/4/42/Bird_singing.ogg"


response = runner.send_chat_completion_request(
    ChatCompletionRequest(
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
)

print(response.choices[0].message.content)
