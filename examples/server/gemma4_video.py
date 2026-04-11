from openai import OpenAI

client = OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

# Video input example for Gemma 4
# Start the server with: mistralrs serve -m google/gemma-4-E4B-it

completion = client.chat.completions.create(
    model="default",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {"url": "https://example.com/sample_video.mp4"},
                },
                {
                    "type": "text",
                    "text": "Describe what happens in this video.",
                },
            ],
        }
    ],
    max_tokens=512,
)
print(completion.choices[0].message.content)
