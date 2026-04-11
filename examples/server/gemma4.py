from openai import OpenAI

client = OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

# Image input example for Gemma 4
# Start the server with: mistralrs serve -m google/gemma-4-E4B-it

completion = client.chat.completions.create(
    model="default",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://www.nhmagazine.com/content/uploads/2019/05/mtwashingtonFranconia-2-19-18-108-Edit-Edit.jpg"
                    },
                },
                {
                    "type": "text",
                    "text": "What is this?",
                },
            ],
        },
    ],
    max_tokens=256,
    frequency_penalty=1.0,
    top_p=0.1,
    temperature=0,
)
print(completion.choices[0].message.content)
