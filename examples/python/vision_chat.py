import argparse
from mistralrs import Runner, Which, ChatCompletionRequest, VisionArchitecture

parser = argparse.ArgumentParser(description="Vision model chat example")
parser.add_argument("--model-id", required=True, help="HuggingFace model id")
parser.add_argument("--arch", required=True, help="VisionArchitecture name")
parser.add_argument(
    "--image-url",
    default="https://www.nhmagazine.com/content/uploads/2019/05/mtwashingtonFranconia-2-19-18-108-Edit-Edit.jpg",
)
args = parser.parse_args()

runner = Runner(
    which=Which.VisionPlain(
        model_id=args.model_id,
        arch=VisionArchitecture[args.arch],
    ),
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model=args.arch.lower(),
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": args.image_url},
                    },
                    {
                        "type": "text",
                        "text": "<|image_1|>\nWhat is shown in this image? Write a detailed response analyzing the scene.",
                    },
                ],
            }
        ],
        max_tokens=256,
        presence_penalty=1.0,
        top_p=0.1,
        temperature=0.1,
    )
)
print(res.choices[0].message.content)
print(res.usage)
