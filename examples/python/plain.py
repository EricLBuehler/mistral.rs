import argparse
from mistralrs import Runner, Which, ChatCompletionRequest, Architecture

parser = argparse.ArgumentParser(description="Text model chat example")
parser.add_argument("--model-id", required=True, help="HuggingFace model id")
parser.add_argument("--arch", required=True, help="Architecture name")
args = parser.parse_args()

runner = Runner(
    which=Which.Plain(
        model_id=args.model_id,
        arch=Architecture[args.arch],
    ),
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model=args.arch.lower(),
        messages=[
            {"role": "user", "content": "Tell me a story about the Rust type system."}
        ],
        max_tokens=256,
        presence_penalty=1.0,
        top_p=0.1,
        temperature=0.1,
    )
)
print(res.choices[0].message.content)
print(res.usage)
