from mistralrs import Runner, Which, ChatCompletionRequest, Architecture
from json import dumps

runner = Runner(
    which=Which.Plain(
        model_id="microsoft/Phi-3.5-mini-instruct",
    ),
    num_device_layers=["500"],
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[{"role": "user", "content": "Tell me a short joke."}],
        max_tokens=30,
        temperature=0.1,
        grammar_type="regex",
        grammar=r"[0-9A-Z ]+",
    )
)
print(res.choices[0].message.content)
print(res.usage)
