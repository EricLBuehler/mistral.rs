from mistralrs import (
    Runner,
    Which,
    ChatCompletionRequest,
    Architecture,
    WebSearchOptions,
)

runner = Runner(
    which=Which.Plain(
        model_id="NousResearch/Hermes-3-Llama-3.1-8B",
        arch=Architecture.Llama,
    ),
    enable_search=True,
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="mistral",
        messages=[
            {
                "role": "user",
                "content": "Can you show me some code using mistral.rs for running Llama 3.2 Vision?",
            }
        ],
        max_tokens=256,
        presence_penalty=1.0,
        top_p=0.1,
        temperature=0.1,
        web_search_options=WebSearchOptions(
            search_context_size=None, user_location=None
        ),
    )
)
print(res.choices[0].message.content)
print(res.usage)
