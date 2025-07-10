from mistralrs import (
    Runner,
    Which,
    ChatCompletionRequest,
    Architecture,
    WebSearchOptions,
)
import os


def local_search(query: str):
    results = []
    for root, _, files in os.walk("."):
        for f in files:
            if query in f:
                path = os.path.join(root, f)
                try:
                    content = open(path).read()
                except Exception:
                    content = ""
                results.append(
                    {
                        "title": f,
                        "description": path,
                        "url": path,
                        "content": content,
                    }
                )
    results.sort(key=lambda r: r["title"], reverse=True)
    return results


runner = Runner(
    which=Which.Plain(
        model_id="NousResearch/Hermes-3-Llama-3.1-8B",
        arch=Architecture.Llama,
    ),
    enable_search=True,
    search_callback=local_search,
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[{"role": "user", "content": "Where is Cargo.toml in this repo?"}],
        max_tokens=64,
        web_search_options=WebSearchOptions(
            search_description="Local filesystem search"
        ),
    )
)
print(res.choices[0].message.content)
