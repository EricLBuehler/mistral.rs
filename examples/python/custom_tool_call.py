import json
import os
from mistralrs import (
    Runner,
    Which,
    ChatCompletionRequest,
    Architecture,
    ToolChoice,
)


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


def tool_cb(name: str, args: dict) -> str:
    if name == "local_search":
        return json.dumps(local_search(args.get("query", "")))
    return ""


schema = json.dumps(
    {
        "type": "function",
        "function": {
            "name": "local_search",
            "description": "Local filesystem search",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
)

runner = Runner(
    which=Which.Plain(
        model_id="NousResearch/Hermes-3-Llama-3.1-8B", arch=Architecture.Llama
    ),
    tool_callbacks={"local_search": tool_cb},
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[{"role": "user", "content": "Where is Cargo.toml in this repo?"}],
        max_tokens=64,
        tool_schemas=[schema],
        tool_choice=ToolChoice.Auto,
    )
)
print(res.choices[0].message.content)
