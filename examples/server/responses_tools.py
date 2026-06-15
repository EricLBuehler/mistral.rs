"""
Responses hosted tools with web search and code interpreter.

Start the server:
    mistralrs serve --agent -p 1234 -m Qwen/Qwen3-4B

Then run this script:
    python examples/server/responses_tools.py
"""

from openai import OpenAI

client = OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

search_response = client.responses.create(
    model="default",
    input="Find one current source about mistral.rs and summarize it in two sentences.",
    tools=[
        {
            "type": "web_search",
            "search_context_size": "low",
            "return_token_budget": "default",
        }
    ],
    tool_choice="required",
)

print("Search response:")
print(search_response.output_text)

code_response = client.responses.create(
    model="default",
    input="Use Python to calculate the first ten Fibonacci numbers.",
    tools=[{"type": "code_interpreter", "container": {"type": "auto"}}],
    tool_choice="required",
)

print("\nCode response:")
print(code_response.output_text)
