import openai
from datetime import datetime

Runs = 4

ENDPOINT = "http://localhost:1234/v1/"


def request(stream: bool):
    client = openai.Client(api_key="foobar", base_url=ENDPOINT)
    return client.chat.completions.create(
        model="mistral",
        messages=[
            {
                "role": "user",
                "content": "What is the meaning of life? Write a long story.",
            }
        ],
        stream=stream,
        max_tokens=400,
        temperature=0.0,
    )


def run():
    for run in range(Runs):
        print("\nCompletion: ")
        print("=" * 15)

        now = datetime.now()
        request(stream=False)
        finished = datetime.now()

        print(f"Duration: {finished-now}")

        print("\nStreaming: ")
        print("=" * 15)

        now = datetime.now()
        stream = request(stream=True)
        for _message in stream:
            pass
        finished = datetime.now()

        print(f"Duration: {finished-now}")


if __name__ == "__main__":
    run()
