from openai import OpenAI

client = OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

result = client.images.generate(
    model="flux",
    prompt='A rusty robot walking on a beach holding a small torch, the robot has the word "rust" written on it, high quality, 4k',
    n=1,
)
print(result.data[0].url)
