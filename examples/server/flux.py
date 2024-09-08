from openai import OpenAI

client = OpenAI()

result = client.images.generate(
    model="flux",
    prompt='A rusty robot walking on a beach holding a small torch, the robot has the word "rust" written on it, high quality, 4k',
    n=1,
)
print(result.data[0].url)
