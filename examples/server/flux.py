from openai import OpenAI

client = OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

result = client.images.generate(
    model="flux",
    prompt="A majestic snow-capped mountain which soars above a valley. 4k, high quality.",
    n=1,
)
print(result.data[0].url)
