import asyncio
from pathlib import Path
from openai import AsyncOpenAI

# Initialize the asynchronous OpenAI client
client = AsyncOpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

# List of texts to convert to speech
texts_to_speak = [
    "[S1] mistral r s is a local LLM inference engine. [S2] You can run text and vision models, and also image generation and speech generation. [S1] There is agentic web search, tool calling, and a convenient Python API. [S2] Check it out on github.",
    "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on GitHub or Hugging Face.",
    # Add more texts as needed
]

# Asynchronous function to generate speech
async def generate_speech(text, index):
    response = await client.audio.speech.create(
        model="dia", voice="N/A", input=text, response_format="wav"
    )
    output_path = Path(f"output_{index}.wav")
    output_path.write_bytes(await response.read())
    print(f"WAV audio written to {output_path.resolve()}")

# Main asynchronous function to run all tasks
async def main():
    tasks = [generate_speech(text, idx) for idx, text in enumerate(texts_to_speak)]
    await asyncio.gather(*tasks)

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
