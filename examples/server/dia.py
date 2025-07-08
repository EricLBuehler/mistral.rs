from pathlib import Path
from openai import OpenAI

client = OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

# text_to_speak = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face."
text_to_speak = "[S1] mistral r s is a local LLM inference engine. [S2] You can run text and vision models, and also image generation and speech generation. [S1] There is agentic web search, tool calling, and a convenient Python API. [S2] Check it out on github."

response = client.audio.speech.create(
    model="default", voice="N/A", input=text_to_speak, response_format="wav"
)

output_path = Path("output.wav")
output_path.write_bytes(response.read())
print(f"WAV audio written to {output_path.resolve()}")
