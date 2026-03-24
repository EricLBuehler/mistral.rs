# Dia 1.6b Model: [`nari-labs/Dia-1.6B`](https://huggingface.co/nari-labs/Dia-1.6B)

Dia is a 1.6B parameter text to speech model created by Nari Labs. You can condition the output on audio, enabling emotion and tone control. The model can also produce nonverbal communications like laughter, coughing, clearing throat, etc.

- Generate dialogue via the [S1] and [S2] tags
- Generate non-verbal like (laughs), (coughs), etc.
- Below verbal tags will be recognized, but might result in unexpected output. (laughs), (clears throat), (sighs), (gasps), (coughs), (singing), (sings), (mumbles), (beep), (groans), (sniffs), (claps), (screams), (inhales), (exhales), (applause), (burps), (humming), (sneezes), (chuckle), (whistles)

> Note: voice cloning support is coming!

## HTTP server

The OpenAI HTTP server provides a drop-in compatible way to easily use Dia locally!

> Note: we only support `pcm` and `wav` outputs.

```
mistralrs run speech -m nari-labs/Dia-1.6B -a dia
```

After this, you can send requests via the HTTP server:
```py
from pathlib import Path
from openai import OpenAI

client = OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

# text_to_speak = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face."
text_to_speak = "[S1] mistral r s is a local LLM inference engine. [S2] You can run text and vision models, and also image generation and speech generation. [S1] There is agentic web search, tool calling, and a convenient Python SDK. [S2] Check it out on github."

response = client.audio.speech.create(
    model="default", voice="N/A", input=text_to_speak, response_format="wav"
)

output_path = Path("output.wav")
output_path.write_bytes(response.read())
print(f"WAV audio written to {output_path.resolve()}")
```

## Rust example
```rust
use std::time::Instant;

use anyhow::Result;
use mistralrs::{speech_utils, SpeechLoaderType, SpeechModelBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    let model = SpeechModelBuilder::new("nari-labs/Dia-1.6B", SpeechLoaderType::Dia)
        .with_logging()
        .build()
        .await?;

    let start = Instant::now();

    // let text_to_speak = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face.";
    let text_to_speak = "[S1] mistral r s is a local LLM inference engine. [S2] You can run text and vision models, and also image generation and speech generation. [S1] There is agentic web search, tool calling, and a convenient Python SDK. [S2] Check it out on github.";

    let (pcm, rate, channels) = model.generate_speech(text_to_speak).await?;

    let finished = Instant::now();

    let mut output = std::fs::File::create("out.wav").unwrap();
    speech_utils::write_pcm_as_wav(&mut output, &pcm, rate as u32, channels as u16).unwrap();

    println!(
        "Done! Took {} s. Audio saved at `out.wav`.",
        finished.duration_since(start).as_secs_f32(),
    );

    Ok(())
}
```

## Python example
```py
from mistralrs import (
    Runner,
    Which,
    SpeechLoaderType,
)
from pathlib import Path
import wave, struct

# text_to_speak = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face."
text_to_speak = "[S1] mistral r s is a local LLM inference engine. [S2] You can run text and vision models, and also image generation and speech generation. [S1] There is agentic web search, tool calling, and a convenient Python SDK. [S2] Check it out on github."

runner = Runner(
    which=Which.Speech(
        model_id="nari-labs/Dia-1.6B",
        arch=SpeechLoaderType.Dia,
    ),
)

res = runner.generate_speech(text_to_speak)
print(res.choices[0].url)

pcm_data = res.pcm  # list of floats between -1.0 and 1.0
output_path = Path("output.wav")

# convert floats to 16-bit PCM ints
pcm_ints = [int(max(-32768, min(32767, int(sample * 32767)))) for sample in pcm_data]
with wave.open(output_path, "wb") as wf:
    wf.setnchannels(1)  # mono
    wf.setsampwidth(2 * res.channels)  # 2 bytes per sample (16-bit)
    wf.setframerate(res.rate)  # set sample rate (adjust if needed)
    wf.writeframes(b"".join(struct.pack("<h", s) for s in pcm_ints))

print(f"WAV audio written to {output_path.resolve()}")
```