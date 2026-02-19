# Voxtral Model: [`mistralai/Voxtral-Mini-4B-Realtime-2602`](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)

Voxtral Mini is a 4.4B parameter real-time automatic speech recognition (ASR) model created by Mistral AI. It features a causal Whisper-based audio encoder, a temporal adapter, and a Mistral decoder. The model accepts audio input and produces text output (speech-to-text).

The Voxtral Model has support in the Rust, Python, and HTTP APIs. Additionally, the Voxtral Model supports ISQ for increased performance.

> Note: Voxtral uses Mistral's native format (`params.json`, `consolidated.safetensors`, `tekken.json`), which mistral.rs handles automatically.

## HTTP server

We support an OpenAI compatible HTTP API for audio models.

1) Start the server

```
mistralrs serve vision -m mistralai/Voxtral-Mini-4B-Realtime-2602
```

2) Send a request

```py
import base64
from openai import OpenAI

client = OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

# Load a local audio file
with open("audio.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode("utf-8")

completion = client.chat.completions.create(
    model="ignore",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": f"data:audio/wav;base64,{audio_b64}"
                    },
                },
                {
                    "type": "text",
                    "text": "Transcribe this audio.",
                },
            ],
        },
    ],
    max_tokens=256,
    temperature=0,
)
resp = completion.choices[0].message.content
print(resp)
```

## Rust
You can find this example [here](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/models/asr/main.rs).

```rust
use anyhow::Result;
use mistralrs::{AudioInput, TextMessageRole, VisionMessages, VisionModelBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    let model = VisionModelBuilder::new("mistralai/Voxtral-Mini-4B-Realtime-2602")
        .with_logging()
        .build()
        .await?;

    let audio_bytes = std::fs::read("sample_audio.wav")?;
    let audio = AudioInput::from_bytes(&audio_bytes)?;

    let messages = VisionMessages::new().add_multimodal_message(
        TextMessageRole::User,
        "Transcribe this audio.",
        vec![],
        vec![audio],
    );

    let response = model.send_chat_request(messages).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    dbg!(
        response.usage.avg_prompt_tok_per_sec,
        response.usage.avg_compl_tok_per_sec
    );

    Ok(())
}
```

## Python

```py
from mistralrs import Runner, Which, ChatCompletionRequest, VisionArchitecture

runner = Runner(
    which=Which.VisionPlain(
        model_id="mistralai/Voxtral-Mini-4B-Realtime-2602",
        arch=VisionArchitecture.Voxtral,
    ),
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="ignore",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {
                            "url": "path/to/audio.wav"
                        },
                    },
                    {
                        "type": "text",
                        "text": "Transcribe this audio.",
                    },
                ],
            }
        ],
        max_tokens=256,
        temperature=0.1,
    )
)
print(res.choices[0].message.content)
print(res.usage)
```
