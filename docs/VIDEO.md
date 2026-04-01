# Video Input Support

mistral.rs supports video input for compatible multimodal models. Videos are decoded into frames and passed through the model's vision encoder alongside any image or audio inputs.

**Supported models:** Gemma 4

## FFmpeg requirement

Non-GIF video formats require the [FFmpeg](https://ffmpeg.org/) binary to be available on your `PATH`. FFmpeg is used to decode video files into individual frames for processing.

GIF files are decoded natively using the `image` crate and do not require FFmpeg.

### Installation

**Linux (Debian/Ubuntu):**
```bash
sudo apt install ffmpeg
```

**Linux (Fedora/RHEL):**
```bash
sudo dnf install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**

Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html) and add the binary to your `PATH`.

**Docker:**
```dockerfile
RUN apt-get update && apt-get install -y ffmpeg
```

## Frame sampling

Videos are uniformly sampled to 32 frames by default. The sampled frames are evenly spaced across the full duration of the video, preserving temporal coverage regardless of the original frame rate or video length.

## Supported formats

Any format that FFmpeg can decode is supported, including:
- mp4
- avi
- mov
- mkv
- webm
- m4v
- gif (decoded natively without FFmpeg)

## API format

Video inputs use the `video_url` content type in the OpenAI-compatible chat completion API:

```json
{
    "type": "video_url",
    "video_url": {
        "url": "path/to/video.mp4"
    }
}
```

The `url` field accepts either a local file path or a URL.

## CLI example

Use `--video` with `-i` for one-shot video queries from the command line:

```bash
# Describe a local video
mistralrs run -m google/gemma-4-12b-it --video clip.mp4 -i "What happens in this video?"

# Use a URL
mistralrs run -m google/gemma-4-12b-it --video https://example.com/video.mp4 -i "Summarize this video"

# Multiple videos
mistralrs run -m google/gemma-4-12b-it --video clip1.mp4 --video clip2.mp4 -i "Compare these two videos"
```

Or use video files directly in interactive mode by including the path in your prompt:

```
> What happens in this video? clip.mp4
```

## HTTP API example

```py
from openai import OpenAI

client = OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

completion = client.chat.completions.create(
    model="default",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {
                        "url": "path/to/video.mp4"
                    },
                },
                {
                    "type": "text",
                    "text": "What happens in this video?",
                },
            ],
        }
    ],
    max_tokens=256,
)
print(completion.choices[0].message.content)
```

See [GEMMA4.md](GEMMA4.md) for full examples across all APIs.
