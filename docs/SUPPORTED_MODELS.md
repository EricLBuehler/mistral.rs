# Supported Models

Complete reference for model support in mistral.rs.

## Model Categories

### Text Models

- Granite 4.0
- SmolLM 3
- DeepSeek V3
- GPT-OSS
- DeepSeek V2
- Qwen 3 Next
- Qwen 3 MoE
- Phi 3.5 MoE
- Qwen 3
- GLM 4
- GLM-4.7-Flash
- GLM-4.7 (MoE)
- Gemma 2
- Qwen 2
- Starcoder 2
- Phi 3
- Mixtral
- Phi 2
- Gemma
- Llama
- Mistral

### Vision Models

- Qwen 3-VL
- Qwen 3-VL MoE
- Gemma 3n
- Llama 4
- Gemma 3
- Mistral 3
- Phi 4 multimodal
- Qwen 2.5-VL
- MiniCPM-O
- Llama 3.2 Vision
- Qwen 2-VL
- Idefics 3
- Idefics 2
- LLaVA Next
- LLaVA
- Phi 3V

### Speech Models

- Voxtral (ASR/speech-to-text)
- Dia

### Image Generation Models

- FLUX

### Embedding Models

- Embedding Gemma
- Qwen 3 Embedding

[Request a new model](https://github.com/EricLBuehler/mistral.rs/issues/156)

### Supported GGUF Architectures

**Plain:**
- llama
- phi2
- phi3
- starcoder2
- qwen2
- qwen3

**With adapters:**
- llama
- phi3

## Quantization Support

|Model|GGUF|GGML|ISQ|
|--|--|--|--|
|Mistral|✅| |✅|
|Gemma| | |✅|
|Llama|✅|✅|✅|
|Mixtral|✅| |✅|
|Phi 2|✅| |✅|
|Phi 3|✅| |✅|
|Phi 3.5 MoE| | |✅|
|Qwen 2.5| | |✅|
|Phi 3 Vision| | |✅|
|Idefics 2| | |✅|
|Gemma 2| | |✅|
|GLM4| | |✅|
|GLM-4.7-Flash (MoE)| | |✅|
|GLM-4.7 (MoE)| | |✅|
|Starcoder 2| |✅|✅|
|LLaVa Next| | |✅|
|LLaVa| | |✅|
|Llama 3.2 Vision| | |✅|
|Qwen2-VL| | |✅|
|Idefics 3| | |✅|
|Deepseek V2| | |✅|
|Deepseek V3| | |✅|
|MiniCPM-O 2.6| | |✅|
|Qwen2.5-VL| | |✅|
|Gemma 3| | |✅|
|Mistral 3| | |✅|
|Llama 4| | |✅|
|Qwen 3|✅| |✅|
|SmolLM3| | |✅|
|Dia 1.6b| | |✅|
|Voxtral| | |✅|
|Gemma 3n| | |✅|
|Qwen 3 VL| | |✅|
|Qwen 3-VL MoE| | |✅|
|Qwen 3 Next| | |✅|
|Phi 4 Multimodal| | |✅|
|Granite 4.0| | |✅|
|GPT-OSS| | |✅|

## Device Mapping Support

|Model category|Supported|
|--|--|
|Plain|✅|
|GGUF|✅|
|GGML| |
|Vision Plain|✅|

## X-LoRA and LoRA Support

|Model|X-LoRA|X-LoRA+GGUF|X-LoRA+GGML|
|--|--|--|--|
|Mistral|✅|✅| |
|Gemma|✅| | |
|Llama|✅|✅|✅|
|Mixtral|✅|✅| |
|Phi 2|✅| | |
|Phi 3|✅|✅| |
|Phi 3.5 MoE| | | |
|Qwen 2.5| | | |
|Phi 3 Vision| | | |
|Idefics 2| | | |
|Gemma 2|✅| | |
|GLM4|✅| | |
|GLM-4.7-Flash (MoE)| | | |
|GLM-4.7 (MoE)| | | |
|Starcoder 2|✅| | |
|LLaVa Next| | | |
|LLaVa| | | |
|Qwen2-VL| | | |
|Idefics 3| | | |
|Deepseek V2| | | |
|Deepseek V3| | | |
|MiniCPM-O 2.6| | | |
|Qwen2.5-VL| | | |
|Gemma 3| | | |
|Mistral 3| | | |
|Llama 4| | | |
|Qwen 3| | | |
|SmolLM3|✅| | |
|Gemma 3n| | | |
|Voxtral| | | |
|Qwen 3 VL| | | |
|Qwen 3-VL MoE| | | |
|Qwen 3 Next| | | |
|Phi 4 Multimodal| | | |
|Llama 3.2 Vision| | | |
|Granite 4.0| | | |
|GPT-OSS| | | |

## AnyMoE Support

|Model|AnyMoE|
|--|--|
|Mistral 7B|✅|
|Gemma|✅|
|Llama|✅|
|Mixtral| |
|Phi 2|✅|
|Phi 3|✅|
|Phi 3.5 MoE| |
|Qwen 2.5|✅|
|Phi 3 Vision| |
|Idefics 2| |
|Gemma 2|✅|
|GLM-4.7-Flash (MoE)| |
|GLM-4.7 (MoE)| |
|Starcoder 2|✅|
|LLaVa Next|✅|
|LLaVa|✅|
|Llama 3.2 Vision| |
|Qwen2-VL| |
|Idefics 3|✅|
|Deepseek V2| |
|Deepseek V3| |
|MiniCPM-O 2.6| |
|Qwen2.5-VL| |
|Gemma 3|✅|
|Mistral 3|✅|
|Llama 4| |
|Qwen 3| |
|SmolLM3|✅|
|Gemma 3n| |
|Voxtral| |
|Qwen 3 VL| |
|Qwen 3-VL MoE| |
|Qwen 3 Next| |
|Phi 4 Multimodal| |
|Dia 1.6b| |
|Granite 4.0| |
|GPT-OSS| |

## Using Derivative Models

Model type is auto-detected. Use flags for quantized models and adapters:

| Model Type          | Required Arguments                                                     |
|---------------------|-----------------------------------------------------------------------|
| Plain               | `-m <model-id>`                                                       |
| GGUF Quantized      | `-m <model-id> --format gguf -f <file>`                               |
| ISQ Quantized       | `-m <model-id> --isq <level>`                                         |
| UQFF Quantized      | `-m <model-id> --from-uqff <file>`                                    |
| LoRA                | `-m <model-id> --lora <adapter>`                                      |
| X-LoRA              | `-m <model-id> --xlora <adapter> --xlora-order <file>`                |

### Example: Zephyr GGUF model

```bash
mistralrs serve -p 1234 --log output.txt --format gguf -t HuggingFaceH4/zephyr-7b-beta -m TheBloke/zephyr-7B-beta-GGUF -f zephyr-7b-beta.Q5_0.gguf
```

### Chat Templates and Tokenizer

Mistral.rs will attempt to automatically load a chat template and tokenizer. This enables high flexibility across models and ensures accurate and flexible chat templating. However, this behavior can be customized.

- [Adapter models documentation](ADAPTER_MODELS.md)
- [Chat templates documentation](CHAT_TOK.md)
- [LoRA and X-LoRA examples](LORA_XLORA.md)
