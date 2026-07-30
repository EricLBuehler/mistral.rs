//! Static per-architecture metadata powering the supported-models reference.
//! One architecture often serves several brand names (Qwen 3.5/3.6, LFM2/LFM2.5);
//! `families` carries those names while the loader enum stays the machine-matchable key.
//! `supported_models_matches_committed` keeps docs/.../reference/supported-models.md in sync;
//! refresh with `cargo test -p mistralrs-core regenerate_supported_models -- --ignored`.

use std::fmt::Write as _;

use strum::IntoEnumIterator;

use crate::pipeline::SupportedModality;
use crate::speech_models::SpeechLoaderType;
use crate::{DiffusionLoaderType, EmbeddingLoaderType, MultimodalLoaderType, NormalLoaderType};

use SupportedModality::{Audio, Embedding, Text, Video, Vision};

#[derive(Clone, Copy)]
pub struct ModelExample {
    pub repo: &'static str,
    pub label: &'static str,
}

// A macro so each call expands to a struct literal, which const-promotes to `&'static`
// (a `const fn` call in an `&[..]` literal would not promote).
macro_rules! ex {
    ($repo:expr) => {
        ModelExample {
            repo: $repo,
            label: "",
        }
    };
    ($repo:expr, $label:expr) => {
        ModelExample {
            repo: $repo,
            label: $label,
        }
    };
}

pub struct ArchMetadata {
    pub families: &'static [&'static str],
    pub modalities: &'static [SupportedModality],
    pub examples: &'static [ModelExample],
}

impl NormalLoaderType {
    pub fn arch_metadata(&self) -> ArchMetadata {
        let m = &[Text][..];
        match self {
            Self::Mistral => ArchMetadata {
                families: &["Mistral"],
                modalities: m,
                examples: &[ex!("mistralai/Mistral-7B-Instruct-v0.3")],
            },
            Self::Gemma => ArchMetadata {
                families: &["Gemma"],
                modalities: m,
                examples: &[ex!("google/gemma-7b-it")],
            },
            Self::Mixtral => ArchMetadata {
                families: &["Mixtral"],
                modalities: m,
                examples: &[ex!("mistralai/Mixtral-8x7B-Instruct-v0.1")],
            },
            Self::Llama => ArchMetadata {
                families: &["Llama 2", "Llama 3.x"],
                modalities: m,
                examples: &[ex!("meta-llama/Llama-3.1-8B-Instruct")],
            },
            Self::Phi2 => ArchMetadata {
                families: &["Phi-2"],
                modalities: m,
                examples: &[ex!("microsoft/phi-2")],
            },
            Self::Phi3 => ArchMetadata {
                families: &["Phi-3", "Phi-3.5"],
                modalities: m,
                examples: &[ex!("microsoft/Phi-3-medium-4k-instruct")],
            },
            Self::Qwen2 => ArchMetadata {
                families: &["Qwen2", "Qwen2.5"],
                modalities: m,
                examples: &[
                    ex!("Qwen/Qwen2.5-7B-Instruct", "2.5"),
                    ex!("Qwen/Qwen2-7B-Instruct", "2"),
                ],
            },
            Self::Gemma2 => ArchMetadata {
                families: &["Gemma 2"],
                modalities: m,
                examples: &[ex!("google/gemma-2-9b-it")],
            },
            Self::Starcoder2 => ArchMetadata {
                families: &["Starcoder2"],
                modalities: m,
                examples: &[ex!("bigcode/starcoder2-7b")],
            },
            Self::Phi3_5MoE => ArchMetadata {
                families: &["Phi-3.5-MoE"],
                modalities: m,
                examples: &[ex!("microsoft/Phi-3.5-MoE-instruct")],
            },
            Self::DeepSeekV2 => ArchMetadata {
                families: &["DeepSeek-V2"],
                modalities: m,
                examples: &[ex!("deepseek-ai/DeepSeek-V2-Chat")],
            },
            Self::DeepSeekV3 => ArchMetadata {
                families: &["DeepSeek-V3", "DeepSeek-R1"],
                modalities: m,
                examples: &[
                    ex!("deepseek-ai/DeepSeek-V3", "V3"),
                    ex!("deepseek-ai/DeepSeek-R1", "R1"),
                ],
            },
            Self::Qwen3 => ArchMetadata {
                families: &["Qwen3"],
                modalities: m,
                examples: &[ex!("Qwen/Qwen3-4B")],
            },
            Self::GLM4 => ArchMetadata {
                families: &["GLM-4"],
                modalities: m,
                examples: &[ex!("zai-org/GLM-4-32B-0414")],
            },
            Self::GLM4MoeLite => ArchMetadata {
                families: &["GLM-4.7-Flash"],
                modalities: m,
                examples: &[ex!("zai-org/GLM-4.7-Flash")],
            },
            Self::GLM4Moe => ArchMetadata {
                families: &["GLM-4.7"],
                modalities: m,
                examples: &[ex!("zai-org/GLM-4.7")],
            },
            Self::Qwen3Moe => ArchMetadata {
                families: &["Qwen3 MoE"],
                modalities: m,
                examples: &[ex!("Qwen/Qwen3-30B-A3B")],
            },
            Self::SmolLm3 => ArchMetadata {
                families: &["SmolLM3"],
                modalities: m,
                examples: &[ex!("HuggingFaceTB/SmolLM3-3B")],
            },
            Self::GraniteMoeHybrid => ArchMetadata {
                families: &["Granite 4.0"],
                modalities: m,
                examples: &[ex!("ibm-granite/granite-4.0-micro")],
            },
            Self::GptOss => ArchMetadata {
                families: &["GPT-OSS"],
                modalities: m,
                examples: &[
                    ex!("openai/gpt-oss-20b", "20b"),
                    ex!("openai/gpt-oss-120b", "120b"),
                ],
            },
            Self::HunYuanDenseV1 => ArchMetadata {
                families: &["HunYuan"],
                modalities: m,
                examples: &[ex!("tencent/Hunyuan-7B-Instruct")],
            },
            Self::HunYuanMoEV1 => ArchMetadata {
                families: &["HunYuan MoE"],
                modalities: m,
                examples: &[ex!("tencent/Hunyuan-A13B-Instruct")],
            },
            Self::Qwen3Next => ArchMetadata {
                families: &["Qwen3-Next", "Qwen3-Coder-Next"],
                modalities: m,
                examples: &[ex!("Qwen/Qwen3-Next-80B-A3B-Instruct")],
            },
            Self::Lfm2 => ArchMetadata {
                families: &["LFM2", "LFM2.5"],
                modalities: m,
                examples: &[
                    ex!("LiquidAI/LFM2.5-1.2B-Instruct", "LFM2.5"),
                    ex!("LiquidAI/LFM2-1.2B", "LFM2"),
                ],
            },
            Self::Lfm2Moe => ArchMetadata {
                families: &["LFM2 MoE", "LFM2.5 MoE"],
                modalities: m,
                examples: &[
                    ex!("LiquidAI/LFM2.5-8B-A1B", "LFM2.5"),
                    ex!("LiquidAI/LFM2-8B-A1B", "LFM2"),
                ],
            },
        }
    }
}

impl MultimodalLoaderType {
    pub fn arch_metadata(&self) -> ArchMetadata {
        match self {
            Self::Phi3V => ArchMetadata {
                families: &["Phi-3.5-Vision"],
                modalities: &[Text, Vision],
                examples: &[ex!("microsoft/Phi-3.5-vision-instruct")],
            },
            Self::Idefics2 => ArchMetadata {
                families: &["Idefics2"],
                modalities: &[Text, Vision],
                examples: &[ex!("HuggingFaceM4/idefics2-8b")],
            },
            Self::LLaVANext => ArchMetadata {
                families: &["LLaVA-NeXT"],
                modalities: &[Text, Vision],
                examples: &[ex!("llava-hf/llava-v1.6-mistral-7b-hf")],
            },
            Self::LLaVA => ArchMetadata {
                families: &["LLaVA 1.5"],
                modalities: &[Text, Vision],
                examples: &[ex!("llava-hf/llava-1.5-7b-hf")],
            },
            Self::Lfm2Vl => ArchMetadata {
                families: &["LFM2-VL", "LFM2.5-VL"],
                modalities: &[Text, Vision],
                examples: &[
                    ex!("LiquidAI/LFM2.5-VL-1.6B", "1.6B"),
                    ex!("LiquidAI/LFM2.5-VL-450M", "450M"),
                ],
            },
            Self::VLlama => ArchMetadata {
                families: &["Llama 3.2 Vision"],
                modalities: &[Text, Vision],
                examples: &[ex!("meta-llama/Llama-3.2-11B-Vision-Instruct")],
            },
            Self::Qwen2VL => ArchMetadata {
                families: &["Qwen2-VL"],
                modalities: &[Text, Vision, Video],
                examples: &[ex!("Qwen/Qwen2-VL-7B-Instruct")],
            },
            Self::Idefics3 => ArchMetadata {
                families: &["Idefics3", "SmolVLM"],
                modalities: &[Text, Vision],
                examples: &[ex!("HuggingFaceM4/Idefics3-8B-Llama3")],
            },
            Self::MiniCpmO => ArchMetadata {
                families: &["MiniCPM-o"],
                modalities: &[Text, Vision, Audio],
                examples: &[ex!("openbmb/MiniCPM-o-2_6")],
            },
            Self::Phi4MM => ArchMetadata {
                families: &["Phi-4-multimodal"],
                modalities: &[Text, Vision, Audio],
                examples: &[ex!("microsoft/Phi-4-multimodal-instruct")],
            },
            Self::Qwen2_5VL => ArchMetadata {
                families: &["Qwen2.5-VL"],
                modalities: &[Text, Vision, Video],
                examples: &[ex!("Qwen/Qwen2.5-VL-7B-Instruct")],
            },
            Self::Gemma3 => ArchMetadata {
                families: &["Gemma 3"],
                modalities: &[Text, Vision],
                examples: &[ex!("google/gemma-3-12b-it")],
            },
            Self::Mistral3 => ArchMetadata {
                families: &["Mistral Small 3"],
                modalities: &[Text, Vision],
                examples: &[ex!("mistralai/Mistral-Small-3.2-24B-Instruct-2506")],
            },
            Self::Llama4 => ArchMetadata {
                families: &["Llama 4"],
                modalities: &[Text, Vision],
                examples: &[ex!("meta-llama/Llama-4-Scout-17B-16E-Instruct")],
            },
            Self::Gemma3n => ArchMetadata {
                families: &["Gemma 3n"],
                modalities: &[Text, Vision, Audio, Video],
                examples: &[ex!("google/gemma-3n-E4B-it")],
            },
            Self::Qwen3VL => ArchMetadata {
                families: &["Qwen3-VL"],
                modalities: &[Text, Vision, Video],
                examples: &[ex!("Qwen/Qwen3-VL-4B-Instruct")],
            },
            Self::Qwen3VLMoE => ArchMetadata {
                families: &["Qwen3-VL MoE"],
                modalities: &[Text, Vision, Video],
                examples: &[ex!("Qwen/Qwen3-VL-235B-A22B-Instruct")],
            },
            Self::Qwen3_5 => ArchMetadata {
                families: &["Qwen 3.5", "Qwen 3.6"],
                modalities: &[Text, Vision],
                examples: &[
                    ex!("Qwen/Qwen3.5-27B", "3.5"),
                    ex!("Qwen/Qwen3.6-27B", "3.6"),
                ],
            },
            Self::Qwen3_5Moe => ArchMetadata {
                families: &["Qwen 3.5 MoE", "Qwen 3.6 MoE"],
                modalities: &[Text, Vision],
                examples: &[
                    ex!("Qwen/Qwen3.5-35B-A3B", "3.5"),
                    ex!("Qwen/Qwen3.6-35B-A3B", "3.6"),
                ],
            },
            Self::Voxtral => ArchMetadata {
                families: &["Voxtral"],
                modalities: &[Text, Audio],
                examples: &[ex!("mistralai/Voxtral-Mini-3B-2507")],
            },
            Self::Gemma4 => ArchMetadata {
                families: &["Gemma 4"],
                modalities: &[Text, Vision, Audio, Video],
                examples: &[
                    ex!("google/gemma-4-E4B-it", "E4B"),
                    ex!("google/gemma-4-26B-A4B-it", "26B-A4B MoE"),
                    ex!("google/gemma-4-31B-it", "31B dense"),
                ],
            },
            Self::DiffusionGemma => ArchMetadata {
                families: &["DiffusionGemma"],
                modalities: &[Text, Vision],
                examples: &[ex!("google/diffusiongemma-26B-A4B-it")],
            },
        }
    }
}

impl EmbeddingLoaderType {
    pub fn arch_metadata(&self) -> ArchMetadata {
        match self {
            Self::EmbeddingGemma => ArchMetadata {
                families: &["EmbeddingGemma"],
                modalities: &[Embedding],
                examples: &[ex!("google/embeddinggemma-300m")],
            },
            Self::Qwen3Embedding => ArchMetadata {
                families: &["Qwen3 Embedding"],
                modalities: &[Embedding],
                examples: &[ex!("Qwen/Qwen3-Embedding-0.6B")],
            },
        }
    }
}

impl DiffusionLoaderType {
    pub fn arch_metadata(&self) -> ArchMetadata {
        match self {
            Self::Flux => ArchMetadata {
                families: &["FLUX.1"],
                modalities: &[Text, Vision],
                examples: &[ex!("black-forest-labs/FLUX.1-schnell")],
            },
            Self::FluxOffloaded => ArchMetadata {
                families: &["FLUX.1 (offloaded)"],
                modalities: &[Text, Vision],
                examples: &[ex!("black-forest-labs/FLUX.1-schnell")],
            },
        }
    }
}

impl SpeechLoaderType {
    pub fn arch_metadata(&self) -> ArchMetadata {
        match self {
            Self::Dia => ArchMetadata {
                families: &["Dia"],
                modalities: &[Text, Audio],
                examples: &[ex!("nari-labs/Dia-1.6B")],
            },
        }
    }
}

// The canonical HF `config.json` `architectures` string for each loader, so the table's
// leftmost column is the value users Ctrl-F. Must round-trip through `from_causal_lm_name`
// (enforced by `config_archs_round_trip`).
impl NormalLoaderType {
    pub fn config_arch(&self) -> &'static str {
        match self {
            Self::Mistral => "MistralForCausalLM",
            Self::Gemma => "GemmaForCausalLM",
            Self::Mixtral => "MixtralForCausalLM",
            Self::Llama => "LlamaForCausalLM",
            Self::Phi2 => "PhiForCausalLM",
            Self::Phi3 => "Phi3ForCausalLM",
            Self::Qwen2 => "Qwen2ForCausalLM",
            Self::Gemma2 => "Gemma2ForCausalLM",
            Self::Starcoder2 => "Starcoder2ForCausalLM",
            Self::Phi3_5MoE => "PhiMoEForCausalLM",
            Self::DeepSeekV2 => "DeepseekV2ForCausalLM",
            Self::DeepSeekV3 => "DeepseekV3ForCausalLM",
            Self::Qwen3 => "Qwen3ForCausalLM",
            Self::GLM4 => "Glm4ForCausalLM",
            Self::GLM4MoeLite => "Glm4MoeLiteForCausalLM",
            Self::GLM4Moe => "Glm4MoeForCausalLM",
            Self::Qwen3Moe => "Qwen3MoeForCausalLM",
            Self::SmolLm3 => "SmolLM3ForCausalLM",
            Self::GraniteMoeHybrid => "GraniteMoeHybridForCausalLM",
            Self::GptOss => "GptOssForCausalLM",
            Self::HunYuanDenseV1 => "HunYuanDenseV1ForCausalLM",
            Self::HunYuanMoEV1 => "HunYuanMoEV1ForCausalLM",
            Self::Qwen3Next => "Qwen3NextForCausalLM",
            Self::Lfm2 => "Lfm2ForCausalLM",
            Self::Lfm2Moe => "Lfm2MoeForCausalLM",
        }
    }
}

impl MultimodalLoaderType {
    pub fn config_arch(&self) -> &'static str {
        match self {
            Self::Phi3V => "Phi3VForCausalLM",
            Self::Idefics2 => "Idefics2ForConditionalGeneration",
            Self::LLaVANext => "LlavaNextForConditionalGeneration",
            Self::LLaVA => "LlavaForConditionalGeneration",
            Self::Lfm2Vl => "Lfm2VlForConditionalGeneration",
            Self::VLlama => "MllamaForConditionalGeneration",
            Self::Qwen2VL => "Qwen2VLForConditionalGeneration",
            Self::Idefics3 => "Idefics3ForConditionalGeneration",
            Self::MiniCpmO => "MiniCPMO",
            Self::Phi4MM => "Phi4MMForCausalLM",
            Self::Qwen2_5VL => "Qwen2_5_VLForConditionalGeneration",
            Self::Gemma3 => "Gemma3ForConditionalGeneration",
            Self::Mistral3 => "Mistral3ForConditionalGeneration",
            Self::Llama4 => "Llama4ForConditionalGeneration",
            Self::Gemma3n => "Gemma3nForConditionalGeneration",
            Self::Qwen3VL => "Qwen3VLForConditionalGeneration",
            Self::Qwen3VLMoE => "Qwen3VLMoeForConditionalGeneration",
            Self::Qwen3_5 => "Qwen3_5ForConditionalGeneration",
            Self::Qwen3_5Moe => "Qwen3_5MoeForConditionalGeneration",
            Self::Voxtral => "VoxtralForConditionalGeneration",
            Self::Gemma4 => "Gemma4ForConditionalGeneration",
            Self::DiffusionGemma => "DiffusionGemmaForBlockDiffusion",
        }
    }
}

impl EmbeddingLoaderType {
    pub fn config_arch(&self) -> &'static str {
        match self {
            Self::EmbeddingGemma => "Gemma3TextModel",
            Self::Qwen3Embedding => "Qwen3ForCausalLM",
        }
    }
}

const HEADER: &str = r#"---
title: Supported models
description: Architectures supported by mistral.rs, and how to tell if yours is one of them.
---

<!-- Generated from the loader registry by mistralrs-core model_metadata. Do not edit by hand. -->

## Is my model supported?

mistral.rs auto-detects the architecture from a repo's `config.json`. To check yours:

1. Open the model's `config.json` on Hugging Face and read the `architectures` field (e.g. `"Qwen3ForCausalLM"`, `"Gemma4ForConditionalGeneration"`).
2. Find the matching row below. Each architecture covers every checkpoint that reports that class, including future fine-tunes and sizes, so the families and examples here are a sample, not the full list.
3. Not listed? You can still try it: force a known architecture with `--arch`, load a [GGUF](/mistral.rs/guides/models/run-any-model/) build, or [request the model](https://github.com/EricLBuehler/mistral.rs/issues/156).

```bash
mistralrs run -m <model>     # interactive
mistralrs serve -m <model>   # OpenAI-compatible server
```

Expand the example in any row to copy a ready-to-run command. One loader often serves several brand names (Qwen 3.5 and 3.6 share `Qwen3_5`; LFM2 and LFM2.5 share `Lfm2`) - the `Model families` column lists them. Behavior that differs from the defaults is collected in [model family notes](/mistral.rs/guides/models/model-family-notes/).

The `Architecture` column is the `config.json` `architectures` value. Per-family quantization, thinking, gated-repo, and tool-calling details live in [model family notes](/mistral.rs/guides/models/model-family-notes/).

"#;

const FOOTER: &str = r#"## Format and quantization notes

Text, multimodal, speech, and embedding models support ISQ at load time. Diffusion models (FLUX) do not; they load at native precision. Pre-quantized format availability (GGUF, [UQFF](/mistral.rs/reference/uqff-format/), GPTQ, AWQ) is per-model on Hugging Face.

## Speculative decoding

| Mode | Target architecture | Assistant checkpoint family | Guide |
|---|---|---|---|
| MTP | `Gemma4` | Gemma 4 assistant checkpoints, PagedAttention required | [Speculative decoding (MTP)](/mistral.rs/guides/perf/speculative-decoding/) |
"#;

fn md_cell(s: &str) -> String {
    s.replace('|', "\\|")
}

fn families_cell(meta: &ArchMetadata) -> String {
    md_cell(&meta.families.join(", "))
}

fn examples_cell(meta: &ArchMetadata) -> String {
    let label = |e: &ModelExample| {
        if e.label.is_empty() {
            format!("<code>{}</code>", e.repo)
        } else {
            format!("<code>{}</code> ({})", e.repo, e.label)
        }
    };
    let summary = meta
        .examples
        .iter()
        .map(label)
        .collect::<Vec<_>>()
        .join(", ");
    let body = meta
        .examples
        .iter()
        .map(|e| format!("<code>mistralrs run -m {}</code>", e.repo))
        .collect::<Vec<_>>()
        .join("<br>");
    md_cell(&format!(
        "<details><summary>{summary}</summary>{body}</details>"
    ))
}

fn simple_table(md: &mut String, rows: impl Iterator<Item = (String, ArchMetadata)>) {
    md.push_str("| Architecture | Model families | Example |\n|---|---|---|\n");
    for (name, meta) in rows {
        writeln!(
            md,
            "| `{name}` | {} | {} |",
            families_cell(&meta),
            examples_cell(&meta),
        )
        .unwrap();
    }
    md.push('\n');
}

fn variant_name<T: std::fmt::Debug>(v: &T) -> String {
    format!("{v:?}")
}

/// Render the full `supported-models.md` page from the loader registry.
pub fn render_supported_models_markdown() -> String {
    let mut md = String::new();
    md.push_str(HEADER);

    md.push_str("## Text models\n\n");
    simple_table(
        &mut md,
        NormalLoaderType::iter().map(|t| (t.config_arch().to_string(), t.arch_metadata())),
    );

    md.push_str("## Multimodal models\n\n");
    simple_table(
        &mut md,
        MultimodalLoaderType::iter().map(|t| (t.config_arch().to_string(), t.arch_metadata())),
    );

    md.push_str("## Image generation\n\n");
    simple_table(
        &mut md,
        DiffusionLoaderType::iter().map(|t| (variant_name(&t), t.arch_metadata())),
    );

    md.push_str("## Speech\n\n");
    simple_table(
        &mut md,
        SpeechLoaderType::iter().map(|t| (variant_name(&t), t.arch_metadata())),
    );

    md.push_str("## Embedding\n\n");
    simple_table(
        &mut md,
        EmbeddingLoaderType::iter().map(|t| (t.config_arch().to_string(), t.arch_metadata())),
    );

    md.push_str(FOOTER);
    md
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    const COMMITTED: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../docs/src/content/docs/reference/supported-models.md"
    );
    const REGEN_HINT: &str =
        "cargo test -p mistralrs-core regenerate_supported_models -- --ignored";

    #[test]
    fn supported_models_matches_committed() {
        let rendered = render_supported_models_markdown();
        // Windows runners check out with autocrlf, so normalize before comparing.
        let on_disk = std::fs::read_to_string(COMMITTED)
            .expect("supported-models.md exists")
            .replace("\r\n", "\n");
        assert!(
            on_disk == rendered,
            "committed supported-models.md is out of date; regenerate with: {REGEN_HINT}",
        );
    }

    #[test]
    #[ignore = "writes the committed supported-models.md"]
    fn regenerate_supported_models() {
        std::fs::write(Path::new(COMMITTED), render_supported_models_markdown()).unwrap();
    }

    #[test]
    fn config_archs_round_trip() {
        for t in NormalLoaderType::iter() {
            assert_eq!(
                NormalLoaderType::from_causal_lm_name(t.config_arch()).unwrap(),
                t,
                "{t:?} config_arch does not map back through from_causal_lm_name",
            );
        }
        for t in MultimodalLoaderType::iter() {
            assert_eq!(
                MultimodalLoaderType::from_causal_lm_name(t.config_arch()).unwrap(),
                t,
                "{t:?} config_arch does not map back through from_causal_lm_name",
            );
        }
        for t in EmbeddingLoaderType::iter() {
            assert_eq!(
                EmbeddingLoaderType::from_causal_lm_name(t.config_arch()).unwrap(),
                t,
                "{t:?} config_arch does not map back through from_causal_lm_name",
            );
        }
    }
}
