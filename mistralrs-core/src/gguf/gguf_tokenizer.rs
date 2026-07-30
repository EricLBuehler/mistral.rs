// https://github.com/huggingface/transformers/blob/8685b3c5d2dd2550527773d2a02499495a759e31/src/transformers/convert_slow_tokenizer.py

use std::{collections::HashMap, sync::atomic::Ordering};

use crate::utils::gguf_metadata::ContentMetadata;
use crate::DEBUG;
use ahash::AHashMap;
use anyhow::Result;
use candle_core::quantized::gguf_file::Value;
use tokenizers::pre_tokenizers::{
    sequence::Sequence,
    split::{Split, SplitPattern},
    PreTokenizerWrapper,
};
use tokenizers::tokenizer::normalizer::SplitDelimiterBehavior;
use tokenizers::{
    decoders::{
        self, byte_fallback::ByteFallback, byte_level::ByteLevel, fuse::Fuse, strip::Strip,
    },
    models::{bpe::BpeBuilder, unigram::Unigram},
    normalizers::{self, Prepend, Replace, NFC},
    processors, AddedToken, DecoderWrapper, ModelWrapper, NormalizerWrapper, Tokenizer,
};
use tracing::info;

use super::Content;

const GPT2_REGEX: &str =
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)";
const LLAMA3_REGEX: &str = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const QWEN2_REGEX: &str = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const QWEN35_REGEX: &str = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const DEEPSEEK_LLM_REGEXES: &[&str] = &[
    r"[\r\n]",
    r"\s?[A-Za-z\x{b5}\x{c0}-\x{d6}\x{d8}-\x{f6}\x{f8}-\x{1ba}\x{1bc}-\x{1bf}\x{1c4}-\x{293}\x{295}-\x{2af}\x{370}-\x{373}\x{376}\x{377}\x{37b}-\x{37d}\x{37f}\x{386}\x{388}-\x{38a}\x{38c}\x{38e}-\x{3a1}\x{3a3}-\x{3f5}\x{3f7}-\x{481}\x{48a}-\x{52f}\x{531}-\x{556}\x{10a0}-\x{10c5}\x{13a0}-\x{13f5}\x{13f8}-\x{13fd}\x{1c90}-\x{1cba}\x{1cbd}-\x{1cbf}\x{1d00}-\x{1d2b}\x{1d6b}-\x{1d77}\x{1d79}-\x{1d9a}\x{1e00}-\x{1f15}\x{1f18}-\x{1f1d}\x{1f20}-\x{1f45}\x{1f48}-\x{1f4d}\x{1f50}-\x{1f57}\x{1f59}\x{1f5b}\x{1f5d}\x{1f5f}-\x{1f7d}\x{1f80}-\x{1fb4}\x{1fb6}-\x{1fbc}\x{1fbe}\x{1fc2}-\x{1fc4}\x{1fc6}-\x{1fcc}\x{1fd0}-\x{1fd3}\x{1fd6}-\x{1fdb}\x{1fe0}-\x{1fec}\x{1ff2}-\x{1ff4}\x{1ff6}-\x{1ffc}\x{2102}\x{2107}\x{210a}-\x{2113}\x{2115}\x{2119}-\x{211d}\x{2124}\x{2126}\x{2128}\x{212a}-\x{212d}\x{212f}-\x{2134}\x{2139}\x{213c}-\x{213f}\x{2145}-\x{2149}\x{214e}\x{2183}\x{2184}\x{2c00}-\x{2c7b}\x{2c7e}-\x{2ce4}\x{2ceb}-\x{2cee}\x{2cf2}\x{2cf3}\x{a640}-\x{a66d}\x{a680}-\x{a69b}\x{a722}-\x{a76f}\x{a771}-\x{a787}\x{a78b}-\x{a78e}\x{ab70}-\x{abbf}\x{fb00}-\x{fb06}\x{fb13}-\x{fb17}\x{ff21}-\x{ff3a}\x{ff41}-\x{ff5a}\x{10400}-\x{1044f}\x{104b0}-\x{104d3}\x{104d8}-\x{104fb}\x{10c80}-\x{10cb2}\x{10cc0}-\x{10cf2}\x{118a0}-\x{118df}\x{1e900}-\x{1e943}]+",
    r"\s?[!-/:-~\x{ff01}-\x{ff0f}\x{ff1a}-\x{ff5e}\x{2018}-\x{201f}\x{3000}-\x{3002}]+",
    r"\s+$",
    r"[\x{4e00}-\x{9fa5}\x{800}-\x{4e00}\x{ac00}-\x{d7ff}]+",
    r"\p{N}+",
];
const DEEPSEEK_CODER_REGEXES: &[&str] = &[
    r"[\r\n]",
    r"\s?\p{L}+",
    r"\s?\p{P}+",
    r"[\x{4e00}-\x{9fa5}\x{800}-\x{4e00}\x{ac00}-\x{d7ff}]+",
    r"\p{N}",
];
const DEEPSEEK_V3_REGEXES: &[&str] = &[
    "\\p{N}{1,3}",
    "[\\x{4e00}-\\x{9fa5}\\x{3040}-\\x{309f}\\x{30a0}-\\x{30ff}]+",
    "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\\r\\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
];
const GPT4O_REGEX: &str = "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))*((?=[\\p{L}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))+((?=[\\p{L}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const TEKKEN_REGEX: &str = "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))*((?=[\\p{L}])([^A-Z]))+|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))+((?=[\\p{L}])([^A-Z]))*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

pub(crate) struct GgufTokenizerConversion {
    pub tokenizer: Tokenizer,
    pub bos: Option<String>,
    pub eos: Option<String>,
    pub unk: Option<String>,
}

struct PropsGGUF {
    model: String,
    pre: Option<String>,
    tokens: Vec<String>,
    added_tokens: Option<Vec<String>>,
    scores: Option<Vec<f32>>,
    merges: Option<Vec<String>>,
    unk: Option<u32>,
    bos: Option<u32>,
    eos: u32,
}

impl TryFrom<ContentMetadata<'_>> for PropsGGUF {
    type Error = anyhow::Error;

    fn try_from(c: ContentMetadata) -> Result<Self, Self::Error> {
        let required = ["model", "tokens", "eos_token_id"];
        c.has_required_keys(&required)?;

        let props = Self {
            model: c.get_value("model")?,
            pre: c.get_option_value("pre")?,
            tokens: c.get_value("tokens")?,
            added_tokens: c.get_value("added_tokens").ok(),
            scores: c.get_value("scores").ok(),
            merges: c.get_value("merges").ok(),
            unk: c.get_value("unknown_token_id").ok(),
            eos: c.get_value("eos_token_id")?,
            bos: c.get_value("bos_token_id").ok(),
        };

        // Special token ids come from untrusted GGUF metadata; reject out-of-range ids
        // so a malformed file errors instead of panicking when indexing `tokens`.
        let vocab_size = props.tokens.len();
        let check = |name: &str, id: u32| -> Result<()> {
            anyhow::ensure!(
                (id as usize) < vocab_size,
                "GGUF `{name}` token id {id} is out of bounds for vocab size {vocab_size}"
            );
            Ok(())
        };
        check("eos", props.eos)?;
        if let Some(bos) = props.bos {
            check("bos", bos)?;
        }
        if let Some(unk) = props.unk {
            check("unk", unk)?;
        }

        Ok(props)
    }
}

pub fn convert_gguf_to_hf_tokenizer<R: std::io::Seek + std::io::Read>(
    content: &Content<'_, R>,
) -> Result<GgufTokenizerConversion> {
    convert_gguf_metadata_to_hf_tokenizer(content.get_metadata())
}

pub(crate) fn convert_gguf_metadata_to_hf_tokenizer(
    values: &HashMap<String, Value>,
) -> Result<GgufTokenizerConversion> {
    let metadata = ContentMetadata {
        path_prefix: "tokenizer.ggml",
        metadata: values,
    };

    let md_get = |s: &str| match metadata.metadata.get(s) {
        None => candle_core::bail!("cannot find {s} in metadata"),
        Some(v) => Ok(v),
    };

    let mut token_types = Vec::<i32>::new();
    if metadata.metadata.contains_key("tokenizer.ggml.token_type") {
        let vtypes: &Vec<Value> = md_get("tokenizer.ggml.token_type")
            .unwrap()
            .to_vec()
            .unwrap();
        let v: Vec<i32> = vtypes.iter().map(|v| v.to_i32().unwrap()).collect();
        token_types.extend(v);
    }

    let props = PropsGGUF::try_from(metadata)?;

    let (mut tokenizer, kind) = match props.model.as_str() {
        "llama" | "replit" => unigram_tokenizer(&props)?,
        "gpt2" => bpe_tokenizer(&props)?,
        other => {
            anyhow::bail!("Tokenizer model `{other}` not supported.");
        }
    };

    // Token types other than `NORMAL` (1) are treated as special tokens.
    // Batch them so `AddedVocabulary` refreshes its matchers once.
    let mut special = Vec::new();
    if token_types.len() == props.tokens.len() {
        for (i, ty) in token_types.iter().enumerate() {
            if *ty != 1i32 {
                special.push(AddedToken::from(props.tokens[i].clone(), true));
            }
        }
    }
    let num_special_tokens = special.len();
    if !special.is_empty() {
        tokenizer.add_special_tokens(&special);
    }

    info!(
        "GGUF tokenizer model is `{model}`, kind: `{kind:?}`, num tokens: {}, num special tokens {}, num added tokens: {}, num merges: {}, num scores: {}",
        tokenizer.get_vocab_size(true),
        num_special_tokens,
        props.added_tokens.as_ref().map(|x| x.len()).unwrap_or(0),
        props.merges.as_ref().map(|x| x.len()).unwrap_or(0),
        props.scores.as_ref().map(|x| x.len()).unwrap_or(0),
        model = props.model,
    );
    if DEBUG.load(Ordering::Relaxed) {
        info!("Tokenizer: {tokenizer:?}");
    }

    let unk = match props.unk {
        Some(u) => Some(props.tokens[u as usize].clone()),
        _ => None,
    };

    let bos = match props.bos {
        Some(b) => Some(props.tokens[b as usize].clone()),
        None => None,
    };

    Ok(GgufTokenizerConversion {
        tokenizer,
        bos,
        eos: Some(props.tokens[props.eos as usize].clone()),
        unk,
    })
}

// TODO: Add support for additional tokenizer models: WordPiece, WordLevel
// https://docs.rs/tokenizers/latest/tokenizers/models/enum.ModelWrapper.html
#[derive(Debug)]
enum TokenizerKind {
    Unigram,
    Bpe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BpePreTokenizerKind {
    Llama3,
    Dbrx,
    Qwen2,
    Gpt2,
    Starcoder,
    Glm4,
    DeepSeekLlm,
    DeepSeekCoder,
    DeepSeekV3,
    Gpt4O,
    Tekken,
}

#[derive(Debug)]
struct BpePreTokenizerSpec {
    kind: BpePreTokenizerKind,
    regexes: &'static [&'static str],
    ignore_merges: bool,
    normalize_nfc: bool,
}

fn bpe_pre_tokenizer_spec(pre: Option<&str>) -> Result<BpePreTokenizerSpec> {
    let pre = pre.ok_or_else(|| {
        anyhow::anyhow!(
            "GGUF BPE tokenizer is missing required metadata `tokenizer.ggml.pre`; refusing to guess a pre-tokenizer regex"
        )
    })?;
    let spec = match pre {
        "llama3" | "llama-v3" | "llama-bpe" | "lfm2" | "pixtral" => BpePreTokenizerSpec {
            kind: BpePreTokenizerKind::Llama3,
            regexes: &[LLAMA3_REGEX],
            ignore_merges: true,
            normalize_nfc: false,
        },
        "dbrx" => BpePreTokenizerSpec {
            kind: BpePreTokenizerKind::Dbrx,
            regexes: &[LLAMA3_REGEX],
            ignore_merges: false,
            normalize_nfc: false,
        },
        "qwen2" | "deepseek-r1-qwen" => BpePreTokenizerSpec {
            kind: BpePreTokenizerKind::Qwen2,
            regexes: &[QWEN2_REGEX],
            ignore_merges: false,
            normalize_nfc: false,
        },
        "qwen35" => BpePreTokenizerSpec {
            kind: BpePreTokenizerKind::Qwen2,
            regexes: &[QWEN35_REGEX],
            ignore_merges: false,
            normalize_nfc: true,
        },
        "gpt-2" | "phi-2" => BpePreTokenizerSpec {
            kind: BpePreTokenizerKind::Gpt2,
            regexes: &[GPT2_REGEX],
            ignore_merges: false,
            normalize_nfc: false,
        },
        "starcoder" | "smollm" => BpePreTokenizerSpec {
            kind: BpePreTokenizerKind::Starcoder,
            regexes: &["\\p{N}", GPT2_REGEX],
            ignore_merges: false,
            normalize_nfc: false,
        },
        "glm4" | "chatglm-bpe" => BpePreTokenizerSpec {
            kind: BpePreTokenizerKind::Glm4,
            regexes: &[LLAMA3_REGEX],
            ignore_merges: false,
            normalize_nfc: false,
        },
        "deepseek-llm" => BpePreTokenizerSpec {
            kind: BpePreTokenizerKind::DeepSeekLlm,
            regexes: DEEPSEEK_LLM_REGEXES,
            ignore_merges: false,
            normalize_nfc: false,
        },
        "deepseek-coder" => BpePreTokenizerSpec {
            kind: BpePreTokenizerKind::DeepSeekCoder,
            regexes: DEEPSEEK_CODER_REGEXES,
            ignore_merges: false,
            normalize_nfc: false,
        },
        "deepseek-v3" | "hunyuan-dense" => BpePreTokenizerSpec {
            kind: BpePreTokenizerKind::DeepSeekV3,
            regexes: DEEPSEEK_V3_REGEXES,
            ignore_merges: false,
            normalize_nfc: false,
        },
        "hunyuan" => BpePreTokenizerSpec {
            kind: BpePreTokenizerKind::Qwen2,
            regexes: &[QWEN2_REGEX],
            ignore_merges: false,
            normalize_nfc: false,
        },
        "gpt-4o" | "llama4" => BpePreTokenizerSpec {
            kind: BpePreTokenizerKind::Gpt4O,
            regexes: &[GPT4O_REGEX],
            ignore_merges: false,
            normalize_nfc: false,
        },
        "tekken" => BpePreTokenizerSpec {
            kind: BpePreTokenizerKind::Tekken,
            regexes: &[TEKKEN_REGEX],
            ignore_merges: true,
            normalize_nfc: false,
        },
        _ => {
            anyhow::bail!(
                "GGUF BPE pre-tokenizer `{pre}` is not supported for standalone conversion; use the original tokenizer.json or add its exact tokenizer.ggml.pre profile"
            )
        }
    };
    Ok(spec)
}

fn unigram_tokenizer(p: &PropsGGUF) -> Result<(Tokenizer, TokenizerKind)> {
    let PropsGGUF { unk, eos, bos, .. } = *p;
    // Unigram (SentencePiece) default UNK is 0
    let unk = unk.unwrap_or(0);

    // Create the Tokenizer model:
    let model = {
        let vocab: Vec<(String, f64)> = {
            let Some(s) = p.scores.as_ref() else {
                anyhow::bail!(
                    "`llama` unigram tokenizer is missing required metadata `tokenizer.ggml.scores`"
                );
            };
            let scores = s.iter().cloned().map(|f_32| f_32 as f64);

            p.tokens.iter().cloned().zip(scores).collect()
        };

        Unigram::from(vocab, Some(unk as usize), true).map_err(anyhow::Error::msg)?
    };

    // Decoder + Normalizer config reference:
    // https://github.com/EricLBuehler/mistral.rs/pull/389#discussion_r1630620763
    let decoder = Decoder::Sequence(vec![
        Decoder::Replace("▁", " "),
        Decoder::ByteFallback,
        Decoder::Fuse,
        Decoder::Strip(' ', 1, 0),
    ]);

    let normalizer = Normalizer::Sequence(vec![
        Normalizer::Prepend("▁"),
        Normalizer::Replace(" ", "▁"),
    ]);

    let mut tokenizer: Tokenizer = TokenizerX::new(
        ModelWrapper::Unigram(model),
        Some(decoder),
        Some(normalizer),
    )?;

    // Add special tokens (bos, eos, unk):
    for v in [bos, Some(eos), Some(unk)].iter().flatten() {
        let tk = p.tokens[*v as usize].clone();
        tokenizer.add_special_tokens(&[AddedToken::from(tk.to_string(), true)]);
    }
    Ok((tokenizer, TokenizerKind::Unigram))
}

fn bpe_tokenizer(p: &PropsGGUF) -> Result<(Tokenizer, TokenizerKind)> {
    let pre_tokenizer = bpe_pre_tokenizer_spec(p.pre.as_deref())?;
    tracing::debug!(
        "Using GGUF BPE pre-tokenizer `{}` as {:?}",
        p.pre.as_deref().unwrap(),
        pre_tokenizer.kind
    );

    // BPE merges have each string item as a space-delimited pair:
    // https://github.com/EricLBuehler/mistral.rs/pull/397#discussion_r1631988370
    let merges = p
        .merges
        .as_ref()
        .ok_or(anyhow::Error::msg("BPE tokenizer must include merges"))?
        .iter()
        .map(|merge| {
            let (left, right) = merge.split_once(' ').ok_or_else(|| {
                anyhow::anyhow!(
                    "Invalid GGUF BPE merge `{merge}`; expected two space-delimited tokens"
                )
            })?;
            anyhow::ensure!(
                !left.is_empty() && !right.is_empty(),
                "Invalid GGUF BPE merge `{merge}`; both tokens must be non-empty"
            );
            Ok((left.to_string(), right.to_string()))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut vocab = AHashMap::new();
    for (i, token) in p.tokens.iter().enumerate() {
        #[allow(clippy::cast_possible_truncation)]
        vocab.insert(token.clone(), i as u32);
    }

    let PropsGGUF { bos, eos, unk, .. } = *p;

    let mut bpe = BpeBuilder::new()
        .vocab_and_merges(vocab, merges)
        .ignore_merges(pre_tokenizer.ignore_merges);
    if let Some(unk) = unk {
        bpe = bpe.unk_token(p.tokens[unk as usize].to_string());
    };

    let bpe = bpe.build().map_err(anyhow::Error::msg)?;

    let mut tokenizer = TokenizerX::new(
        ModelWrapper::BPE(bpe),
        Some(Decoder::ByteLevel(true, true, true)),
        pre_tokenizer.normalize_nfc.then_some(Normalizer::Nfc),
    )?;

    let mut pre_tokenizers = pre_tokenizer
        .regexes
        .iter()
        .map(|regex| {
            Split::new(
                SplitPattern::Regex((*regex).to_string()),
                SplitDelimiterBehavior::Isolated,
                false,
            )
            .map(PreTokenizerWrapper::Split)
            .map_err(|error| {
                anyhow::anyhow!(
                    "Invalid regex for GGUF BPE pre-tokenizer `{}`: {error}",
                    p.pre.as_deref().unwrap()
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    pre_tokenizers.push(PreTokenizerWrapper::ByteLevel(ByteLevel::new(
        false, false, false,
    )));
    tokenizer.with_pre_tokenizer(Some(Sequence::new(pre_tokenizers)));

    tokenizer.with_decoder(Some(decoders::byte_level::ByteLevel::new(
        false, false, false,
    )));
    tokenizer.with_post_processor(Some(processors::byte_level::ByteLevel::new(
        false, false, false,
    )));

    for v in [bos, Some(eos), unk].iter().flatten() {
        let tk = p.tokens[*v as usize].clone();
        tokenizer.add_special_tokens(&[AddedToken::from(tk.to_string(), true)]);
    }

    Ok((tokenizer, TokenizerKind::Bpe))
}

// This is a workaround to have a better builder API.
// Upstream `TokenizerBuilder` is difficult to work with:
// https://github.com/huggingface/tokenizers/issues/1549
struct TokenizerX;

impl TokenizerX {
    #[allow(clippy::new_ret_no_self)]
    fn new<'a>(
        model: ModelWrapper,
        decoder: Option<Decoder<'a>>,
        normalizer: Option<Normalizer<'a>>,
    ) -> Result<Tokenizer> {
        let mut tokenizer = Tokenizer::new(model);

        // Handle local enum to remote enum type:
        if let Some(decoder) = decoder {
            let d = DecoderWrapper::try_from(decoder)?;
            tokenizer.with_decoder(Some(d));
        }
        if let Some(normalizer) = normalizer {
            let n: NormalizerWrapper = NormalizerWrapper::try_from(normalizer)?;
            tokenizer.with_normalizer(Some(n));
        }

        Ok(tokenizer)
    }
}

// Convenient alternative to upstream:
// https://docs.rs/tokenizers/latest/tokenizers/decoders/enum.DecoderWrapper.html
enum Decoder<'a> {
    ByteFallback,
    Fuse,
    Replace(&'a str, &'a str),
    Strip(char, usize, usize),
    Sequence(Vec<Self>),
    ByteLevel(bool, bool, bool),
}

// Convert into upstream type wrapped enum variants:
impl TryFrom<Decoder<'_>> for DecoderWrapper {
    type Error = anyhow::Error;

    fn try_from(variant: Decoder) -> Result<Self, Self::Error> {
        let value: DecoderWrapper = match variant {
            Decoder::ByteFallback => ByteFallback::default().into(),
            Decoder::Fuse => Fuse::default().into(),
            Decoder::Replace(pattern, content) => Replace::new(pattern, content)
                .map_err(anyhow::Error::msg)?
                .into(),
            Decoder::Strip(content, start, stop) => Strip::new(content, start, stop).into(),
            Decoder::Sequence(decoders) => {
                let seq = decoders
                    .into_iter()
                    .map(DecoderWrapper::try_from)
                    .collect::<Result<Vec<DecoderWrapper>>>()?;

                decoders::sequence::Sequence::new(seq).into()
            }
            Decoder::ByteLevel(add_prefix_space, trim_offsets, use_regex) => {
                ByteLevel::new(add_prefix_space, trim_offsets, use_regex).into()
            }
        };

        Ok(value)
    }
}

// Convenient alternative to upstream:
// https://docs.rs/tokenizers/latest/tokenizers/normalizers/enum.NormalizerWrapper.html
enum Normalizer<'a> {
    Nfc,
    Prepend(&'a str),
    Replace(&'a str, &'a str),
    Sequence(Vec<Self>),
}

impl TryFrom<Normalizer<'_>> for NormalizerWrapper {
    type Error = anyhow::Error;

    fn try_from(variant: Normalizer) -> Result<Self, Self::Error> {
        let value: NormalizerWrapper = match variant {
            Normalizer::Nfc => NFC.into(),
            Normalizer::Prepend(prepend) => Prepend::new(prepend.to_owned()).into(),
            Normalizer::Replace(pattern, content) => Replace::new(pattern, content)
                .map_err(anyhow::Error::msg)?
                .into(),
            Normalizer::Sequence(decoders) => {
                let seq = decoders
                    .into_iter()
                    .map(NormalizerWrapper::try_from)
                    .collect::<Result<Vec<NormalizerWrapper>>>()?;

                normalizers::Sequence::new(seq).into()
            }
        };

        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::{bpe_pre_tokenizer_spec, bpe_tokenizer, BpePreTokenizerKind, PropsGGUF};
    use anyhow::Result;
    use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
    use tokenizers::Tokenizer;

    #[allow(dead_code)]
    #[derive(Debug)]
    enum TokenizerType {
        /// Mistral v0.1 tokenizer
        Llama,
        Replit,
        Gpt2,
        Rwkv,
    }

    fn get_gguf_tokenizer(tokenizer: TokenizerType) -> Result<Tokenizer> {
        match tokenizer {
            TokenizerType::Llama => {
                let api = ApiBuilder::new().with_progress(true).build().unwrap();
                let api = api.repo(Repo::with_revision(
                    "EricB/mistralrs_tests".to_string(),
                    RepoType::Model,
                    "main".to_string(),
                ));

                let filename = api.get("llama_gguf_tokenizer.json").unwrap();
                let tokenizer = Tokenizer::from_file(filename).expect("Valid tokenizer");
                Ok(tokenizer)
            }
            TokenizerType::Gpt2 => {
                let api = ApiBuilder::new().with_progress(true).build().unwrap();
                let api = api.repo(Repo::with_revision(
                    "EricB/mistralrs_tests".to_string(),
                    RepoType::Model,
                    "main".to_string(),
                ));

                let filename = api.get("gpt2_gguf_tokenizer.json").unwrap();
                let tokenizer = Tokenizer::from_file(filename).expect("Valid tokenizer");
                Ok(tokenizer)
            }
            other => anyhow::bail!("Cannot get testing HF tokenizer for type {other:?}"),
        }
    }

    fn get_hf_tokenizer(tokenizer: TokenizerType) -> Result<Tokenizer> {
        match tokenizer {
            TokenizerType::Llama => {
                let api = ApiBuilder::new().with_progress(true).build().unwrap();
                let api = api.repo(Repo::with_revision(
                    "EricB/mistralrs_tests".to_string(),
                    RepoType::Model,
                    "main".to_string(),
                ));

                let tokenizer_filename = api.get("tokenizer.json").unwrap();
                Ok(Tokenizer::from_file(tokenizer_filename).unwrap())
            }
            TokenizerType::Gpt2 => {
                let api = ApiBuilder::new().with_progress(true).build().unwrap();
                let api = api.repo(Repo::with_revision(
                    "EricB/mistralrs_tests".to_string(),
                    RepoType::Model,
                    "main".to_string(),
                ));

                let tokenizer_filename = api.get("tokenizer_gpt2.json").unwrap();
                Ok(Tokenizer::from_file(tokenizer_filename).unwrap())
            }
            other => anyhow::bail!("Cannot get testing HF tokenizer for type {other:?}"),
        }
    }

    // Content based upon https://github.com/ggerganov/llama.cpp/blob/master/tests/test-tokenizer-random.py#L99-L161
    fn get_test_passage() -> String {
        let passage = "Hello, world! \n🚀 (normal) 😶‍🌫️ (compound emoji, zwj sequence) ✅ (emoji as single token)\n你好世界！\nNǐ hǎo shìjiè!";

        passage.to_owned()
    }

    // The provided passage should encode and decode back into the same passage string:
    fn codec_roundtrip(
        tokenizer: &Tokenizer,
        passage: &str,
        add_special_tokens: bool,
    ) -> Result<String> {
        let tokenized = tokenizer
            .encode_fast(passage, add_special_tokens)
            .map_err(anyhow::Error::msg)?;

        // NOTE: The special tokens bool param meaning differs between encode() / decode():
        decode(tokenizer, tokenized.get_ids(), !add_special_tokens)
    }

    fn decode(
        tokenizer: &Tokenizer,
        token_ids: &[u32],
        skip_special_tokens: bool,
    ) -> Result<String> {
        tokenizer
            .decode(token_ids, skip_special_tokens)
            .map_err(anyhow::Error::msg)
    }

    fn test_bpe_props(pre: Option<&str>) -> PropsGGUF {
        PropsGGUF {
            model: "gpt2".to_string(),
            pre: pre.map(str::to_string),
            tokens: ["<eos>", "a", "b", "ab"]
                .into_iter()
                .map(str::to_string)
                .collect(),
            added_tokens: None,
            scores: None,
            merges: Some(Vec::new()),
            unk: None,
            bos: None,
            eos: 0,
        }
    }

    #[test]
    fn bpe_pre_tokenizer_profiles_cover_normal_families() -> Result<()> {
        let cases = [
            ("llama3", BpePreTokenizerKind::Llama3, true, false, 1),
            ("llama-bpe", BpePreTokenizerKind::Llama3, true, false, 1),
            ("lfm2", BpePreTokenizerKind::Llama3, true, false, 1),
            ("dbrx", BpePreTokenizerKind::Dbrx, false, false, 1),
            ("qwen2", BpePreTokenizerKind::Qwen2, false, false, 1),
            ("qwen35", BpePreTokenizerKind::Qwen2, false, true, 1),
            ("gpt-2", BpePreTokenizerKind::Gpt2, false, false, 1),
            ("phi-2", BpePreTokenizerKind::Gpt2, false, false, 1),
            ("starcoder", BpePreTokenizerKind::Starcoder, false, false, 2),
            ("smollm", BpePreTokenizerKind::Starcoder, false, false, 2),
            ("glm4", BpePreTokenizerKind::Glm4, false, false, 1),
            (
                "deepseek-llm",
                BpePreTokenizerKind::DeepSeekLlm,
                false,
                false,
                6,
            ),
            (
                "deepseek-coder",
                BpePreTokenizerKind::DeepSeekCoder,
                false,
                false,
                5,
            ),
            (
                "deepseek-v3",
                BpePreTokenizerKind::DeepSeekV3,
                false,
                false,
                3,
            ),
            (
                "hunyuan-dense",
                BpePreTokenizerKind::DeepSeekV3,
                false,
                false,
                3,
            ),
            ("hunyuan", BpePreTokenizerKind::Qwen2, false, false, 1),
            ("gpt-4o", BpePreTokenizerKind::Gpt4O, false, false, 1),
            ("tekken", BpePreTokenizerKind::Tekken, true, false, 1),
        ];

        for (pre, kind, ignore_merges, normalize_nfc, regex_count) in cases {
            let spec = bpe_pre_tokenizer_spec(Some(pre))?;
            assert_eq!(spec.kind, kind, "{pre}");
            assert_eq!(spec.ignore_merges, ignore_merges, "{pre}");
            assert_eq!(spec.normalize_nfc, normalize_nfc, "{pre}");
            assert_eq!(spec.regexes.len(), regex_count, "{pre}");
            bpe_tokenizer(&test_bpe_props(Some(pre)))?;
        }
        Ok(())
    }

    #[test]
    fn llama3_and_lfm2_enable_ignore_merges() -> Result<()> {
        for pre in ["llama3", "lfm2"] {
            let (tokenizer, _) = bpe_tokenizer(&test_bpe_props(Some(pre)))?;
            let encoding = tokenizer
                .encode_fast("ab", false)
                .map_err(anyhow::Error::msg)?;
            assert_eq!(encoding.get_ids(), &[3], "{pre}");
        }

        let (tokenizer, _) = bpe_tokenizer(&test_bpe_props(Some("qwen2")))?;
        let encoding = tokenizer
            .encode_fast("ab", false)
            .map_err(anyhow::Error::msg)?;
        assert_eq!(encoding.get_ids(), &[1, 2]);
        Ok(())
    }

    #[test]
    fn dbrx_profile_keeps_three_digit_chunks() -> Result<()> {
        let mut props = test_bpe_props(Some("dbrx"));
        props.tokens = ["<eos>", "1", "2", "3", "4", "12", "123", "1234"]
            .into_iter()
            .map(str::to_string)
            .collect();
        props.merges = Some(
            ["1 2", "12 3", "123 4"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        );
        let (tokenizer, _) = bpe_tokenizer(&props)?;
        let encoding = tokenizer
            .encode_fast("1234", false)
            .map_err(anyhow::Error::msg)?;
        assert_eq!(encoding.get_ids(), &[6, 4]);
        Ok(())
    }

    #[test]
    fn missing_or_unknown_bpe_pre_tokenizer_fails_explicitly() {
        let missing = bpe_tokenizer(&test_bpe_props(None)).unwrap_err();
        assert!(missing.to_string().contains("tokenizer.ggml.pre"));
        assert!(missing.to_string().contains("refusing to guess"));

        let unknown = bpe_tokenizer(&test_bpe_props(Some("future-tokenizer"))).unwrap_err();
        assert!(unknown.to_string().contains("future-tokenizer"));
        assert!(unknown.to_string().contains("original tokenizer.json"));
    }

    #[test]
    fn test_encode_decode_llama() -> Result<()> {
        use rand::rng;
        use rand::seq::SliceRandom;

        let passage = get_test_passage();
        let hf_tokenizer = get_hf_tokenizer(TokenizerType::Llama)?;
        let gguf_tokenizer = get_gguf_tokenizer(TokenizerType::Llama)?;

        // Without adding special tokens
        let hf_decoded = codec_roundtrip(&hf_tokenizer, passage.as_str(), false)?;
        let gguf_decoded = codec_roundtrip(&gguf_tokenizer, passage.as_str(), false)?;
        assert_eq!(hf_decoded, gguf_decoded);
        assert_eq!(passage, gguf_decoded);

        // With special tokens added
        // SKIPPED:
        // - Bugged the GGUF tokenizer does not prepend `<s> `
        // - Due to HF tokenizer using BPE (tokenizer.json) while GGUF tokenizer uses Unigram (metadata)?
        /*
        let hf_decoded = codec_roundtrip(&hf_tokenizer, passage.as_str(), true)?;
        let gguf_decoded = codec_roundtrip(&gguf_tokenizer, passage.as_str(), true)?;
        assert_eq!(hf_decoded, gguf_decoded);
        */

        #[allow(clippy::cast_possible_truncation)]
        let mut tokens = (0..hf_tokenizer.get_vocab_size(false) as u32).collect::<Vec<_>>();
        tokens.shuffle(&mut rng());

        // Without skipping special tokens
        let hf_decoded = decode(&hf_tokenizer, &tokens, false)?;
        let gguf_decoded = decode(&gguf_tokenizer, &tokens, false)?;
        assert_eq!(hf_decoded, gguf_decoded);

        // With skipping special tokens
        let hf_decoded = decode(&hf_tokenizer, &tokens, true)?;
        let gguf_decoded = decode(&gguf_tokenizer, &tokens, true)?;
        assert_eq!(hf_decoded, gguf_decoded);

        Ok(())
    }

    #[test]
    fn test_encode_decode_gpt2() -> Result<()> {
        use rand::rng;
        use rand::seq::SliceRandom;

        let passage = get_test_passage();
        let hf_tokenizer = get_hf_tokenizer(TokenizerType::Gpt2)?;
        let gguf_tokenizer = get_gguf_tokenizer(TokenizerType::Gpt2)?;

        // Without adding special tokens
        let hf_decoded = codec_roundtrip(&hf_tokenizer, passage.as_str(), false)?;
        let gguf_decoded = codec_roundtrip(&gguf_tokenizer, passage.as_str(), false)?;
        assert_eq!(hf_decoded, gguf_decoded);
        assert_eq!(passage, gguf_decoded);

        // With special tokens added
        // SKIPPED:
        // - Bugged the GGUF tokenizer does not prepend `<s> `
        // - Due to HF tokenizer using BPE (tokenizer.json) while GGUF tokenizer uses Unigram (metadata)?
        /*
        let hf_decoded = codec_roundtrip(&hf_tokenizer, passage.as_str(), true)?;
        let gguf_decoded = codec_roundtrip(&gguf_tokenizer, passage.as_str(), true)?;
        assert_eq!(hf_decoded, gguf_decoded);
        */

        #[allow(clippy::cast_possible_truncation)]
        let mut tokens = (0..hf_tokenizer.get_vocab_size(false) as u32).collect::<Vec<_>>();
        tokens.shuffle(&mut rng());

        // Without skipping special tokens
        let hf_decoded = decode(&hf_tokenizer, &tokens, false)?;
        let gguf_decoded = decode(&gguf_tokenizer, &tokens, false)?;
        assert_eq!(hf_decoded, gguf_decoded);

        // With skipping special tokens
        let hf_decoded = decode(&hf_tokenizer, &tokens, true)?;
        let gguf_decoded = decode(&gguf_tokenizer, &tokens, true)?;
        assert_eq!(hf_decoded, gguf_decoded);

        Ok(())
    }
}
