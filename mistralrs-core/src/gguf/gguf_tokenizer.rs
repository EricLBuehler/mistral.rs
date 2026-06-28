// https://github.com/huggingface/transformers/blob/8685b3c5d2dd2550527773d2a02499495a759e31/src/transformers/convert_slow_tokenizer.py

use std::sync::atomic::Ordering;

use crate::utils::gguf_metadata::ContentMetadata;
use crate::DEBUG;
use ahash::AHashMap;
use anyhow::Result;
use candle_core::quantized::gguf_file::Value;
use itertools::Itertools;
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
    normalizers::{self, Prepend, Replace},
    processors, AddedToken, DecoderWrapper, ModelWrapper, NormalizerWrapper, Tokenizer,
};
use tracing::info;

use super::Content;

pub(crate) struct GgufTokenizerConversion {
    pub tokenizer: Tokenizer,
    pub bos: Option<String>,
    pub eos: Option<String>,
    pub unk: Option<String>,
}

struct PropsGGUF {
    model: String,
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

impl PropsGGUF {
    fn token_text(&self, id: u32, what: &str) -> Result<&String> {
        self.tokens.get(id as usize).ok_or_else(|| {
            anyhow::anyhow!(
                "GGUF tokenizer `{what}` id {id} is out of range for a vocab of {} tokens",
                self.tokens.len()
            )
        })
    }
}

fn token_type_as_i32(v: &Value) -> Result<i32> {
    let ty = match v {
        Value::I8(x) => *x as i64,
        Value::I16(x) => *x as i64,
        Value::I32(x) => *x as i64,
        Value::I64(x) => *x,
        Value::U8(x) => *x as i64,
        Value::U16(x) => *x as i64,
        Value::U32(x) => *x as i64,
        Value::U64(x) => i64::try_from(*x)
            .map_err(|_| anyhow::anyhow!("GGUF `tokenizer.ggml.token_type` entry {x} overflows"))?,
        other => anyhow::bail!(
            "GGUF `tokenizer.ggml.token_type` entries must be integers, got {:?}",
            other.value_type()
        ),
    };
    i32::try_from(ty).map_err(|_| anyhow::anyhow!("GGUF token type {ty} is out of i32 range"))
}

pub fn convert_gguf_to_hf_tokenizer<R: std::io::Seek + std::io::Read>(
    content: &Content<'_, R>,
) -> Result<GgufTokenizerConversion> {
    let metadata = ContentMetadata {
        path_prefix: "tokenizer.ggml",
        metadata: content.get_metadata(),
    };

    let md_get = |s: &str| match metadata.metadata.get(s) {
        None => candle_core::bail!("cannot find {s} in metadata"),
        Some(v) => Ok(v),
    };

    let mut token_types = Vec::<i32>::new();
    if metadata.metadata.contains_key("tokenizer.ggml.token_type") {
        let vtypes: &Vec<Value> = md_get("tokenizer.ggml.token_type")?
            .to_vec()
            .map_err(|_| anyhow::anyhow!("GGUF `tokenizer.ggml.token_type` must be an array"))?;
        for v in vtypes {
            token_types.push(token_type_as_i32(v)?);
        }
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
        Some(u) => Some(props.token_text(u, "unknown_token_id")?.clone()),
        _ => None,
    };

    let bos = match props.bos {
        Some(b) => Some(props.token_text(b, "bos_token_id")?.clone()),
        None => None,
    };

    let eos = props.token_text(props.eos, "eos_token_id")?.clone();

    Ok(GgufTokenizerConversion {
        tokenizer,
        bos,
        eos: Some(eos),
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

fn unigram_tokenizer(p: &PropsGGUF) -> Result<(Tokenizer, TokenizerKind)> {
    let PropsGGUF { unk, eos, bos, .. } = *p;
    // Unigram (SentencePiece) default UNK is 0
    let unk = unk.unwrap_or(0);
    p.token_text(unk, "unknown_token_id")?;

    // Create the Tokenizer model:
    let model = {
        let vocab: Vec<(String, f64)> = {
            let Some(s) = p.scores.as_ref() else {
                anyhow::bail!(
                    "`llama` unigram tokenizer is missing required metadata `tokenizer.ggml.scores`"
                );
            };
            if s.len() != p.tokens.len() {
                anyhow::bail!(
                    "GGUF tokenizer has {} tokens but {} scores; the file is malformed",
                    p.tokens.len(),
                    s.len()
                );
            }
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
        let tk = p.token_text(*v, "special token")?.clone();
        tokenizer.add_special_tokens(&[AddedToken::from(tk.to_string(), true)]);
    }
    Ok((tokenizer, TokenizerKind::Unigram))
}

fn bpe_tokenizer(p: &PropsGGUF) -> Result<(Tokenizer, TokenizerKind)> {
    // BPE merges have each string item as a space-delimited pair:
    // https://github.com/EricLBuehler/mistral.rs/pull/397#discussion_r1631988370
    let merges = p
        .merges
        .as_ref()
        .ok_or(anyhow::Error::msg("BPE tokenizer must include merges"))?
        .iter()
        .map(|merge| {
            merge
                .splitn(2, ' ')
                .collect_tuple()
                .map(|(a, b): (&str, &str)| (a.to_string(), b.to_string()))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "malformed GGUF BPE merge entry {merge:?}, expected `<left> <right>`"
                    )
                })
        })
        .collect::<Result<Vec<_>>>()?;

    let mut vocab = AHashMap::new();
    for (i, token) in p.tokens.iter().enumerate() {
        #[allow(clippy::cast_possible_truncation)]
        vocab.insert(token.clone(), i as u32);
    }

    let PropsGGUF { bos, eos, unk, .. } = *p;

    let mut bpe = BpeBuilder::new().vocab_and_merges(vocab, merges);
    if let Some(unk) = unk {
        bpe = bpe.unk_token(p.token_text(unk, "unknown_token_id")?.to_string());
    };

    let bpe = bpe.build().map_err(anyhow::Error::msg)?;

    let mut tokenizer = TokenizerX::new(
        ModelWrapper::BPE(bpe),
        Some(Decoder::ByteLevel(true, true, true)),
        None,
    )?;

    let split = Split::new(
        SplitPattern::Regex("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+".to_string()),
        SplitDelimiterBehavior::Isolated,
        false,
    ).unwrap();

    // example:
    // "type": "ByteLevel",
    // "add_prefix_space": false,
    // "trim_offsets": false,
    // "use_regex": false
    let pre_tokenizer = Sequence::new(vec![
        PreTokenizerWrapper::Split(split),
        PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, false, false)),
    ]);

    tokenizer.with_pre_tokenizer(Some(pre_tokenizer));

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
    Prepend(&'a str),
    Replace(&'a str, &'a str),
    Sequence(Vec<Self>),
}

impl TryFrom<Normalizer<'_>> for NormalizerWrapper {
    type Error = anyhow::Error;

    fn try_from(variant: Normalizer) -> Result<Self, Self::Error> {
        let value: NormalizerWrapper = match variant {
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

#[cfg(test)]
mod malformed_gguf_tests {
    use std::io::Cursor;

    use candle_core::quantized::gguf_file;

    use super::super::Content;
    use super::convert_gguf_to_hf_tokenizer;
    use crate::utils::gguf_metadata::ContentConfig;

    fn v_str(s: &str) -> gguf_file::Value {
        gguf_file::Value::String(s.to_string())
    }

    fn v_str_arr(items: &[&str]) -> gguf_file::Value {
        gguf_file::Value::Array(items.iter().map(|s| v_str(s)).collect())
    }

    fn v_f32_arr(items: &[f32]) -> gguf_file::Value {
        gguf_file::Value::Array(items.iter().map(|f| gguf_file::Value::F32(*f)).collect())
    }

    fn gguf_reader(metadata: &[(&str, gguf_file::Value)]) -> Cursor<Vec<u8>> {
        let mut buf = Cursor::new(Vec::new());
        let metadata: Vec<(&str, &gguf_file::Value)> =
            metadata.iter().map(|(k, v)| (*k, v)).collect();
        gguf_file::write(&mut buf, &metadata, &[]).unwrap();
        buf.set_position(0);
        buf
    }

    fn base_unigram() -> Vec<(&'static str, gguf_file::Value)> {
        vec![
            ("general.architecture", v_str("llama")),
            ("tokenizer.ggml.model", v_str("llama")),
            ("tokenizer.ggml.tokens", v_str_arr(&["<unk>", "a", "b"])),
            ("tokenizer.ggml.scores", v_f32_arr(&[0.0, -1.0, -2.0])),
            ("tokenizer.ggml.eos_token_id", gguf_file::Value::U32(2)),
        ]
    }

    fn convert(
        metadata: Vec<(&'static str, gguf_file::Value)>,
    ) -> anyhow::Result<super::GgufTokenizerConversion> {
        let mut reader = gguf_reader(&metadata);
        let mut readers = [&mut reader];
        let content = Content::from_readers(&mut readers).unwrap();
        convert_gguf_to_hf_tokenizer(&content)
    }

    fn expect_err_containing(metadata: Vec<(&'static str, gguf_file::Value)>, needle: &str) {
        let err = match convert(metadata) {
            Ok(_) => panic!("expected an error mentioning {needle:?}"),
            Err(e) => format!("{e:#}"),
        };
        assert!(
            err.contains(needle),
            "error {err:?} should mention {needle:?}"
        );
    }

    #[test]
    fn valid_unigram_converts() {
        let conv = convert(base_unigram()).unwrap();
        assert_eq!(conv.eos.as_deref(), Some("b"));
        assert_eq!(conv.unk, None);
    }

    #[test]
    fn valid_bpe_converts() {
        let metadata = vec![
            ("general.architecture", v_str("llama")),
            ("tokenizer.ggml.model", v_str("gpt2")),
            ("tokenizer.ggml.tokens", v_str_arr(&["a", "b", "ab"])),
            ("tokenizer.ggml.merges", v_str_arr(&["a b"])),
            ("tokenizer.ggml.eos_token_id", gguf_file::Value::U32(2)),
        ];
        let conv = convert(metadata).unwrap();
        assert_eq!(conv.eos.as_deref(), Some("ab"));
    }

    #[test]
    fn eos_id_out_of_range_is_an_error() {
        let mut m = base_unigram();
        m.retain(|(k, _)| *k != "tokenizer.ggml.eos_token_id");
        m.push(("tokenizer.ggml.eos_token_id", gguf_file::Value::U32(99)));
        expect_err_containing(m, "out of range");
    }

    #[test]
    fn unk_id_out_of_range_is_an_error() {
        let mut m = base_unigram();
        m.push(("tokenizer.ggml.unknown_token_id", gguf_file::Value::U32(99)));
        expect_err_containing(m, "out of range");
    }

    #[test]
    fn bos_id_out_of_range_is_an_error() {
        let mut m = base_unigram();
        m.push(("tokenizer.ggml.bos_token_id", gguf_file::Value::U32(99)));
        expect_err_containing(m, "out of range");
    }

    #[test]
    fn scores_length_mismatch_is_an_error() {
        let mut m = base_unigram();
        m.retain(|(k, _)| *k != "tokenizer.ggml.scores");
        m.push(("tokenizer.ggml.scores", v_f32_arr(&[0.0])));
        expect_err_containing(m, "scores");
    }

    #[test]
    fn malformed_merge_entry_is_an_error() {
        let metadata = vec![
            ("general.architecture", v_str("llama")),
            ("tokenizer.ggml.model", v_str("gpt2")),
            ("tokenizer.ggml.tokens", v_str_arr(&["a", "b", "ab"])),
            ("tokenizer.ggml.merges", v_str_arr(&["nospace"])),
            ("tokenizer.ggml.eos_token_id", gguf_file::Value::U32(2)),
        ];
        expect_err_containing(metadata, "merge");
    }

    #[test]
    fn token_type_accepts_any_integer_width() {
        let mut m = base_unigram();
        m.push((
            "tokenizer.ggml.token_type",
            gguf_file::Value::Array(vec![
                gguf_file::Value::U32(3),
                gguf_file::Value::U32(1),
                gguf_file::Value::U32(1),
            ]),
        ));
        convert(m).unwrap();
    }

    #[test]
    fn token_type_non_integer_is_an_error() {
        let mut m = base_unigram();
        m.push((
            "tokenizer.ggml.token_type",
            v_str_arr(&["not", "an", "int"]),
        ));
        expect_err_containing(m, "token_type");
    }

    #[test]
    fn split_count_with_wrong_type_is_an_error() {
        let mut m = base_unigram();
        m.push(("split.count", v_str("two")));
        let mut reader = gguf_reader(&m);
        let mut readers = [&mut reader];
        let err = match Content::from_readers(&mut readers) {
            Ok(_) => panic!("expected an error"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("split.count"), "{err}");
    }

    #[test]
    fn content_config_missing_key_is_an_error() {
        let mut reader = gguf_reader(&base_unigram());
        let mut readers = [&mut reader];
        let content = Content::from_readers(&mut readers).unwrap();
        let err = match ContentConfig::try_from(&content) {
            Ok(_) => panic!("expected an error"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("llama.context_length"), "{err}");
    }

    #[test]
    fn content_config_wrong_type_is_an_error() {
        let mut m = base_unigram();
        m.push(("llama.context_length", v_str("long")));
        m.push(("llama.embedding_length", gguf_file::Value::U32(16)));
        m.push(("llama.attention.head_count", gguf_file::Value::U32(2)));
        m.push(("llama.attention.head_count_kv", gguf_file::Value::U32(2)));
        m.push(("llama.block_count", gguf_file::Value::U32(1)));
        let mut reader = gguf_reader(&m);
        let mut readers = [&mut reader];
        let content = Content::from_readers(&mut readers).unwrap();
        let err = match ContentConfig::try_from(&content) {
            Ok(_) => panic!("expected an error"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("unsigned integer"), "{err}");
    }
}
