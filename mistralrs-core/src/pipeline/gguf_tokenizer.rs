use std::sync::atomic::Ordering;

use anyhow::Result;
use candle_core::quantized::gguf_file::Content;
use tokenizers::{
    decoders::{self, byte_fallback::ByteFallback, fuse::Fuse, strip::Strip},
    models::unigram::Unigram,
    normalizers::{self, Prepend, Replace},
    AddedToken, DecoderWrapper, ModelWrapper, NormalizerWrapper, Tokenizer,
};
use tracing::info;

use crate::utils::gguf_metadata::MetadataContext;
use crate::DEBUG;

pub struct ConversionResult {
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
    eos: u32,
    bos: u32,
}

// This approach is a workaround for candles GGUF `Value` enum type wrapper,
// a better upstream approach would be to have serialize/deserialize support.
impl TryFrom<MetadataContext<'_>> for PropsGGUF {
    type Error = anyhow::Error;

    fn try_from(c: MetadataContext) -> Result<Self, Self::Error> {
        let required = ["model", "tokens", "eos_token_id", "bos_token_id"];
        c.has_required_keys(&required)?;

        let tokenizer_ggml = PropsGGUF {
            model: c.get_value("model")?,
            tokens: c.get_value("tokens")?,
            added_tokens: c.get_value("added_tokens").ok(),
            scores: c.get_value("scores").ok(),
            merges: c.get_value("merges").ok(),
            unk: c.get_value("unknown_token_id").ok(),
            eos: c.get_value("eos_token_id")?,
            bos: c.get_value("bos_token_id")?,
        };

        Ok(tokenizer_ggml)
    }
}

pub fn convert_ggml_to_hf_tokenizer(content: &Content) -> Result<ConversionResult> {
    let metadata = MetadataContext {
        path_prefix: "tokenizer.ggml".to_string(),
        metadata: &content.metadata,
    };
    let props = PropsGGUF::try_from(metadata)?;

    let (tokenizer, kind, special_tokens) = match props.model.as_str() {
        "llama" | "replit" => unigram_tokenizer(&props)?,
        other => {
            anyhow::bail!("Tokenizer model `{other}` not supported.");
        }
    };

    info!(
        "GGUF tokenizer model is `{model}`, kind: `{kind:?}`, num tokens: {}, num added tokens: {}, num merges: {}, num scores: {}",
        tokenizer.get_vocab_size(true),
        props.added_tokens.as_ref().map(|x| x.len()).unwrap_or(0),
        props.merges.as_ref().map(|x| x.len()).unwrap_or(0),
        props.scores.as_ref().map(|x| x.len()).unwrap_or(0),
        model = props.model,
    );
    if DEBUG.load(Ordering::Relaxed) {
        info!("Tokenizer: {tokenizer:?}");
    }

    let [bos_str, eos_str, unk_str] = special_tokens
        .try_into()
        .or_else(|_| anyhow::bail!("Tokenizer is missing required special tokens"))?;

    Ok(ConversionResult {
        tokenizer,
        bos: Some(bos_str),
        eos: Some(eos_str),
        unk: Some(unk_str),
    })
}

// TODO: Add support for additional tokenizer models: BPE, WordPiece, WordLevel
// https://docs.rs/tokenizers/latest/tokenizers/models/enum.ModelWrapper.html
#[derive(Debug)]
enum TokenizerKind {
    Unigram,
}

fn unigram_tokenizer(p: &PropsGGUF) -> Result<(Tokenizer, TokenizerKind, Vec<String>)> {
    let PropsGGUF { unk, eos, bos, .. } = *p;
    // Unigram (SentencePiece) default UNK is 0
    let unk = unk.unwrap_or(0);

    let vocab: Vec<(String, f64)> = {
        let Some(s) = p.scores.as_ref() else {
            anyhow::bail!(
                "`llama` unigram tokenizer is missing required metadata `tokenizer.ggml.scores`"
            );
        };
        let scores = s.iter().cloned().map(|f_32| f_32 as f64);

        p.tokens.iter().cloned().zip(scores).collect()
    };

    let unigram = Unigram::from(vocab, Some(unk as usize), true).map_err(anyhow::Error::msg)?;
    let mut tokenizer = Tokenizer::new(ModelWrapper::Unigram(unigram));
    tokenizer.with_decoder(decoders::sequence::Sequence::new(vec![
        DecoderWrapper::Replace(Replace::new("▁", " ").map_err(anyhow::Error::msg)?),
        DecoderWrapper::ByteFallback(ByteFallback::new()),
        DecoderWrapper::Fuse(Fuse::new()),
        DecoderWrapper::Strip(Strip::new(' ', 1, 0)),
    ]));
    tokenizer.with_normalizer(normalizers::Sequence::new(vec![
        NormalizerWrapper::Prepend(Prepend::new("▁".to_string())),
        NormalizerWrapper::Replace(Replace::new(" ", "▁").map_err(anyhow::Error::msg)?),
    ]));

    let mut special_tokens = Vec::<String>::new();
    for token_id in [bos, eos, unk] {
        let token = p.tokens[token_id as usize].as_str();

        special_tokens.push(token.to_owned());
        tokenizer.add_special_tokens(&[AddedToken::from(token.to_owned(), true)]);
    }

    Ok((tokenizer, TokenizerKind::Unigram, special_tokens))
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use candle_core::quantized::gguf_file::Content;
    use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
    use tokenizers::Tokenizer;

    use super::convert_ggml_to_hf_tokenizer;

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
                    "TheBloke/Mistral-7B-Instruct-v0.1-GGUF".to_string(),
                    RepoType::Model,
                    "main".to_string(),
                ));

                let filename = api.get("mistral-7b-instruct-v0.1.Q2_K.gguf").unwrap();
                let mut file = std::fs::File::open(&filename)?;
                convert_ggml_to_hf_tokenizer(
                    &Content::read(&mut file)
                        .map_err(|e| e.with_path(filename))
                        .map_err(anyhow::Error::msg)?,
                )
                .map_err(anyhow::Error::msg)
                .map(|res| res.tokenizer)
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
            other => anyhow::bail!("Cannot get testing HF tokenizer for type {other:?}"),
        }
    }

    fn get_test_passage() -> String {
        // TODO: Why is it necessary to depend on this for a multi-line test string?
        let passage = reqwest::blocking::get("https://loripsum.net/api")
            .expect("Failed to download sample text")
            .bytes()
            .expect("Failed to get bytes");

        String::from_utf8(passage.to_vec()).expect("Failed to convert sample text to string.")
    }

    // The provided passage should encode and decode back into the same passage string:
    fn codec_roundtrip(
        tokenizer: &Tokenizer,
        passage: &str,
        add_special_tokens: bool,
    ) -> Result<String> {
        let tokenized = tokenizer
            .encode(passage, add_special_tokens)
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
    fn test_encode_llama() -> Result<()> {
        let passage = get_test_passage();
        let hf_tokenizer = get_hf_tokenizer(TokenizerType::Llama)?;
        let gguf_tokenizer = get_gguf_tokenizer(TokenizerType::Llama)?;

        // Without adding special tokens
        let hf_decoded = codec_roundtrip(&hf_tokenizer, passage.as_str(), false)?;
        let gguf_decoded = codec_roundtrip(&gguf_tokenizer, passage.as_str(), false)?;
        assert_eq!(hf_decoded, gguf_decoded);

        // With special tokens added
        let hf_decoded = codec_roundtrip(&hf_tokenizer, passage.as_str(), true)?;
        let gguf_decoded = codec_roundtrip(&gguf_tokenizer, passage.as_str(), true)?;
        assert_eq!(hf_decoded, gguf_decoded);

        Ok(())
    }

    #[test]
    fn test_decode_llama() -> Result<()> {
        use rand::seq::SliceRandom;
        use rand::thread_rng;

        let hf_tokenizer = get_hf_tokenizer(TokenizerType::Llama)?;
        let gguf_tokenizer = get_gguf_tokenizer(TokenizerType::Llama)?;

        #[allow(clippy::cast_possible_truncation)]
        let mut tokens = (0..hf_tokenizer.get_vocab_size(false) as u32).collect::<Vec<_>>();
        tokens.shuffle(&mut thread_rng());

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
