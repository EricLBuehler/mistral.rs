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

use crate::DEBUG;

pub fn convert_ggml_to_hf_tokenizer(content: &Content) -> Result<Tokenizer> {
    let model = content.metadata["tokenizer.ggml.model"]
        .to_string()
        .expect("GGUF tokenizer model is not a string.")
        .clone();
    let tokens = content.metadata["tokenizer.ggml.tokens"]
        .to_vec()
        .expect("GGUF tokenizer tokens is not a vec.")
        .iter()
        .map(|t| t.to_string().expect("GGUF token is not a string.").clone())
        .collect::<Vec<_>>();
    let added_tokens = content
        .metadata
        .get("tokenizer.ggml.added_tokens")
        .map(|items| {
            items
                .to_vec()
                .expect("GGUF tokenizer added_tokens is not a vec.")
                .iter()
                .map(|t| {
                    t.to_string()
                        .expect("GGUF added_token is not a string.")
                        .clone()
                })
                .collect::<Vec<_>>()
        });
    let scores = content.metadata.get("tokenizer.ggml.scores").map(|items| {
        items
            .to_vec()
            .expect("GGUF tokenizer scores is not a vec.")
            .iter()
            .map(|t| t.to_f32().expect("GGUF score is not a f32."))
            .collect::<Vec<_>>()
    });
    let merges = content.metadata.get("tokenizer.ggml.merges").map(|items| {
        items
            .to_vec()
            .expect("GGUF tokenizer merges is not a vec.")
            .iter()
            .map(|t| t.to_string().expect("GGUF merges is not a string.").clone())
            .collect::<Vec<_>>()
    });

    let unk = content.metadata["tokenizer.ggml.unknown_token_id"]
        .to_u32()
        .expect("GGUF unk token is not u32");

    let eos = content.metadata["tokenizer.ggml.eos_token_id"]
        .to_u32()
        .expect("GGUF unk token is not u32");

    let bos = content.metadata["tokenizer.ggml.bos_token_id"]
        .to_u32()
        .expect("GGUF unk token is not u32");

    let (tokenizer, ty) = match model.as_str() {
        "llama" | "replit" => {
            // unigram
            let scores = scores
                .as_ref()
                .expect("Expect `tokenizer.ggml.scores` for `llama` unigram tokeizer.");
            let mut vocab = Vec::new();
            for (token, score) in tokens.iter().zip(scores) {
                vocab.push((token.clone(), *score as f64));
            }
            let unigram =
                Unigram::from(vocab, Some(unk as usize), true).map_err(anyhow::Error::msg)?;
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

            tokenizer.add_special_tokens(&[AddedToken::from(tokens[bos as usize].clone(), true)]);
            tokenizer.add_special_tokens(&[AddedToken::from(tokens[eos as usize].clone(), true)]);
            tokenizer.add_special_tokens(&[AddedToken::from(tokens[unk as usize].clone(), true)]);

            (tokenizer, "unigram")
        }
        other => {
            anyhow::bail!("Tokenizer model `{other}` not supported.");
        }
    };
    info!(
        "GGUF tokenizer model is `{model}`, kind: `{}`, num tokens: {}, num added tokens: {}, num merges: {}, num scores: {}",
        ty,
        tokenizer.get_vocab_size(true),
        added_tokens.as_ref().map(|x| x.len()).unwrap_or(0),
        merges.as_ref().map(|x| x.len()).unwrap_or(0),
        scores.as_ref().map(|x| x.len()).unwrap_or(0)
    );
    if DEBUG.load(Ordering::Relaxed) {
        info!("Tokenizer: {tokenizer:?}");
    }
    Ok(tokenizer)
}

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

    #[allow(dead_code)]
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
            }
            other => anyhow::bail!("Cannot get testing HF tokenizer for type {other:?}"),
        }
    }

    #[allow(dead_code)]
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

    #[allow(dead_code)]
    fn get_test_passage() -> String {
        let passage = reqwest::blocking::get("https://loripsum.net/api")
            .expect("Failed to download sample text")
            .bytes()
            .expect("Failed to get bytes");
        String::from_utf8(passage.to_vec()).expect("Failed to convert sample text to string.")
    }

    #[test]
    fn test_encode_llama() -> Result<()> {
        let passage = get_test_passage();
        let hf_tokenizer = get_hf_tokenizer(TokenizerType::Llama)?;
        let gguf_tokenizer = get_gguf_tokenizer(TokenizerType::Llama)?;

        // Without special tokens
        let hf_tokenized = hf_tokenizer
            .encode(passage.as_str(), false)
            .map_err(anyhow::Error::msg)?;
        let gguf_tokenized = gguf_tokenizer
            .encode(passage.as_str(), false)
            .map_err(anyhow::Error::msg)?;
        let hf_decoded = hf_tokenizer
            .decode(hf_tokenized.get_ids(), false)
            .map_err(anyhow::Error::msg)?;
        let gguf_decoded = gguf_tokenizer
            .decode(gguf_tokenized.get_ids(), false)
            .map_err(anyhow::Error::msg)?;
        assert_eq!(hf_decoded, gguf_decoded);

        // With special tokens
        let hf_tokenized = hf_tokenizer
            .encode(passage.as_str(), true)
            .map_err(anyhow::Error::msg)?;
        let gguf_tokenized = gguf_tokenizer
            .encode(passage.as_str(), true)
            .map_err(anyhow::Error::msg)?;
        let hf_decoded = hf_tokenizer
            .decode(hf_tokenized.get_ids(), true)
            .map_err(anyhow::Error::msg)?;
        let gguf_decoded = gguf_tokenizer
            .decode(gguf_tokenized.get_ids(), true)
            .map_err(anyhow::Error::msg)?;
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
        let hf_decoded = hf_tokenizer
            .decode(&tokens, false)
            .map_err(anyhow::Error::msg)?;
        let gguf_decoded = gguf_tokenizer
            .decode(&tokens, false)
            .map_err(anyhow::Error::msg)?;
        assert_eq!(hf_decoded, gguf_decoded);

        // With skipping special tokens
        let hf_decoded = hf_tokenizer
            .decode(&tokens, true)
            .map_err(anyhow::Error::msg)?;
        let gguf_decoded = gguf_tokenizer
            .decode(&tokens, true)
            .map_err(anyhow::Error::msg)?;
        assert_eq!(hf_decoded, gguf_decoded);
        Ok(())
    }
}
