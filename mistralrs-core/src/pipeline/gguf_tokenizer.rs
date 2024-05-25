use std::collections::HashMap;

use anyhow::Result;
use candle_core::quantized::gguf_file::Content;
use tokenizers::{
    decoders::{byte_fallback::ByteFallback, fuse::Fuse, sequence::Sequence, strip::Strip},
    models::bpe::BpeBuilder,
    normalizers::{self, Prepend, Replace},
    processors::template::{self, Template, TemplateProcessing, Tokens},
    AddedToken, DecoderWrapper, ModelWrapper, NormalizerWrapper, Tokenizer,
};
use tracing::info;

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
    let merges = content.metadata.get("tokenizer.ggml.merges").map(|items| {
        items
            .to_vec()
            .expect("GGUF tokenizer merges is not a vec.")
            .iter()
            .map(|t| t.to_string().expect("GGUF merges is not a string.").clone())
            .collect::<Vec<_>>()
    });

    info!(
        "Converting GGML tokenizer. Model: `{model}`, num tokens: {}, num added tokens: {}, num merges: {}",
        tokens.len(),
        added_tokens.as_ref().map(|x| x.len()).unwrap_or(0),
        merges.as_ref().map(|x| x.len()).unwrap_or(0)
    );
    let unk = content.metadata["tokenizer.ggml.unknown_token_id"]
        .to_u32()
        .expect("GGUF unk token is not u32");

    let eos = content.metadata["tokenizer.ggml.eos_token_id"]
        .to_u32()
        .expect("GGUF unk token is not u32");

    let bos = content.metadata["tokenizer.ggml.bos_token_id"]
        .to_u32()
        .expect("GGUF unk token is not u32");

    let tokenizer = match model.as_str() {
        "llama" | "replit" | "gpt2" | "rwkv" => {
            // BPE, as seen in relevant tokenizer.json files
            let bpe_builder = BpeBuilder::new().unk_token(tokens[unk as usize].clone());
            info!("Loading as BPE tokenizer.");

            let mut vocab = HashMap::new();
            for (i, tok) in tokens.iter().enumerate() {
                #[allow(clippy::cast_possible_truncation)]
                vocab.insert(tok.clone(), i as u32);
            }
            let mut merges_vec = Vec::new();
            if let Some(merges) = merges {
                for tok in merges {
                    let split = tok.splitn(2, ' ').collect::<Vec<_>>();
                    merges_vec.push((split[0].to_string(), split[1].to_string()));
                }
            }
            let bpe = bpe_builder
                .vocab_and_merges(vocab, merges_vec)
                .fuse_unk(true)
                .build()
                .map_err(anyhow::Error::msg)?;
            let mut tokenizer = Tokenizer::new(ModelWrapper::BPE(bpe));
            tokenizer.with_decoder(Sequence::new(vec![
                DecoderWrapper::Replace(Replace::new("▁", " ").map_err(anyhow::Error::msg)?),
                DecoderWrapper::ByteFallback(ByteFallback::default()),
                DecoderWrapper::Fuse(Fuse::new()),
                DecoderWrapper::Strip(Strip::new(' ', 1, 0)),
            ]));
            if let Some(added_tokens) = added_tokens {
                for added_token in added_tokens {
                    tokenizer.add_special_tokens(&[AddedToken::from(added_token, true)]);
                }
            }
            tokenizer.add_special_tokens(&[AddedToken::from(tokens[bos as usize].clone(), true)]);
            tokenizer.add_special_tokens(&[AddedToken::from(tokens[eos as usize].clone(), true)]);
            tokenizer.add_special_tokens(&[AddedToken::from(tokens[unk as usize].clone(), true)]);

            tokenizer.with_post_processor(
                TemplateProcessing::builder()
                    .special_tokens(Tokens::from(vec![template::SpecialToken::new(
                        tokens[bos as usize].clone(),
                        vec![bos],
                        vec![tokens[bos as usize].clone()],
                    )
                    .map_err(anyhow::Error::msg)?]))
                    .pair(
                        Template::try_from(vec![
                            tokens[bos as usize].clone(),
                            "$A".to_string(),
                            tokens[bos as usize].clone(),
                            "$B:1".to_string(),
                        ])
                        .unwrap(),
                    )
                    .single(
                        Template::try_from(vec![tokens[bos as usize].clone(), "$A".to_string()])
                            .unwrap(),
                    )
                    .build()?,
            );
            tokenizer.with_normalizer(normalizers::Sequence::new(vec![
                NormalizerWrapper::Prepend(Prepend::new("▁".to_string())),
                NormalizerWrapper::Replace(Replace::new(" ", "▁").map_err(anyhow::Error::msg)?),
            ]));
            info!("Decoder is: {:?}", tokenizer.get_decoder());
            tokenizer
        }
        other => {
            anyhow::bail!("Tokenizer model `{other}` not supported.");
        }
    };
    Ok(tokenizer)
}
