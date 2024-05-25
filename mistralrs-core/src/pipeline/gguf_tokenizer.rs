use std::collections::HashMap;

use anyhow::Result;
use candle_core::quantized::gguf_file::Content;
use tokenizers::{models::bpe::BpeBuilder, AddedToken, ModelWrapper, Tokenizer};

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

    let _bos = content.metadata["tokenizer.ggml.bos_token_id"]
        .to_u32()
        .expect("GGUF bos token is not u32");
    let _eos = content.metadata["tokenizer.ggml.eos_token_id"]
        .to_u32()
        .expect("GGUF eos token is not u32");
    let unk = content.metadata["tokenizer.ggml.unknown_token_id"]
        .to_u32()
        .expect("GGUF unk token is not u32");
    let _sep = content.metadata["tokenizer.ggml.separator_token_id"]
        .to_u32()
        .expect("GGUF sep token is not u32");
    let _pad = content.metadata["tokenizer.ggml.padding_token_id"]
        .to_u32()
        .expect("GGUF pad token is not u32");

    let tokenizer = match model.as_str() {
        "llama" | "replit" | "gpt2" | "rwkv" => {
            // BPE, as seen in relevant tokenizer.json files
            let bpe_builder = BpeBuilder::new().unk_token(tokens[unk as usize].clone());

            let mut vocab = HashMap::new();
            for (i, tok) in tokens.into_iter().enumerate() {
                #[allow(clippy::cast_possible_truncation)]
                vocab.insert(tok, i as u32);
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
                .build()
                .map_err(anyhow::Error::msg)?;
            let mut tokenizer = Tokenizer::new(ModelWrapper::BPE(bpe));
            if let Some(added_tokens) = added_tokens {
                for added_token in added_tokens {
                    tokenizer.add_special_tokens(&[AddedToken::from(added_token, true)]);
                }
            }
            tokenizer
        }
        other => {
            anyhow::bail!("Tokenizer model `{other}` not supported.");
        }
    };
    Ok(tokenizer)
}
