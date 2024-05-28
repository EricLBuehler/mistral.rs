use anyhow::Result;
use candle_core::quantized::gguf_file::Content;
use tokenizers::{models::unigram::Unigram, ModelWrapper, Tokenizer};
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

    info!(
        "Converting GGML tokenizer. Model: `{model}`, num tokens: {}, num added tokens: {}, num merges: {}, num scores: {}",
        tokens.len(),
        added_tokens.as_ref().map(|x| x.len()).unwrap_or(0),
        merges.as_ref().map(|x| x.len()).unwrap_or(0),
        scores.as_ref().map(|x| x.len()).unwrap_or(0)
    );
    let unk = content.metadata["tokenizer.ggml.unknown_token_id"]
        .to_u32()
        .expect("GGUF unk token is not u32");

    let _eos = content.metadata["tokenizer.ggml.eos_token_id"]
        .to_u32()
        .expect("GGUF unk token is not u32");

    let _bos = content.metadata["tokenizer.ggml.bos_token_id"]
        .to_u32()
        .expect("GGUF unk token is not u32");

    let tokenizer = match model.as_str() {
        "llama" => {
            let scores =
                scores.expect("Expect `tokenizer.ggml.scores` for `llama` unigram tokeizer.");
            let mut vocab = Vec::new();
            for (token, score) in tokens.into_iter().zip(scores) {
                vocab.push((token, score as f64));
            }
            let unigram =
                Unigram::from(vocab, Some(unk as usize), true).map_err(anyhow::Error::msg)?;
            Tokenizer::new(ModelWrapper::Unigram(unigram))
        }
        other => {
            anyhow::bail!("Tokenizer model `{other}` not supported.");
        }
    };
    Ok(tokenizer)
}
