use ahash::AHashMap;
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use kbnf::{
    engine_like::{AcceptTokenError, MaskLogitsError},
    AcceptTokenResult, Engine, EngineLike, Token, Vocabulary,
};
use tokenizers::Tokenizer;

pub struct KbnfGrammar {
    engine: Engine,
    vocab_size: usize,
}

pub enum KbnfGrammarBias {
    /// Token was accepted, it can be added to the sequence. No need for resampling.
    Accepted,
    /// Token was rejected. Resample with these new logits.
    /// The token sampled with the bias can be added to the sequence.
    Resample { new_logits: Tensor },
    /// Generation was finished, the token can be added to the sequence,
    /// but no more generation is necessary.
    FinishedGeneration,
}

impl KbnfGrammar {
    pub fn new(grammar: &str, tokenizer: &Tokenizer) -> Result<Self> {
        let tokenizer_vocab = tokenizer.get_vocab(true);
        let mut id_to_tok = AHashMap::new();
        let mut id_to_tok_str = AHashMap::new();
        for (tok_str, id) in tokenizer_vocab {
            id_to_tok.insert(id, Token(tok_str.as_bytes().to_vec().into_boxed_slice()));
            id_to_tok_str.insert(id, tok_str);
        }
        let vocab = Vocabulary::new(id_to_tok, id_to_tok_str)?;
        Ok(Self {
            engine: Engine::new(grammar, vocab)?,
            vocab_size: tokenizer.get_vocab_size(true),
        })
    }

    /// Compute the bias if this token were to be added.
    /// If the token can be added
    pub fn compute_bias_for(
        &mut self,
        tok: u32,
        logits: &Tensor,
    ) -> candle_core::Result<KbnfGrammarBias> {
        // Try to accept the new token
        match self.engine.try_accept_new_token(tok) {
            Ok(AcceptTokenResult::Ongoing) => {
                // Token was accepted, no resampling needed
                self.engine.compute_allowed_token_ids();
                Ok(KbnfGrammarBias::Accepted)
            }
            Err(AcceptTokenError::Rejected) => {
                self.engine.compute_allowed_token_ids();
                let mut bias = vec![0f32; self.vocab_size];
                match self.engine.mask_logits(&mut bias) {
                    Ok(()) => {
                        let new_logits = (logits.to_device(&Device::Cpu)?.to_dtype(DType::F32)?
                            + Tensor::from_vec(bias, (self.vocab_size,), &Device::Cpu)?)?;
                        Ok(KbnfGrammarBias::Resample { new_logits })
                    }
                    Err(MaskLogitsError::InvalidLogitsLength) => {
                        // This should really be unreachable.
                        candle_core::bail!("Invalid logits length {}", bias.len())
                    }
                }
            }
            Ok(AcceptTokenResult::Finished) | Err(AcceptTokenError::Finished) => {
                Ok(KbnfGrammarBias::FinishedGeneration)
            }
            Err(AcceptTokenError::UnknownTokenID) => candle_core::bail!("Unknown token ID {tok}"),
        }
    }
}
