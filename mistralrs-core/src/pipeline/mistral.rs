use super::{Loader, ModelPaths, Pipeline, SimpleModelPaths, TokenSource};
use crate::{deref_mut_refcell, deref_refcell};
use crate::{
    models::mistral::{Config, Model},
    sequence::Sequence,
    utils::{
        dtype::get_dtype_from_torch_dtype, tokens::get_token,
        varbuilder_utils::from_mmaped_safetensors,
    },
};
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::Activation;
use candle_sampling::logits_processor::Logprobs;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use serde::Deserialize;
use std::cell::RefCell;
use std::{iter::repeat, rc::Rc, sync::Mutex};
use thiserror::Error;
use tokenizers::Tokenizer;

pub struct MistralPipeline {
    model: Model,
    tokenizer: Tokenizer,
    config: MistralSpecificConfig,
}

pub struct MistralLoader {
    model_id: String,
    default_dtype: DType,
    config: MistralSpecificConfig,
    forced_dtype: Option<DType>,
}

#[derive(Clone, Copy)]
pub struct MistralSpecificConfig {
    pub use_flash_attn: bool,
    pub repeat_last_n: usize,
}

#[derive(Deserialize)]
pub struct BasicConfig {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    hidden_act: Activation,
    max_position_embeddings: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    sliding_window: usize,
    torch_dtype: Option<String>,
}

#[derive(Error, Debug)]
enum TokenizerError {
    #[error("`{0}`")]
    Error(String),
}

impl MistralLoader {
    pub fn new(
        model_id: String,
        config: MistralSpecificConfig,
        forced_dtype: Option<DType>,
    ) -> Self {
        Self {
            model_id,
            default_dtype: DType::BF16,
            config,
            forced_dtype,
        }
    }
}

impl Loader for MistralLoader {
    fn download_model(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
    ) -> Result<Box<dyn ModelPaths>> {
        let api = ApiBuilder::new()
            .with_progress(true)
            .with_token(Some(get_token(&token_source)?))
            .build()?;
        let revision = revision.unwrap_or("main".to_string());
        let api = api.repo(Repo::with_revision(
            self.model_id.clone(),
            RepoType::Model,
            revision,
        ));

        let tokenizer_filename = api.get("tokenizer.json")?;

        let config_filename = api.get("config.json")?;

        let mut filenames = vec![];
        for rfilename in api
            .info()?
            .siblings
            .iter()
            .map(|x| x.rfilename.clone())
            .filter(|x| x.ends_with(".safetensors"))
        {
            let filename = api.get(&rfilename)?;
            filenames.push(filename);
        }

        Ok(Box::new(SimpleModelPaths {
            tokenizer_filename,
            config_filename,
            filenames,
        }))
    }

    fn _setup_model(
        &self,
        paths: &dyn ModelPaths,
        dtype: Option<DType>,
        device: &Device,
    ) -> Result<Box<Mutex<dyn Pipeline>>> {
        let basic_config: BasicConfig =
            serde_json::from_slice(&std::fs::read(paths.get_config_filename())?)?;
        let config = Config {
            vocab_size: basic_config.vocab_size,
            hidden_size: basic_config.hidden_size,
            intermediate_size: basic_config.intermediate_size,
            num_hidden_layers: basic_config.num_hidden_layers,
            num_attention_heads: basic_config.num_attention_heads,
            num_key_value_heads: basic_config.num_key_value_heads,
            hidden_act: basic_config.hidden_act,
            max_position_embeddings: basic_config.max_position_embeddings,
            rms_norm_eps: basic_config.rms_norm_eps,
            rope_theta: basic_config.rope_theta,
            sliding_window: basic_config.sliding_window,
            use_flash_attn: self.config.use_flash_attn,
        };
        let default_dtype = match (basic_config.torch_dtype, self.forced_dtype) {
            (_, Some(forced)) => forced,
            (Some(value), _) => get_dtype_from_torch_dtype(value)?,
            (None, _) => self.default_dtype,
        };

        let vb = from_mmaped_safetensors(
            paths.get_weight_filenames(),
            dtype.unwrap_or(default_dtype),
            device,
            false,
        )?;

        let model = Model::new(&config, vb)?;

        let tokenizer = Tokenizer::from_file(paths.get_tokenizer_filename())
            .map_err(|e| TokenizerError::Error(e.to_string()))?;

        Ok(Box::new(Mutex::new(MistralPipeline {
            model,
            tokenizer,
            config: self.config,
        })))
    }
}

impl Pipeline for MistralPipeline {
    fn forward(&mut self, input_toks: Box<[Rc<RefCell<Sequence>>]>) -> Result<Tensor> {
        // NOTE(EricLBuehler): Unwrap reasoning: Get the maximum sequence length.
        let max_len = input_toks
            .iter()
            .map(|seq| deref_refcell!(seq).len())
            .max()
            .unwrap();
        let padding_tok = 0;
        // Pad each sequence by the padding token to the max len.
        let mut seqs_tensors = Vec::new();
        let mut seqlen_offsets = Vec::new();
        for seq in input_toks.iter() {
            let context_size = if *deref_mut_refcell!(seq).gen_idx() > 0 {
                1
            } else {
                deref_refcell!(seq).get_toks().len()
            };
            let start_pos = deref_refcell!(seq)
                .get_toks()
                .len()
                .saturating_sub(context_size);
            let mut ctxt = deref_refcell!(seq).get_toks()[start_pos..].to_vec();
            seqlen_offsets.push(start_pos);
            *deref_mut_refcell!(seq).gen_idx() += 1;

            ctxt.extend(repeat(padding_tok).take(max_len - ctxt.len()));

            // NOTE(EricLBuehler): Unwrap reasoning: The dimensions must match.
            seqs_tensors.push(
                Tensor::new(ctxt, self.device())
                    .unwrap()
                    .unsqueeze(0)
                    .unwrap(),
            );
        }
        // NOTE(EricLBuehler): Unwrap reasoning: Correct dimensions are provided.
        let input_ids = Tensor::cat(&seqs_tensors, 0).unwrap();

        Ok(self.model.forward(&input_ids, &seqlen_offsets)?)
    }
    fn tokenize_prompt(&self, prompt: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow::Error::msg(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }
    fn device(&self) -> &Device {
        &self.model.device
    }
    fn num_hidden_layers(&self) -> usize {
        self.model.cache.lock().len()
    }
    fn cache(&self) -> &crate::models::Cache {
        &self.model.cache
    }
    fn sample(&mut self, logits: Tensor, seq: Rc<RefCell<Sequence>>) -> Result<Logprobs> {
        let logits = logits.squeeze(0).unwrap().to_dtype(DType::F32).unwrap();
        let start_at = deref_refcell!(seq)
            .get_toks()
            .len()
            .saturating_sub(self.config.repeat_last_n);
        let ctxt = deref_refcell!(seq).get_toks()[start_at..].to_vec();

        Ok(deref_mut_refcell!(seq)
            .logits_processor()
            .sample(&logits, Some(&ctxt))?)
    }
    fn tokenizer(&self) -> Tokenizer {
        self.tokenizer.clone()
    }
    fn eos_tok(&self) -> u32 {
        self.tokenizer
            .get_vocab(true)
            .get("</s>")
            .copied()
            .expect("Unable to extract `</s>` EOS token.")
    }
}
