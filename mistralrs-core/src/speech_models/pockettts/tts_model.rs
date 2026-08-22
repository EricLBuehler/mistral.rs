//! High-level pocket-tts pipeline: text -> latents (FlowLM) -> audio (Mimi codec).
//! Ported from the upstream `pocket-tts` crate against candle 0.11. The
//! `without-voice-cloning` checkpoint runs with an empty voice state (default speaker).

use std::path::Path;

use super::conditioners::text::LUTConditioner;
use super::config::{defaults, PocketTtsConfig};
use super::models::flow_lm::FlowLMModel;
use super::models::mimi::MimiModel;
use super::models::seanet::{SEANetDecoder, SEANetEncoder};
use super::models::transformer::{ProjectedTransformer, StreamingTransformer};
use super::modules::mlp::SimpleMLPAdaLN;
use super::voice_state::{increment_steps, init_states, ModelState};

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

#[derive(Clone)]
pub struct TTSModel {
    pub flow_lm: FlowLMModel,
    pub mimi: MimiModel,
    pub conditioner: LUTConditioner,
    pub temp: f32,
    pub lsd_decode_steps: usize,
    pub eos_threshold: f32,
    pub sample_rate: usize,
    pub dim: usize,
    pub ldim: usize,
    pub device: Device,
}

impl TTSModel {
    pub fn new(config: &PocketTtsConfig, vb: VarBuilder, tokenizer_path: &Path) -> Result<Self> {
        let device = vb.device().clone();

        let conditioner = LUTConditioner::new(
            config.flow_lm.lookup_table.n_bins,
            tokenizer_path,
            config.flow_lm.lookup_table.dim,
            config.flow_lm.transformer.d_model,
            vb.pp("flow_lm.conditioner"),
        )?;

        let dim = config.flow_lm.transformer.d_model;
        let ldim = config.mimi.quantizer.dimension;
        let hidden_dim = dim * config.flow_lm.transformer.hidden_scale;

        let flow_net = SimpleMLPAdaLN::new(
            ldim,
            config.flow_lm.flow.dim,
            ldim,
            dim,
            config.flow_lm.flow.depth,
            2,
            config.flow_lm.transformer.max_period as f32,
            vb.pp("flow_lm.flow_net"),
        )?;

        let transformer = StreamingTransformer::new(
            dim,
            config.flow_lm.transformer.num_heads,
            config.flow_lm.transformer.num_layers,
            None,
            hidden_dim,
            None,
            config.flow_lm.transformer.max_period as f32,
            "kv",
            "flow_lm.transformer",
            vb.pp("flow_lm.transformer"),
        )?;

        let flow_lm = FlowLMModel::new(flow_net, transformer, ldim, dim, vb.pp("flow_lm"))?;

        let seanet_cfg = &config.mimi.seanet;
        let encoder = SEANetEncoder::new(
            seanet_cfg.channels,
            seanet_cfg.dimension,
            seanet_cfg.n_filters,
            seanet_cfg.n_residual_layers,
            &seanet_cfg.ratios,
            seanet_cfg.kernel_size,
            seanet_cfg.last_kernel_size,
            seanet_cfg.residual_kernel_size,
            seanet_cfg.dilation_base,
            &seanet_cfg.pad_mode,
            seanet_cfg.compress,
            "mimi.encoder",
            vb.pp("mimi.encoder"),
        )?;

        let decoder = SEANetDecoder::new(
            seanet_cfg.channels,
            seanet_cfg.dimension,
            seanet_cfg.n_filters,
            seanet_cfg.n_residual_layers,
            &seanet_cfg.ratios,
            seanet_cfg.kernel_size,
            seanet_cfg.last_kernel_size,
            seanet_cfg.residual_kernel_size,
            seanet_cfg.dilation_base,
            &seanet_cfg.pad_mode,
            seanet_cfg.compress,
            "mimi.decoder",
            vb.pp("mimi.decoder"),
        )?;

        let mimi_tr_cfg = &config.mimi.transformer;
        let encoder_transformer = ProjectedTransformer::new(
            mimi_tr_cfg.input_dimension,
            mimi_tr_cfg.output_dimensions.clone(),
            mimi_tr_cfg.d_model,
            mimi_tr_cfg.num_heads,
            mimi_tr_cfg.num_layers,
            mimi_tr_cfg.layer_scale as f32,
            mimi_tr_cfg.context,
            mimi_tr_cfg.max_period as f32,
            mimi_tr_cfg.dim_feedforward,
            "mimi.encoder_transformer",
            vb.pp("mimi.encoder_transformer"),
        )?;

        let decoder_transformer = ProjectedTransformer::new(
            mimi_tr_cfg.input_dimension,
            mimi_tr_cfg.output_dimensions.clone(),
            mimi_tr_cfg.d_model,
            mimi_tr_cfg.num_heads,
            mimi_tr_cfg.num_layers,
            mimi_tr_cfg.layer_scale as f32,
            mimi_tr_cfg.context,
            mimi_tr_cfg.max_period as f32,
            mimi_tr_cfg.dim_feedforward,
            "mimi.decoder_transformer",
            vb.pp("mimi.decoder_transformer"),
        )?;

        let hop_length: usize = seanet_cfg.ratios.iter().product();
        let encoder_frame_rate = config.mimi.sample_rate as f64 / hop_length as f64;

        let mimi = MimiModel::new(
            encoder,
            decoder,
            encoder_transformer,
            decoder_transformer,
            config.mimi.frame_rate,
            encoder_frame_rate,
            config.mimi.sample_rate,
            config.mimi.channels,
            config.mimi.quantizer.dimension,
            config.mimi.quantizer.output_dimension,
            "mimi",
            vb.pp("mimi"),
        )?;

        Ok(Self {
            flow_lm,
            mimi,
            conditioner,
            temp: defaults::TEMPERATURE,
            lsd_decode_steps: defaults::LSD_DECODE_STEPS,
            eos_threshold: defaults::EOS_THRESHOLD,
            sample_rate: config.mimi.sample_rate,
            dim,
            ldim,
            device,
        })
    }

    /// Load a precomputed speaker latent prompt (`.safetensors` with an `audio_prompt` tensor,
    /// shape `[1, T, d_model]`) and prime a `ModelState` by running it through FlowLM.
    pub fn voice_state_from_prompt_file(&self, path: &Path) -> Result<ModelState> {
        let tensors = candle_core::safetensors::load(path, &self.device)?;
        let prompt = tensors
            .get("audio_prompt")
            .ok_or_else(|| anyhow::anyhow!("'audio_prompt' not found in {path:?}"))?;
        let prompt = if prompt.device().same_device(&self.device) {
            prompt.clone()
        } else {
            prompt.to_device(&self.device)?
        };
        let mut state = init_states(1, 1000);
        self.run_flow_lm_prompt(&prompt, &mut state)?;
        Ok(state)
    }

    fn run_flow_lm_prompt(&self, conditioning: &Tensor, state: &mut ModelState) -> Result<()> {
        let empty_text = Tensor::zeros((1, 0), DType::I64, &self.device)?;
        let text_embeddings = self.conditioner.forward(&empty_text)?;
        let input = Tensor::cat(&[conditioning, &text_embeddings], 1)?;
        let _ = self.flow_lm.transformer.forward(&input, state, 0)?;
        let increment_by = conditioning.dims()[1];
        increment_steps(state, "offset", increment_by);
        Ok(())
    }

    /// Token-length-aware sentence chunking, keeping each chunk under `MAX_TOKENS_PER_CHUNK` to
    /// preserve O(N) attention for long inputs.
    pub fn split_into_best_sentences(&self, text: &str) -> Vec<String> {
        const MAX_TOKENS_PER_CHUNK: usize = 50;

        let prepared_text = prepare_text_prompt(text);

        let raw_sentences: Vec<&str> = prepared_text
            .split_inclusive(['.', '!', '?', ';', ':'])
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        if raw_sentences.is_empty() {
            return vec![prepared_text];
        }

        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut current_token_count = 0;

        for sentence in raw_sentences {
            let sentence_tokens = self
                .conditioner
                .count_tokens(sentence)
                .unwrap_or(MAX_TOKENS_PER_CHUNK);

            if sentence_tokens > MAX_TOKENS_PER_CHUNK {
                if !current_chunk.is_empty() {
                    chunks.push(current_chunk);
                    current_chunk = String::new();
                    current_token_count = 0;
                }

                let words: Vec<&str> = sentence.split_whitespace().collect();
                const WORDS_PER_BATCH: usize = 35;

                for word_batch in words.chunks(WORDS_PER_BATCH) {
                    let chunk_str = word_batch.join(" ");
                    let actual_tokens = self
                        .conditioner
                        .count_tokens(&chunk_str)
                        .unwrap_or(MAX_TOKENS_PER_CHUNK);

                    if actual_tokens <= MAX_TOKENS_PER_CHUNK {
                        chunks.push(chunk_str);
                    } else {
                        let mid = word_batch.len() / 2;
                        chunks.push(word_batch[..mid].join(" "));
                        chunks.push(word_batch[mid..].join(" "));
                    }
                }
                continue;
            }

            if current_chunk.is_empty() {
                current_chunk = sentence.to_string();
                current_token_count = sentence_tokens;
            } else if current_token_count + sentence_tokens > MAX_TOKENS_PER_CHUNK {
                chunks.push(current_chunk);
                current_chunk = sentence.to_string();
                current_token_count = sentence_tokens;
            } else {
                current_chunk.push(' ');
                current_chunk.push_str(sentence);
                current_token_count += sentence_tokens;
            }
        }

        if !current_chunk.is_empty() {
            chunks.push(current_chunk);
        }

        chunks
    }

    /// Generate mono audio for `text` conditioned on `voice_state` (a primed speaker prompt).
    pub fn generate(&self, text: &str, voice_state: &ModelState) -> Result<Tensor> {
        let mut audio_chunks = Vec::new();
        for chunk in self.generate_stream(text, voice_state) {
            audio_chunks.push(chunk?);
        }
        if audio_chunks.is_empty() {
            anyhow::bail!("No audio generated");
        }
        let audio = Tensor::cat(&audio_chunks, 2)?;
        Ok(audio.squeeze(0)?)
    }

    fn generate_stream<'a>(
        &'a self,
        text: &str,
        voice_state: &ModelState,
    ) -> Box<dyn Iterator<Item = Result<Tensor>> + 'a> {
        let chunks = self.split_into_best_sentences(text);
        let voice_state_owned = voice_state.clone();
        let iterator = chunks.into_iter().flat_map(move |chunk_text| {
            self.generate_stream_segment(chunk_text, &voice_state_owned)
        });
        Box::new(iterator)
    }

    fn generate_stream_segment(
        &self,
        text: String,
        voice_state: &ModelState,
    ) -> Box<dyn Iterator<Item = Result<Tensor>>> {
        let mut state = voice_state.clone();
        let mut mimi_state = init_states(1, 1000);

        let prepared_text = prepare_text_prompt(&text);

        let tokens = match self.conditioner.prepare(&prepared_text, &self.device) {
            Ok(t) => t,
            Err(e) => return Box::new(std::iter::once(Err(e))),
        };

        let text_embeddings = match self.conditioner.forward(&tokens) {
            Ok(e) => e,
            Err(e) => return Box::new(std::iter::once(Err(e))),
        };

        if let Err(e) = self
            .flow_lm
            .transformer
            .forward(&text_embeddings, &mut state, 0)
        {
            return Box::new(std::iter::once(Err(anyhow::Error::from(e))));
        }

        let max_gen_len = (prepared_text.split_whitespace().count() + 2) * 13;
        let frames_after_eos = estimate_frames_after_eos(&text);

        let mut backbone_input = match self.flow_lm.bos_emb.clone().reshape((1, 1, self.ldim)) {
            Ok(t) => t,
            Err(e) => return Box::new(std::iter::once(Err(anyhow::Error::from(e)))),
        };

        let mut eos_step: Option<usize> = None;
        let mut finished = false;

        let model = self.clone();

        let time_embeddings = match model.flow_lm.flow_net.compute_time_embeddings(
            model.lsd_decode_steps,
            &model.device,
            DType::F32,
        ) {
            Ok(te) => te,
            Err(e) => return Box::new(std::iter::once(Err(anyhow::Error::from(e)))),
        };

        let empty_text_embeddings =
            Tensor::zeros((1, 0, model.dim), DType::F32, &model.device).unwrap();

        Box::new((0..max_gen_len).map_while(move |step| {
            if finished {
                return None;
            }

            let (next_latent, is_eos) = match model.flow_lm.forward(
                &backbone_input,
                &empty_text_embeddings,
                &mut state,
                &time_embeddings,
                model.temp,
                model.eos_threshold,
                step,
            ) {
                Ok(res) => res,
                Err(e) => return Some(Err(anyhow::anyhow!(e))),
            };

            let audio_frame = match (|| -> Result<Tensor> {
                let next_latent_denorm = next_latent
                    .broadcast_mul(&model.flow_lm.emb_std)?
                    .broadcast_add(&model.flow_lm.emb_mean)?;

                let mimi_input = next_latent_denorm.unsqueeze(1)?.transpose(1, 2)?;
                let quantized = model.mimi.quantize(&mimi_input)?;
                let audio = model
                    .mimi
                    .decode_from_latent(&quantized, &mut mimi_state, step)
                    .map_err(|e| anyhow::anyhow!(e))?;
                Ok(audio)
            })() {
                Ok(frame) => frame,
                Err(e) => return Some(Err(e)),
            };

            if is_eos && eos_step.is_none() {
                eos_step = Some(step);
            }

            if let Some(e_step) = eos_step {
                if step >= e_step + frames_after_eos {
                    finished = true;
                }
            }

            backbone_input = next_latent.unsqueeze(1).unwrap();

            Some(Ok(audio_frame))
        }))
    }
}

fn prepare_text_prompt(text: &str) -> String {
    let text = super::pause::strip_pause_markers(text);

    let mut text = text.trim().to_string();
    if text.is_empty() {
        return ".".to_string();
    }

    text = text.replace(['\n', '\r'], " ").replace("  ", " ");

    let word_count = text.split_whitespace().count();

    if let Some(first) = text.chars().next() {
        if !first.is_uppercase() {
            text = format!("{}{}", first.to_uppercase(), &text[first.len_utf8()..]);
        }
    }

    if let Some(last) = text.chars().last() {
        if last.is_alphanumeric() {
            text.push('.');
        }
    }

    if word_count < 5 {
        text = format!("{}{}", " ".repeat(8), text);
    }

    text
}

fn estimate_frames_after_eos(text: &str) -> usize {
    let word_count = text.split_whitespace().count();
    if word_count <= 4 {
        3 + 2
    } else {
        1 + 2
    }
}
