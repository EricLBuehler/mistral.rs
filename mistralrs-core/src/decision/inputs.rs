use std::{any::Any, sync::Arc};

use anyhow::{ensure, Context, Result};
use candle_core::Device;
use tokenizers::Tokenizer;

use crate::{
    device_map::DeviceMapper,
    pipeline::{
        text_models_inputs_processor::PagedAttentionMeta, InputProcessorOutput, InputsProcessor,
        InputsProcessorType, InputsProcessorValidationError, MessagesAction, Processor,
    },
    sequence::{MultimodalData, Sequence},
    vision_models::{
        lfm3_d1::{config::Config, DecisionAttention},
        lfm3_omni::Lfm3OmniProcessor,
        lfm3_vl::Lfm3VlProcessor,
        preprocessor_config::PreProcessorConfig,
        ModelInputs,
    },
};

use super::{
    DecisionCalibration, DecisionConfig, DecisionInstructions, DecisionMedia, DecisionPlan,
    DecisionPrompt,
};

pub(crate) struct DecisionRow {
    pub inputs: ModelInputs,
    pub plan: DecisionPlan,
    pub attention: DecisionAttention,
}

pub(crate) struct DecisionBatch {
    pub rows: Vec<Vec<DecisionRow>>,
    pub calibration: DecisionCalibration,
}

#[derive(Clone)]
pub(crate) struct DecisionProcessor {
    media: Arc<dyn Processor + Send + Sync>,
    config: DecisionConfig,
    max_seq_len: usize,
    supports_audio: bool,
}

impl DecisionProcessor {
    pub(crate) fn new(
        model_config: &str,
        mut preprocessor: PreProcessorConfig,
        max_edge: Option<u32>,
    ) -> Self {
        let model: Config = serde_json::from_str(model_config).expect("Invalid d1 config");
        let config = serde_json::from_str(model_config).expect("Invalid d1 readout config");
        preprocessor.max_pixels = Some(model.d1_vision_max_pixels);
        if let Some(size) = preprocessor.size.as_mut() {
            size.insert(
                "longest_edge".to_owned(),
                u32::try_from(model.d1_vision_max_pixels)
                    .expect("d1 maximum image size must fit u32"),
            );
        }
        let supports_audio = model.audio_encoder_config.is_some();
        let media: Arc<dyn Processor + Send + Sync> = if supports_audio {
            Arc::new(Lfm3OmniProcessor::new(
                &model.vision,
                &preprocessor,
                max_edge,
            ))
        } else {
            Arc::new(Lfm3VlProcessor::new(&model.vision, &preprocessor, max_edge))
        };
        Self {
            media,
            config,
            max_seq_len: model.vision.text_config.max_position_embeddings,
            supports_audio,
        }
    }

    fn prepare_rows(
        &self,
        seq: &mut Sequence,
        tokenizer: &Arc<Tokenizer>,
        device: &Device,
        other_config: Option<Arc<dyn Any>>,
        mapper: Option<&dyn DeviceMapper>,
    ) -> Result<Vec<DecisionRow>> {
        let request = seq.decision.clone().context("Missing decision request")?;
        if !self.supports_audio
            && (request
                .media
                .iter()
                .any(|item| matches!(item, DecisionMedia::Audio(_)))
                || request.questions.values().any(|question| {
                    matches!(question.instructions(), DecisionInstructions::Audio(_))
                }))
        {
            return Err(InputsProcessorValidationError(
                "This decision model does not support audio".into(),
            )
            .into());
        }
        let prompts = DecisionPrompt::prepare(&request, tokenizer, &self.config)
            .map_err(|error| InputsProcessorValidationError(error.to_string()))?;
        let processor = self.media.inputs_processor();
        let mut rows = Vec::with_capacity(prompts.len());
        for prompt in prompts {
            let mut images = Vec::new();
            let mut audios = Vec::new();
            let mut videos = Vec::new();
            for media in &prompt.media {
                match media {
                    DecisionMedia::Image(image) => images.push(image.clone()),
                    DecisionMedia::Audio(audio) => audios.push(audio.clone()),
                    DecisionMedia::Video(video) => videos.push(video.clone()),
                }
            }
            seq.multimodal =
                MultimodalData::new(Some(images), Some(audios), Some(videos), None, None, None);
            let ids = tokenizer
                .encode_fast(prompt.text.clone(), false)
                .map_err(anyhow::Error::msg)?;
            seq.set_toks_and_reallocate(ids.get_ids().to_vec(), None);
            seq.set_initial_prompt(prompt.text.clone());
            processor.prepare_for_paged_prompt_planning(
                Some(tokenizer.clone()),
                &mut [seq],
                device,
                other_config.clone(),
                None,
            )?;
            let expanded = tokenizer
                .decode(seq.get_toks(), false)
                .map_err(anyhow::Error::msg)?;
            let plan = prompt.plan(tokenizer, &expanded)?;
            ensure!(
                plan.ids == seq.get_toks(),
                "Decision slot layout does not match expanded media tokens"
            );
            if plan.ids.len() > self.max_seq_len {
                return Err(InputsProcessorValidationError(format!(
                    "Decision prompt has {} tokens, exceeding model context length {}",
                    plan.ids.len(),
                    self.max_seq_len,
                ))
                .into());
            }
            let output = processor.process_inputs(
                Some(tokenizer.clone()),
                &mut [seq],
                true,
                false,
                device,
                false,
                None,
                false,
                None,
                other_config.clone(),
                None,
                mapper,
            )?;
            let inputs = *output
                .inputs
                .downcast::<ModelInputs>()
                .expect("Decision media inputs");
            let attention = DecisionAttention {
                sequence_lengths: vec![plan.ids.len()],
                state_lengths: (self.config.d1_attention != "causal").then(|| vec![plan.state_len]),
                question_ids: (self.config.d1_attention == "set")
                    .then(|| vec![plan.question_ids.clone()]),
            };
            rows.push(DecisionRow {
                inputs,
                plan,
                attention,
            });
        }
        Ok(rows)
    }
}

impl Processor for DecisionProcessor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(self.clone())
    }
    fn get_special_tokens(&self) -> &[&'static str] {
        self.media.get_special_tokens()
    }
    fn template_action(&self) -> MessagesAction {
        MessagesAction::Keep
    }
}

impl InputsProcessor for DecisionProcessor {
    fn get_type(&self) -> InputsProcessorType {
        InputsProcessorType::Vision
    }
    fn process_inputs(
        &self,
        tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut Sequence],
        _is_prompt: bool,
        _is_xlora: bool,
        device: &Device,
        _no_kv_cache: bool,
        _last_n_context_len: Option<(usize, usize)>,
        _return_raw_logits: bool,
        _sliding_window: Option<usize>,
        other_config: Option<Arc<dyn Any>>,
        _paged_attn_metadata: Option<PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
    ) -> Result<InputProcessorOutput> {
        let tokenizer = tokenizer.context("Decision models require a tokenizer")?;
        let rows = input_seqs
            .iter_mut()
            .map(|seq| self.prepare_rows(seq, &tokenizer, device, other_config.clone(), mapper))
            .collect::<Result<Vec<_>>>()?;
        Ok(InputProcessorOutput {
            inputs: Box::new(DecisionBatch {
                rows,
                calibration: self.config.d1_calibration.clone(),
            }),
            seq_indices: (0..input_seqs.len()).collect(),
        })
    }
}
