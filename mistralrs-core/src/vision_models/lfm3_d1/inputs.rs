use std::{ops::Range, sync::Arc};

use anyhow::{ensure, Context, Result};
use candle_core::Tensor;
use tokenizers::{Encoding, Tokenizer};

use crate::{
    pipeline::{
        DecisionForwardContext, InputsProcessor, InputsProcessorValidationError, MessagesAction,
        Processor,
    },
    sequence::{MultimodalData, Sequence},
    vision_models::{
        grid_processor::GridVisionSpecificArgs,
        lfm3_omni::{inputs_processor::Lfm3OmniSpecificArgs, Lfm3OmniProcessor},
        lfm3_vl::Lfm3VlProcessor,
        preprocessor_config::PreProcessorConfig,
        ModelInputs,
    },
};

use crate::decision::{DecisionInstructions, DecisionMedia, DecisionRequest};

use super::{
    config::{Config, DecisionConfig},
    prompt::{DecisionPlan, DecisionPrompt},
};

pub(crate) struct DecisionRow {
    pub inputs: ModelInputs,
    pub plan: DecisionPlan,
}

#[derive(Clone)]
pub(crate) struct DecisionProcessor {
    media: Arc<dyn Processor + Send + Sync>,
    pub(super) config: DecisionConfig,
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
            config: model.decision,
            max_seq_len: model.vision.text_config.max_position_embeddings,
            supports_audio,
        }
    }

    pub(super) fn prompts<'a>(
        &'a self,
        request: &'a DecisionRequest,
        tokenizer: &'a Tokenizer,
    ) -> Result<impl Iterator<Item = Result<DecisionPrompt>> + 'a> {
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
        let prompts = DecisionPrompt::prepare(request, tokenizer, &self.config)
            .map_err(|error| InputsProcessorValidationError(error.to_string()))?;
        Ok(prompts.map(|prompt| {
            prompt.map_err(|error| InputsProcessorValidationError(error.to_string()).into())
        }))
    }

    pub(super) fn prepare_row(
        &self,
        seq: &mut Sequence,
        prompt: DecisionPrompt,
        context: &DecisionForwardContext<'_>,
    ) -> Result<DecisionRow> {
        let tokenizer = context.tokenizer;
        let processor = self.media.inputs_processor();
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
            context.device,
            context.processor_config.clone(),
            None,
        )?;
        let expanded = tokenizer
            .decode(seq.get_toks(), false)
            .map_err(anyhow::Error::msg)?;
        let (plan, offsets) = prompt.plan(tokenizer, &expanded)?;
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
            context.device,
            false,
            None,
            false,
            None,
            context.processor_config.clone(),
            None,
            context.mapper,
        )?;
        let mut inputs = *output
            .inputs
            .downcast::<ModelInputs>()
            .expect("Decision media inputs");
        if plan.ids != seq.get_toks() {
            let original = tokenizer
                .encode(expanded, false)
                .map_err(anyhow::Error::msg)?;
            repack_media_inputs(&mut inputs, seq, &plan, &original, &offsets)?;
        }
        Ok(DecisionRow { inputs, plan })
    }
}

fn repack_media_inputs(
    inputs: &mut ModelInputs,
    seq: &mut Sequence,
    plan: &DecisionPlan,
    original: &Encoding,
    offsets: &[(usize, usize)],
) -> Result<()> {
    ensure!(
        original.get_ids() == seq.get_toks(),
        "Expanded decision media tokens changed during preprocessing"
    );
    let remap = |range: Range<usize>| -> Result<Range<usize>> {
        let (start, end) = (
            original.get_offsets()[range.start].0,
            original.get_offsets()[range.end - 1].1,
        );
        let first = offsets.partition_point(|&(a, _)| a < start);
        let last = offsets.partition_point(|&(_, b)| b <= end);
        ensure!(
            first < last && offsets[first].0 == start && offsets[last - 1].1 == end,
            "Decision media boundary intersects a question boundary"
        );
        ensure!(
            original.get_ids()[range] == plan.ids[first..last],
            "Question packing changed decision media tokens"
        );
        Ok(first..last)
    };
    let ids = Tensor::new(plan.ids.as_slice(), inputs.input_ids.device())?.unsqueeze(0)?;
    let remap_vision = |vision: &mut GridVisionSpecificArgs| -> Result<()> {
        for (start, end) in vision
            .continuous_img_pad
            .iter_mut()
            .chain(&mut vision.continuous_vid_pad)
            .flatten()
        {
            let range = remap(*start..*end)?;
            (*start, *end) = (range.start, range.end);
        }
        vision.input_ids_full = ids.clone();
        vision.seqlens = vec![plan.ids.len()];
        Ok(())
    };
    if let Some(args) = inputs
        .model_specific_args
        .downcast_mut::<Lfm3OmniSpecificArgs>()
    {
        remap_vision(args.vision.as_mut().context("Missing d1 vision inputs")?)?;
        for audio in &mut args.audio {
            audio.token_range = remap(audio.token_range.clone())?;
        }
    } else {
        remap_vision(
            inputs
                .model_specific_args
                .downcast_mut::<GridVisionSpecificArgs>()
                .context("Invalid d1 model inputs")?,
        )?;
    }
    let mut features = seq.mm_features().to_vec();
    for feature in &mut features {
        let range = remap(feature.offset..feature.end())?;
        feature.offset = range.start;
        feature.length = range.len();
    }
    inputs.input_ids = ids;
    seq.set_toks_and_reallocate(plan.ids.clone(), None);
    seq.set_mm_features(features);
    Ok(())
}

impl Processor for DecisionProcessor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        self.media.inputs_processor()
    }
    fn get_special_tokens(&self) -> &[&'static str] {
        self.media.get_special_tokens()
    }
    fn template_action(&self) -> MessagesAction {
        MessagesAction::Keep
    }
}
