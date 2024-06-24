use std::{any::Any, sync::Arc};

use candle_core::{quantized::GgmlDType, DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use indicatif::{ProgressBar, ProgressIterator, ProgressStyle};
use rand::{seq::SliceRandom, thread_rng};
use rand_isaac::Isaac64Rng;

use crate::{
    amoe::{AnyMoeConfig, AnyMoeTrainingInputs, AnyMoeTrainingResult},
    get_mut_arcmutex,
    prefix_cacher::PrefixCacheManager,
    sampler::Sampler,
    sequence::{Sequence, SequenceGroup, SequenceRecognizer},
    DeviceMapMetadata, Loader, ModelCategory, ModelKind, ModelPaths, Pipeline, Response,
    TokenSource, TryIntoDType,
};

use super::{
    AdapterActivationMixin, AnyMoePipelineMixin, CacheManagerMixin, IsqPipelineMixin,
    MetadataMixin, PreProcessingMixin,
};

/// A loader for a speculative pipeline using 2 [`Loader`]s.
pub struct AnyMoeLoader {
    pub target: Box<dyn Loader>,
    pub config: AnyMoeConfig,
}

pub struct AnyMoePipeline {
    target: Arc<tokio::sync::Mutex<dyn Pipeline>>,
    config: AnyMoeConfig,
}

impl Loader for AnyMoeLoader {
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model_from_hf(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mapper: DeviceMapMetadata,
        in_situ_quant: Option<GgmlDType>,
    ) -> anyhow::Result<Arc<tokio::sync::Mutex<dyn Pipeline + Send + Sync>>> {
        let target = self.target.load_model_from_hf(
            revision.clone(),
            token_source.clone(),
            dtype,
            device,
            silent,
            mapper.clone(),
            in_situ_quant,
        )?;
        Ok(Arc::new(tokio::sync::Mutex::new(AnyMoePipeline::new(
            target,
            self.config.clone(),
        ))))
    }

    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model_from_path(
        &self,
        paths: &Box<dyn ModelPaths>,
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mapper: DeviceMapMetadata,
        in_situ_quant: Option<GgmlDType>,
    ) -> anyhow::Result<Arc<tokio::sync::Mutex<dyn Pipeline + Send + Sync>>> {
        let target = self.target.load_model_from_path(
            paths,
            dtype,
            device,
            silent,
            mapper.clone(),
            in_situ_quant,
        )?;
        Ok(Arc::new(tokio::sync::Mutex::new(AnyMoePipeline::new(
            target,
            self.config.clone(),
        ))))
    }
    fn get_id(&self) -> String {
        format!("AnyMoE: tgt = `{}`", self.target.get_id(),)
    }
    fn get_kind(&self) -> ModelKind {
        ModelKind::AnyMoe {
            target: Box::new(self.target.get_kind()),
        }
    }
}

impl AnyMoePipeline {
    pub fn new(target: Arc<tokio::sync::Mutex<dyn Pipeline>>, config: AnyMoeConfig) -> Self {
        Self { target, config }
    }

    fn train(&mut self, inputs: AnyMoeTrainingInputs) -> candle_core::Result<AnyMoeTrainingResult> {
        let layer_vars = get_mut_arcmutex!(self.target).layer_vars();
        let device = get_mut_arcmutex!(self.target).device();
        let inputs_processor = get_mut_arcmutex!(self.target)
            .get_processor()
            .inputs_processor();
        let tokenizer = get_mut_arcmutex!(self.target).tokenizer();
        let metadata = get_mut_arcmutex!(self.target).get_metadata().clone();
        let input_processor_cfg = get_mut_arcmutex!(self.target)
            .get_input_processor_config()
            .clone();

        let AnyMoeConfig {
            hidden_size: _,
            lr,
            epochs,
            batch_size,
        } = self.config;
        let mut steps = 0;

        let mut optimizers = layer_vars
            .into_iter()
            .map(|vars| {
                AdamW::new(
                    vars,
                    ParamsAdamW {
                        lr,
                        beta1: 0.9,
                        beta2: 0.999,
                        eps: 1e-8,
                        weight_decay: 0.0,
                    },
                )
            })
            .collect::<candle_core::Result<Vec<_>>>()?;

        let bar = ProgressBar::new(epochs as u64);
        bar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] [{bar:40.green}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        let mut rng = thread_rng();
        let mut samples = inputs.0;

        // Create several dummy objects for the sequences.
        let (dummy_sender, _) = tokio::sync::mpsc::channel(10000);
        let dummy_sampler = Sampler::new(None, 0, tokenizer.clone(), None, None, None, -1, 0.0);
        let dummy_group = Arc::new(tokio::sync::Mutex::new(SequenceGroup::new(
            1, false, false, 0,
        )));

        // Clear KV cache in prep for training
        get_mut_arcmutex!(self.target).set_none_cache(true, true);

        let mut latest_loss = vec![Tensor::new(0.0f32, &device)?; optimizers.len()];

        for _ in (0..epochs).progress_with(bar) {
            samples.as_mut_slice().shuffle(&mut rng);
            for batch in samples.chunks(batch_size).into_iter() {
                steps += 1;

                // === PREPARE INPUTS ==
                let mut seqs = Vec::new();
                for (prompt, _) in batch {
                    let tokens = tokenizer
                        .encode(prompt.clone(), true)
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?
                        .get_ids()
                        .to_vec();
                    seqs.push(new_dummy_seq(
                        tokens,
                        dummy_sender.clone(),
                        dummy_sampler.clone(),
                        dummy_group.clone(),
                    ));
                }
                let mut input_seqs = seqs.iter_mut().collect::<Vec<_>>();
                let inputs = inputs_processor
                    .process_inputs(
                        tokenizer.clone(),
                        &mut input_seqs,
                        true, // Always a prompt
                        metadata.is_xlora,
                        &device,
                        metadata.has_no_kv_cache,
                        None,
                        input_processor_cfg.clone(),
                    )
                    .unwrap();

                // === PREPARE AND RUN MODEL ==

                // Run the model, ignoring the logits
                let _ = get_mut_arcmutex!(self.target).forward_inputs(inputs)?;

                // Clear the KV cache
                get_mut_arcmutex!(self.target).set_none_cache(true, true);

                // === BACKWARD STEP ==
                let labels = Tensor::from_vec(
                    batch.iter().map(|(_, x)| *x as u32).collect::<Vec<_>>(),
                    (batch.len(),),
                    &device,
                )?;

                let cached = get_mut_arcmutex!(self.target).get_cached_gating_outputs();
                for (layer, (optimizer, output)) in optimizers.iter_mut().zip(cached).enumerate() {
                    let loss = candle_nn::loss::cross_entropy(&output, &labels)?;
                    optimizer.backward_step(&loss)?;
                    latest_loss[layer] = loss.to_dtype(DType::F32)?;
                }
            }
        }

        Ok(AnyMoeTrainingResult {
            steps,
            final_loss: latest_loss
                .into_iter()
                .map(|loss| loss.to_scalar::<f32>())
                .collect::<candle_core::Result<Vec<_>>>()?,
        })
    }
}

impl AdapterActivationMixin for AnyMoePipeline {
    fn activate_adapters(&mut self, adapters: Vec<String>) -> anyhow::Result<usize> {
        get_mut_arcmutex!(self.target).activate_adapters(adapters)
    }
}

impl CacheManagerMixin for AnyMoePipeline {
    fn cache(&self) -> &super::Cache {
        unreachable!()
    }
    fn clone_in_cache(&mut self, seqs: &mut [&mut Sequence], modify_draft_cache: bool) {
        get_mut_arcmutex!(self.target).clone_in_cache(seqs, modify_draft_cache)
    }
    fn clone_out_cache(&mut self, seqs: &mut [&mut Sequence], modify_draft_cache: bool) {
        get_mut_arcmutex!(self.target).clone_out_cache(seqs, modify_draft_cache)
    }
    fn set_none_cache(&mut self, reset_non_granular: bool, modify_draft_cache: bool) {
        get_mut_arcmutex!(self.target).set_none_cache(reset_non_granular, modify_draft_cache)
    }
}

impl IsqPipelineMixin for AnyMoePipeline {
    fn re_isq_model(&mut self, dtype: GgmlDType) -> anyhow::Result<()> {
        get_mut_arcmutex!(self.target).re_isq_model(dtype)
    }
}

impl PreProcessingMixin for AnyMoePipeline {
    fn get_chat_template(&self) -> Arc<crate::ChatTemplate> {
        get_mut_arcmutex!(self.target).get_chat_template()
    }
    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>> {
        get_mut_arcmutex!(self.target).get_input_processor_config()
    }
    fn get_processor(&self) -> Arc<dyn super::Processor> {
        get_mut_arcmutex!(self.target).get_processor()
    }
}

impl MetadataMixin for AnyMoePipeline {
    fn device(&self) -> Device {
        get_mut_arcmutex!(self.target).device()
    }
    fn get_metadata(&self) -> Arc<super::GeneralMetadata> {
        get_mut_arcmutex!(self.target).get_metadata()
    }
    fn name(&self) -> String {
        get_mut_arcmutex!(self.target).name()
    }
    fn reset_non_granular_state(&self) {
        get_mut_arcmutex!(self.target).reset_non_granular_state()
    }
    fn tokenizer(&self) -> Arc<tokenizers::Tokenizer> {
        get_mut_arcmutex!(self.target).tokenizer()
    }
}

#[async_trait::async_trait]
impl Pipeline for AnyMoePipeline {
    fn forward_inputs(&mut self, inputs: Box<dyn Any>) -> Result<Tensor, candle_core::Error> {
        get_mut_arcmutex!(self.target).forward_inputs(inputs)
    }

    async fn sample(
        &self,
        seqs: &mut [&mut Sequence],
        logits: Tensor,
        prefix_cacher: &mut PrefixCacheManager,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> Result<(), candle_core::Error> {
        get_mut_arcmutex!(self.target)
            .sample(seqs, logits, prefix_cacher, disable_eos_stop, rng)
            .await
    }

    fn category(&self) -> ModelCategory {
        get_mut_arcmutex!(self.target).category()
    }
}

// TODO
impl AnyMoePipelineMixin for AnyMoePipeline {}

/// Create a dummy sequence containing just the prompt. This is OK because we just want a sequence that
/// has no information other than the input tokens (and maybe images).
fn new_dummy_seq(
    tokens: Vec<u32>,
    dummy_sender: tokio::sync::mpsc::Sender<Response>,
    dummy_sampler: Sampler,
    dummy_group: Arc<tokio::sync::Mutex<SequenceGroup>>,
) -> Sequence {
    Sequence::new_waiting(
        tokens,
        0,
        0,
        0, // Don't allocate any cache
        dummy_sender,
        dummy_sampler,
        vec![],
        vec![],
        None,
        false,
        false,
        dummy_group,
        0,
        0,
        SequenceRecognizer::None,
        None,
        None,
        None,
        None, // TODO support images
    )
}
