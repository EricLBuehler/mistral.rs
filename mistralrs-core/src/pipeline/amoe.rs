use std::sync::Arc;

use candle_core::{DType, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use indicatif::{ProgressBar, ProgressIterator, ProgressStyle};
use rand::{seq::SliceRandom, thread_rng};

use crate::{
    amoe::{AnyMoeTrainingInputs, AnyMoeTrainingResult},
    get_mut_arcmutex,
    sampler::Sampler,
    sequence::{Sequence, SequenceGroup, SequenceRecognizer},
    Pipeline, Response,
};

use super::{AnyMoePipelineMixin, AnyMoeTrainerMixin};

pub struct AnyMoePipeline {
    target: Arc<tokio::sync::Mutex<dyn Pipeline>>,
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

impl AnyMoeTrainerMixin for AnyMoePipeline {
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

        let lr = 1e-3;
        let num_epochs = 100;
        let batch_size = 4;
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

        let bar = ProgressBar::new(num_epochs as u64);
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

        for _ in (0..num_epochs).progress_with(bar) {
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
    fn trainable_params(&self) -> usize {
        get_mut_arcmutex!(self.target).base_model_trainable_params()
    }
}
