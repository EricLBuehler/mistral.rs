mod classifier;
mod config;
mod gemma;
mod llama;
mod mistral;
mod mixtral;
mod phi2;
mod phi3;
mod quantized_llama;
mod quantized_phi3;

use std::sync::Arc;

use crate::lora::Ordering;
use candle_core::{DType, Device, Result, Tensor};
pub(crate) use config::XLoraConfig;
pub(crate) use gemma::XLoraModel as XLoraGemma;
pub(crate) use llama::XLoraLlama;
pub(crate) use mistral::XLoraModel as XLoraMistral;
pub(crate) use mixtral::XLoraModel as XLoraMixtral;
pub(crate) use phi2::Model as XLoraPhi2;
pub(crate) use phi3::Model as XLoraPhi3;
pub(crate) use quantized_llama::ModelWeights as XLoraQLlama;
pub(crate) use quantized_phi3::ModelWeights as XLoraQPhi3;
use tokio::sync::Mutex;

use crate::{get_mut_arcmutex, pipeline::Cache};

use self::classifier::XLoraClassifier;

pub struct NonGranularState {
    pub non_granular_index: Arc<Mutex<usize>>,
    pub tgt_non_granular_index: usize,
}

trait ScalingsMaker {
    fn get_classifier(&self) -> &XLoraClassifier;
    /// For dummy scalings
    fn dtype(&self) -> DType;
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &mut self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        scalings: Tensor,
        is_full_pass: bool,
        no_kv_cache: bool,
        is_scaling_pass: Option<f64>,
        context_lens: &[usize],
    ) -> Result<Tensor>;
    fn get_cache(&self) -> &Cache;

    #[allow(clippy::too_many_arguments)]
    fn get_scalings(
        &mut self,
        input_ids: &Tensor,
        input_ids_full: &Tensor,
        seqlen_offsets: &[usize],
        seqlen_offsets_full: &[usize],
        start_offsets_kernel: &Tensor,
        start_offsets_kernel_full: &Tensor,
        no_kv_cache: bool,
        non_granular_state: &Option<NonGranularState>,
        position_ids: &[usize],
    ) -> Result<Tensor> {
        let (b_size, _) = input_ids_full.dims2()?;
        let (_, seq_len) = input_ids.dims2()?;

        if let Some(ref non_granular_state) = non_granular_state {
            if let Some(scalings_cache) = &*self.get_cache().get_scalings_cache() {
                return Ok(scalings_cache.clone());
            }
            if seq_len == 1 {
                *get_mut_arcmutex!(non_granular_state.non_granular_index) += 1;
            }
        }

        let dummy_scalings = self.get_classifier().get_dummy_scalings(
            b_size,
            seq_len,
            input_ids.device(),
            self.dtype(),
        )?;
        // Using X-LoRA cache here
        let hidden_states = if no_kv_cache {
            let res = self.forward(
                input_ids_full,
                seqlen_offsets_full,
                start_offsets_kernel_full.clone(),
                dummy_scalings,
                true,
                no_kv_cache,
                Some(self.get_classifier().config.scaling_pass_value),
                position_ids,
            )?;

            let mut new_cache = Vec::new();
            for _ in 0..self.get_cache().xlora_lock().len() {
                new_cache.push(Some((
                    Tensor::zeros((1,), DType::U8, &Device::Cpu)?,
                    Tensor::zeros((1,), DType::U8, &Device::Cpu)?,
                )));
            }
            self.get_cache().lock().clone_from(&new_cache);

            res
        } else {
            self.forward(
                input_ids,
                seqlen_offsets,
                start_offsets_kernel.clone(),
                dummy_scalings,
                false,
                no_kv_cache,
                Some(self.get_classifier().config.scaling_pass_value),
                position_ids,
            )?
        };

        let scalings = self.get_classifier().forward(hidden_states)?;
        if let Some(ref non_granular_state) = non_granular_state {
            if *get_mut_arcmutex!(non_granular_state.non_granular_index)
                == non_granular_state.tgt_non_granular_index
            {
                *self.get_cache().get_scalings_cache() = Some(scalings.clone());
            }
        }
        Ok(scalings)
    }
}

fn verify_sanity_adapters(ordering: &Ordering, supported_layers: &[&str]) -> Result<()> {
    if ordering.layers.is_none() {
        return Ok(());
    }
    for path in ordering.layers.as_ref().unwrap().keys() {
        if !supported_layers.iter().any(|layer| path.ends_with(layer)) {
            candle_core::bail!("Got a layer name `{path}` in the ordering, expected it to end with one of {supported_layers:?}");
        }
    }
    Ok(())
}
