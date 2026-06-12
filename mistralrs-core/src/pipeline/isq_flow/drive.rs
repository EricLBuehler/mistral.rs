//! The offline calibration driver: tokenize a calibration file and drive text chunks
//! through any pipeline's model via [`CalibrationDrive`], collecting per-layer statistics.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Result;
use candle_core::Device;
use mistralrs_quant::{QuantMethod, TrackedModule};
use tokenizers::Tokenizer;
use tracing::info;

use super::super::isq::load_imatrix_map;
use super::super::text_models_inputs_processor::{make_prompt_chunk, InputMetadata};
use super::super::{
    EitherCache, EmbeddingModel, ModelForwardContext, MultimodalModel, NormalModel,
};
use super::harvest_imatrix;

pub(crate) trait CalibrationDrive {
    fn calibration_forward(&self, inputs: &InputMetadata) -> candle_core::Result<()>;
    fn reset_cache(&self) {}
    fn sliding_window(&self) -> Option<usize> {
        None
    }
}

pub(crate) struct NormalCalibrationDrive<'a>(pub &'a dyn NormalModel);

impl CalibrationDrive for NormalCalibrationDrive<'_> {
    fn calibration_forward(&self, inputs: &InputMetadata) -> candle_core::Result<()> {
        let input = inputs.input.to_device(self.0.device())?;
        let mut ctx = ModelForwardContext::new(
            &inputs.positions,
            &inputs.context_lens,
            &inputs.position_ids,
            None,
            &inputs.flash_meta,
        );
        self.0.forward(&input, &mut ctx)?;
        Ok(())
    }

    fn reset_cache(&self) {
        reset_either_cache(self.0.cache());
    }

    fn sliding_window(&self) -> Option<usize> {
        self.0.config().sliding_window
    }
}

pub(crate) struct MultimodalCalibrationDrive<'a>(pub &'a dyn MultimodalModel);

impl CalibrationDrive for MultimodalCalibrationDrive<'_> {
    fn calibration_forward(&self, inputs: &InputMetadata) -> candle_core::Result<()> {
        let input = inputs.input.to_device(self.0.device())?;
        let mut ctx = ModelForwardContext::new(
            &inputs.positions,
            &inputs.context_lens,
            &inputs.position_ids,
            None,
            &inputs.flash_meta,
        );
        // Text-only drive: the vision tower sees no calibration data, so its
        // layers quantize without imatrix weights.
        let args = self.0.default_model_specific_args(&input);
        self.0.forward(&input, None, args, &mut ctx)?;
        Ok(())
    }

    fn reset_cache(&self) {
        reset_either_cache(self.0.cache());
    }

    fn sliding_window(&self) -> Option<usize> {
        self.0.config().sliding_window
    }
}

pub(crate) struct EmbeddingCalibrationDrive<'a>(pub &'a dyn EmbeddingModel);

impl CalibrationDrive for EmbeddingCalibrationDrive<'_> {
    fn calibration_forward(&self, inputs: &InputMetadata) -> candle_core::Result<()> {
        let input = inputs.input.to_device(self.0.device())?;
        self.0.forward(&input, &inputs.flash_meta)?;
        Ok(())
    }
}

fn reset_either_cache(cache: &EitherCache) {
    match cache {
        EitherCache::Full(full) => {
            for layer in &mut *full.lock() {
                *layer = None
            }
        }
        EitherCache::Normal(normal) => {
            for layer in &mut *normal.lock().unwrap().0 {
                layer.reset();
            }
        }
        EitherCache::Hybrid(hybrid) => {
            hybrid.lock().unwrap().reset();
        }
    }
}

pub(crate) struct CalibrationCtx<'a> {
    pub tokenizer: &'a Tokenizer,
    pub bos_tok_id: Option<u32>,
    pub load_device: &'a Device,
    pub mapper: Option<&'a dyn crate::device_map::DeviceMapper>,
}

const CALIBRATION_CHUNK_SIZE: usize = 1024;

/// Produce the per-layer imatrix map: collect via calibration forwards, or load from a file.
pub(crate) fn resolve_imatrix_map(
    drive: &dyn CalibrationDrive,
    modules: &[TrackedModule],
    imatrix_path: Option<&PathBuf>,
    calibration_file: Option<&PathBuf>,
    ctx: &CalibrationCtx<'_>,
) -> Result<HashMap<String, Vec<f32>>> {
    if let Some(calibration_file) = calibration_file {
        let calibration_data = std::fs::read_to_string(calibration_file)?;
        // Tokenize without bos; it is inserted per chunk below
        let tokens = ctx
            .tokenizer
            .encode_fast(calibration_data, false)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();
        info!(
            "Collecting imatrix from calibration file `{}` of {} tokens.",
            calibration_file.display(),
            tokens.len()
        );

        for module in modules {
            module.ct.begin_track_stats()?;
        }

        let n_chunks = tokens.len().div_ceil(CALIBRATION_CHUNK_SIZE);
        let collect_start = std::time::Instant::now();
        for (i, chunk) in tokens.chunks(CALIBRATION_CHUNK_SIZE).enumerate() {
            let mut chunk = chunk.to_vec();
            if let Some(bos_tok_id) = ctx.bos_tok_id {
                chunk.insert(0, bos_tok_id);
            }
            let chunk_len = chunk.len();

            let chunk_start = std::time::Instant::now();
            let inputs = make_prompt_chunk(
                0,
                vec![&chunk],
                &[0],
                ctx.load_device,
                None,
                false,
                None,
                ctx.mapper,
                None,
                drive.sliding_window(),
            )?;
            drive.calibration_forward(&inputs)?;
            drive.reset_cache();

            info!(
                "Processed chunk {}/{n_chunks} ({chunk_len} tokens), {:.2}s",
                i + 1,
                chunk_start.elapsed().as_secs_f32()
            );
        }
        ctx.load_device.synchronize()?;
        info!(
            "Finished collecting imatrix in {:.2}s",
            collect_start.elapsed().as_secs_f32()
        );

        harvest_imatrix(modules)
    } else if let Some(imatrix_path) = imatrix_path {
        load_imatrix_map(imatrix_path, modules)
    } else {
        unreachable!("wants_imatrix requires imatrix or calibration_file")
    }
}
