use std::{any::Any, collections::HashMap, sync::Arc};

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::VarBuilder;
pub use config::MiniCpmOConfig;
use mistralrs_quant::QuantMethod;
use resampler::Resampler;

use crate::{
    amoe::AnyMoeBaseModelMixin,
    device_map::DeviceMapper,
    models::qwen2,
    paged_attention::{AttentionImplementation, ModelConfigMetadata},
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, NormalLoadingMetadata, NormalModel, VisionModel,
    },
};

use self::siglip::SiglipVisionTransformer;

use super::siglip;

mod config;
mod resampler;

pub struct MiniCpmOModel {
    cfg: MiniCpmOConfig,
    llm: qwen2::Model,
    vpm: SiglipVisionTransformer,
    resampler: Resampler,
}

impl MiniCpmOModel {
    pub fn new(
        cfg: &MiniCpmOConfig,
        vb: VarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let llm = qwen2::Model::new(
            &cfg.text_config,
            vb.pp("llm"),
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )?;
        let vpm = SiglipVisionTransformer::new(&cfg.vision_config, vb.pp("vpm"))?;
        let resampler = Resampler::new(
            cfg.query_num,
            cfg.text_config.hidden_size,
            cfg.text_config.hidden_size / 128,
            cfg.vision_config.hidden_size,
            true,
            None,
            vb.pp("resampler"),
        )?;
        Ok(Self {
            cfg: cfg.clone(),
            llm,
            vpm,
            resampler,
        })
    }

    fn get_vllm_embedding(
        &self,
        pixel_values_all: Option<Tensor>,
        tgt_sizes_all: Option<Tensor>,
    ) -> Result<Tensor> {
        if let Some(pixel_values_all) = pixel_values_all {
            let tgt_sizes_all = tgt_sizes_all.expect("Need tgt_sizes");

            let mut all_pixel_values = Vec::new();
            let mut img_cnt = Vec::new();
            for pixel_values in pixel_values_all.chunk(pixel_values_all.dim(0)?, 0)? {
                img_cnt.push(pixel_values.dim(0)?);
                let mut imgs = Vec::new();
                for i in pixel_values.chunk(pixel_values.dim(0)?, 0)? {
                    // Assume channel dimension first
                    imgs.push(i.flatten_to(1)?.permute((1, 0))?);
                }
                all_pixel_values.extend(imgs);
            }

            let tgt_sizes = Tensor::cat(&tgt_sizes_all.chunk(tgt_sizes_all.dim(0)?, 0)?, 0)?
                .to_dtype(DType::I32)?;
            let tgt_sizes_vec = tgt_sizes.to_vec2::<i32>()?;

            let max_patches = (tgt_sizes.i((.., 0))? * tgt_sizes.i((.., 1))?)?
                .max(0)?
                .to_scalar::<i32>()? as usize;

            // Original code does padding of the pixel values here
            let lens = all_pixel_values
                .iter()
                .map(|pixel_values| pixel_values.dim(0))
                .collect::<Result<Vec<_>>>()?;
            let max_len = lens.into_iter().max().expect("No pixe values somehow?");
            all_pixel_values = all_pixel_values
                .into_iter()
                .map(|pixel_values| {
                    pixel_values.pad_with_zeros(0, 0, max_len - pixel_values.dim(0)?)
                })
                .collect::<Result<Vec<_>>>()?;
            let mut all_pixel_values = Tensor::stack(&all_pixel_values, 0)?;

            let (b, l, _) = all_pixel_values.dims3()?;
            all_pixel_values = all_pixel_values
                .permute((0, 2, 1))?
                .reshape((b, 3, (), l))?;

            let mut patch_attn_mask =
                Tensor::zeros((b, 1, max_patches), DType::U8, pixel_values_all.device())?;
            for i in 0..b {
                let n = (tgt_sizes_vec[i][0] * tgt_sizes_vec[i][1]) as usize;
                patch_attn_mask = patch_attn_mask.slice_assign(
                    &[&i, &0, &(..n)],
                    &Tensor::ones((1, 1, n), DType::U8, pixel_values_all.device())?,
                )?;
            }

            let vision_batch_size = self.cfg.vision_batch_size;
            all_pixel_values = all_pixel_values.to_dtype(self.llm.embed_dtype())?;

            let mut vision_embedding = if b > vision_batch_size {
                let mut hs = Vec::new();
                for i in (0..b).step_by(vision_batch_size) {
                    let start_idx = i;
                    let end_idx = i + vision_batch_size;
                    let tmp_hs = self.vpm.forward(
                        &all_pixel_values.i(start_idx..end_idx)?,
                        Some(&patch_attn_mask.i(start_idx..end_idx)?),
                        Some(&tgt_sizes.i(start_idx..end_idx)?),
                    )?;
                    hs.push(tmp_hs);
                }
                Tensor::cat(&hs, 0)?
            } else {
                self.vpm
                    .forward(&all_pixel_values, Some(&patch_attn_mask), Some(&tgt_sizes))?
            };
            vision_embedding = self.resampler.forward(&vision_embedding, &tgt_sizes)?;
        }
        todo!()
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        tgt_sizes: Option<Tensor>,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &mut PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let vllm_embedding = self.get_vllm_embedding(pixel_values, tgt_sizes)?;
        self.llm.forward_embed(
            input_ids,
            vllm_embedding,
            seqlen_offsets,
            start_offsets_kernel,
            context_lens,
            metadata,
            flash_params,
        )
    }
}

impl VisionModel for MiniCpmOModel {
    fn cache(&self) -> &EitherCache {
        self.llm.cache()
    }
    fn cache_mut(&mut self) -> &mut EitherCache {
        self.llm.cache_mut()
    }
    fn config(&self) -> &ModelConfigMetadata {
        self.llm.config()
    }
    fn device(&self) -> &Device {
        self.llm.device()
    }
    fn has_conv2d(&self) -> bool {
        true
    }
    fn max_seq_len(&self) -> usize {
        self.llm.max_seq_len()
    }
    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        model_specific_args: Box<dyn Any>, // pixel attention mask, or image sizes, or anything else
        metadata: Option<(Vec<(Tensor, Tensor)>, &mut PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        todo!()
    }
    fn default_model_specific_args(&self, _input_ids: &Tensor) -> Box<dyn Any> {
        todo!()
    }
}

impl IsqModel for MiniCpmOModel {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        self.llm.get_layers()
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        todo!()
    }

    // NOTE: We ONLY calibrate the text bits of these models, so we should only track/return those parts!!

    /// This is used for imatrix generation internally. Begin stats tracking.
    fn begin_track_stats(&mut self) -> anyhow::Result<()> {
        self.llm.begin_track_stats()
    }

    /// End stats tracking and return the imatrix data
    fn extract_imatrix_data(&mut self) -> candle_core::Result<HashMap<usize, Option<Vec<f32>>>> {
        self.llm.extract_imatrix_data()
    }
}

impl AnyMoeBaseModelMixin for MiniCpmOModel {}
