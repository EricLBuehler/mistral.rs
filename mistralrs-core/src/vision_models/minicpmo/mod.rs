use std::{
    any::Any,
    sync::{Arc, Mutex},
};

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
pub use config::MiniCpmOConfig;
pub use inputs_processor::MiniCpmOProcessor;
use mistralrs_quant::{CollectedImatrixData, QuantMethod, ShardedVarBuilder};
use resampler::Resampler;

use crate::{
    amoe::AnyMoeBaseModelMixin,
    device_map::DeviceMapper,
    models::qwen2,
    paged_attention::{
        encoder_cache::EncoderCacheManager, AttentionImplementation, ModelConfigMetadata,
    },
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, NormalLoadingMetadata, NormalModel, VisionModel,
    },
    utils::unvarbuilder::UnVarBuilder,
};

use self::siglip::SiglipVisionTransformer;

use super::siglip;

mod config;
mod inputs_processor;
mod resampler;

pub struct MiniCpmOModel {
    cfg: MiniCpmOConfig,
    llm: qwen2::Model,
    vpm: SiglipVisionTransformer,
    resampler: Resampler,
    encoder_cache: Arc<Mutex<EncoderCacheManager>>,
}

impl MiniCpmOModel {
    pub fn new(
        cfg: &MiniCpmOConfig,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let real_device = normal_loading_metadata.real_device.clone();
        let llm = qwen2::Model::new(
            &cfg.text_config,
            vb.pp("llm"),
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )?;
        let vpm = SiglipVisionTransformer::new(
            &cfg.vision_config,
            vb.pp("vpm").set_device(real_device.clone()),
        )?;
        let resampler = Resampler::new(
            cfg.query_num,
            cfg.text_config.hidden_size,
            cfg.text_config.hidden_size / 128,
            cfg.vision_config.hidden_size,
            true,
            None,
            vb.pp("resampler").set_device(real_device.clone()),
        )?;
        Ok(Self {
            cfg: cfg.clone(),
            llm,
            vpm,
            resampler,
            encoder_cache: Arc::new(Mutex::new(EncoderCacheManager::new(32))),
        })
    }

    fn get_vllm_embedding(
        &self,
        input_ids: &Tensor,
        device: &Device,
        pixel_values_all: Option<Vec<Vec<Tensor>>>,
        tgt_sizes: Option<Vec<Tensor>>,
        image_bound: Option<Vec<Tensor>>,
        image_hashes: &[u64],
    ) -> Result<Tensor> {
        let mut vllm_embedding = self.llm.get_input_embeddings(input_ids)?;

        if let Some(pixel_values_all) = pixel_values_all {
            let tgt_sizes_all = tgt_sizes.as_ref().expect("Need tgt_sizes");
            let image_bound = image_bound.expect("Need image_bound");
            let image_bound_vec = image_bound
                .into_iter()
                .map(|x| x.to_vec2::<u32>())
                .collect::<Result<Vec<_>>>()?;

            // Flatten per-batch images and their tgt_sizes into a single list,
            // tracking which batch element each image belongs to.
            let mut all_pixel_values_raw = Vec::new();
            let mut all_tgt_sizes_raw = Vec::new();
            let mut img_cnts = Vec::new();
            {
                for (tgt_idx, pixel_values) in pixel_values_all.iter().enumerate() {
                    img_cnts.push(pixel_values.len());
                    let tgt = &tgt_sizes_all[tgt_idx];
                    let tgt_rows = tgt.dim(0)?;
                    for (j, pv) in pixel_values.iter().enumerate() {
                        all_pixel_values_raw.push(pv.clone());
                        all_tgt_sizes_raw.push(tgt.get(j)?);
                    }
                    let _ = tgt_rows;
                }
            }
            let n_total_images = all_pixel_values_raw.len();

            // Per-image caching
            let n_hashes = image_hashes.len();
            let vision_embedding = if n_hashes > 0 && n_hashes == n_total_images {
                let mut per_image_features: Vec<Option<Tensor>> = vec![None; n_total_images];
                let mut miss_indices = Vec::new();
                {
                    let mut guard = self
                        .encoder_cache
                        .lock()
                        .expect("encoder cache lock poisoned");
                    for (i, &hash) in image_hashes.iter().enumerate() {
                        if let Some(cached) = guard.get(hash) {
                            per_image_features[i] = Some(cached[0].clone());
                        } else {
                            miss_indices.push(i);
                        }
                    }
                }

                if !miss_indices.is_empty() {
                    // Encode misses one at a time (images can have different sizes/patches)
                    for &idx in &miss_indices {
                        let pv = &all_pixel_values_raw[idx];
                        let tgt_size_tensor = all_tgt_sizes_raw[idx].unsqueeze(0)?;
                        let tgt_size_vec = tgt_size_tensor.to_vec2::<u32>()?;

                        // Prepare single-image pixel values
                        let single_pv = pv.flatten_to(1)?.permute((1, 0))?;
                        let single_pv = single_pv.unsqueeze(0)?; // (1, L, 3)
                        let (_, l, _) = single_pv.dims3()?;
                        let single_pv = single_pv.permute((0, 2, 1))?.reshape((1, 3, (), l))?;
                        let single_pv = single_pv.to_dtype(self.llm.embed_dtype())?;

                        let max_patches = (tgt_size_vec[0][0] * tgt_size_vec[0][1]) as usize;
                        let mut patch_attn_mask =
                            Tensor::zeros((1, 1, max_patches), DType::U8, device)?;
                        patch_attn_mask = patch_attn_mask.slice_assign(
                            &[0..1, 0..1, 0..max_patches],
                            &Tensor::ones((1, 1, max_patches), DType::U8, device)?,
                        )?;

                        let vpm_out = self.vpm.forward(
                            &single_pv,
                            Some(&patch_attn_mask),
                            Some(&tgt_size_tensor),
                        )?;
                        let feats = self.resampler.forward(&vpm_out, &tgt_size_vec)?;
                        // feats shape: (1, query_num, hidden_size)
                        let feats = feats.get(0)?; // (query_num, hidden_size)
                        {
                            let mut guard = self
                                .encoder_cache
                                .lock()
                                .expect("encoder cache lock poisoned");
                            guard.insert(image_hashes[idx], vec![feats.clone()]);
                        }
                        per_image_features[idx] = Some(feats);
                    }
                }

                // Stack all per-image features
                let all_feats: Vec<Tensor> = per_image_features
                    .into_iter()
                    .map(|f| f.expect("all images should be resolved"))
                    .collect();
                Tensor::stack(&all_feats, 0)?
            } else {
                // Original path: no hashes, encode everything
                let mut all_pixel_values = Vec::new();
                for pv in &all_pixel_values_raw {
                    all_pixel_values.push(pv.flatten_to(1)?.permute((1, 0))?);
                }

                let tgt_sizes = Tensor::cat(tgt_sizes_all, 0)?;
                let tgt_sizes_vec = tgt_sizes.to_vec2::<u32>()?;

                let max_patches = (tgt_sizes.i((.., 0))? * tgt_sizes.i((.., 1))?)?
                    .max(0)?
                    .to_scalar::<u32>()? as usize;

                let lens = all_pixel_values
                    .iter()
                    .map(|pixel_values| pixel_values.dim(0))
                    .collect::<Result<Vec<_>>>()?;
                let max_len = lens.into_iter().max().expect("No pixel values somehow?");
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

                let mut patch_attn_mask = Tensor::zeros((b, 1, max_patches), DType::U8, device)?;
                for (i, tgt_sizes_vec_i) in tgt_sizes_vec.iter().enumerate().take(b) {
                    let n = (tgt_sizes_vec_i[0] * tgt_sizes_vec_i[1]) as usize;
                    patch_attn_mask = patch_attn_mask.slice_assign(
                        &[i..i + 1, 0..1, 0..n],
                        &Tensor::ones((1, 1, n), DType::U8, device)?,
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
                vision_embedding = self.resampler.forward(&vision_embedding, &tgt_sizes_vec)?;
                vision_embedding
            };

            let mut start = 0;
            let mut vision_hidden_states = Vec::new();
            for pixel_values in &pixel_values_all {
                let img_cnt = pixel_values.len();
                if img_cnt > 0 {
                    vision_hidden_states.push(Some(
                        vision_embedding
                            .i(start..start + img_cnt)?
                            .to_dtype(vllm_embedding.dtype())?,
                    ));
                    start += img_cnt;
                } else {
                    vision_hidden_states.push(None);
                }
            }

            let mut new_vllm_embedding = Vec::new();
            for i in 0..input_ids.dim(0)? {
                if let Some(cur_vs_hs) = &vision_hidden_states[i] {
                    let mut cur_vllm_emb = vllm_embedding.i(i)?;
                    let cur_image_bound = &image_bound_vec[i];
                    if !cur_image_bound.is_empty() {
                        let mut image_indices = Vec::new();
                        for r in cur_image_bound {
                            image_indices.push(Tensor::arange(r[0], r[1], device)?);
                        }
                        let image_indices = Tensor::stack(&image_indices, 0)?;

                        let indices = image_indices
                            .reshape(((), 1))?
                            .repeat((1, cur_vllm_emb.dim(D::Minus1)?))?;
                        // Zero out the current data
                        let cur_vllm_emb_neg = cur_vllm_emb.gather(&indices, 0)?.neg()?;
                        cur_vllm_emb = cur_vllm_emb.scatter_add(&indices, &cur_vllm_emb_neg, 0)?;
                        // Add the image data
                        cur_vllm_emb = cur_vllm_emb.scatter_add(
                            &indices,
                            &cur_vs_hs.reshape(((), cur_vs_hs.dim(D::Minus1)?))?,
                            0,
                        )?;
                        new_vllm_embedding.push(cur_vllm_emb);
                    }
                }
            }
            vllm_embedding = Tensor::stack(&new_vllm_embedding, 0)?;
        }

        Ok(vllm_embedding)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values_all: Option<Vec<Vec<Tensor>>>,
        tgt_sizes: Option<Vec<Tensor>>,
        image_bound: Option<Vec<Tensor>>,
        image_hashes: &[u64],
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let vllm_embedding = self.get_vllm_embedding(
            input_ids,
            self.llm.device(),
            pixel_values_all,
            tgt_sizes,
            image_bound,
            image_hashes,
        )?;

        self.llm.forward_embed(
            input_ids,
            vllm_embedding,
            seqlen_offsets,
            context_lens,
            metadata,
            flash_params,
        )
    }
}

#[derive(Default)]
pub(crate) struct MiniCpmOSpecificArgs {
    pub(crate) pixel_values_all: Option<Vec<Vec<Tensor>>>,
    pub(crate) tgt_sizes: Option<Vec<Tensor>>,
    pub(crate) image_bound: Option<Vec<Tensor>>,
    pub(crate) image_hashes: Vec<u64>,
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
    fn max_seq_len(&self) -> usize {
        self.llm.max_seq_len()
    }
    fn forward(
        &self,
        input_ids: &Tensor,
        _pixel_values: Option<Tensor>,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        model_specific_args: Box<dyn Any>, // pixel attention mask, or image sizes, or anything else
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let MiniCpmOSpecificArgs {
            pixel_values_all,
            tgt_sizes,
            image_bound,
            image_hashes,
        } = *model_specific_args
            .downcast()
            .expect("Cannot downcast into `MiniCpmOSpecificArgs`");
        self.forward(
            input_ids,
            pixel_values_all,
            tgt_sizes,
            image_bound,
            &image_hashes,
            seqlen_offsets,
            context_lens,
            metadata,
            flash_params,
        )
    }
    fn default_model_specific_args(&self, _input_ids: &Tensor) -> Box<dyn Any> {
        Box::new(MiniCpmOSpecificArgs {
            pixel_values_all: None,
            tgt_sizes: None,
            image_bound: None,
            image_hashes: vec![],
        })
    }
    fn encoder_cache_counters(
        &self,
    ) -> Option<(
        Arc<std::sync::atomic::AtomicUsize>,
        Arc<std::sync::atomic::AtomicUsize>,
    )> {
        Some(
            self.encoder_cache
                .lock()
                .expect("encoder cache poisoned")
                .counters(),
        )
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
        let uvb = UnVarBuilder::new();

        uvb.pp("llm").extend(self.llm.residual_tensors());
        uvb.pp("vpm").extend(self.vpm.residual_tensors());
        uvb.pp("resampler")
            .extend(self.resampler.residual_tensors());

        uvb.to_safetensors()
    }

    // NOTE: We ONLY calibrate the text bits of these models, so we should only track/return those parts!!

    /// This is used for imatrix generation internally. Begin stats tracking.
    fn begin_track_stats(&mut self) -> anyhow::Result<()> {
        self.llm.begin_track_stats()
    }

    /// End stats tracking and return the imatrix data
    fn extract_imatrix_data(&mut self) -> candle_core::Result<CollectedImatrixData> {
        self.llm.extract_imatrix_data()
    }
}

impl AnyMoeBaseModelMixin for MiniCpmOModel {}
