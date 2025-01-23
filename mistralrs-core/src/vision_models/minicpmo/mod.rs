use std::{any::Any, collections::HashMap, sync::Arc};

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Linear, Module, VarBuilder};
pub use config::MiniCpmOConfig;
pub use inputs_processor::MiniCpmOProcessor;
use mistralrs_quant::QuantMethod;
use resampler::Resampler;

use crate::{
    amoe::AnyMoeBaseModelMixin,
    device_map::DeviceMapper,
    layers::{Activation, AvgPool1d, GetFloatInfo},
    models::qwen2,
    paged_attention::{AttentionImplementation, ModelConfigMetadata},
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, NormalLoadingMetadata, NormalModel, VisionModel,
    },
    utils::unvarbuilder::UnVarBuilder,
};

use self::siglip::SiglipVisionTransformer;

use super::common::{siglip, whisper::WhisperEncoder};

mod config;
mod inputs_processor;
mod resampler;

pub struct MultiModalProjector {
    act: Activation,
    linear1: Linear,
    linear2: Linear,
}

impl MultiModalProjector {
    fn new(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            act: Activation::Relu,
            linear1: candle_nn::linear(in_dim, out_dim, vb.pp("linear1"))?,
            linear2: candle_nn::linear(in_dim, out_dim, vb.pp("linear2"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.linear1)?
            .apply(&self.act)?
            .apply(&self.linear2)
    }
}

pub struct MiniCpmOModel {
    cfg: MiniCpmOConfig,
    llm: qwen2::Model,
    vpm: SiglipVisionTransformer,
    resampler: Resampler,
    apm: WhisperEncoder,
    audio_projection_layer: MultiModalProjector,
    audio_avg_pooler: AvgPool1d,
}

impl MiniCpmOModel {
    pub fn new(
        cfg: &MiniCpmOConfig,
        vb: VarBuilder,
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
        // Vision
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
        // Audio
        let apm = WhisperEncoder::new(&cfg.audio_config, vb.pp("apm"))?;
        let audio_projection_layer = MultiModalProjector::new(
            cfg.audio_config.encoder_ffn_dim / 4,
            cfg.text_config.hidden_size,
            vb.pp("audio_projection_layer"),
        )?;
        let audio_avg_pooler = AvgPool1d {
            kernel_size: cfg.audio_pool_step,
            stride: cfg.audio_pool_step,
        };
        Ok(Self {
            cfg: cfg.clone(),
            llm,
            vpm,
            resampler,
            apm,
            audio_projection_layer,
            audio_avg_pooler,
        })
    }

    fn get_vllm_embedding(
        &self,
        input_ids: &Tensor,
        device: &Device,
        pixel_values_all: Option<Vec<Vec<Tensor>>>,
        tgt_sizes: Option<Vec<Tensor>>,
        image_bound: Option<Vec<Tensor>>,
    ) -> Result<Tensor> {
        let mut vllm_embedding = self.llm.get_input_embeddings(input_ids)?;

        if let Some(pixel_values_all) = pixel_values_all {
            let tgt_sizes_all = tgt_sizes.as_ref().expect("Need tgt_sizes");
            let image_bound = image_bound.expect("Need image_bound");
            let image_bound_vec = image_bound
                .into_iter()
                .map(|x| x.to_vec2::<u32>())
                .collect::<Result<Vec<_>>>()?;

            let mut all_pixel_values = Vec::new();
            let mut img_cnts = Vec::new();
            for pixel_values in &pixel_values_all {
                img_cnts.push(pixel_values.len());
                let mut imgs = Vec::new();
                for i in pixel_values {
                    // Assume channel dimension first
                    imgs.push(i.flatten_to(1)?.permute((1, 0))?);
                }
                all_pixel_values.extend(imgs);
            }

            let tgt_sizes = Tensor::cat(tgt_sizes_all, 0)?;
            let tgt_sizes_vec = tgt_sizes.to_vec2::<u32>()?;

            let max_patches = (tgt_sizes.i((.., 0))? * tgt_sizes.i((.., 1))?)?
                .max(0)?
                .to_scalar::<u32>()? as usize;

            // Original code does padding of the pixel values here
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
                    &[&i, &0, &(..n)],
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

    fn get_feat_extract_output_lengths(&self, input_lengths: &Tensor) -> Result<(Tensor, Tensor)> {
        let input_lengths = input_lengths.to_dtype(DType::I32)?;
        let input_lengths_after_cnn = (((input_lengths - 1.)? / 2.)?.floor()? + 1.)?;
        let input_lengths_after_pooling = (((&input_lengths_after_cnn
            - self.cfg.audio_pool_step as f64)?
            / self.cfg.audio_pool_step as f64)?
            .floor()?
            + 1.)?;

        Ok((input_lengths_after_cnn, input_lengths_after_pooling))
    }

    fn get_audio_embedding(
        &self,
        audio_features: &Tensor,
        audio_feature_lens_raw: Vec<Tensor>,
    ) -> Result<Vec<Vec<Tensor>>> {
        let audio_feature_lens = Tensor::cat(&audio_feature_lens_raw, 0)?;
        let (bs, _, max_mel_seq_len) = audio_features.dims3()?;
        let max_seq_len = (max_mel_seq_len - 1) / 2 + 1;

        // Create a sequence tensor of shape (bs, max_seq_len)
        let seq_range = Tensor::arange(0, max_seq_len as u32, audio_features.device())?
            .unsqueeze(1)?
            .expand((bs, max_seq_len))?;
        let lengths_expand = audio_feature_lens.unsqueeze(1)?.expand((bs, max_seq_len))?;

        // Create mask: 1 for padded values
        let padding_mask = seq_range.ge(&lengths_expand)?;
        let audio_attention_mask = padding_mask.reshape((bs, 1, 1, max_seq_len))?.expand((
            bs,
            1,
            max_seq_len,
            max_seq_len,
        ))?;
        let apm_dtype = self.apm.dtype();
        // 1 -> -inf, 0 -> 0
        let audio_attention_mask =
            (audio_attention_mask.to_dtype(apm_dtype)? * apm_dtype.finfo()?.min)?;

        let audio_states = self
            .apm
            .forward(audio_features, Some(&audio_attention_mask))?;
        let mut audio_embeds = self.audio_projection_layer.forward(&audio_states)?;

        audio_embeds = audio_embeds.transpose(1, 2)?;
        audio_embeds = self.audio_avg_pooler.forward(&audio_embeds)?;
        audio_embeds = audio_embeds.transpose(1, 2)?;

        let (_, feature_lens_after_pooling) =
            self.get_feat_extract_output_lengths(&audio_feature_lens)?;

        let num_audio_tokens = feature_lens_after_pooling.to_vec1::<i32>()?;

        let mut final_audio_embeds = Vec::new();
        let mut idx = 0;
        for lens_i in &audio_feature_lens_raw {
            let mut target_audio_embeds = Vec::new();
            for _ in 0..lens_i.dim(0)? {
                target_audio_embeds.push(audio_embeds.i((
                    idx,
                    ..num_audio_tokens[idx] as usize,
                    ..,
                ))?);
                idx += 1;
            }
            final_audio_embeds.push(target_audio_embeds)
        }

        Ok(final_audio_embeds)
    }

    fn get_omni_embedding(
        &self,
        input_embeddings: &Tensor,
        audio_features: &Tensor,
        audio_feature_lens_raw: Vec<Tensor>,
        audio_bound: Vec<Tensor>,
    ) -> Result<Tensor> {
        let audio_embeddings = self.get_audio_embedding(audio_features, audio_feature_lens_raw)?;

        assert_eq!(audio_embeddings.len(), audio_bound.len());
        let audio_bound_vec = audio_bound
            .into_iter()
            .map(|x| x.to_vec2::<u32>())
            .collect::<Result<Vec<_>>>()?;

        let mut new_embeddings = input_embeddings.clone();
        for ((i, audio_embs), bounds) in audio_embeddings.iter().enumerate().zip(audio_bound_vec) {
            assert_eq!(audio_embs.len(), bounds.len());
            for (embs, bound) in audio_embs.iter().zip(bounds) {
                let audio_indices_len = bound[1] - bound[0];

                if embs.dim(0)? != audio_indices_len as usize {
                    candle_core::bail!(
                        "Shape mismatch: Trying to assign embeddings of shape {:?} to input indices of length {audio_indices_len}",
                        embs.dims(),
                    );
                }

                new_embeddings = new_embeddings.slice_assign(
                    &[&i, &(bound[0] as usize..bound[1] as usize), &..],
                    &embs.to_dtype(input_embeddings.dtype())?,
                )?;
            }
        }

        Ok(new_embeddings)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values_all: Option<Vec<Vec<Tensor>>>,
        tgt_sizes: Option<Vec<Tensor>>,
        image_bound: Option<Vec<Tensor>>,
        audio_features: Option<Tensor>,
        audio_feature_lens_raw: Option<Vec<Tensor>>,
        audio_bound: Option<Vec<Tensor>>,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &mut PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let mut embedding = self.get_vllm_embedding(
            input_ids,
            self.llm.device(),
            pixel_values_all,
            tgt_sizes,
            image_bound,
        )?;

        if let Some(audio_features) = audio_features {
            let audio_feature_lens_raw =
                audio_feature_lens_raw.expect("Require audio_feature_lens_raw");
            let audio_bound = audio_bound.expect("Require audio_feature_lens_raw");
            embedding = self.get_omni_embedding(
                &embedding,
                &audio_features,
                audio_feature_lens_raw,
                audio_bound,
            )?;
        }

        self.llm.forward_embed(
            input_ids,
            embedding,
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
    pub(crate) audio_features: Option<Tensor>,
    pub(crate) audio_feature_lens_raw: Option<Vec<Tensor>>,
    pub(crate) audio_bound: Option<Vec<Tensor>>,
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
        _pixel_values: Option<Tensor>,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        model_specific_args: Box<dyn Any>, // pixel attention mask, or image sizes, or anything else
        metadata: Option<(Vec<(Tensor, Tensor)>, &mut PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let MiniCpmOSpecificArgs {
            pixel_values_all,
            tgt_sizes,
            image_bound,
            audio_features,
            audio_feature_lens_raw,
            audio_bound,
        } = *model_specific_args
            .downcast()
            .expect("Cannot downcast into `MiniCpmOSpecificArgs`");

        self.forward(
            input_ids,
            pixel_values_all,
            tgt_sizes,
            image_bound,
            audio_features,
            audio_feature_lens_raw,
            audio_bound,
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
            audio_features: None,
            audio_feature_lens_raw: None,
            audio_bound: None,
        })
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
    fn extract_imatrix_data(&mut self) -> candle_core::Result<HashMap<usize, Option<Vec<f32>>>> {
        self.llm.extract_imatrix_data()
    }
}

impl AnyMoeBaseModelMixin for MiniCpmOModel {}
