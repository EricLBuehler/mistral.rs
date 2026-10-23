#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

pub(crate) mod config;
pub(crate) mod generation;

use std::sync::Arc;

use candle_core::{DType, Module, Result, Tensor, D};
use mistralrs_quant::{NonZeroOp, QuantMethod, ShardedVarBuilder};

use crate::{
    layers::{Activation, RmsNorm},
    paged_attention::AttentionImplementation,
    pipeline::{ModelForwardContext, NormalLoadingMetadata},
    vision_models::gemma4::{
        multimodal_embedding::Gemma4MultimodalEmbedder, text::TextModel, vision::VisionTower,
    },
};

pub(crate) use config::DiffusionGemmaConfig;

/// Gated MLP over the previous denoising step's soft embeddings; its output is added to
/// the canvas input embeddings and post-normalized (norm without learned scale).
struct SelfConditioning {
    pre_norm: RmsNorm,
    post_norm: RmsNorm,
    gate_proj: Arc<dyn QuantMethod>,
    up_proj: Arc<dyn QuantMethod>,
    down_proj: Arc<dyn QuantMethod>,
    act: Activation,
}

impl SelfConditioning {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        eps: f64,
        act: Activation,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let pre_norm = RmsNorm::new(hidden_size, eps, vb.pp("pre_norm"))?;
        let post_norm = RmsNorm::from_w(Tensor::ones(hidden_size, vb.dtype(), vb.device())?, eps)?;
        let gate_proj = mistralrs_quant::linear_no_bias(
            hidden_size,
            intermediate_size,
            &None,
            vb.pp("gate_proj"),
        )?;
        let up_proj = mistralrs_quant::linear_no_bias(
            hidden_size,
            intermediate_size,
            &None,
            vb.pp("up_proj"),
        )?;
        let down_proj = mistralrs_quant::linear_no_bias(
            intermediate_size,
            hidden_size,
            &None,
            vb.pp("down_proj"),
        )?;
        Ok(Self {
            pre_norm,
            post_norm,
            gate_proj,
            up_proj,
            down_proj,
            act,
        })
    }

    fn forward(&self, inputs_embeds: &Tensor, soft_embeds: &Tensor) -> Result<Tensor> {
        let normed = self.pre_norm.forward(soft_embeds)?;
        let gated = self.gate_proj.forward(&normed)?.apply(&self.act)?;
        let up = self.up_proj.forward(&normed)?;
        let sc_signal = self.down_proj.forward(&(gated * up)?)?;
        self.post_norm.forward(&(inputs_embeds + sc_signal)?)
    }
}

/// DiffusionGemma: a single Gemma4 backbone run in two modes. Encoder mode is causal and
/// writes the KV cache (prompt prefill, then each accepted canvas); decoder mode denoises
/// a canvas with bidirectional attention, reading the cache without writing it.
///
/// Checkpoint layout: the backbone lives once under `model.decoder.*` (the encoder ties to
/// it, except per-layer `layer_scalar` buffers); vision under `model.encoder.vision_tower.*`.
pub struct DiffusionGemmaModel {
    text: TextModel,
    vision: Option<(VisionTower, Gemma4MultimodalEmbedder)>,
    self_conditioning: SelfConditioning,
    encoder_layer_scalars: Vec<Tensor>,
    cfg: DiffusionGemmaConfig,
    embed_scale: f64,
    vision_dtype: DType,
    diffusion_params: std::sync::OnceLock<generation::DiffusionParams>,
    last_denoise_micros: std::sync::atomic::AtomicU64,
}

impl DiffusionGemmaModel {
    pub fn new(
        cfg: &DiffusionGemmaConfig,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let vb = vb.pp("model");
        let text_cfg = &cfg.text_config;

        let vision_dtype = if vb.dtype() == DType::F16 {
            DType::F32
        } else {
            vb.dtype()
        };

        let vb_encoder = vb.pp("encoder");
        let vision = if let Some(ref vision_cfg) = cfg.vision_config {
            let tower = VisionTower::new(
                vision_cfg,
                normal_loading_metadata
                    .mapper
                    .set_nm_device(vb_encoder.pp("vision_tower"), false)
                    .set_dtype(vision_dtype),
            )?;
            let embedder = Gemma4MultimodalEmbedder::new(
                vision_cfg.hidden_size,
                text_cfg.hidden_size,
                vision_cfg.rms_norm_eps,
                normal_loading_metadata
                    .mapper
                    .set_nm_device(vb_encoder.pp("embed_vision"), false)
                    .set_dtype(vision_dtype),
            )?;
            Some((tower, embedder))
        } else {
            None
        };

        // Encoder layers tie all weights to the decoder's; only `layer_scalar` is its own.
        let vb_enc_layers = vb_encoder.pp("language_model").pp("layers");
        let mut encoder_layer_scalars = Vec::with_capacity(text_cfg.num_hidden_layers);
        for i in 0..text_cfg.num_hidden_layers {
            let scalar = normal_loading_metadata
                .mapper
                .set_device(i, vb_enc_layers.pp(i), false)
                .get((1,), "layer_scalar")?;
            encoder_layer_scalars.push(scalar);
        }

        let vb_decoder = vb.pp("decoder");
        let self_conditioning = SelfConditioning::new(
            text_cfg.hidden_size,
            text_cfg.intermediate_size,
            text_cfg.rms_norm_eps,
            text_cfg.hidden_activation,
            normal_loading_metadata
                .mapper
                .set_nm_device(vb_decoder.pp("self_conditioning"), false),
        )?;

        let text = TextModel::new(
            text_cfg,
            Some(cfg.image_token_id),
            None,
            vb_decoder,
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )?;

        Ok(Self {
            text,
            vision,
            self_conditioning,
            encoder_layer_scalars,
            embed_scale: (text_cfg.hidden_size as f64).sqrt(),
            cfg: cfg.clone(),
            vision_dtype,
            diffusion_params: std::sync::OnceLock::new(),
            last_denoise_micros: std::sync::atomic::AtomicU64::new(0),
        })
    }

    pub fn canvas_length(&self) -> usize {
        self.cfg.canvas_length
    }

    pub fn config(&self) -> &DiffusionGemmaConfig {
        &self.cfg
    }

    /// Encoder pass: causal, writes the KV cache. `xs` are the (vision-merged) input
    /// embeddings for the uncached tokens (prompt at prefill, last accepted canvas after).
    pub fn encoder_forward_embeds(
        &self,
        input_ids: &Tensor,
        xs: Tensor,
        ctx: &mut ModelForwardContext<'_>,
        has_images: bool,
    ) -> Result<Tensor> {
        self.text.forward_embeds_scaled(
            input_ids,
            input_ids,
            xs,
            ctx,
            has_images,
            Some(&self.encoder_layer_scalars),
        )
    }

    /// One denoising pass: embed the canvas, inject self-conditioning, run the decoder
    /// bidirectionally over [cache + canvas]. Returns softcapped logits [b, CL, vocab].
    /// `self_conditioning_logits` carries the processed logits from the prior step.
    pub fn denoise_step(
        &self,
        canvas_ids: &Tensor,
        self_conditioning_logits: Option<&Tensor>,
        rope_positions: &Tensor,
        canvas_kv: &[(Tensor, Tensor)],
    ) -> Result<Tensor> {
        let inputs_embeds = self.text.embed_tokens(canvas_ids)?;
        let soft = match self_conditioning_logits {
            Some(logits) => self.soft_embed(logits)?,
            None => inputs_embeds.zeros_like()?,
        };
        let xs = self.self_conditioning.forward(&inputs_embeds, &soft)?;
        self.text
            .forward_canvas_embeds(xs, rope_positions, canvas_kv)
    }

    pub fn soft_embed(&self, logits: &Tensor) -> Result<Tensor> {
        let embed_w = self.text.embedding_weight()?;
        let probs = candle_nn::ops::softmax(&logits.to_dtype(DType::F32)?, D::Minus1)?;
        let soft = probs
            .to_dtype(embed_w.dtype())?
            .broadcast_matmul(&embed_w)?;
        soft * self.embed_scale
    }

    pub fn self_conditioning_logits(&self, logits: &Tensor) -> Result<Tensor> {
        logits.to_dtype(self.text.embedding_dtype())
    }

    pub fn set_diffusion_params(&self, params: generation::DiffusionParams) {
        let _ = self.diffusion_params.set(params);
    }

    fn diffusion_params(&self) -> generation::DiffusionParams {
        self.diffusion_params.get().cloned().unwrap_or_default()
    }

    fn merge_vision_embeds(
        &self,
        input_ids: &Tensor,
        mut input_embeds: Tensor,
        pixel_values: &Tensor,
        image_sizes: &[(u32, u32)],
    ) -> Result<Tensor> {
        let (tower, embedder) = self.vision.as_ref().ok_or_else(|| {
            candle_core::Error::Msg(
                "DiffusionGemma model was loaded without a vision encoder.".to_string(),
            )
        })?;

        let n_images = pixel_values.dim(0)?;
        let per_image = (0..n_images)
            .map(|i| {
                let pv = pixel_values.get(i)?.unsqueeze(0)?;
                let pv = if let Some((h, w)) = image_sizes.get(i).copied() {
                    pv.narrow(2, 0, h as usize)?.narrow(3, 0, w as usize)?
                } else {
                    pv
                };
                pv.to_dtype(self.vision_dtype)
            })
            .collect::<Result<Vec<_>>>()?;
        let features = tower.forward(&per_image)?;
        let image_embeds = embedder
            .forward(&features)?
            .to_dtype(input_embeds.dtype())?
            .squeeze(0)?;

        let image_mask = input_ids
            .to_dtype(DType::F32)?
            .eq(self.cfg.image_token_id as f64)?;
        let image_mask_expanded = image_mask
            .unsqueeze(D::Minus1)?
            .broadcast_as(input_embeds.shape())?
            .to_dtype(DType::U32)?;
        let indices = image_mask_expanded.flatten_all()?.nonzero()?.squeeze(1)?;
        if indices.dim(0)? > 0 {
            let mut x_flat = input_embeds.flatten_all()?;
            let src_flat = image_embeds.flatten_all()?;
            let current_vals = x_flat.gather(&indices, 0)?;
            let diff = (src_flat - current_vals)?;
            x_flat = x_flat.scatter_add(&indices, &diff, 0)?;
            input_embeds = x_flat.reshape(input_embeds.shape())?;
        }
        Ok(input_embeds)
    }
}

impl crate::pipeline::IsqModel for DiffusionGemmaModel {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let mut tensors: Vec<(String, Tensor)> = self
            .text
            .residual_tensors()
            .into_iter()
            .map(|(name, t)| (format!("model.decoder.{name}"), t))
            .collect();

        let uvb = crate::utils::unvarbuilder::UnVarBuilder::new();
        uvb.pp("model")
            .pp("decoder")
            .pp("self_conditioning")
            .pp("pre_norm")
            .add(&self.self_conditioning.pre_norm);
        let uvb_enc_layers = uvb
            .pp("model")
            .pp("encoder")
            .pp("language_model")
            .pp("layers");
        for (i, scalar) in self.encoder_layer_scalars.iter().enumerate() {
            uvb_enc_layers
                .pp(i)
                .add_tensor("layer_scalar", scalar.clone());
        }
        tensors.extend(uvb.to_safetensors());

        if let Some((tower, embedder)) = &self.vision {
            tensors.extend(
                tower
                    .residual_tensors()
                    .into_iter()
                    .map(|(name, t)| (format!("model.encoder.vision_tower.{name}"), t)),
            );
            tensors.extend(
                embedder
                    .residual_tensors()
                    .into_iter()
                    .map(|(name, t)| (format!("model.encoder.embed_vision.{name}"), t)),
            );
        }

        tensors
    }
}

impl crate::amoe::AnyMoeBaseModelMixin for DiffusionGemmaModel {}
impl crate::speculative::SpeculativeTargetMixin for DiffusionGemmaModel {}

impl crate::pipeline::MultimodalModel for DiffusionGemmaModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        model_specific_args: Box<dyn std::any::Any>,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let (b_sz, _q_len) = input_ids.dims2()?;
        let args = model_specific_args
            .downcast::<crate::vision_models::gemma4::Gemma4SpecificArgs>()
            .expect("DiffusionGemma expects Gemma4SpecificArgs");

        let mut input_embeds = self.text.embed_tokens(input_ids)?;
        if let Some(ref pixel_values) = pixel_values {
            input_embeds =
                self.merge_vision_embeds(input_ids, input_embeds, pixel_values, &args.image_sizes)?;
        }

        ctx.require_full_prefill_queries();
        // Encoder pass fills the KV cache; its next-token logits are unused.
        let _ =
            self.encoder_forward_embeds(input_ids, input_embeds, ctx, pixel_values.is_some())?;

        // Chunked prompts encode chunk by chunk; only the final chunk denoises a canvas.
        if !ctx.is_final_prompt_chunk() {
            return Tensor::from_vec(Vec::<u32>::new(), (b_sz, 0), &candle_core::Device::Cpu);
        }

        if let Ok(dump_path) = std::env::var("MISTRALRS_DIFFUSION_DEBUG_DUMP") {
            if input_ids.dim(1)? > 1 {
                input_ids
                    .to_dtype(DType::I64)?
                    .to_device(&candle_core::Device::Cpu)?
                    .write_npy(format!("{dump_path}.prompt_ids.npy"))?;
            }
        }

        // The whole batch denoises in lockstep: the scheduler buckets sequences by context
        // length, so KV is rectangular and one batched gather + one batched denoise loop
        // serve every sequence. Offsets may differ only via prefix-cache trims.
        let params = self.diffusion_params();
        let offsets = ctx.seqlen_offsets().to_vec();
        let context_lens = ctx.context_lens().to_vec();
        let mut cache_offsets = Vec::with_capacity(b_sz);
        for seq_idx in 0..b_sz {
            // (start, len) selects this row's real last position, so start + len is its
            // unpadded query length.
            let (sel_start, sel_len) = context_lens[seq_idx];
            cache_offsets.push(offsets[seq_idx] + sel_start + sel_len);
        }
        let kv_len = cache_offsets[0];
        if cache_offsets.iter().any(|&len| len != kv_len) {
            candle_core::bail!(
                "block-diffusion batch must share one context length, got {cache_offsets:?}"
            );
        }
        let canvas_kv = self.text.gather_canvas_kv(ctx, b_sz, kv_len)?;
        let (blocks, denoise_time) = generation::generate_canvas(
            self,
            &params,
            &canvas_kv,
            &cache_offsets,
            input_ids.device(),
            args.block_denoising_progress.as_deref(),
        )?;
        self.last_denoise_micros.store(
            denoise_time.as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        let canvas_length = blocks[0].len();
        Tensor::from_vec(
            blocks.into_iter().flatten().collect::<Vec<u32>>(),
            (b_sz, canvas_length),
            &candle_core::Device::Cpu,
        )
    }

    fn default_model_specific_args(&self, _input_ids: &Tensor) -> Box<dyn std::any::Any> {
        Box::new(crate::vision_models::gemma4::Gemma4SpecificArgs::default())
    }

    fn cache(&self) -> &crate::pipeline::EitherCache {
        crate::pipeline::MultimodalModel::cache(&self.text)
    }

    fn device(&self) -> &candle_core::Device {
        crate::pipeline::MultimodalModel::device(&self.text)
    }

    fn max_seq_len(&self) -> usize {
        crate::pipeline::MultimodalModel::max_seq_len(&self.text)
    }

    fn config(&self) -> &crate::paged_attention::ModelConfigMetadata {
        crate::pipeline::MultimodalModel::config(&self.text)
    }

    fn model_config(&self) -> Arc<dyn crate::paged_attention::ModelConfigLike + Send + Sync> {
        self.text.model_config_like()
    }
}

impl crate::block_diffusion::BlockDiffusionMixin for DiffusionGemmaModel {
    fn is_block_diffusion(&self) -> bool {
        true
    }

    fn configure_block_diffusion(&self, generation_config_json: &str) {
        self.set_diffusion_params(generation::DiffusionParams::from_generation_config(
            generation_config_json,
        ));
    }

    fn take_block_denoise_time(&self) -> Option<std::time::Duration> {
        let micros = self
            .last_denoise_micros
            .swap(0, std::sync::atomic::Ordering::Relaxed);
        (micros > 0).then(|| std::time::Duration::from_micros(micros))
    }
}
