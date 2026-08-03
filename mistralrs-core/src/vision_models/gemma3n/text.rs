use std::{collections::HashMap, sync::Arc};

use crate::layers_masker::CausalMaskConfig;
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::Linear;
use mistralrs_quant::{
    softcap, ColumnParallelLayer, QuantMethod, ReplicatedLayer, RowParallelLayer, ShardedVarBuilder,
};
use statrs::distribution::{ContinuousCDF, Normal};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::{AttentionMask, SdpaParams},
    device_map::{DeviceMappedMask, DeviceMapper},
    layers::{
        self, dense_embedding, embedding_with_legacy_tied_uqff, Activation, CausalMasker,
        Gemma3nRotaryEmbedding, RmsNorm, RotaryEmbedding, ScaledEmbedding, Sdpa,
    },
    matformer::MatformerSliceConfig,
    paged_attention::{
        AttentionImplementation, KvCacheTopology, ModelConfigLike, ModelConfigMetadata,
        PagedAttention,
    },
    pipeline::{
        extract_logits,
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, ModelForwardContext, MultimodalModel, NormalCache,
        NormalCacheType, NormalLoadingMetadata,
    },
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

use super::config::{Gemma3nTextConfig, IntermediateSize};

macro_rules! is_sliding {
    ($layer_idx:expr, $cfg:expr) => {
        $cfg.layer_types[$layer_idx] == "sliding_attention"
    };
}

const EPS: f64 = 1e-8;

fn sliding_decode_kv_window(
    use_sliding_window: bool,
    query_len: usize,
    sliding_window: Option<usize>,
    kv_len: usize,
) -> Option<(usize, usize)> {
    let window = sliding_window?;
    if !use_sliding_window || query_len != 1 || kv_len <= window {
        return None;
    }
    Some((kv_len - window, window))
}

fn is_paged_decode_forward(
    is_paged: bool,
    query_len: usize,
    is_first_prompt_chunk: bool,
    has_prompt_cache_metadata: bool,
) -> bool {
    is_paged && query_len > 0 && !is_first_prompt_chunk && !has_prompt_cache_metadata
}

fn kv_shared_layer_index_for_layout(
    layer_types: &[String],
    num_kv_shared_layers: usize,
    layer_idx: usize,
) -> Result<Option<usize>> {
    let first_kv_shared_layer_idx = layer_types
        .len()
        .checked_sub(num_kv_shared_layers)
        .ok_or_else(|| candle_core::Error::msg("Gemma 3n has more shared KV layers than layers"))?;
    let attention_type = layer_types.get(layer_idx).ok_or_else(|| {
        candle_core::Error::msg(format!("Gemma 3n layer index {layer_idx} is out of bounds"))
    })?;
    if num_kv_shared_layers == 0 || layer_idx < first_kv_shared_layer_idx {
        return Ok(None);
    }
    if first_kv_shared_layer_idx == 0 {
        candle_core::bail!("Gemma 3n shared KV layers have no donor layers");
    }

    layer_types[..first_kv_shared_layer_idx]
        .iter()
        .rposition(|candidate| candidate == attention_type)
        .map(Some)
        .ok_or_else(|| {
            candle_core::Error::msg(format!(
                "Gemma 3n layer {layer_idx} shares KV without a prior `{attention_type}` donor"
            ))
        })
}

fn kv_shared_layer_index(cfg: &Gemma3nTextConfig, layer_idx: usize) -> Result<Option<usize>> {
    if cfg.layer_types.len() != cfg.num_hidden_layers {
        candle_core::bail!(
            "Gemma 3n has {} layer types for {} layers",
            cfg.layer_types.len(),
            cfg.num_hidden_layers
        );
    }
    kv_shared_layer_index_for_layout(&cfg.layer_types, cfg.num_kv_shared_layers, layer_idx)
}

struct AttentionForwardContext<'a> {
    attention_mask: &'a AttentionMask,
    sliding_attention_mask: &'a AttentionMask,
    rope_positions: &'a Tensor,
    kv_caches: &'a mut [KvCache],
    metadata: Option<((Tensor, Tensor), &'a PagedAttentionInputMetadata)>,
    flash_params: Option<&'a FlashParams>,
}

#[derive(Clone)]
struct Gemma3nModelConfigLike {
    base: ModelConfigMetadata,
    kv_cache_topology: KvCacheTopology,
}

impl ModelConfigLike for Gemma3nModelConfigLike {
    fn max_seq_len(&self) -> usize {
        self.base.max_seq_len
    }

    fn num_layers(&self) -> usize {
        self.base.num_layers
    }

    fn hidden_size(&self) -> usize {
        self.base.hidden_size
    }

    fn num_kv_heads(&self) -> usize {
        self.base.num_kv_heads
    }

    fn num_attn_heads(&self) -> usize {
        self.base.num_attn_heads
    }

    fn k_head_dim(&self) -> usize {
        self.base.k_head_dim
    }

    fn v_head_dim(&self) -> usize {
        self.base.v_head_dim
    }

    fn has_kv_cache_sharing(&self) -> bool {
        self.kv_cache_topology.has_shared_layers()
    }

    fn kv_cache_topology(&self) -> KvCacheTopology {
        self.kv_cache_topology.clone()
    }
}

#[derive(Clone)]
pub struct Mlp {
    gate: Arc<dyn QuantMethod>,
    up: Arc<dyn QuantMethod>,
    down: Arc<dyn QuantMethod>,
    activation_sparsity: f64,
    act: Activation,
    std_multiplier: f64,
}

impl Mlp {
    fn new(
        vb: ShardedVarBuilder,
        cfg: &Gemma3nTextConfig,
        comm: &Arc<mistralrs_quant::Comm>,
        layer_idx: usize,
    ) -> Result<Self> {
        let std_multiplier = Self::std_multiplier(cfg.activation_sparsity_pattern[layer_idx]);
        let (intermediate_size, orig_intermediate_size) = match &cfg.intermediate_size {
            IntermediateSize::Single(size) => (*size, None),
            IntermediateSize::PerLayer(sizes) => (sizes[layer_idx], None),
            IntermediateSize::Matformer(sizes, orig_sizes) => {
                (sizes[layer_idx], Some(orig_sizes[layer_idx]))
            }
        };

        if let Some(orig_intermediate_size) = orig_intermediate_size {
            Ok(Self {
                gate: ColumnParallelLayer::new_matformer(
                    cfg.hidden_size,
                    intermediate_size,
                    orig_intermediate_size,
                    &cfg.quantization_config,
                    false,
                    comm,
                    vb.pp("gate_proj"),
                )?,
                up: ColumnParallelLayer::new_matformer(
                    cfg.hidden_size,
                    intermediate_size,
                    orig_intermediate_size,
                    &cfg.quantization_config,
                    false,
                    comm,
                    vb.pp("up_proj"),
                )?,
                down: RowParallelLayer::new_matformer(
                    intermediate_size,
                    cfg.hidden_size,
                    orig_intermediate_size,
                    &cfg.quantization_config,
                    false,
                    comm,
                    vb.pp("down_proj"),
                )?,
                activation_sparsity: cfg.activation_sparsity_pattern[layer_idx],
                act: cfg.hidden_activation,
                std_multiplier,
            })
        } else {
            Ok(Self {
                gate: ColumnParallelLayer::new(
                    cfg.hidden_size,
                    intermediate_size,
                    &cfg.quantization_config,
                    false,
                    comm,
                    vb.pp("gate_proj"),
                )?,
                up: ColumnParallelLayer::new(
                    cfg.hidden_size,
                    intermediate_size,
                    &cfg.quantization_config,
                    false,
                    comm,
                    vb.pp("up_proj"),
                )?,
                down: RowParallelLayer::new(
                    intermediate_size,
                    cfg.hidden_size,
                    &cfg.quantization_config,
                    false,
                    comm,
                    vb.pp("down_proj"),
                )?,
                activation_sparsity: cfg.activation_sparsity_pattern[layer_idx],
                act: cfg.hidden_activation,
                std_multiplier,
            })
        }
    }

    fn std_multiplier(p: f64) -> f64 {
        let normal = Normal::new(0.0, 1.0).unwrap();
        normal.inverse_cdf(p)
    }

    fn gaussian_topk(&self, xs: &Tensor) -> Result<Tensor> {
        // Cast to float32 for better precision in statistical calculations
        let xs_f32 = xs.to_dtype(DType::F32)?;
        let xs_mean = xs_f32.mean_keepdim(D::Minus1)?;
        let xs_sq_mean = xs_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let var = (&xs_sq_mean - xs_mean.sqr()?)?;
        let xs_std = (var + EPS)?.sqrt()?;
        let cutoff_xs = (xs_mean + (xs_std * self.std_multiplier)?)?;
        // Convert back to original dtype after computation
        xs.broadcast_sub(&cutoff_xs.to_dtype(xs.dtype())?)?.relu()
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut gate = self.gate.forward(xs)?;
        if self.activation_sparsity > 0. {
            gate = self.gaussian_topk(&gate)?;
        }
        let up = self.up.forward(xs)?;
        let res = self
            .down
            .forward(&crate::ops::mul_and_act(&gate, &up, self.act)?)?;
        Ok(res)
    }
}

struct Attention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb_global: Arc<Gemma3nRotaryEmbedding>,
    rotary_emb_local: Arc<RotaryEmbedding>,
    use_sliding_window: bool,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    v_norm: RmsNorm,
    kv_shared_layer_index: Option<usize>,
    layer_idx: usize,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb_global: Arc<Gemma3nRotaryEmbedding>,
        rotary_emb_local: Arc<RotaryEmbedding>,
        cfg: &Gemma3nTextConfig,
        layer_idx: usize,
        mapper: &dyn DeviceMapper,
        vb: ShardedVarBuilder,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let bias = cfg.attention_bias;
        let q_proj = ColumnParallelLayer::new(
            hidden_sz,
            num_heads * head_dim,
            &cfg.quantization_config,
            bias,
            comm,
            vb.pp("q_proj"),
        )?;
        let kv_shard = mistralrs_quant::compute_kv_shard(
            cfg.num_key_value_heads,
            cfg.hidden_size / cfg.num_attention_heads,
            comm,
        )?;
        let k_proj = ColumnParallelLayer::new_with_shard(
            hidden_sz,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            bias,
            comm,
            kv_shard,
            vb.pp("k_proj"),
        )?;
        let v_proj = ColumnParallelLayer::new_with_shard(
            hidden_sz,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            bias,
            comm,
            kv_shard,
            vb.pp("v_proj"),
        )?;
        let o_proj = RowParallelLayer::new(
            num_heads * head_dim,
            hidden_sz,
            &cfg.quantization_config,
            bias,
            comm,
            vb.pp("o_proj"),
        )?;
        let sliding_window = if is_sliding!(layer_idx, cfg) {
            Some(cfg.sliding_window)
        } else {
            None
        };

        let q_norm = RmsNorm::new_gemma_3n(
            cfg.head_dim,
            cfg.rms_norm_eps,
            true,
            mapper.set_device(layer_idx, vb.pp("q_norm"), false),
        )?;
        let k_norm = RmsNorm::new_gemma_3n(
            cfg.head_dim,
            cfg.rms_norm_eps,
            true,
            mapper.set_device(layer_idx, vb.pp("k_norm"), false),
        )?;
        let v_norm = RmsNorm::new_gemma_3n(
            cfg.head_dim,
            cfg.rms_norm_eps,
            false, // this is unique, it is false
            mapper.set_device(layer_idx, vb.pp("v_norm"), false),
        )?;

        let kv_shared_layer_index = kv_shared_layer_index(cfg, layer_idx)?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: num_heads / comm.world_size(),
            num_kv_heads: (num_kv_heads / comm.world_size()).max(1),
            head_dim,
            rotary_emb_global,
            rotary_emb_local,
            use_sliding_window: sliding_window.is_some(),
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: mistralrs_quant::compute_n_kv_groups(
                    cfg.num_key_value_heads,
                    cfg.num_attention_heads,
                    comm,
                )?,
                softcap: None,
                softmax_scale: 1.0,
                sliding_window,
                sinks: None,
            },
            q_norm,
            k_norm,
            v_norm,
            kv_shared_layer_index,
            layer_idx,
        })
    }

    fn apply_rope_positions(&self, xs: &Tensor, positions: &Tensor) -> Result<Tensor> {
        let (cos, sin) = if self.use_sliding_window {
            self.rotary_emb_local.get_cos_sin()?
        } else {
            self.rotary_emb_global.get_cos_sin()?
        };
        let dtype = xs.dtype();
        mistralrs_quant::rotary::apply_rotary_q(
            &xs.transpose(1, 2)?.to_dtype(DType::F32)?,
            &cos.to_dtype(DType::F32)?,
            &sin.to_dtype(DType::F32)?,
            positions,
            true,
        )?
        .to_dtype(dtype)?
        .transpose(1, 2)
    }

    fn forward(&self, xs: &Tensor, ctx: &mut AttentionForwardContext<'_>) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let is_shared = self.kv_shared_layer_index.is_some();
        let qkv = if !is_shared {
            Some(crate::ops::qkv_projections(
                xs,
                &*self.q_proj,
                &*self.k_proj,
                &*self.v_proj,
            )?)
        } else {
            None
        };

        let mut q = if let Some((q, _, _)) = qkv.as_ref() {
            q.clone()
        } else {
            self.q_proj.forward(xs)?
        };
        q = q.reshape((b_sz, q_len, self.num_heads, self.head_dim))?;
        q = q.apply(&self.q_norm)?;
        q = self.apply_rope_positions(&q, ctx.rope_positions)?;
        q = q.transpose(1, 2)?;

        let (k, v) = if is_shared {
            (None, None)
        } else {
            let (mut k, mut v) = if let Some((_, k, v)) = qkv {
                (k, v)
            } else {
                (self.k_proj.forward(xs)?, self.v_proj.forward(xs)?)
            };

            k = k.reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?;
            k = k.apply(&self.k_norm)?;
            k = self.apply_rope_positions(&k, ctx.rope_positions)?;
            k = k.transpose(1, 2)?;

            v = v.reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?;
            v = v.apply(&self.v_norm)?;
            v = v.transpose(1, 2)?;

            (Some(k), Some(v))
        };

        let mask = if self.use_sliding_window {
            ctx.sliding_attention_mask
        } else {
            ctx.attention_mask
        };

        let mut attn_output = match &self.paged_attn {
            Some(paged_attn) => match &ctx.metadata {
                Some(((key_cache, value_cache), input_metadata)) if is_shared => paged_attn
                    .forward_donor_cache(
                        &q,
                        key_cache,
                        value_cache,
                        mask,
                        input_metadata,
                        &self.sdpa_params,
                        ctx.flash_params,
                    )?,
                Some(((key_cache, value_cache), input_metadata)) => paged_attn.forward(
                    &q,
                    k.as_ref().unwrap(),
                    v.as_ref().unwrap(),
                    mask,
                    Some(key_cache.clone()),
                    Some(value_cache.clone()),
                    input_metadata,
                    &self.sdpa_params,
                    ctx.flash_params,
                )?,
                None if is_shared => {
                    candle_core::bail!("Gemma 3n shared KV attention is missing paged metadata")
                }
                None => {
                    let input_metadata = PagedAttentionInputMetadata::dummy(q.device())?;
                    if matches!(mask, AttentionMask::None) {
                        candle_core::bail!("Gemma 3n paged prompt is missing an attention mask");
                    }
                    paged_attn.forward(
                        &q,
                        k.as_ref().unwrap(),
                        v.as_ref().unwrap(),
                        mask,
                        None,
                        None,
                        &input_metadata,
                        &self.sdpa_params,
                        ctx.flash_params,
                    )?
                }
            },
            None => {
                let (mut k, mut v) = if let Some(kv_shared_layer_index) = self.kv_shared_layer_index
                {
                    let shared_cache = &ctx.kv_caches[kv_shared_layer_index];
                    let k = shared_cache.appended_k()?.ok_or_else(|| {
                        candle_core::Error::msg("Gemma 3n shared KV donor has no key cache")
                    })?;
                    let v = shared_cache.appended_v()?.ok_or_else(|| {
                        candle_core::Error::msg("Gemma 3n shared KV donor has no value cache")
                    })?;
                    (k.to_device(q.device())?, v.to_device(q.device())?)
                } else {
                    ctx.kv_caches[self.layer_idx]
                        .append(k.as_ref().unwrap(), v.as_ref().unwrap())?
                };

                if let Some((start, len)) = sliding_decode_kv_window(
                    self.use_sliding_window,
                    q_len,
                    self.sdpa_params.sliding_window,
                    k.dim(2)?,
                ) {
                    k = k.narrow(2, start, len)?;
                    v = v.narrow(2, start, len)?;
                }

                let mask = if let AttentionMask::Custom(mask) = mask {
                    let kv_len = k.dim(2)?;
                    let mask_dims = mask.dims();
                    let narrowed = match mask.rank() {
                        2 if mask_dims[1] > kv_len => {
                            mask.narrow(1, mask_dims[1] - kv_len, kv_len)?
                        }
                        3 if mask_dims[2] > kv_len => {
                            mask.narrow(2, mask_dims[2] - kv_len, kv_len)?
                        }
                        4 if mask_dims[3] > kv_len => {
                            mask.narrow(3, mask_dims[3] - kv_len, kv_len)?
                        }
                        _ => mask.clone(),
                    };
                    AttentionMask::Custom(narrowed)
                } else {
                    mask.clone()
                };

                Sdpa.run_attention(&q, &k, &v, &mask, ctx.flash_params, &self.sdpa_params)?
            }
        };

        attn_output = if matches!(mask, AttentionMask::None) {
            attn_output.reshape((b_sz, q_len, ()))?
        } else {
            attn_output.transpose(1, 2)?.reshape((b_sz, q_len, ()))?
        };
        self.o_proj.forward(&attn_output)
    }
}

struct TextAltUp {
    correct_output_scale: Tensor,
    correction_coefs: Linear,
    prediction_coefs: Linear,
    modality_router: Linear,
    router_norm: RmsNorm,
    router_input_scale: f64,
    altup_active_idx: usize,
    altup_num_inputs: usize,
}

impl TextAltUp {
    fn new(cfg: &Gemma3nTextConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let correct_output_scale = vb.get(cfg.hidden_size, "correct_output_scale")?;
        let mut correction_coefs = layers::linear_no_bias(
            cfg.altup_num_inputs,
            cfg.altup_num_inputs,
            vb.pp("correction_coefs"),
        )?;
        if let Some(altup_coef_clip) = cfg.altup_coef_clip {
            correction_coefs = Linear::new(
                correction_coefs
                    .weight()
                    .clamp(-altup_coef_clip, altup_coef_clip)?,
                None,
            );
        }
        let prediction_coefs = layers::linear_no_bias(
            cfg.altup_num_inputs,
            cfg.altup_num_inputs.pow(2),
            vb.pp("prediction_coefs"),
        )?;
        let modality_router = layers::linear_no_bias(
            cfg.hidden_size,
            cfg.altup_num_inputs,
            vb.pp("modality_router"),
        )?;
        let router_norm = RmsNorm::new_gemma_3n(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            true,
            vb.pp("router_norm"),
        )?;

        Ok(Self {
            correct_output_scale,
            correction_coefs,
            prediction_coefs,
            modality_router,
            router_norm,
            router_input_scale: 1. / (cfg.hidden_size as f64),
            altup_active_idx: cfg.altup_active_idx,
            altup_num_inputs: cfg.altup_num_inputs,
        })
    }

    fn compute_router_modalities(&self, xs: &Tensor) -> Result<Tensor> {
        // Perform routing calculations in float32 for better precision
        let router_inputs_normed = self.router_norm.forward(xs)?;
        let router_inputs_f32 = router_inputs_normed.to_dtype(DType::F32)?;
        let router_inputs = (router_inputs_f32 * self.router_input_scale)?.to_dtype(xs.dtype())?;
        let routed = self.modality_router.forward(&router_inputs)?;
        routed.to_dtype(DType::F32)?.tanh()?.to_dtype(xs.dtype())
    }

    fn predict(&self, xs: &Tensor) -> Result<Tensor> {
        let modalities = self.compute_router_modalities(&xs.i(self.altup_active_idx)?)?;

        // Project and then transpose all 2D matrices contained so that mulmat gives the correct result
        let shape = [
            modalities.dims()[..modalities.dims().len() - 1].to_vec(),
            vec![self.altup_num_inputs, self.altup_num_inputs],
        ]
        .concat();
        let all_coefs = self
            .prediction_coefs
            .forward(&modalities)?
            .reshape(shape)?
            .permute((0, 1, 3, 2))?;

        // permute hidden_states to [batch_size, num_tokens, hidden_size, altup_num_inputs]
        let mut predictions = xs
            .permute((1, 2, 3, 0))?
            .contiguous()?
            .matmul(&all_coefs.contiguous()?)?;
        predictions = predictions.permute((3, 0, 1, 2))?;
        // Perform addition in float32 for precision
        let predictions_f32 = predictions.to_dtype(DType::F32)?;
        let xs_f32 = xs.to_dtype(DType::F32)?;
        predictions = (predictions_f32 + xs_f32)?.to_dtype(predictions.dtype())?;
        predictions.contiguous()
    }

    fn correct(&self, predictions: &Tensor, activated: &Tensor) -> Result<Tensor> {
        let modalities = self.compute_router_modalities(activated)?;
        // Compute innovation in float32 for better precision
        let activated_f32 = activated.to_dtype(DType::F32)?;
        let pred_active_f32 = predictions.i(self.altup_active_idx)?.to_dtype(DType::F32)?;
        let innovation_f32 = activated_f32.broadcast_sub(&pred_active_f32)?;
        let innovation =
            innovation_f32
                .to_dtype(activated.dtype())?
                .repeat((self.altup_num_inputs, 1, 1, 1))?;

        // Perform coefficient computation in float32
        let coefs = self.correction_coefs.forward(&modalities)?;
        let coefs_f32 = coefs.to_dtype(DType::F32)?;
        let all_coefs = (coefs_f32 + 1.)?
            .to_dtype(coefs.dtype())?
            .permute((2, 0, 1))?
            .unsqueeze(D::Minus1)?;

        innovation
            .broadcast_mul(&all_coefs)?
            .broadcast_add(predictions)?
            .contiguous()
    }

    fn scale_corrected_output(&self, xs: &Tensor) -> Result<Tensor> {
        xs.to_dtype(self.correct_output_scale.dtype())?
            .broadcast_mul(&self.correct_output_scale)?
            .to_dtype(xs.dtype())
    }
}

struct TextLaurelBlock {
    left: Linear,
    right: Linear,
    post_norm: RmsNorm,
}

impl TextLaurelBlock {
    fn new(cfg: &Gemma3nTextConfig, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            left: layers::linear_no_bias(cfg.hidden_size, cfg.laurel_rank, vb.pp("linear_left"))?,
            right: layers::linear_no_bias(cfg.laurel_rank, cfg.hidden_size, vb.pp("linear_right"))?,
            post_norm: RmsNorm::new_gemma_3n(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                true,
                vb.pp("post_laurel_norm"),
            )?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut laurel_xs = self.left.forward(xs)?;
        laurel_xs = self.right.forward(&laurel_xs)?;
        laurel_xs = self.post_norm.forward(&laurel_xs)?;
        // Perform addition in float32 for precision
        let xs_f32 = xs.to_dtype(DType::F32)?;
        let laurel_xs_f32 = laurel_xs.to_dtype(DType::F32)?;
        (xs_f32 + laurel_xs_f32)?.to_dtype(xs.dtype())
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    altup: TextAltUp,
    laurel: TextLaurelBlock,
    per_layer_input_gate: Linear,
    per_layer_projection: Linear,
    post_per_layer_input_norm: RmsNorm,
    altup_active_idx: usize,
    altup_correct_scale: bool,
    act: Activation,
}

impl DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb_global: Arc<Gemma3nRotaryEmbedding>,
        rotary_emb_local: Arc<RotaryEmbedding>,
        cfg: &Gemma3nTextConfig,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let self_attn = Attention::new(
            rotary_emb_global,
            rotary_emb_local,
            cfg,
            layer_idx,
            mapper,
            mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq),
            paged_attn,
            comm,
        )?;
        let mlp = Mlp::new(
            mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq),
            cfg,
            comm,
            layer_idx,
        )?;
        let input_layernorm = RmsNorm::new_gemma_3n(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            true,
            mapper.set_device(layer_idx, vb.pp("input_layernorm"), false),
        )?;
        let post_attention_layernorm = RmsNorm::new_gemma_3n(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            true,
            mapper.set_device(layer_idx, vb.pp("post_attention_layernorm"), false),
        )?;
        let pre_feedforward_layernorm = RmsNorm::new_gemma_3n(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            true,
            mapper.set_device(layer_idx, vb.pp("pre_feedforward_layernorm"), false),
        )?;
        let post_feedforward_layernorm = RmsNorm::new_gemma_3n(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            true,
            mapper.set_device(layer_idx, vb.pp("post_feedforward_layernorm"), false),
        )?;

        let altup = TextAltUp::new(cfg, mapper.set_device(layer_idx, vb.pp("altup"), false))?;
        let laurel =
            TextLaurelBlock::new(cfg, mapper.set_device(layer_idx, vb.pp("laurel"), false))?;
        let per_layer_input_gate = layers::linear_no_bias(
            cfg.hidden_size,
            cfg.hidden_size_per_layer_input,
            mapper.set_device(layer_idx, vb.pp("per_layer_input_gate"), false),
        )?;
        let per_layer_projection = layers::linear_no_bias(
            cfg.hidden_size_per_layer_input,
            cfg.hidden_size,
            mapper.set_device(layer_idx, vb.pp("per_layer_projection"), false),
        )?;
        let post_per_layer_input_norm = RmsNorm::new_gemma_3n(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            true,
            mapper.set_device(layer_idx, vb.pp("post_per_layer_input_norm"), false),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            altup,
            laurel,
            per_layer_input_gate,
            per_layer_projection,
            post_per_layer_input_norm,
            altup_active_idx: cfg.altup_active_idx,
            altup_correct_scale: cfg.altup_correct_scale,
            act: cfg.hidden_activation,
        })
    }

    fn has_active_dynamic_lora(&self) -> bool {
        [
            &self.self_attn.q_proj,
            &self.self_attn.k_proj,
            &self.self_attn.v_proj,
            &self.self_attn.o_proj,
            &self.mlp.gate,
            &self.mlp.up,
            &self.mlp.down,
        ]
        .into_iter()
        .any(|layer| layer.is_dynamic_lora_active())
    }

    fn forward(
        &self,
        xs: &Tensor,
        per_layer_input: &Tensor,
        ctx: &mut AttentionForwardContext<'_>,
    ) -> Result<Tensor> {
        let predictions = self.altup.predict(xs)?;
        let active_prediction = predictions.i(self.altup_active_idx)?;

        let active_prediction_normed = self.input_layernorm.forward(&active_prediction)?;
        let laurel_output = self.laurel.forward(&active_prediction_normed)?;

        let attn = self
            .self_attn
            .forward(&active_prediction_normed, ctx)?
            .apply(&self.post_attention_layernorm)?;

        // Perform addition and scaling in float32 for precision
        let active_pred_f32 = active_prediction.to_dtype(DType::F32)?;
        let attn_f32 = attn.to_dtype(DType::F32)?;
        let attn_gated_f32 = (active_pred_f32 + attn_f32)?;
        let attn_gated = attn_gated_f32.to_dtype(active_prediction.dtype())?;
        let attn_laurel_f32 = (attn_gated_f32 + laurel_output.to_dtype(DType::F32)?)?;
        let attn_laurel = (attn_laurel_f32 / 2f64.sqrt())?.to_dtype(attn_gated.dtype())?;

        let attn_norm = self.pre_feedforward_layernorm.forward(&attn_laurel)?;
        let attn_ffw = self.mlp.forward(&attn_norm)?;
        let attn_ffw_norm = self.post_feedforward_layernorm.forward(&attn_ffw)?;
        // Perform addition in float32 for precision
        let attn_laurel_f32 = attn_laurel.to_dtype(DType::F32)?;
        let attn_ffw_norm_f32 = attn_ffw_norm.to_dtype(DType::F32)?;
        let attn_ffw_laurel_gated =
            (attn_laurel_f32 + attn_ffw_norm_f32)?.to_dtype(attn_laurel.dtype())?;
        let mut corrected_predictions = self.altup.correct(&predictions, &attn_ffw_laurel_gated)?;

        let mut first_prediction = corrected_predictions.i(self.altup_active_idx)?;
        if self.altup_correct_scale {
            first_prediction = self.altup.scale_corrected_output(&first_prediction)?;
        }
        first_prediction = self.per_layer_input_gate.forward(&first_prediction)?;
        first_prediction = self.act.forward(&first_prediction)?;
        // Perform multiplication in float32
        let first_pred_f32 = first_prediction.to_dtype(DType::F32)?;
        let per_layer_input_f32 = per_layer_input.to_dtype(DType::F32)?;
        first_prediction =
            (first_pred_f32 * per_layer_input_f32)?.to_dtype(first_prediction.dtype())?;

        first_prediction = self.per_layer_projection.forward(&first_prediction)?;
        first_prediction = self.post_per_layer_input_norm.forward(&first_prediction)?;

        // Perform broadcast add in float32 for precision
        let slice_f32 = corrected_predictions
            .i((1.., .., .., ..))?
            .to_dtype(DType::F32)?;
        let first_pred_f32 = first_prediction.to_dtype(DType::F32)?;
        let added = slice_f32
            .broadcast_add(&first_pred_f32)?
            .to_dtype(corrected_predictions.dtype())?;
        let cp_dim0 = corrected_predictions.dim(0)?;
        let cp_dim1 = corrected_predictions.dim(1)?;
        let cp_dim2 = corrected_predictions.dim(2)?;
        let cp_dim3 = corrected_predictions.dim(3)?;
        corrected_predictions = corrected_predictions
            .slice_assign(&[1..cp_dim0, 0..cp_dim1, 0..cp_dim2, 0..cp_dim3], &added)?;

        Ok(corrected_predictions)
    }
}

type MatformerSliceResult = (
    Gemma3nTextConfig,
    Option<Tensor>,
    usize,
    Option<HashMap<usize, usize>>,
    Option<Vec<usize>>,
);

pub(crate) fn handle_matformer_slicing(
    cfg: &Gemma3nTextConfig,
    matformer_slicing_config: &Option<MatformerSliceConfig>,
    mapper: &(dyn DeviceMapper + Send + Sync),
) -> Result<MatformerSliceResult> {
    match matformer_slicing_config {
        Some(slicing_config) => {
            let matformer_slice = slicing_config.get_slicing().ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "Matformer slice '{}' not found in config",
                    slicing_config.slice_name
                ))
            })?;
            let mut cfg = cfg.clone();

            let layers_skipped = matformer_slice.layers_skipped.clone().unwrap_or_default();

            let first_kv_shared_layer_idx = cfg.num_hidden_layers - cfg.num_kv_shared_layers;
            let local_kv_sharing_layer_idx = first_kv_shared_layer_idx - 2;
            let global_kv_sharing_layer_idx = first_kv_shared_layer_idx - 1;

            if layers_skipped.contains(&local_kv_sharing_layer_idx)
                || layers_skipped.contains(&global_kv_sharing_layer_idx)
            {
                candle_core::bail!(
                    "Layers {} and {} are reserved.",
                    local_kv_sharing_layer_idx,
                    global_kv_sharing_layer_idx
                );
            }

            let count_kv_sharing = layers_skipped.iter().filter(|x| **x >= 20).count();
            cfg.num_kv_shared_layers -= count_kv_sharing;

            let count_activation_sparsity = layers_skipped.iter().filter(|x| **x <= 9).count();
            let final_num_layers = cfg.num_hidden_layers - layers_skipped.len();

            let kept_layers_indices = (0..cfg.num_hidden_layers)
                .filter_map(|idx| {
                    if !layers_skipped.contains(&idx) {
                        Some(idx as u32)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();

            cfg.layer_types = cfg
                .layer_types
                .iter()
                .enumerate()
                .filter_map(|(idx, layer_type)| {
                    if !layers_skipped.contains(&idx) {
                        Some(layer_type.clone())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();

            let layer_rename_map = kept_layers_indices
                .iter()
                .enumerate()
                .map(|(new_idx, old_idx)| (*old_idx as usize, new_idx))
                .collect::<HashMap<_, _>>();

            let kept_layers_indices_tensor =
                mapper.cast_nm_device(&Tensor::new(kept_layers_indices, &Device::Cpu)?, false)?;

            let orig_num_hidden_layers = cfg.num_hidden_layers;

            let activation_sparsity_list = [
                vec![0.95f64; 10 - count_activation_sparsity],
                vec![0f64; final_num_layers - 10 + count_activation_sparsity],
            ]
            .concat();
            cfg.activation_sparsity_pattern = activation_sparsity_list;

            cfg.num_hidden_layers = final_num_layers;
            let orig_intermediate_size = match &cfg.intermediate_size {
                IntermediateSize::Single(size) => vec![*size; orig_num_hidden_layers],
                IntermediateSize::PerLayer(sizes) => sizes.clone(),
                IntermediateSize::Matformer(_, orig) => orig.clone(),
            };
            cfg.intermediate_size = IntermediateSize::Matformer(
                matformer_slice.ffn_hidden_dimensions.clone(),
                orig_intermediate_size,
            );

            Ok((
                cfg,
                Some(kept_layers_indices_tensor),
                orig_num_hidden_layers,
                Some(layer_rename_map),
                Some(layers_skipped),
            ))
        }
        None => Ok((cfg.clone(), None, cfg.num_hidden_layers, None, None)),
    }
}

pub struct TextModel {
    embed_tokens: Arc<dyn QuantMethod>,
    embed_tokens_scale: f64,
    dtype: DType,
    embed_tokens_per_layer: ScaledEmbedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Arc<dyn QuantMethod>,
    device: Device,
    cache: EitherCache,
    max_seq_len: usize,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    sliding_window: usize,
    cfg: ModelConfigMetadata,
    model_config: Arc<dyn ModelConfigLike + Send + Sync>,
    per_layer_projection_scale: f64,
    per_layer_input_scale: f64,
    altup_projections: Vec<Arc<dyn QuantMethod>>,
    altup_unembed_projections: Vec<Arc<dyn QuantMethod>>,
    per_layer_model_projection: Arc<dyn QuantMethod>,
    per_layer_projection_norm: RmsNorm,
    hidden_size_per_layer_input: usize,
    final_logit_softcapping: Option<f64>,
}

#[derive(Clone)]
struct PrefillQuerySelection {
    source_context_lens: Vec<(usize, usize)>,
    seqlen_offsets: Vec<usize>,
    num_cached_tokens: Vec<usize>,
    query_lens: Vec<usize>,
}

impl PrefillQuerySelection {
    fn from_logits_context(
        query_len: usize,
        context_lens: &[(usize, usize)],
        seqlen_offsets: &[usize],
    ) -> Option<Self> {
        if context_lens.is_empty() || context_lens.len() != seqlen_offsets.len() {
            return None;
        }

        let mut tail_offsets = Vec::with_capacity(context_lens.len());
        let mut num_cached_tokens = Vec::with_capacity(context_lens.len());
        let mut query_lens = Vec::with_capacity(context_lens.len());
        for ((start, len), offset) in context_lens.iter().zip(seqlen_offsets) {
            if *len != 1 || start.checked_add(*len)? != query_len {
                return None;
            }
            tail_offsets.push(offset + start);
            num_cached_tokens.push(offset + start);
            query_lens.push(*len);
        }

        Some(Self {
            source_context_lens: context_lens.to_vec(),
            seqlen_offsets: tail_offsets,
            num_cached_tokens,
            query_lens,
        })
    }

    fn reduce(&self, tensor: &Tensor) -> Result<Tensor> {
        extract_logits(tensor, self.source_context_lens.clone())
    }

    fn reduce_altup(&self, tensor: &Tensor) -> Result<Tensor> {
        let streams = tensor.dim(0)?;
        let mut reduced = Vec::with_capacity(streams);
        for stream in 0..streams {
            reduced.push(self.reduce(&tensor.i(stream)?)?);
        }
        Tensor::stack(&reduced, 0)
    }
}

struct KvSharingFastPrefillPlan {
    first_shared_layer: usize,
    query_selection: PrefillQuerySelection,
    paged_metadata: Option<PagedAttentionInputMetadata>,
}

impl TextModel {
    pub fn new(
        cfg: &Gemma3nTextConfig,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        if let Some(ref quant_cfg) = &cfg.quantization_config {
            tracing::info!(
                "Using {} quantization: {}.",
                quant_cfg.name(),
                quant_cfg.get_bits_name(&vb)
            );
        }
        let mapper = normal_loading_metadata.mapper;

        // Implement the matformer slicing.
        // https://colab.research.google.com/github/google-gemini/gemma-cookbook/blob/main/Gemma/[Gemma_3n]MatFormer_Lab.ipynb
        let (cfg, kept_layers_indices, orig_num_hidden_layers, layer_rename_map, layers_skipped) =
            handle_matformer_slicing(
                cfg,
                &normal_loading_metadata.matformer_slicing_config,
                &*mapper,
            )?;
        let cfg = &cfg;

        // Use float32 for embedding scale factor
        let embed_scale = (cfg.hidden_size as f64).sqrt();
        let dtype = vb.dtype();
        let embed_tokens = embedding_with_legacy_tied_uqff(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb.pp("embed_tokens"), normal_loading_metadata.loading_isq),
            cfg.tie_word_embeddings.then(|| {
                mapper.set_nm_device(vb.pp("lm_head"), normal_loading_metadata.loading_isq)
            }),
            &cfg.quantization_config,
        )?;
        // Use float32 for per-layer embedding scale factor
        let per_layer_embed_scale = (cfg.hidden_size_per_layer_input as f64).sqrt();
        // Keep embed_tokens_per_layer on CPU if not using Metal
        let embed_tokens_per_layer_vb = if normal_loading_metadata.real_device.is_metal() {
            mapper.set_nm_device(vb.pp("embed_tokens_per_layer"), false)
        } else {
            vb.pp("embed_tokens_per_layer").set_device(Device::Cpu)
        };
        let mut embed_tokens_per_layer = ScaledEmbedding::new(
            per_layer_embed_scale,
            dense_embedding(
                cfg.vocab_size_per_layer_input,
                cfg.hidden_size_per_layer_input * orig_num_hidden_layers,
                embed_tokens_per_layer_vb,
                &cfg.quantization_config,
            )?,
        );
        if let Some(kept_layers_indices) = &kept_layers_indices {
            let embedding = embed_tokens_per_layer.embedding.clone();
            let embedding_reshaped = embedding.reshape((
                embedding.dim(0)?,
                orig_num_hidden_layers,
                embedding.dim(1)? / orig_num_hidden_layers,
            ))?;

            embed_tokens_per_layer.embedding = embedding_reshaped
                .index_select(kept_layers_indices, 1)?
                .reshape((embedding_reshaped.dim(0)?, ()))?
                .contiguous()?;
        }

        let mut global_ropes = HashMap::new();
        for layer_idx in 0..orig_num_hidden_layers {
            let layer_idx = if let Some(layer_rename_map) = &layer_rename_map {
                if layers_skipped.as_ref().unwrap().contains(&layer_idx) {
                    continue;
                }
                layer_rename_map[&layer_idx]
            } else {
                layer_idx
            };
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            global_ropes.insert(
                device.location(),
                Arc::new(Gemma3nRotaryEmbedding::new(
                    is_gptx,
                    vb.dtype(),
                    cfg,
                    device,
                )?),
            );
        }

        let mut local_ropes = HashMap::new();
        for layer_idx in 0..orig_num_hidden_layers {
            let layer_idx = if let Some(layer_rename_map) = &layer_rename_map {
                if layers_skipped.as_ref().unwrap().contains(&layer_idx) {
                    continue;
                }
                layer_rename_map[&layer_idx]
            } else {
                layer_idx
            };
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            local_ropes.insert(
                device.location(),
                Arc::new(RotaryEmbedding::new(
                    cfg.rope_local_base_freq as f32,
                    cfg.head_dim,
                    cfg.max_position_embeddings,
                    device,
                    is_gptx,
                    vb.dtype(),
                )?),
            );
        }

        let vb_l = vb.pp("layers");

        let layers = NiceProgressBar::<_, 'b'>(
            0..orig_num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .into_iter()
        .filter(|layer_idx| {
            !layers_skipped
                .as_ref()
                .is_some_and(|layers_skipped| layers_skipped.contains(layer_idx))
        })
        .map(|layer_idx| {
            let layer_idx_effective = if let Some(layer_rename_map) = &layer_rename_map {
                layer_rename_map[&layer_idx]
            } else {
                layer_idx
            };
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let rotary_emb_global = global_ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();
            let rotary_emb_local = local_ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => {
                    Some(PagedAttention::new(cfg.head_dim, device, None)?)
                }
            };
            let comm = mapper.get_comm_for(layer_idx)?;
            DecoderLayer::new(
                rotary_emb_global,
                rotary_emb_local,
                cfg,
                vb_l.pp(layer_idx),
                &*mapper,
                layer_idx_effective,
                normal_loading_metadata.loading_isq,
                paged_attn,
                &comm,
            )
        })
        .collect::<Result<Vec<_>>>()?;

        let norm = RmsNorm::new_gemma_3n(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            true,
            mapper.set_nm_device(vb.pp("norm"), false),
        )?;

        let lm_head = if !cfg.tie_word_embeddings {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &cfg.quantization_config,
                false,
                mapper.set_nm_device(vb.pp("lm_head"), normal_loading_metadata.loading_isq),
            )?
        } else {
            embed_tokens.clone()
        };

        let per_layer_model_projection = ReplicatedLayer::new_layers_matformer_indices(
            cfg.hidden_size,
            orig_num_hidden_layers * cfg.hidden_size_per_layer_input,
            kept_layers_indices.as_ref(),
            orig_num_hidden_layers,
            &cfg.quantization_config,
            false,
            mapper.set_nm_device(
                vb.pp("per_layer_model_projection"),
                normal_loading_metadata.loading_isq,
            ),
        )?;
        let per_layer_projection_norm = RmsNorm::new_gemma_3n(
            cfg.hidden_size_per_layer_input,
            cfg.rms_norm_eps,
            true,
            mapper.set_nm_device(vb.pp("per_layer_projection_norm"), false),
        )?;

        let mut altup_projections = Vec::new();
        let mut altup_unembed_projections = Vec::new();
        for i in 1..cfg.altup_num_inputs {
            altup_projections.push(ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.hidden_size,
                &cfg.quantization_config,
                false,
                mapper.set_nm_device(
                    vb.pp("altup_projections").pp(i - 1),
                    normal_loading_metadata.loading_isq,
                ),
            )?);
            altup_unembed_projections.push(ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.hidden_size,
                &cfg.quantization_config,
                false,
                mapper.set_nm_device(
                    vb.pp("altup_unembed_projections").pp(i - 1),
                    normal_loading_metadata.loading_isq,
                ),
            )?);
        }

        let mut kv_cache_layer_owners = Vec::with_capacity(cfg.num_hidden_layers);
        let cache_types = (0..cfg.num_hidden_layers)
            .map(|layer_idx| {
                if let Some(owner) = kv_shared_layer_index(cfg, layer_idx)? {
                    kv_cache_layer_owners.push(owner);
                    Ok(NormalCacheType::Shared { owner })
                } else if is_sliding!(layer_idx, cfg) {
                    kv_cache_layer_owners.push(layer_idx);
                    Ok(NormalCacheType::SlidingWindow {
                        window: cfg.sliding_window,
                    })
                } else {
                    kv_cache_layer_owners.push(layer_idx);
                    Ok(NormalCacheType::Normal {
                        max_seq_len: cfg.max_position_embeddings,
                    })
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let cfg_metadata = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_attn_heads: cfg.num_attention_heads / mapper.get_comm_for(0)?.world_size(),
            num_kv_heads: (cfg.num_key_value_heads / mapper.get_comm_for(0)?.world_size()).max(1),
            sliding_window: Some(cfg.sliding_window),
            k_head_dim: cfg.head_dim,
            v_head_dim: cfg.head_dim,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };
        let model_config = Arc::new(Gemma3nModelConfigLike {
            base: cfg_metadata.clone(),
            kv_cache_topology: KvCacheTopology::from_layer_owners(kv_cache_layer_owners),
        });

        Ok(Self {
            embed_tokens,
            embed_tokens_scale: embed_scale,
            dtype,
            embed_tokens_per_layer,
            layers,
            norm,
            lm_head,
            device: normal_loading_metadata.real_device,
            cache: EitherCache::Normal(NormalCache::from_types(cache_types)),
            max_seq_len: cfg.max_position_embeddings,
            sliding_window: cfg.sliding_window,
            cfg: cfg_metadata,
            model_config,
            mapper,
            per_layer_input_scale: 1. / (2f64.sqrt()),
            // Keep scale factors in float64 for maximum precision
            per_layer_projection_scale: 1. / (cfg.hidden_size as f64).sqrt(),
            altup_projections,
            altup_unembed_projections,
            per_layer_model_projection,
            per_layer_projection_norm,
            hidden_size_per_layer_input: cfg.hidden_size_per_layer_input,
            final_logit_softcapping: cfg.final_logit_softcapping,
        })
    }

    pub fn model_config_like(&self) -> Arc<dyn ModelConfigLike + Send + Sync> {
        self.model_config.clone()
    }

    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.embedding_forward(input_ids, self.dtype)? * self.embed_tokens_scale
    }

    fn get_per_layer_inputs(&self, input_ids: &Tensor) -> Result<Tensor> {
        let shape = [
            input_ids.dims().to_vec(),
            vec![self.layers.len(), self.hidden_size_per_layer_input],
        ]
        .concat();
        // Cast to float32 for better precision
        let input_ids_device = input_ids.device();

        // Only cast to CPU if not using Metal
        let per_layer_embeds = if input_ids_device.is_metal() {
            // On Metal, use input_ids directly
            self.embed_tokens_per_layer.forward(input_ids)?
        } else {
            // On non-Metal devices, cast input to CPU for embedding computation
            let input_ids_cpu = input_ids.to_device(&Device::Cpu)?;
            self.embed_tokens_per_layer.forward(&input_ids_cpu)?
        };

        let original_dtype = per_layer_embeds.dtype();
        // Move result back to original device after computation (if needed)
        let result = per_layer_embeds
            .to_dtype(DType::F32)?
            .reshape(shape)?
            .to_dtype(original_dtype)?;

        // Only move back to device if we computed on CPU
        if !input_ids_device.is_metal() && input_ids_device.location() != Device::Cpu.location() {
            result.to_device(input_ids_device)
        } else {
            Ok(result)
        }
    }
    fn project_per_layer_inputs(
        &self,
        xs: &Tensor,
        per_layer_inputs: Option<Tensor>,
    ) -> Result<Tensor> {
        // Cast to float32 for per-layer projection scaling
        let mut per_layer_projection = self.per_layer_model_projection.forward(xs)?;
        let original_dtype = per_layer_projection.dtype();
        let per_layer_projection_f32 = per_layer_projection.to_dtype(DType::F32)?;
        per_layer_projection = (per_layer_projection_f32 * self.per_layer_projection_scale)?;
        let shape = [
            xs.dims()[..xs.dims().len() - 1].to_vec(),
            vec![self.layers.len(), self.hidden_size_per_layer_input],
        ]
        .concat();
        per_layer_projection = per_layer_projection
            .reshape(shape)?
            .to_dtype(original_dtype)?; // Convert back to model dtype before RMSNorm
        per_layer_projection = self
            .per_layer_projection_norm
            .forward(&per_layer_projection)?;

        let Some(mut per_layer_inputs) = per_layer_inputs else {
            return Ok(per_layer_projection.clone());
        };

        if per_layer_projection.shape() != per_layer_inputs.shape() {
            // per-layer inputs are sometimes padded with zeros, slice the relevant embeddings.
            per_layer_inputs = per_layer_inputs.narrow(D::Minus2, 0, self.layers.len())?;
        }
        // Perform addition and scaling in float32 for better precision
        let per_layer_projection_f32 = per_layer_projection.to_dtype(DType::F32)?;
        let per_layer_inputs_f32 = per_layer_inputs.to_dtype(DType::F32)?;
        let result_f32 = (per_layer_projection_f32 + per_layer_inputs_f32)?;
        (result_f32 * self.per_layer_input_scale)?.to_dtype(per_layer_projection.dtype())
    }

    fn kv_sharing_fast_prefill_plan(
        &self,
        input_ids: &Tensor,
        ctx: &ModelForwardContext<'_>,
    ) -> Result<Option<KvSharingFastPrefillPlan>> {
        if ctx.requires_full_prefill_queries() {
            return Ok(None);
        }
        let (batch, query_len) = input_ids.dims2()?;
        if query_len <= 1 || ctx.context_lens().len() != batch {
            return Ok(None);
        }
        let Some(first_shared_layer) = self
            .layers
            .iter()
            .position(|layer| layer.self_attn.kv_shared_layer_index.is_some())
        else {
            return Ok(None);
        };
        if !self.layers[first_shared_layer..]
            .iter()
            .all(|layer| layer.self_attn.kv_shared_layer_index.is_some())
            || self.layers[first_shared_layer..]
                .iter()
                .any(DecoderLayer::has_active_dynamic_lora)
        {
            return Ok(None);
        }
        let Some(query_selection) = PrefillQuerySelection::from_logits_context(
            query_len,
            ctx.context_lens(),
            ctx.seqlen_offsets(),
        ) else {
            return Ok(None);
        };

        let paged_metadata = if let Some(metadata) = ctx.paged_input_metadata() {
            if metadata.block_tables.is_none() {
                return Ok(None);
            }
            Some(
                metadata
                    .for_reduced_prefill_queries(
                        &self.mapper.get_unique_devices(),
                        &query_selection.num_cached_tokens,
                        &query_selection.query_lens,
                    )
                    .map_err(|error| candle_core::Error::Msg(error.to_string()))?,
            )
        } else {
            None
        };
        Ok(Some(KvSharingFastPrefillPlan {
            first_shared_layer,
            query_selection,
            paged_metadata,
        }))
    }

    pub fn forward_embeds(
        &self,
        input_ids: &Tensor,
        ple_input_ids: &Tensor,
        mut xs: Tensor,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let flash_params = ctx.flash_params().clone();
        let per_layer_inputs = Some(self.get_per_layer_inputs(ple_input_ids)?);
        let per_layer_inputs = self.project_per_layer_inputs(&xs, per_layer_inputs)?;
        // Cast per_layer_inputs back to model dtype after float32 operations
        let per_layer_inputs = per_layer_inputs.to_dtype(xs.dtype())?;

        let cache = &mut self.cache.normal().0;
        let mask_cache = ctx.mask_cache(cache);
        let mut attention_mask = CausalMasker.make_causal_mask(
            input_ids,
            &mask_cache,
            xs.dtype(),
            &CausalMaskConfig::default(),
        )?;
        let mut sliding_attention_mask = CausalMasker.make_causal_mask(
            input_ids,
            &mask_cache,
            xs.dtype(),
            &CausalMaskConfig {
                sliding_window: Some(self.sliding_window),
                ..Default::default()
            },
        )?;
        let is_paged_decode = is_paged_decode_forward(
            ctx.is_paged(),
            input_ids.dim(1)?,
            ctx.is_first_prompt_chunk(),
            ctx.paged_input_metadata()
                .is_some_and(|metadata| metadata.num_cached_tokens.is_some()),
        );
        if is_paged_decode {
            attention_mask = AttentionMask::None;
            sliding_attention_mask = AttentionMask::None;
        }

        // Already using float32 for magnitude calculations
        let target_magnitude = xs
            .to_dtype(DType::F32)?
            .sqr()?
            .mean_keepdim(D::Minus1)?
            .sqrt()?;
        let eps = Tensor::new(&[EPS as f32], target_magnitude.device())?;

        let mut temp_hidden_states = vec![xs.clone()];
        for altup_proj in &self.altup_projections {
            let altup_proj = altup_proj.forward(&xs)?;
            let new_magnitude = altup_proj
                .to_dtype(DType::F32)?
                .sqr()?
                .mean_keepdim(D::Minus1)?
                .sqrt()?;

            let current_hidden_state = altup_proj
                .to_dtype(DType::F32)?
                .broadcast_mul(
                    &target_magnitude.broadcast_div(&new_magnitude.broadcast_maximum(&eps)?)?,
                )?
                .to_dtype(altup_proj.dtype())?;
            temp_hidden_states.push(current_hidden_state);
        }
        xs = Tensor::stack(&temp_hidden_states, 0)?;

        let attention_mask = DeviceMappedMask::new(attention_mask, &*self.mapper)?;
        let sliding_attention_mask = DeviceMappedMask::new(sliding_attention_mask, &*self.mapper)?;
        let fast_prefill_tail = self.kv_sharing_fast_prefill_plan(input_ids, ctx)?;
        let mut reduced_to_logits = false;
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(plan) = fast_prefill_tail
                .as_ref()
                .filter(|plan| i == plan.first_shared_layer)
            {
                xs = plan.query_selection.reduce_altup(&xs)?;
                reduced_to_logits = true;
            }

            let per_layer_input = per_layer_inputs.i((.., .., i, ..))?;
            let per_layer_input = if reduced_to_logits {
                fast_prefill_tail
                    .as_ref()
                    .expect("missing active fast prefill plan")
                    .query_selection
                    .reduce(&per_layer_input)?
            } else {
                per_layer_input
            };
            xs = self.mapper.map(xs, i)?;
            let rope_positions = if reduced_to_logits {
                let plan = fast_prefill_tail
                    .as_ref()
                    .expect("missing active fast prefill plan");
                if let Some(metadata) = plan.paged_metadata.as_ref() {
                    crate::pipeline::metadata_rope_positions(metadata, xs.device())
                        .ok_or_else(|| candle_core::Error::msg("missing RoPE positions"))?
                        .clone()
                } else {
                    ctx.text_positions_from_offsets(
                        &plan.query_selection.seqlen_offsets,
                        xs.dim(2)?,
                        xs.device(),
                    )?
                }
            } else {
                ctx.text_positions(xs.device(), xs.dim(2)?)?
                    .ok_or_else(|| candle_core::Error::msg("missing RoPE positions"))?
                    .clone()
            };
            let (layer_attention_mask, layer_sliding_attention_mask) = if reduced_to_logits {
                (AttentionMask::None, AttentionMask::None)
            } else {
                (
                    attention_mask.get(xs.device()),
                    sliding_attention_mask.get(xs.device()),
                )
            };
            let cache_idx = layer.self_attn.kv_shared_layer_index.unwrap_or(i);
            let metadata = ctx.paged_layer(cache_idx).map(|(kv_cache, metadata)| {
                let metadata = if reduced_to_logits {
                    fast_prefill_tail
                        .as_ref()
                        .and_then(|plan| plan.paged_metadata.as_ref())
                        .unwrap_or(metadata)
                } else {
                    metadata
                };
                (kv_cache, metadata)
            });
            let mut layer_ctx = AttentionForwardContext {
                attention_mask: &layer_attention_mask,
                sliding_attention_mask: &layer_sliding_attention_mask,
                rope_positions: &rope_positions,
                kv_caches: &mut *cache,
                metadata,
                flash_params: (!reduced_to_logits).then_some(&flash_params),
            };
            xs = layer.forward(
                &xs,
                &per_layer_input.to_device(xs.device())?,
                &mut layer_ctx,
            )?;
        }
        xs = xs.to_device(&self.device)?;

        let target_magnitude = xs
            .i(0)?
            .to_dtype(DType::F32)?
            .sqr()?
            .mean_keepdim(D::Minus1)?
            .sqrt()?;

        let mut temp_hidden_states = vec![xs.i(0)?];
        for (i, altup_proj) in self.altup_unembed_projections.iter().enumerate() {
            let altup_proj = altup_proj.forward(&xs.i(i + 1)?)?;
            let new_magnitude = altup_proj
                .to_dtype(DType::F32)?
                .sqr()?
                .mean_keepdim(D::Minus1)?
                .sqrt()?;

            let current_hidden_state = altup_proj
                .to_dtype(DType::F32)?
                .broadcast_mul(
                    &target_magnitude.broadcast_div(&new_magnitude.broadcast_maximum(&eps)?)?,
                )?
                .to_dtype(altup_proj.dtype())?;
            temp_hidden_states.push(current_hidden_state);
        }
        // Perform mean operation in float32 for better precision
        let stacked = Tensor::stack(&temp_hidden_states, 0)?;
        let stacked_f32 = stacked.to_dtype(DType::F32)?;
        xs = stacked_f32.mean(0)?.to_dtype(stacked.dtype())?;

        xs = xs.apply(&self.norm)?;
        let mut xs = if reduced_to_logits {
            xs
        } else {
            ctx.logits(&xs)?
        };
        xs = self.lm_head.forward(&xs)?;

        if let Some(final_logit_softcapping) = self.final_logit_softcapping {
            let dtype = xs.dtype();
            xs = softcap(&xs, final_logit_softcapping as f32)?.to_dtype(dtype)?;
        }

        Ok(xs)
    }
}

impl IsqModel for TextModel {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        // Embeddings
        uvb.pp("embed_tokens").add(&self.embed_tokens);
        uvb.pp("embed_tokens_per_layer")
            .add(&self.embed_tokens_per_layer);

        // Layer components
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let layer_uvb = uvb.pp(format!("layers.{layer_idx}"));

            // Layer norms
            layer_uvb.pp("input_layernorm").add(&layer.input_layernorm);
            layer_uvb
                .pp("post_attention_layernorm")
                .add(&layer.post_attention_layernorm);
            layer_uvb
                .pp("pre_feedforward_layernorm")
                .add(&layer.pre_feedforward_layernorm);
            layer_uvb
                .pp("post_feedforward_layernorm")
                .add(&layer.post_feedforward_layernorm);
            layer_uvb
                .pp("post_per_layer_input_norm")
                .add(&layer.post_per_layer_input_norm);

            // Attention norms
            let attn_uvb = layer_uvb.pp("self_attn");
            attn_uvb.pp("q_norm").add(&layer.self_attn.q_norm);
            attn_uvb.pp("k_norm").add(&layer.self_attn.k_norm);
            attn_uvb.pp("v_norm").add(&layer.self_attn.v_norm);

            // AltUp components
            let altup_uvb = layer_uvb.pp("altup");
            altup_uvb.add_tensor(
                "correct_output_scale",
                layer.altup.correct_output_scale.clone(),
            );
            altup_uvb
                .pp("correction_coefs")
                .add(&layer.altup.correction_coefs);
            altup_uvb
                .pp("prediction_coefs")
                .add(&layer.altup.prediction_coefs);
            altup_uvb
                .pp("modality_router")
                .add(&layer.altup.modality_router);
            altup_uvb.pp("router_norm").add(&layer.altup.router_norm);

            // Laurel block
            let laurel_uvb = layer_uvb.pp("laurel");
            laurel_uvb.pp("linear_left").add(&layer.laurel.left);
            laurel_uvb.pp("linear_right").add(&layer.laurel.right);
            laurel_uvb
                .pp("post_laurel_norm")
                .add(&layer.laurel.post_norm);

            // Per-layer input components
            layer_uvb
                .pp("per_layer_input_gate")
                .add(&layer.per_layer_input_gate);
            layer_uvb
                .pp("per_layer_projection")
                .add(&layer.per_layer_projection);
        }

        // Final norm
        uvb.pp("norm").add(&self.norm);

        // Per-layer projection components
        uvb.pp("per_layer_projection_norm")
            .add(&self.per_layer_projection_norm);

        uvb.to_safetensors()
    }
}

impl crate::speculative::SpeculativeTargetMixin for TextModel {}

impl crate::block_diffusion::BlockDiffusionMixin for TextModel {}

impl MultimodalModel for TextModel {
    fn forward(
        &self,
        _input_ids: &Tensor,
        _pixel_values: Option<Tensor>,
        _model_specific_args: Box<dyn std::any::Any>, // pixel attention mask, or image sizes, or anything else
        _ctx: &mut crate::pipeline::ModelForwardContext<'_>,
    ) -> candle_core::Result<Tensor> {
        unreachable!()
    }
    fn default_model_specific_args(&self, _input_ids: &Tensor) -> Box<dyn std::any::Any> {
        unreachable!()
    }
    fn cache(&self) -> &EitherCache {
        &self.cache
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    fn config(&self) -> &ModelConfigMetadata {
        &self.cfg
    }
    fn model_config(&self) -> Arc<dyn ModelConfigLike + Send + Sync> {
        self.model_config_like()
    }
}

impl AnyMoeBaseModelMixin for TextModel {}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{DType, Device, Tensor};

    use super::{
        is_paged_decode_forward, kv_shared_layer_index_for_layout, sliding_decode_kv_window,
        KvCacheTopology, PagedAttentionInputMetadata, PrefillQuerySelection,
    };

    fn layer_types(types: &[&str]) -> Vec<String> {
        types.iter().map(|ty| (*ty).to_string()).collect()
    }

    #[test]
    fn shared_kv_layers_use_the_latest_matching_donor() {
        let types = layer_types(&[
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ]);
        let owners = (0..types.len())
            .map(|layer| {
                kv_shared_layer_index_for_layout(&types, 2, layer)
                    .unwrap()
                    .unwrap_or(layer)
            })
            .collect::<Vec<_>>();
        let topology = KvCacheTopology::from_layer_owners(owners);

        assert_eq!(topology.owner_for_layer(4), 2);
        assert_eq!(topology.owner_for_layer(5), 3);
        assert!(topology.uses_own_kv_cache_for_layer(2));
        assert!(!topology.uses_own_kv_cache_for_layer(4));
    }

    #[test]
    fn malformed_shared_kv_layouts_fail_closed() {
        let types = layer_types(&["sliding_attention", "full_attention"]);
        assert!(kv_shared_layer_index_for_layout(&types, 3, 0).is_err());
        assert!(kv_shared_layer_index_for_layout(&types, 2, 0).is_err());
        assert!(kv_shared_layer_index_for_layout(&types, 0, 2).is_err());

        let missing_donor =
            layer_types(&["sliding_attention", "sliding_attention", "full_attention"]);
        assert!(kv_shared_layer_index_for_layout(&missing_donor, 1, 2).is_err());
    }

    #[test]
    fn sliding_donor_decode_uses_the_trailing_window() {
        assert_eq!(sliding_decode_kv_window(true, 1, Some(4), 11), Some((7, 4)));
        assert_eq!(sliding_decode_kv_window(true, 2, Some(4), 11), None);
        assert_eq!(sliding_decode_kv_window(false, 1, Some(4), 11), None);
        assert_eq!(sliding_decode_kv_window(true, 1, Some(4), 4), None);
    }

    #[test]
    fn paged_decode_does_not_depend_on_physical_query_width() {
        assert!(is_paged_decode_forward(true, 1, false, false));
        assert!(is_paged_decode_forward(true, 8, false, false));
        assert!(!is_paged_decode_forward(true, 8, true, false));
        assert!(!is_paged_decode_forward(true, 8, false, true));
        assert!(!is_paged_decode_forward(false, 8, false, false));
    }

    #[test]
    fn reduced_shared_prefill_keeps_query_rows_and_positions_aligned() {
        let selection =
            PrefillQuerySelection::from_logits_context(4, &[(3, 1), (3, 1)], &[5, 10]).unwrap();
        let source = Tensor::from_vec(
            (0u32..8).map(|value| value as f32).collect::<Vec<_>>(),
            (2, 4, 1),
            &Device::Cpu,
        )
        .unwrap();
        assert_eq!(
            selection
                .reduce(&source)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![3.0, 7.0]
        );
        let shifted = (source.clone() + 10.0).unwrap();
        let altup = Tensor::stack(&[source, shifted], 0).unwrap();
        assert_eq!(
            selection.reduce_altup(&altup).unwrap().dims(),
            &[2, 2, 1, 1]
        );

        let mut metadata = PagedAttentionInputMetadata::dummy(&Device::Cpu).unwrap();
        metadata.block_size = Some(16);
        metadata.block_tables = Some(HashMap::from([(
            Device::Cpu.location(),
            Tensor::zeros((2, 1), DType::U32, &Device::Cpu).unwrap(),
        )]));
        let reduced = metadata
            .for_reduced_prefill_queries(
                &[Device::Cpu],
                &selection.num_cached_tokens,
                &selection.query_lens,
            )
            .unwrap();
        assert_eq!(reduced.num_cached_tokens, Some(vec![8, 13]));
        assert_eq!(reduced.query_lens, Some(vec![1, 1]));
        assert_eq!(
            reduced
                .rope_positions
                .unwrap()
                .get(&Device::Cpu.location())
                .unwrap()
                .to_vec1::<u32>()
                .unwrap(),
            vec![8, 13]
        );
    }
}
