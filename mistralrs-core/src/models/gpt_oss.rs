#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::{CpuStorage, DType, Device, IndexOp, Result, Shape, Storage, Tensor, D};
use candle_nn::{Embedding, Linear, Module};
use core::num;
use mistralrs_quant::{
    apply_immediate_isq, linear, linear_no_bias, log::once_log_info, should_apply_immediate_isq,
    AfqLayer, ColumnParallelLayer, Comm, IsqType, MXFP4Layer, MatMul, QuantMethod,
    QuantMethodConfig, QuantizeOntoGuard, QuantizedConfig, ReplicatedLayer, RowParallelLayer,
    ShardedVarBuilder, SumAllReduce, UnquantLinear,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{atomic::AtomicUsize, Arc},
};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::{chunked_attention, SdpaParams},
    device_map::DeviceMapper,
    kv_cache::NormalCacheType,
    layers::{
        embedding, repeat_kv, Activation, CausalMasker, GPTOSSRotaryEmbedding, GPTOSSopeScalingConfig, RmsNorm, Sdpa
    },
    layers_masker::PastKvLenCache,
    ops::{TopKLastDimOp, TopKOutput},
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        extract_logits,
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, NormalCache, NormalLoadingMetadata, NormalModel,
    },
    serde_default_fn,
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

macro_rules! is_sliding {
    ($layer_idx:expr, $cfg:expr) => {
        $cfg.layer_types[$layer_idx] == "sliding_window"
    };
}

serde_default_fn!(bool, word_emb_default, false);

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct GPTOSSConfig {
    pub hidden_act: Option<Activation>,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub rope_scaling: Option<GPTOSSopeScalingConfig>,
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default = "word_emb_default")]
    pub tie_word_embeddings: bool,
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
    pub sliding_window: usize,
    pub attention_bias: bool,
    pub layer_types: Vec<String>,
}

impl GPTOSSConfig {
    pub fn hidden_act(&self) -> Activation {
        self.hidden_act.unwrap_or(Activation::Silu)
    }
}

struct CausalSelfAttention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<GPTOSSRotaryEmbedding>,
    max_seq_len: usize,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
    sinks: Tensor,
}

impl CausalSelfAttention {
    #[allow(clippy::too_many_arguments)]
    fn new(
        vb: ShardedVarBuilder,
        cfg: &GPTOSSConfig,
        layer_idx: usize,
        loading_isq: bool,
        mapper: &dyn DeviceMapper,
        rope: Arc<GPTOSSRotaryEmbedding>,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let size_in = cfg.hidden_size;
        let size_q = cfg.head_dim * cfg.num_attention_heads;
        let size_kv = cfg.head_dim * cfg.num_key_value_heads;
        let q_proj = ColumnParallelLayer::new(
            size_in,
            size_q,
            &cfg.quantization_config,
            cfg.attention_bias,
            comm,
            mapper.set_device(layer_idx, vb.pp("q_proj"), loading_isq),
        )?;
        let kv_shard =
            mistralrs_quant::compute_kv_shard(cfg.num_key_value_heads, cfg.head_dim, comm);
        let k_proj = ColumnParallelLayer::new_with_shard(
            size_in,
            size_kv,
            &cfg.quantization_config,
            cfg.attention_bias,
            comm,
            kv_shard,
            mapper.set_device(layer_idx, vb.pp("k_proj"), loading_isq),
        )?;
        let v_proj = ColumnParallelLayer::new_with_shard(
            size_in,
            size_kv,
            &cfg.quantization_config,
            cfg.attention_bias,
            comm,
            kv_shard,
            mapper.set_device(layer_idx, vb.pp("v_proj"), loading_isq),
        )?;
        let o_proj = RowParallelLayer::new(
            size_q,
            size_in,
            &cfg.quantization_config,
            cfg.attention_bias,
            comm,
            mapper.set_device(layer_idx, vb.pp("o_proj"), loading_isq),
        )?;
        let sinks = mapper
            .set_nm_device(vb.clone(), loading_isq)
            .get(cfg.num_attention_heads, "sinks")?;
        let head_dim = cfg.head_dim;
        let sliding_window = if is_sliding!(layer_idx, cfg) {
            Some(cfg.sliding_window)
        } else {
            None
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads / comm.world_size(),
            num_key_value_heads: (cfg.num_key_value_heads / comm.world_size()).max(1),
            head_dim,
            rotary_emb: rope,
            max_seq_len: cfg.max_position_embeddings,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: mistralrs_quant::compute_n_kv_groups(
                    cfg.num_key_value_heads,
                    cfg.num_attention_heads,
                    comm,
                ),
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window,
            },
            sinks,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        attention_mask: Option<&Tensor>,
        sliding_attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let mut q = self.q_proj.forward_autocast(x)?;
        let mut k = self.k_proj.forward_autocast(x)?;
        let mut v = self.v_proj.forward_autocast(x)?;

        q = q
            .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?;
        k = k
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;
        v = v
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;

        (q, k) = self.rotary_emb.forward(&q, &k, seqlen_offsets)?;

        let mask = if self.sdpa_params.sliding_window.is_some() {
            sliding_attention_mask
        } else {
            attention_mask
        };

        let mut y = {
            let (k, v) = kv_cache.append(&k, &v)?;

            let k = repeat_kv(k.clone(), self.sdpa_params.n_kv_groups)?;
            let v = repeat_kv(v.clone(), self.sdpa_params.n_kv_groups)?;
            
            // Use chunked attention with a closure that captures the necessary parameters
            chunked_attention(
                &q.contiguous()?,
                &k.contiguous()?,
                &v.contiguous()?,
                mask,
                |q_chunk, k, v, mask_chunk| {
                    let mut att = MatMul.matmul_affine_mul(
                        q_chunk,
                        &k.t()?,
                        self.sdpa_params.softmax_scale.into(),
                    )?;

                    if let Some(mask) = mask_chunk {
                        att = att.broadcast_add(mask)?;
                    }

                    let sinks =
                        self.sinks
                            .reshape((1, (), 1, 1))?
                            .broadcast_as(Shape::from_dims(&[
                                q.dim(0)?,
                                self.num_attention_heads,
                                q.dim(D::Minus2)?,
                                1,
                            ]))?;
                    att = Tensor::cat(&[&att, &sinks], D::Minus1)?;
                    // TODO eval
                    att = att.broadcast_sub(&att.max_keepdim(D::Minus1)?)?;

                    att = candle_nn::ops::softmax_last_dim(&att)?;
                    // Drop the sink
                    att = att.i((.., .., .., ..att.dim(D::Minus1)? - 1))?;
                    MatMul.matmul(&att, v)
                },
            )?
        };
        y = if attention_mask.is_some() {
            y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?
        } else {
            y.reshape((b_sz, seq_len, ()))?
        };
        self.o_proj.forward_autocast(&y)
    }
}

struct Mlp {
    gate: Arc<dyn QuantMethod>,
    up: Arc<dyn QuantMethod>,
    down: Arc<dyn QuantMethod>,
    act: Activation,
}

impl Mlp {
    fn new(
        vb: ShardedVarBuilder,
        hidden_size: usize,
        intermediate_size: usize,
        quantization_config: &Option<QuantizedConfig>,
        hidden_act: Activation,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        Ok(Self {
            gate: ColumnParallelLayer::new(
                hidden_size,
                intermediate_size,
                quantization_config,
                false,
                comm,
                vb.pp("gate_proj"),
            )?,
            up: ColumnParallelLayer::new(
                hidden_size,
                intermediate_size,
                quantization_config,
                false,
                comm,
                vb.pp("up_proj"),
            )?,
            down: RowParallelLayer::new(
                intermediate_size,
                hidden_size,
                quantization_config,
                false,
                comm,
                vb.pp("down_proj"),
            )?,
            act: hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = self.gate.forward_autocast(xs)?;
        let rhs = self.up.forward_autocast(xs)?;

        self.down
            .forward_autocast(&crate::ops::mul_and_act(&lhs, &rhs, self.act)?)
    }
}

#[derive(Debug)]
pub struct Experts {
    pub gate_up_proj: Arc<dyn QuantMethod>,
    pub down_proj: Arc<dyn QuantMethod>,
    pub gate_up_proj_bias: Tensor,
    pub down_proj_bias: Tensor,
}

impl Experts {
    /// Note: we only support AFQ/MXFP4/unquantized here because they are the only ones that support indexed.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_local_experts: usize,
        hidden_size: usize,
        intermediate_size: usize,
        config: &Option<QuantizedConfig>,
        bias: bool,
        comm: &Arc<Comm>,
        vb: ShardedVarBuilder,
        loading_isq: bool,
    ) -> Result<Self> {
        let base_vb = vb.clone();

        // let is_quantized =
        //     loading_isq || vb.device().is_cuda() || should_apply_immediate_isq(&vb);
        assert!(!(loading_isq || should_apply_immediate_isq(&vb)));
        let is_quantized = true;

        // let vb_gate_up_proj_proj = if is_quantized {
        //     vb.pp("gate_up_proj").set_device(Device::Cpu)
        // } else {
        //     vb.pp("gate_up_proj")
        // };
        // let vb_down_proj_proj = if is_quantized {
        //     vb.pp("down_proj").set_device(Device::Cpu)
        // } else {
        //     vb.pp("down_proj")
        // };
        // let vb = vb.set_device(Device::Cpu);
        let gu_blocks = vb
            .get_with_hints_dtype(
                (
                    num_local_experts,
                    intermediate_size * 2,
                    90,
                    16,
                    // hidden_size * 4 / 32,
                ),
                "gate_up_proj_blocks",
                Default::default(),
                DType::F4,
            )?
            .reshape((num_local_experts, intermediate_size * 2, 1440))?;
        let gu_scales = vb.get_with_hints_dtype(
            (num_local_experts, intermediate_size * 2, 90), //hidden_size * 32),
            "gate_up_proj_scales",
            Default::default(),
            DType::F8E8M0,
        )?;
        let mut gate_up_proj: Arc<dyn QuantMethod> =
            Arc::new(MXFP4Layer::new(QuantMethodConfig::MXFP4 {
                blocks: gu_blocks,
                scales: gu_scales,
                bias: None,
            })?);

        let d_blocks = vb
            .get_with_hints_dtype(
                (num_local_experts, hidden_size, 90, 16), //intermediate_size * 4 / 32),
                "down_proj_blocks",
                Default::default(),
                DType::F4,
            )?
            .reshape((num_local_experts, hidden_size, 1440))?;
        let d_scales = vb.get_with_hints_dtype(
            (num_local_experts, hidden_size, 90), //intermediate_size * 32),
            "down_proj_scales",
            Default::default(),
            DType::F8E8M0,
        )?;
        let mut down_proj: Arc<dyn QuantMethod> =
            Arc::new(MXFP4Layer::new(QuantMethodConfig::MXFP4 {
                blocks: d_blocks,
                scales: d_scales,
                bias: None,
            })?);

        if is_quantized {
            gate_up_proj = Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
                Linear::new(gate_up_proj.dequantize_w()?.t()?.contiguous()?, gate_up_proj.bias().cloned()),
            ))?);
            down_proj = Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
                Linear::new(down_proj.dequantize_w()?.t()?.contiguous()?, down_proj.bias().cloned()),
            ))?);
        }

        // If on cuda and no other quantization, automatic quant to Q4K
        // if vb.device().is_cuda() && !loading_isq && !should_apply_immediate_isq(&vb) {
            once_log_info("Requantizing MoE layers to ISQ 4.");
            gate_up_proj = gate_up_proj.apply_isq(
                Some(IsqType::AFQ4),
                vb.device().clone(),
                &AtomicUsize::new(0),
                None,
                QuantizeOntoGuard::new(),
            )?;
            down_proj = down_proj.apply_isq(
                Some(IsqType::AFQ4),
                vb.device().clone(),
                &AtomicUsize::new(0),
                None,
                QuantizeOntoGuard::new(),
            )?;
        // }

        // gate_up_proj = apply_immediate_isq(gate_up_proj, base_vb.pp("gate_up_proj"))?;
        // down_proj = apply_immediate_isq(down_proj, base_vb.pp("down_proj"))?;

        let gate_up_proj_bias = vb.get(
            (num_local_experts, 2 * intermediate_size),
            "gate_up_proj_bias",
        )?;
        let down_proj_bias = vb.get((num_local_experts, intermediate_size), "down_proj_bias")?;

        Ok(Self {
            gate_up_proj,
            down_proj,
            gate_up_proj_bias,
            down_proj_bias,
        })
    }
}

struct TextExperts {
    gate_up_proj: Arc<dyn QuantMethod>,
    down_proj: Arc<dyn QuantMethod>,
    gate_up_proj_bias: Tensor,
    down_proj_bias: Tensor,
    act: Activation,
    hidden_size: usize,
    intermediate_size: usize,
    sum_all_reduce: SumAllReduce,
    limit: f64,
    alpha: f64,
}

impl TextExperts {
    fn new(
        vb: ShardedVarBuilder,
        cfg: &GPTOSSConfig,
        quantization_config: &Option<QuantizedConfig>,
        comm: &Arc<mistralrs_quant::Comm>,
        loading_isq: bool,
    ) -> Result<Self> {
        let Experts {
            gate_up_proj,
            down_proj,
            gate_up_proj_bias,
            down_proj_bias,
        } = Experts::new(
            cfg.num_local_experts,
            cfg.hidden_size,
            cfg.intermediate_size,
            quantization_config,
            true,
            comm,
            vb,
            loading_isq,
        )?;
        Ok(Self {
            gate_up_proj,
            down_proj,
            gate_up_proj_bias,
            down_proj_bias,
            act: cfg.hidden_act(),
            hidden_size: cfg.hidden_size,
            intermediate_size: cfg.intermediate_size,
            sum_all_reduce: SumAllReduce::new(comm),
            limit: 7.0,
            alpha: 1.702,
        })
    }

    // xs: (bs * seq_len, hidden_size)
    // expert indices: (bs * seq_len)
    fn forward(
        &self,
        bs: usize,
        xs: &Tensor,
        routing_weights: &Tensor,
        // routing_indices: &Tensor,
    ) -> Result<Tensor> {
        let mut xs = xs.unsqueeze(1)?;

        let num_experts = routing_weights.dim(1)?;
        xs = xs
            .repeat((num_experts, 1))?
            .reshape((num_experts, (), xs.dim(D::Minus1)?))?;
        let gate_up = self
            .gate_up_proj
            .forward_autocast(&xs).unwrap()
            .broadcast_add(&self.gate_up_proj_bias.unsqueeze(D::Minus2)?)?
            .reshape((xs.dim(0)?, xs.dim(1)?, (), 2))?;
        let gate = gate_up.i((.., .., .., 0))?.clamp(-self.limit, self.limit)?;
        let up = gate_up.i((.., .., .., 1))?.clamp(-self.limit, self.limit)?;
        let glu = (&gate * candle_nn::ops::sigmoid(&(&gate * self.alpha)?)?)?;
        xs = self
            .down_proj
            .forward_autocast(&((&up + 1.)? * glu)?).unwrap()
            .broadcast_add(&self.down_proj_bias.unsqueeze(D::Minus2)?)?
            .reshape((num_experts, bs, (), self.hidden_size))?;
        xs = xs.broadcast_mul(
            &routing_weights
                .transpose(0, 1)?
                .reshape((num_experts, bs, ()))?
                .unsqueeze(D::Minus1)?,
        )?;
        xs.sum(0)

        // if self.gate_up_proj.len() == 1 {
        //     let gate_up = self.gate_up_proj[0]
        //         .gather_forward_autocast(&xs, indices)?
        //         .clamp(-self.swiglu_limit, self.swiglu_limit)?;
        //     let gate = gate_up.narrow(D::Minus1, 0, self.intermediate_size)?;
        //     let up = gate.narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?;

        //     let mut xs = self.down_proj[0]
        //         .gather_forward_autocast(&(up * gate.apply(&self.act)?)?, indices)?
        //         .clamp(-self.swiglu_limit, self.swiglu_limit)?;
        //     xs = self.sum_all_reduce.sum_all_reduce(&xs)?;
        //     xs.reshape(((), self.hidden_size))
        // } else {
        //     let indices = indices.to_vec1::<u32>()?;
        //     let mut results = Vec::new();
        //     for (tok, id) in indices.into_iter().enumerate() {
        //         let xs = xs.i(tok)?.reshape((1, self.hidden_size))?;

        //         let res = {
        //             let gate_up = self.gate_up_proj[id as usize]
        //                 .forward_autocast(&xs)?
        //                 .clamp(-self.swiglu_limit, self.swiglu_limit)?;
        //             let gate = gate_up.narrow(D::Minus1, 0, self.intermediate_size)?;
        //             let up =
        //                 gate.narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?;

        //             self.down_proj[id as usize]
        //                 .forward_autocast(&(up * gate.apply(&self.act)?)?)?
        //                 .clamp(-self.swiglu_limit, self.swiglu_limit)?
        //         };
        //         results.push(res);
        //     }
        //     let mut xs = Tensor::cat(&results, 0)?;
        //     xs = self.sum_all_reduce.sum_all_reduce(&xs)?;
        //     xs.reshape(((), self.hidden_size))
        // }
    }
}

struct TextMoe {
    experts: TextExperts,
    router: Arc<dyn QuantMethod>,
    topk: usize,
}

impl TextMoe {
    fn new(
        vb: ShardedVarBuilder,
        cfg: &GPTOSSConfig,
        quantization_config: &Option<QuantizedConfig>,
        comm: &Arc<mistralrs_quant::Comm>,
        loading_isq: bool,
    ) -> Result<Self> {
        let experts = TextExperts::new(
            vb.pp("experts"),
            cfg,
            quantization_config,
            comm,
            loading_isq,
        )?;
        let router = linear(
            cfg.hidden_size,
            cfg.num_local_experts,
            quantization_config,
            vb.pp("router"),
        )?;
        Ok(Self {
            experts,
            router,
            topk: cfg.num_experts_per_tok,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (bs, seq_len, hidden_dim) = xs.dims3()?;
        let xs = xs.reshape(((), hidden_dim))?;
        let router_logits = self.router.forward_autocast(&xs)?;

        let TopKOutput {
            values: router_top_value,
            indices: router_indices,
        } = router_logits.topk(self.topk)?;

        let router_scores = candle_nn::ops::sigmoid(&router_top_value.to_dtype(DType::F32)?)?
            .to_dtype(router_top_value.dtype())?;

        let routed_out = self
            .experts
            .forward(bs, &xs, &router_scores)?
            .reshape((bs, seq_len, hidden_dim))?;

        Ok(routed_out)
    }
}

struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    mlp: TextMoe,
}

impl Block {
    #[allow(clippy::too_many_arguments)]
    fn new(
        vb: ShardedVarBuilder,
        cfg: &GPTOSSConfig,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        rope: Arc<GPTOSSRotaryEmbedding>,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let attn = CausalSelfAttention::new(
            vb.pp("self_attn"),
            cfg,
            layer_idx,
            loading_isq,
            mapper,
            rope,
            paged_attn,
            comm,
        )?;
        let mlp = TextMoe::new(
            mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq),
            cfg,
            &cfg.quantization_config,
            comm,
            loading_isq,
        )?;
        let rms_1 = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("input_layernorm"), false),
        )?;
        let rms_2 = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("post_attention_layernorm"), false),
        )?;
        Ok(Self {
            rms_1,
            attn,
            rms_2,
            mlp,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        attention_mask: Option<&Tensor>,
        sliding_attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let x = (self.attn.forward(
            &x,
            attention_mask,
            sliding_attention_mask,
            seqlen_offsets,
            kv_cache,
            metadata,
            flash_params,
        )? + residual)?;
        let residual = &x;
        let x = (self.mlp.forward(&self.rms_2.forward(&x)?)? + residual)?;
        Ok(x)
    }
}

pub struct GPTOSSModel {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: Arc<dyn QuantMethod>,
    kv_cache: crate::pipeline::EitherCache,
    device: Device,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    cfg: ModelConfigMetadata,
    sliding_window: usize,
}

impl GPTOSSModel {
    pub fn new(
        cfg: &GPTOSSConfig,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let vb_lm_head = vb.pp("lm_head");
        Self::new_inner(
            cfg,
            vb_m,
            vb_lm_head,
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )
    }

    pub fn new_inner(
        cfg: &GPTOSSConfig,
        vb_m: ShardedVarBuilder,
        vb_lm_head: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let mut cfg = cfg.clone();
        cfg.quantization_config = None;
        let cfg = &cfg;
        if let Some(ref quant_cfg) = &cfg.quantization_config {
            tracing::info!(
                "Using {} quantization: {}.",
                quant_cfg.name(),
                quant_cfg.get_bits_name(&vb_m)
            );
        }
        assert_eq!(AttentionImplementation::Eager, attention_mechanism);
        let mapper = normal_loading_metadata.mapper;

        let wte = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), false),
            &cfg.quantization_config,
        )?;
        let lm_head = if !cfg.tie_word_embeddings {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &cfg.quantization_config,
                false,
                mapper.set_nm_device(vb_lm_head, normal_loading_metadata.loading_isq),
            )?
        } else {
            ReplicatedLayer::from_linear(candle_nn::Linear::new(
                mapper.cast_nm_device(wte.embeddings(), normal_loading_metadata.loading_isq)?,
                None,
            ))?
        };
        let ln_f = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;
        let head_dim = cfg.head_dim;
        let mut ropes = HashMap::new();
        for i in 0..cfg.num_hidden_layers {
            let device = mapper
                .device_for(i, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            ropes.insert(
                device.location(),
                Arc::new(GPTOSSRotaryEmbedding::new(
                    is_gptx,
                    vb_m.dtype(),
                    cfg,
                    device,
                )?),
            );
        }
        let blocks = NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading text repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .par_iter_if_isq(|i| {
            let device = mapper
                .device_for(i, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let rotary_emb = ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => {
                    Some(PagedAttention::new(head_dim, device, None)?)
                }
            };
            let comm = mapper.get_comm_for(i)?;
            Block::new(
                vb_m.pp(format!("layers.{i}")),
                cfg,
                &*mapper,
                i,
                normal_loading_metadata.loading_isq,
                rotary_emb,
                paged_attn,
                &comm,
            )
        })?;

        let cache_types = (0..cfg.num_hidden_layers)
            .map(|layer_idx| {
                is_sliding!(layer_idx, cfg)
                    .then(|| NormalCacheType::SlidingWindow {
                        window: cfg.sliding_window,
                    })
                    .unwrap_or(NormalCacheType::Normal {
                        max_seq_len: cfg.max_position_embeddings,
                    })
            })
            .collect::<Vec<_>>();
        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
            kv_cache: EitherCache::Normal(NormalCache::from_types(cache_types)),
            device: normal_loading_metadata.real_device,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_kv_heads: (cfg.num_key_value_heads / mapper.get_comm_for(0)?.world_size())
                    .max(1),
                num_attn_heads: cfg.num_attention_heads / mapper.get_comm_for(0)?.world_size(),
                sliding_window: Some(cfg.sliding_window),
                k_head_dim: cfg.head_dim,
                v_head_dim: cfg.head_dim,
            },
            mapper,
            sliding_window: cfg.sliding_window,
        })
    }

    pub fn get_input_embeddings(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.wte.forward(input_ids)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_embeds(
        &self,
        input_ids: &Tensor,
        input_embeds: Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let mut x = input_embeds;
        let cache = &mut self.kv_cache.normal().0;
        let attention_mask = CausalMasker.make_causal_mask_matrix(
            input_ids,
            &*cache,
            x.dtype(),
            self.cfg.num_attn_heads,
        )?;
        // PagedAttention prompt chunking
        let attention_mask = attention_mask.filter(|_| {
            metadata
                .as_ref()
                .map(|(_, meta)| meta.is_first_prompt_chunk)
                .unwrap_or(true)
        });
        let sliding_attention_mask = CausalMasker.make_sliding_window_causal_mask_matrix(
            input_ids,
            &*cache,
            Some(self.sliding_window),
            x.dtype(),
            self.cfg.num_attn_heads,
        )?;
        // PagedAttention prompt chunking
        let sliding_attention_mask = sliding_attention_mask.filter(|_| {
            metadata
                .as_ref()
                .map(|(_, meta)| meta.is_first_prompt_chunk)
                .unwrap_or(true)
        });
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = self.mapper.map(x, block_idx)?;
            x = block.forward(
                &x,
                attention_mask
                    .as_ref()
                    .map(|m| m.to_device(x.device()).unwrap())
                    .as_ref(),
                sliding_attention_mask
                    .as_ref()
                    .map(|m| m.to_device(x.device()).unwrap())
                    .as_ref(),
                seqlen_offsets,
                &mut cache[block_idx],
                metadata
                    .as_ref()
                    .map(|(kv_cache, metadata)| (kv_cache[block_idx].clone(), *metadata)),
                flash_params,
            )?;
        }
        let mut x = x.to_device(&self.device)?;
        x = self.ln_f.forward(&x)?;
        x = self.lm_head.forward_autocast(&x)?;
        extract_logits(&x, context_lens)
    }

    pub fn residual_tensors_m(&self, uvb_m: UnVarBuilder) -> Vec<(String, Tensor)> {
        uvb_m.pp("embed_tokens").add(&self.wte);
        uvb_m.pp("norm").add(&self.ln_f);

        for (layer_idx, layer) in self.blocks.iter().enumerate() {
            let uvb_l = uvb_m.pp("layers").pp(layer_idx);
            uvb_l.pp("input_layernorm").add(&layer.rms_1);
            uvb_l.pp("post_attention_layernorm").add(&layer.rms_2);
        }

        uvb_m.to_safetensors()
    }
}

impl IsqModel for GPTOSSModel {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        let mut tensors = Vec::new();
        tensors.push((&mut self.lm_head, None));
        for (i, layer) in self.blocks.iter_mut().enumerate() {
            tensors.push((&mut layer.attn.q_proj, Some(i)));
            tensors.push((&mut layer.attn.k_proj, Some(i)));
            tensors.push((&mut layer.attn.v_proj, Some(i)));
            tensors.push((&mut layer.attn.o_proj, Some(i)));
            tensors.push((&mut layer.mlp.router, Some(i)));
            tensors.push((&mut layer.mlp.experts.gate_up_proj, Some(i)));
            tensors.push((&mut layer.mlp.experts.down_proj, Some(i)));
        }
        (tensors, &*self.mapper)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        self.residual_tensors_m(uvb.pp("model"))
    }
}

impl NormalModel for GPTOSSModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        self.forward_embeds(
            input_ids,
            self.get_input_embeddings(input_ids)?,
            seqlen_offsets,
            context_lens,
            metadata,
            flash_params,
        )
    }
    fn xlora_forward(
        &self,
        _input_ids: &Tensor,
        _input_ids_full: &Tensor,
        _seqlen_offsets: &[usize],
        _seqlen_offsets_full: &[usize],
        _no_kv_cache: bool,
        _non_granular_state: &Option<crate::xlora_models::NonGranularState>,
        _context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        _flash_params: &FlashParams,
        _flash_params_full: &FlashParams,
    ) -> Result<Tensor> {
        unimplemented!()
    }
    fn cache(&self) -> &crate::pipeline::EitherCache {
        &self.kv_cache
    }
    fn cache_mut(&mut self) -> &mut crate::pipeline::EitherCache {
        &mut self.kv_cache
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn is_xlora(&self) -> bool {
        false
    }
    fn max_seq_len(&self) -> usize {
        self.blocks[0].attn.max_seq_len
    }
    fn config(&self) -> &ModelConfigMetadata {
        &self.cfg
    }
}

impl AnyMoeBaseModelMixin for GPTOSSModel {}
