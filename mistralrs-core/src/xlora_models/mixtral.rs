#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::SdpaParams,
    layers::{self, Activation, RotaryEmbedding, Sdpa},
    lora::{linear_no_bias, LinearLayerLike, LoraConfig, Ordering},
    paged_attention::ModelConfigMetadata,
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, NormalLoadingMetadata,
    },
    utils::progress::NiceProgressBar,
};
/// Mixtral Model
/// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py
/// https://mistral.ai/news/mixtral-of-experts/
use candle_core::{DType, Device, Module, Result, Tensor};
use mistralrs_quant::{QuantMethod, ShardedVarBuilder};
use std::{collections::HashMap, sync::Arc};
use tqdm::Iter;
use tracing::info;

use crate::{
    device_map::{DeviceMappedMask, DeviceMapper},
    layers::{CausalMasker, RmsNorm},
    models::mixtral::Config,
    pipeline::{extract_logits, Cache, NormalModel},
};

use super::{classifier::XLoraClassifier, NonGranularState, ScalingsMaker, XLoraConfig};

struct Attention {
    q_proj: Arc<dyn LinearLayerLike + Send + Sync>,
    k_proj: Arc<dyn LinearLayerLike + Send + Sync>,
    v_proj: Arc<dyn LinearLayerLike + Send + Sync>,
    o_proj: Arc<dyn LinearLayerLike + Send + Sync>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    sliding_window: Option<usize>,
    sdpa_params: SdpaParams,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        vb: ShardedVarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        count: &mut usize,
        ord: &Ordering,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = hidden_sz / num_heads;
        let q_proj = linear_no_bias(
            hidden_sz,
            num_heads * head_dim,
            mapper.set_device(layer_idx, vb.pp("q_proj"), loading_isq),
            mapper.set_device(layer_idx, vb.pp("q_proj"), false),
            lora_config,
            count,
            ord,
            preload_adapters,
        )?;
        let k_proj = linear_no_bias(
            hidden_sz,
            num_kv_heads * head_dim,
            mapper.set_device(layer_idx, vb.pp("k_proj"), loading_isq),
            mapper.set_device(layer_idx, vb.pp("k_proj"), false),
            lora_config,
            count,
            ord,
            preload_adapters,
        )?;
        let v_proj = linear_no_bias(
            hidden_sz,
            num_kv_heads * head_dim,
            mapper.set_device(layer_idx, vb.pp("v_proj"), loading_isq),
            mapper.set_device(layer_idx, vb.pp("v_proj"), false),
            lora_config,
            count,
            ord,
            preload_adapters,
        )?;
        let o_proj = linear_no_bias(
            num_heads * head_dim,
            hidden_sz,
            mapper.set_device(layer_idx, vb.pp("o_proj"), loading_isq),
            mapper.set_device(layer_idx, vb.pp("o_proj"), false),
            lora_config,
            count,
            ord,
            preload_adapters,
        )?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_emb,
            sliding_window: cfg.sliding_window,
            sdpa_params: SdpaParams {
                n_kv_groups: num_heads / num_kv_heads,
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window: cfg.sliding_window,
                sinks: None,
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut Option<(Tensor, Tensor)>,
        scalings: Option<Tensor>,
        global_scaling_weight: f64,
        is_scaling_pass: Option<f64>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let original_dtype = xs.dtype();
        let mut xs = xs.clone();
        if let Some(t) = self.q_proj.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        let mut q = self.q_proj.lora_forward(
            &xs,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?;
        let mut k = self.k_proj.lora_forward(
            &xs,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?;
        let mut v = self.v_proj.lora_forward(
            &xs,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?;
        if self.q_proj.quantized_act_type().is_some() {
            q = q.to_dtype(original_dtype)?;
            k = k.to_dtype(original_dtype)?;
            v = v.to_dtype(original_dtype)?;
        }

        let (q, k, v) = if q_len != 1 {
            let q = q
                .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?;
            let k = k
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        } else {
            let q = q.reshape((b_sz, self.num_heads, q_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.num_kv_heads, q_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.num_kv_heads, q_len, self.head_dim))?;
            (q, k, v)
        };

        let (q, k) = self.rotary_emb.forward(&q, &k, seqlen_offsets)?;

        let (k, v, attn_mask) = Cache::update_kv_cache_sliding_window(
            kv_cache,
            k,
            v,
            attention_mask,
            self.sliding_window,
        )?;

        let mut attn_output = Sdpa.run_attention(
            &q,
            &k,
            &v,
            attn_mask.as_ref(),
            Some(flash_params),
            &self.sdpa_params,
        )?;

        if let Some(t) = self.q_proj.quantized_act_type() {
            attn_output = attn_output.to_dtype(t)?;
        }
        let mut res = self.o_proj.lora_forward(
            &attn_output.transpose(1, 2)?.reshape((b_sz, q_len, ()))?,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?;
        if self.q_proj.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

#[derive(Clone)]
struct BlockSparseTop2MLP {
    w1: Arc<dyn LinearLayerLike + Send + Sync>,
    w2: Arc<dyn LinearLayerLike + Send + Sync>,
    w3: Arc<dyn LinearLayerLike + Send + Sync>,
    act_fn: Activation,
}

impl BlockSparseTop2MLP {
    #[allow(clippy::too_many_arguments)]
    fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        count: &mut usize,
        ord: &Ordering,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let w1 = linear_no_bias(
            hidden_sz,
            intermediate_sz,
            mapper.set_device(layer_idx, vb.pp("w1"), loading_isq),
            mapper.set_device(layer_idx, vb.pp("w1"), false),
            lora_config,
            count,
            ord,
            preload_adapters,
        )?;
        let w2 = linear_no_bias(
            intermediate_sz,
            hidden_sz,
            mapper.set_device(layer_idx, vb.pp("w2"), loading_isq),
            mapper.set_device(layer_idx, vb.pp("w2"), false),
            lora_config,
            count,
            ord,
            preload_adapters,
        )?;
        let w3 = linear_no_bias(
            hidden_sz,
            intermediate_sz,
            mapper.set_device(layer_idx, vb.pp("w3"), loading_isq),
            mapper.set_device(layer_idx, vb.pp("w3"), false),
            lora_config,
            count,
            ord,
            preload_adapters,
        )?;
        Ok(Self {
            w1,
            w2,
            w3,
            act_fn: cfg.hidden_act,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        scalings: Option<Tensor>,
        global_scaling_weight: f64,
        is_scaling_pass: Option<f64>,
    ) -> Result<Tensor> {
        let original_dtype = xs.dtype();
        let mut xs = xs.clone();
        if let Some(t) = self.w1.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        let lhs = self
            .w1
            .lora_forward(
                &xs,
                scalings.clone(),
                global_scaling_weight,
                is_scaling_pass,
            )?
            .apply(&self.act_fn)?;
        let rhs = self.w3.lora_forward(
            &xs,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?;
        let mut res = self.w2.lora_forward(
            &(lhs * rhs)?,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?;
        if self.w1.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

#[derive(Clone)]
struct SparseMoeBlock {
    gate: Arc<dyn LinearLayerLike + Send + Sync>,
    experts: Vec<BlockSparseTop2MLP>,
    num_experts_per_tok: usize,
}

impl SparseMoeBlock {
    #[allow(clippy::too_many_arguments)]
    fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        count: &mut usize,
        ord: &Ordering,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Self> {
        let gate = linear_no_bias(
            cfg.hidden_size,
            cfg.num_local_experts,
            mapper.set_device(layer_idx, vb.pp("gate"), loading_isq),
            mapper.set_device(layer_idx, vb.pp("gate"), false),
            lora_config,
            count,
            ord,
            preload_adapters,
        )?;
        let mut experts = Vec::with_capacity(cfg.num_local_experts);
        let vb = vb.pp("experts");
        for idx in 0..cfg.num_local_experts {
            let expert = BlockSparseTop2MLP::new(
                cfg,
                vb.pp(idx),
                lora_config,
                count,
                ord,
                mapper,
                layer_idx,
                loading_isq,
                preload_adapters,
            )?;
            experts.push(expert)
        }
        Ok(SparseMoeBlock {
            gate,
            experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        scalings: Option<Tensor>,
        global_scaling_weight: f64,
        is_scaling_pass: Option<f64>,
    ) -> Result<Tensor> {
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        let xs = xs.reshape(((), hidden_dim))?;

        let original_dtype = xs.dtype();
        let mut xs = xs.clone();
        if let Some(t) = self.gate.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        let mut router_logits = self.gate.lora_forward(
            &xs,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?;
        if self.gate.quantized_act_type().is_some() {
            router_logits = router_logits.to_dtype(original_dtype)?;
        }

        let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

        // In order to extract topk, we extract the data from the tensor and manipulate it
        // directly. Maybe we will want to use some custom ops instead at some point.
        let routing_weights = routing_weights.to_dtype(DType::F32)?.to_vec2::<f32>()?;

        // routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        // top_x contains the row indexes to evaluate for each expert.
        let mut top_x = vec![vec![]; self.experts.len()];
        let mut selected_rws = vec![vec![]; self.experts.len()];
        for (row_idx, rw) in routing_weights.iter().enumerate() {
            let mut dst = (0..rw.len() as u32).collect::<Vec<u32>>();
            dst.sort_by(|&i, &j| rw[j as usize].total_cmp(&rw[i as usize]));
            let mut sum_routing_weights = 0f32;
            for &expert_idx in dst.iter().take(self.num_experts_per_tok) {
                let expert_idx = expert_idx as usize;
                let routing_weight = rw[expert_idx];
                sum_routing_weights += routing_weight;
                top_x[expert_idx].push(row_idx as u32);
            }
            for &expert_idx in dst.iter().take(self.num_experts_per_tok) {
                let expert_idx = expert_idx as usize;
                let routing_weight = rw[expert_idx];
                selected_rws[expert_idx].push(routing_weight / sum_routing_weights)
            }
        }

        // routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        // expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        let mut ys = xs.zeros_like()?;
        for (expert_idx, expert_layer) in self.experts.iter().enumerate() {
            let top_x = &top_x[expert_idx];
            if top_x.is_empty() {
                continue;
            }
            let top_x = Tensor::new(top_x.as_slice(), xs.device())?;
            let selected_rws =
                Tensor::new(selected_rws[expert_idx].as_slice(), xs.device())?.reshape(((), 1))?;
            // Index the correct hidden states and compute the expert hidden state for
            // the current expert. We need to make sure to multiply the output hidden
            // states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            let current_state = xs.index_select(&top_x, 0)?.reshape(((), hidden_dim))?;
            // current_hidden_states = expert_layer(current_state, routing_weights[top_x_list, idx_list, None])
            let current_hidden_states = expert_layer.forward(
                &current_state,
                scalings.clone(),
                global_scaling_weight,
                is_scaling_pass,
            )?;
            let current_hidden_states = current_hidden_states.broadcast_mul(&selected_rws)?;
            ys = ys.index_add(&top_x, &current_hidden_states, 0)?;
        }

        let ys = ys.reshape((b_size, seq_len, hidden_dim))?;
        Ok(ys)
    }
}

struct DecoderLayer {
    self_attn: Attention,
    block_sparse_moe: SparseMoeBlock,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        vb: ShardedVarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        count: &mut usize,
        ord: &Ordering,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Self> {
        let self_attn = Attention::new(
            rotary_emb,
            cfg,
            vb.pp("self_attn"),
            lora_config,
            count,
            ord,
            mapper,
            layer_idx,
            loading_isq,
            preload_adapters,
        )?;
        let block_sparse_moe = SparseMoeBlock::new(
            cfg,
            vb.pp("block_sparse_moe"),
            lora_config,
            count,
            ord,
            mapper,
            layer_idx,
            loading_isq,
            preload_adapters,
        )?;
        let input_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("input_layernorm"), false),
        )?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("post_attention_layernorm"), false),
        )?;
        Ok(Self {
            self_attn,
            block_sparse_moe,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut Option<(Tensor, Tensor)>,
        scalings: Option<Tensor>,
        global_scaling_weight: f64,
        is_scaling_pass: Option<f64>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(
            &xs,
            attention_mask,
            seqlen_offsets,
            kv_cache,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
            flash_params,
        )?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .block_sparse_moe
            .forward(
                &xs.apply(&self.post_attention_layernorm)?,
                scalings.clone(),
                global_scaling_weight,
                is_scaling_pass,
            )?
            .to_dtype(residual.dtype())?;
        residual + xs
    }
}

pub struct XLoraModel {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Arc<dyn LinearLayerLike + Send + Sync>,
    sliding_window: Option<usize>,
    device: Device,
    cache: EitherCache,
    dtype: DType,
    max_seq_len: usize,
    xlora_classifier: Option<XLoraClassifier>,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    cfg: ModelConfigMetadata,
}

impl XLoraModel {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Self> {
        if let Some(ref quant_cfg) = &cfg.quantization_config {
            tracing::info!(
                "Using {} quantization: {}.",
                quant_cfg.name(),
                quant_cfg.get_bits_name(&vb)
            );
        }
        let mapper = normal_loading_metadata.mapper;
        let vb_m = vb.pp("model");

        let embed_tokens = layers::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), false),
            &cfg.quantization_config,
        )?;
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        let mut ropes = HashMap::new();
        for layer_idx in 0..cfg.num_hidden_layers {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            ropes.insert(
                device.location(),
                Arc::new(RotaryEmbedding::new(
                    cfg.rope_theta as f32,
                    head_dim,
                    cfg.max_position_embeddings,
                    device,
                    is_gptx,
                    vb_m.dtype(),
                )?),
            );
        }

        let mut count = 0;
        for layer_idx in NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        ) {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let rotary_emb = ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();
            let layer = DecoderLayer::new(
                rotary_emb.clone(),
                cfg,
                vb_l.pp(layer_idx),
                lora_config,
                &mut count,
                &xlora_ordering,
                &*mapper,
                layer_idx,
                normal_loading_metadata.loading_isq,
                preload_adapters,
            )?;
            layers.push(layer)
        }
        if xlora_config.is_none() && preload_adapters.is_none() {
            // We are now a LoRA model so we must merge the weights
            info!("Merging LoRA adapters.");
            for layer in layers.iter_mut().tqdm() {
                Arc::get_mut(&mut layer.self_attn.k_proj)
                    .unwrap()
                    .merge_weights()?;
                Arc::get_mut(&mut layer.self_attn.o_proj)
                    .unwrap()
                    .merge_weights()?;
                Arc::get_mut(&mut layer.self_attn.q_proj)
                    .unwrap()
                    .merge_weights()?;
                Arc::get_mut(&mut layer.self_attn.v_proj)
                    .unwrap()
                    .merge_weights()?;

                Arc::get_mut(&mut layer.block_sparse_moe.gate)
                    .unwrap()
                    .merge_weights()?;
                for expert in layer.block_sparse_moe.experts.iter_mut() {
                    Arc::get_mut(&mut expert.w1).unwrap().merge_weights()?;
                    Arc::get_mut(&mut expert.w2).unwrap().merge_weights()?;
                    Arc::get_mut(&mut expert.w3).unwrap().merge_weights()?;
                }
            }
        }
        let norm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;
        let lm_head = linear_no_bias(
            cfg.hidden_size,
            cfg.vocab_size,
            mapper.set_nm_device(vb.pp("lm_head"), normal_loading_metadata.loading_isq),
            mapper.set_nm_device(vb.pp("lm_head"), false),
            lora_config,
            &mut count,
            &xlora_ordering,
            preload_adapters,
        )?;
        if xlora_config.is_some() && lm_head.is_lora() {
            // This is why we can pass dummy values (..., None, 1.0, None)?
            candle_core::bail!("Got an adapter `lm_head` layer, this is unsupported with X-LoRA.");
        }
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            sliding_window: cfg.sliding_window,
            device: normal_loading_metadata.real_device,
            dtype: vb.dtype(),
            cache: EitherCache::Full(Cache::new(cfg.num_hidden_layers, false)),
            max_seq_len: cfg.max_position_embeddings,
            xlora_classifier: xlora_config.map(|xlora_config| {
                XLoraClassifier::new(xlora_config, count, lora_config.len(), vb, false).unwrap()
            }),
            mapper,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_kv_heads: cfg.num_key_value_heads,
                num_attn_heads: cfg.num_attention_heads,
                sliding_window: cfg.sliding_window,
                k_head_dim: cfg.hidden_size / cfg.num_attention_heads,
                v_head_dim: cfg.hidden_size / cfg.num_attention_heads,
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn inner_forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        scalings: Option<Tensor>,
        is_full_pass: bool,
        no_kv_cache: bool,
        is_scaling_pass: Option<f64>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let mut cache = if is_full_pass {
            if no_kv_cache {
                let mut new_cache = Vec::new();
                for _ in 0..self.cache.full().xlora_lock().len() {
                    new_cache.push(None);
                }

                self.cache.full().xlora_lock().clone_from(&new_cache);
            }
            self.cache.full().xlora_lock()
        } else {
            self.cache.full().lock()
        };
        let mut xs = self.embed_tokens.forward(input_ids)?;
        let attention_mask = CausalMasker.make_sliding_window_causal_mask_matrix(
            input_ids,
            &*cache,
            self.sliding_window,
            xs.dtype(),
            self.cfg.num_attn_heads,
        )?;
        let attention_mask = DeviceMappedMask::new(attention_mask, &*self.mapper)?;
        for (i, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;
            xs = layer.forward(
                &xs,
                attention_mask.as_ref().map(|m| m.get(xs.device())),
                seqlen_offsets,
                &mut cache[i],
                scalings.clone(),
                self.xlora_classifier
                    .as_ref()
                    .map(|classifier| classifier.get_global_scaling_weight())
                    .unwrap_or(1.0),
                is_scaling_pass,
                flash_params,
            )?
        }
        let xs = xs.to_device(&self.device)?;
        xs.apply(&self.norm)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        input_ids: &Tensor,
        input_ids_full: &Tensor,
        seqlen_offsets: &[usize],
        seqlen_offsets_full: &[usize],
        no_kv_cache: bool,
        non_granular_state: &Option<NonGranularState>,
        context_lens: Vec<(usize, usize)>,
        flash_params: &FlashParams,
        flash_params_full: &FlashParams,
    ) -> Result<Tensor> {
        if self.xlora_classifier.is_some() {
            let scalings = self.get_scalings(
                input_ids,
                input_ids_full,
                seqlen_offsets,
                seqlen_offsets_full,
                no_kv_cache,
                non_granular_state,
                &vec![usize::MAX; context_lens.len()],
                flash_params,
                flash_params_full,
            )?;

            if no_kv_cache {
                let res = self
                    .inner_forward(
                        input_ids_full,
                        seqlen_offsets_full,
                        Some(scalings),
                        true,
                        no_kv_cache,
                        None,
                        flash_params_full,
                    )?
                    .contiguous()?;
                let mut res = extract_logits(&res, context_lens)?;
                if let Some(t) = self.lm_head.quantized_act_type() {
                    res = res.to_dtype(t)?;
                }
                self.lm_head.lora_forward(&res, None, 1.0, None)
            } else {
                // is_full_pass=true is ok because no_kv_cache=false
                let res = self
                    .inner_forward(
                        input_ids,
                        seqlen_offsets,
                        Some(scalings),
                        true,
                        no_kv_cache,
                        None,
                        flash_params,
                    )?
                    .contiguous()?;
                let mut res = extract_logits(&res, context_lens)?;
                if let Some(t) = self.lm_head.quantized_act_type() {
                    res = res.to_dtype(t)?;
                }
                self.lm_head.lora_forward(&res, None, 1.0, None)
            }
        } else {
            let res = self
                .inner_forward(
                    input_ids,
                    seqlen_offsets,
                    None,
                    false,
                    no_kv_cache,
                    None,
                    flash_params,
                )?
                .contiguous()?;
            let mut res = extract_logits(&res, context_lens)?;
            if let Some(t) = self.lm_head.quantized_act_type() {
                res = res.to_dtype(t)?;
            }
            self.lm_head.lora_forward(&res, None, 1.0, None)
        }
    }
}

impl IsqModel for XLoraModel {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        let mut tensors = Vec::new();
        tensors.push((Arc::get_mut(&mut self.lm_head).unwrap().quant_inner(), None));
        for (i, layer) in self.layers.iter_mut().enumerate() {
            tensors.push((
                Arc::get_mut(&mut layer.self_attn.q_proj)
                    .unwrap()
                    .quant_inner(),
                Some(i),
            ));
            tensors.push((
                Arc::get_mut(&mut layer.self_attn.k_proj)
                    .unwrap()
                    .quant_inner(),
                Some(i),
            ));
            tensors.push((
                Arc::get_mut(&mut layer.self_attn.v_proj)
                    .unwrap()
                    .quant_inner(),
                Some(i),
            ));
            tensors.push((
                Arc::get_mut(&mut layer.self_attn.o_proj)
                    .unwrap()
                    .quant_inner(),
                Some(i),
            ));
            tensors.push((
                Arc::get_mut(&mut layer.block_sparse_moe.gate)
                    .unwrap()
                    .quant_inner(),
                Some(i),
            ));
            for expert in &mut layer.block_sparse_moe.experts {
                tensors.push((Arc::get_mut(&mut expert.w1).unwrap().quant_inner(), Some(i)));
                tensors.push((Arc::get_mut(&mut expert.w2).unwrap().quant_inner(), Some(i)));
                tensors.push((Arc::get_mut(&mut expert.w3).unwrap().quant_inner(), Some(i)));
            }
        }
        (tensors, &*self.mapper)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        panic!("Cannot generate UQFF for an adapter model.")
    }
}

impl NormalModel for XLoraModel {
    fn forward(
        &self,
        _input_ids: &Tensor,
        _seqlen_offsets: &[usize],
        _context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        _metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        _flash_params: &FlashParams,
    ) -> Result<Tensor> {
        unreachable!()
    }
    fn xlora_forward(
        &self,
        input_ids: &Tensor,
        input_ids_full: &Tensor,
        seqlen_offsets: &[usize],
        seqlen_offsets_full: &[usize],
        no_kv_cache: bool,
        non_granular_state: &Option<crate::xlora_models::NonGranularState>,
        context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        flash_params: &FlashParams,
        flash_params_full: &FlashParams,
    ) -> Result<Tensor> {
        self.forward(
            input_ids,
            input_ids_full,
            seqlen_offsets,
            seqlen_offsets_full,
            no_kv_cache,
            non_granular_state,
            context_lens,
            flash_params,
            flash_params_full,
        )
    }
    fn cache(&self) -> &EitherCache {
        &self.cache
    }
    fn cache_mut(&mut self) -> &mut EitherCache {
        &mut self.cache
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn is_xlora(&self) -> bool {
        true
    }
    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    fn config(&self) -> &ModelConfigMetadata {
        &self.cfg
    }
}

impl ScalingsMaker for XLoraModel {
    fn dtype(&self) -> DType {
        self.dtype
    }
    fn get_cache(&self) -> &EitherCache {
        &self.cache
    }
    fn get_classifier(&self) -> &XLoraClassifier {
        self.xlora_classifier.as_ref().unwrap()
    }
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        scalings: Tensor,
        is_full_pass: bool,
        no_kv_cache: bool,
        is_scaling_pass: Option<f64>,
        _context_lens: &[usize],
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        self.inner_forward(
            input_ids,
            seqlen_offsets,
            Some(scalings),
            is_full_pass,
            no_kv_cache,
            is_scaling_pass,
            flash_params,
        )
    }
}

impl AnyMoeBaseModelMixin for XLoraModel {}
