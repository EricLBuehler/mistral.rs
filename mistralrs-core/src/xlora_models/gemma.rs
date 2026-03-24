#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{collections::HashMap, sync::Arc};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::SdpaParams,
    layers::{self, Activation, GemmaRmsNorm, RotaryEmbedding, Sdpa},
    lora::{linear_b as linear, LinearLayerLike, LoraConfig, Ordering},
    paged_attention::ModelConfigMetadata,
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, NormalLoadingMetadata,
    },
    utils::progress::NiceProgressBar,
};
use candle_core::{DType, Device, Module, Result, Tensor};
use mistralrs_quant::{QuantMethod, ShardedVarBuilder};
use tqdm::Iter;
use tracing::info;

use crate::{
    device_map::{DeviceMappedMask, DeviceMapper},
    layers::CausalMasker,
    models::gemma::Config,
    pipeline::{extract_logits, Cache, NormalModel},
};

use super::{classifier::XLoraClassifier, NonGranularState, ScalingsMaker, XLoraConfig};

fn default_max_position_embeddings() -> usize {
    4096
}

#[derive(Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    gate_proj: Arc<dyn LinearLayerLike + Send + Sync>,
    up_proj: Arc<dyn LinearLayerLike + Send + Sync>,
    down_proj: Arc<dyn LinearLayerLike + Send + Sync>,
    act_fn: Activation,
}

impl MLP {
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
        let gate_proj = linear(
            hidden_sz,
            intermediate_sz,
            false,
            mapper.set_device(layer_idx, vb.pp("gate_proj"), loading_isq),
            mapper.set_device(layer_idx, vb.pp("gate_proj"), false),
            lora_config,
            count,
            ord,
            preload_adapters,
        )?;
        let up_proj = linear(
            hidden_sz,
            intermediate_sz,
            false,
            mapper.set_device(layer_idx, vb.pp("up_proj"), loading_isq),
            mapper.set_device(layer_idx, vb.pp("up_proj"), false),
            lora_config,
            count,
            ord,
            preload_adapters,
        )?;
        let down_proj = linear(
            intermediate_sz,
            hidden_sz,
            false,
            mapper.set_device(layer_idx, vb.pp("down_proj"), loading_isq),
            mapper.set_device(layer_idx, vb.pp("down_proj"), false),
            lora_config,
            count,
            ord,
            preload_adapters,
        )?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act()?,
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
        if let Some(t) = self.gate_proj.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        let lhs = self
            .gate_proj
            .lora_forward(
                &xs,
                scalings.clone(),
                global_scaling_weight,
                is_scaling_pass,
            )?
            .apply(&self.act_fn)?;
        let rhs = self.up_proj.lora_forward(
            &xs,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?;
        let mut res = self.down_proj.lora_forward(
            &(lhs * rhs)?,
            scalings,
            global_scaling_weight,
            is_scaling_pass,
        )?;
        if self.gate_proj.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

struct Attention {
    q_proj: Arc<dyn LinearLayerLike + Send + Sync>,
    k_proj: Arc<dyn LinearLayerLike + Send + Sync>,
    v_proj: Arc<dyn LinearLayerLike + Send + Sync>,
    o_proj: Arc<dyn LinearLayerLike + Send + Sync>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
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
        let head_dim = cfg.head_dim;
        let bias = cfg.attention_bias;
        let q_proj = linear(
            hidden_sz,
            num_heads * head_dim,
            bias,
            mapper.set_device(layer_idx, vb.pp("q_proj"), loading_isq),
            mapper.set_device(layer_idx, vb.pp("q_proj"), false),
            lora_config,
            count,
            ord,
            preload_adapters,
        )?;
        let k_proj = linear(
            hidden_sz,
            num_kv_heads * head_dim,
            bias,
            mapper.set_device(layer_idx, vb.pp("k_proj"), loading_isq),
            mapper.set_device(layer_idx, vb.pp("k_proj"), false),
            lora_config,
            count,
            ord,
            preload_adapters,
        )?;
        let v_proj = linear(
            hidden_sz,
            num_kv_heads * head_dim,
            bias,
            mapper.set_device(layer_idx, vb.pp("v_proj"), loading_isq),
            mapper.set_device(layer_idx, vb.pp("v_proj"), false),
            lora_config,
            count,
            ord,
            preload_adapters,
        )?;
        let o_proj = linear(
            num_heads * head_dim,
            hidden_sz,
            bias,
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
            sdpa_params: SdpaParams {
                n_kv_groups: num_heads / num_kv_heads,
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window: None,
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

        let (k, v) = Cache::update_kv_cache(kv_cache, k, v)?;

        let mut attn_output = Sdpa.run_attention(
            &q,
            &k,
            &v,
            attention_mask,
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

struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: GemmaRmsNorm,
    post_attention_layernorm: GemmaRmsNorm,
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
        let mlp = MLP::new(
            cfg,
            vb.pp("mlp"),
            lora_config,
            count,
            ord,
            mapper,
            layer_idx,
            loading_isq,
            preload_adapters,
        )?;
        let input_layernorm = GemmaRmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("input_layernorm"), false),
        )?;
        let post_attention_layernorm = GemmaRmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("post_attention_layernorm"), false),
        )?;
        Ok(Self {
            self_attn,
            mlp,
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
        let xs = self.mlp.forward(
            &xs.apply(&self.post_attention_layernorm)?,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?;
        residual + xs
    }
}

pub struct XLoraModel {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: GemmaRmsNorm,
    lm_head: Arc<dyn LinearLayerLike + Send + Sync>,
    dtype: DType,
    hidden_size: usize,
    device: Device,
    cache: EitherCache,
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
                    cfg.head_dim,
                    cfg.max_position_embeddings,
                    device,
                    is_gptx,
                    vb.dtype(),
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

                Arc::get_mut(&mut layer.mlp.down_proj)
                    .unwrap()
                    .merge_weights()?;
                Arc::get_mut(&mut layer.mlp.gate_proj)
                    .unwrap()
                    .merge_weights()?;
                Arc::get_mut(&mut layer.mlp.up_proj)
                    .unwrap()
                    .merge_weights()?;
            }
        }
        let norm = GemmaRmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;
        let lm_head = linear(
            embed_tokens.embeddings().dim(1)?,
            embed_tokens.embeddings().dim(0)?,
            false,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), normal_loading_metadata.loading_isq),
            mapper.set_nm_device(vb_m.pp("embed_tokens"), false),
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
            device: normal_loading_metadata.real_device,
            dtype: vb.dtype(),
            hidden_size: cfg.hidden_size,
            cache: EitherCache::Full(Cache::new(cfg.num_hidden_layers, true)),
            max_seq_len: default_max_position_embeddings(),
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
                sliding_window: None,
                k_head_dim: cfg.head_dim,
                v_head_dim: cfg.head_dim,
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
        let xs = self.embed_tokens.forward(input_ids)?;
        let attention_mask = CausalMasker.make_causal_mask_matrix(
            input_ids,
            &*cache,
            xs.dtype(),
            self.cfg.num_attn_heads,
        )?;
        let attention_mask = DeviceMappedMask::new(attention_mask, &*self.mapper)?;
        let mut xs = (xs * (self.hidden_size as f64).sqrt())?;
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
                Arc::get_mut(&mut layer.mlp.gate_proj)
                    .unwrap()
                    .quant_inner(),
                Some(i),
            ));
            tensors.push((
                Arc::get_mut(&mut layer.mlp.up_proj).unwrap().quant_inner(),
                Some(i),
            ));
            tensors.push((
                Arc::get_mut(&mut layer.mlp.down_proj)
                    .unwrap()
                    .quant_inner(),
                Some(i),
            ));
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
