#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::SdpaParams,
    layers::{Llama3RotaryEmbedding, Sdpa},
    lora::{linear_no_bias as linear, LinearLayerLike, LoraConfig, Ordering},
    paged_attention::ModelConfigMetadata,
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel,
    },
    utils::progress::NiceProgressBar,
};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, Module};
use mistralrs_quant::{QuantMethod, ShardedVarBuilder};
use std::{collections::HashMap, sync::Arc};
use tqdm::Iter;
use tracing::info;

use crate::{
    device_map::{DeviceMappedMask, DeviceMapper},
    layers::{embedding, CausalMasker, RmsNorm},
    models::llama::Config,
    pipeline::{self, extract_logits, LayerCaches, NormalLoadingMetadata, NormalModel},
};

use super::{classifier::XLoraClassifier, NonGranularState, ScalingsMaker, XLoraConfig};

struct CausalSelfAttention {
    q_proj: Arc<dyn LinearLayerLike + Send + Sync>,
    k_proj: Arc<dyn LinearLayerLike + Send + Sync>,
    v_proj: Arc<dyn LinearLayerLike + Send + Sync>,
    o_proj: Arc<dyn LinearLayerLike + Send + Sync>,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<Llama3RotaryEmbedding>,
    max_seq_len: usize,
    sdpa_params: SdpaParams,
}

impl CausalSelfAttention {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        mask: &Option<Tensor>,
        seqlen_offsets: &[usize],
        block_idx: usize,
        kv_cache: &mut LayerCaches,
        scalings: Option<Tensor>,
        global_scaling_weight: f64,
        is_scaling_pass: Option<f64>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, hidden_size) = x.dims3()?;

        let original_dtype = x.dtype();
        let mut x = x.clone();
        if let Some(t) = self.q_proj.quantized_act_type() {
            x = x.to_dtype(t)?;
        }
        let mut q = self.q_proj.lora_forward(
            &x,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?;
        let mut k = self.k_proj.lora_forward(
            &x,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?;
        let mut v = self.v_proj.lora_forward(
            &x,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?;
        if self.q_proj.quantized_act_type().is_some() {
            q = q.to_dtype(original_dtype)?;
            k = k.to_dtype(original_dtype)?;
            v = v.to_dtype(original_dtype)?;
        }

        let (q, k, v) = if seq_len != 1 {
            let q = q
                .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
                .transpose(1, 2)?;
            let k = k
                .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        } else {
            let q = q.reshape((b_sz, self.num_attention_heads, seq_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.num_key_value_heads, seq_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.num_key_value_heads, seq_len, self.head_dim))?;
            (q, k, v)
        };

        let (q, k) = self.rotary_emb.forward(&q, &k, seqlen_offsets)?;

        let (k, v) = crate::pipeline::Cache::update_kv_cache(&mut kv_cache[block_idx], k, v)?;

        let y = Sdpa.run_attention(
            &q,
            &k,
            &v,
            mask.clone().as_ref(),
            Some(flash_params),
            &self.sdpa_params,
        )?;

        let mut y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, hidden_size])?;
        if let Some(t) = self.q_proj.quantized_act_type() {
            y = y.to_dtype(t)?;
        }
        let mut res = self.o_proj.lora_forward(
            &y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?;
        if self.q_proj.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }

    #[allow(clippy::too_many_arguments)]
    fn load(
        vb: ShardedVarBuilder,
        cfg: &Config,
        lora_config: &[((String, String), LoraConfig)],
        count: &mut usize,
        ord: &Ordering,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        rope: Arc<Llama3RotaryEmbedding>,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Self> {
        let size_in = cfg.hidden_size;
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
        let q_proj = linear(
            size_in,
            size_q,
            mapper.set_device(layer_idx, vb.pp("q_proj"), loading_isq),
            mapper.set_device(layer_idx, vb.pp("q_proj"), false),
            lora_config,
            count,
            ord,
            preload_adapters,
        )?;
        let k_proj = linear(
            size_in,
            size_kv,
            mapper.set_device(layer_idx, vb.pp("k_proj"), loading_isq),
            mapper.set_device(layer_idx, vb.pp("k_proj"), false),
            lora_config,
            count,
            ord,
            preload_adapters,
        )?;
        let v_proj = linear(
            size_in,
            size_kv,
            mapper.set_device(layer_idx, vb.pp("v_proj"), loading_isq),
            mapper.set_device(layer_idx, vb.pp("v_proj"), false),
            lora_config,
            count,
            ord,
            preload_adapters,
        )?;
        let o_proj = linear(
            size_q,
            size_in,
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
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            rotary_emb: rope,
            max_seq_len: cfg.max_position_embeddings,
            sdpa_params: SdpaParams {
                n_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
                softcap: None,
                softmax_scale: 1.0 / ((cfg.hidden_size / cfg.num_attention_heads) as f32).sqrt(),
                sliding_window: None,
                sinks: None,
            },
        })
    }
}

#[derive(Clone)]
struct Mlp {
    c_fc1: Arc<dyn LinearLayerLike + Send + Sync>,
    c_fc2: Arc<dyn LinearLayerLike + Send + Sync>,
    c_proj: Arc<dyn LinearLayerLike + Send + Sync>,
}

impl Mlp {
    fn forward(
        &self,
        x: &Tensor,
        scalings: Option<Tensor>,
        global_scaling_weight: f64,
        is_scaling_pass: Option<f64>,
    ) -> Result<Tensor> {
        let original_dtype = x.dtype();
        let mut x = x.clone();
        if let Some(t) = self.c_fc1.quantized_act_type() {
            x = x.to_dtype(t)?;
        }
        let x = (candle_nn::ops::silu(&self.c_fc1.lora_forward(
            &x,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?)? * self.c_fc2.lora_forward(
            &x,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?)?;
        let mut res = self.c_proj.lora_forward(
            &x,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?;
        if self.c_fc1.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }

    #[allow(clippy::too_many_arguments)]
    fn load(
        vb: ShardedVarBuilder,
        cfg: &Config,
        lora_config: &[((String, String), LoraConfig)],
        count: &mut usize,
        ord: &Ordering,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Self> {
        let h_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let c_fc1 = linear(
            h_size,
            i_size,
            mapper.set_device(layer_idx, vb.pp("gate_proj"), loading_isq),
            mapper.set_device(layer_idx, vb.pp("gate_proj"), false),
            lora_config,
            count,
            ord,
            preload_adapters,
        )?;
        let c_fc2 = linear(
            h_size,
            i_size,
            mapper.set_device(layer_idx, vb.pp("up_proj"), loading_isq),
            mapper.set_device(layer_idx, vb.pp("up_proj"), false),
            lora_config,
            count,
            ord,
            preload_adapters,
        )?;
        let c_proj = linear(
            i_size,
            h_size,
            mapper.set_device(layer_idx, vb.pp("down_proj"), loading_isq),
            mapper.set_device(layer_idx, vb.pp("down_proj"), false),
            lora_config,
            count,
            ord,
            preload_adapters,
        )?;
        Ok(Self {
            c_fc1,
            c_fc2,
            c_proj,
        })
    }
}

struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    mlp: Mlp,
}

impl Block {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        mask: &Option<Tensor>,
        seqlen_offsets: &[usize],
        block_idx: usize,
        kv_cache: &mut LayerCaches,
        scalings: Option<Tensor>,
        global_scaling_weight: f64,
        is_scaling_pass: Option<f64>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let x = (self.attn.forward(
            &x,
            mask,
            seqlen_offsets,
            block_idx,
            kv_cache,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
            flash_params,
        )? + residual)?;
        let residual = &x;
        let x = (self.mlp.forward(
            &self.rms_2.forward(&x)?,
            scalings,
            global_scaling_weight,
            is_scaling_pass,
        )? + residual)?;
        Ok(x)
    }

    #[allow(clippy::too_many_arguments)]
    fn load(
        vb: ShardedVarBuilder,
        cfg: &Config,
        lora_config: &[((String, String), LoraConfig)],
        count: &mut usize,
        ord: &Ordering,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        rope: Arc<Llama3RotaryEmbedding>,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Self> {
        let attn = CausalSelfAttention::load(
            vb.pp("self_attn"),
            cfg,
            lora_config,
            count,
            ord,
            mapper,
            layer_idx,
            loading_isq,
            rope,
            preload_adapters,
        )?;
        let mlp = Mlp::load(
            vb.pp("mlp"),
            cfg,
            lora_config,
            count,
            ord,
            mapper,
            layer_idx,
            loading_isq,
            preload_adapters,
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
}

pub struct XLoraLlama {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: Arc<dyn LinearLayerLike + Send + Sync>,
    kv_cache: pipeline::EitherCache,
    device: Device,
    xlora_classifier: Option<XLoraClassifier>,
    dtype: DType,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    cfg: ModelConfigMetadata,
}

impl XLoraLlama {
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
        let mut x = self.wte.forward(input_ids)?;
        let mut cache = if is_full_pass {
            if no_kv_cache {
                let mut new_cache = Vec::new();
                for _ in 0..self.kv_cache.full().xlora_lock().len() {
                    new_cache.push(None);
                }

                self.kv_cache.full().xlora_lock().clone_from(&new_cache);
            }
            self.kv_cache.full().xlora_lock()
        } else {
            self.kv_cache.full().lock()
        };
        let mask = CausalMasker.make_causal_mask_matrix(
            input_ids,
            &*cache,
            x.dtype(),
            self.cfg.num_attn_heads,
        )?;
        let mask = DeviceMappedMask::new(mask, &*self.mapper)?;
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = self.mapper.map(x, block_idx)?;
            x = block.forward(
                &x,
                &mask.as_ref().map(|m| m.get(x.device()).clone()),
                seqlen_offsets,
                block_idx,
                &mut cache,
                scalings.clone(),
                self.xlora_classifier
                    .as_ref()
                    .map(|classifier| classifier.get_global_scaling_weight())
                    .unwrap_or(1.0),
                is_scaling_pass,
                flash_params,
            )?;
        }
        let x = x.to_device(&self.device)?;
        self.ln_f.forward(&x)
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
        let dtype = vb.dtype();
        let mut count = 0;

        let wte = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb.pp("model.embed_tokens"), false),
            &cfg.quantization_config,
        )?;
        let lm_head = linear(
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
        let ln_f = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb.pp("model.norm"), false),
        )?;
        let mut ropes = HashMap::new();
        for i in 0..cfg.num_hidden_layers {
            let device = mapper
                .device_for(i, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            ropes.insert(
                device.location(),
                Arc::new(Llama3RotaryEmbedding::new_llama3(
                    vb.dtype(),
                    cfg,
                    device,
                    is_gptx,
                )?),
            );
        }
        let mut blocks: Vec<_> = NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .into_iter()
        .map(|i| {
            let device = mapper
                .device_for(i, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let rotary_emb = ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();
            Block::load(
                vb.pp(format!("model.layers.{i}")),
                cfg,
                lora_config,
                &mut count,
                &xlora_ordering,
                &*mapper,
                i,
                normal_loading_metadata.loading_isq,
                rotary_emb,
                preload_adapters,
            )
            .expect("Failed to load block.")
        })
        .collect();
        if xlora_config.is_none() && preload_adapters.is_none() {
            // We are now a LoRA model so we must merge the weights
            info!("Merging LoRA adapters.");
            for layer in blocks.iter_mut().tqdm() {
                Arc::get_mut(&mut layer.attn.k_proj)
                    .unwrap()
                    .merge_weights()?;
                Arc::get_mut(&mut layer.attn.o_proj)
                    .unwrap()
                    .merge_weights()?;
                Arc::get_mut(&mut layer.attn.q_proj)
                    .unwrap()
                    .merge_weights()?;
                Arc::get_mut(&mut layer.attn.v_proj)
                    .unwrap()
                    .merge_weights()?;

                Arc::get_mut(&mut layer.mlp.c_fc1)
                    .unwrap()
                    .merge_weights()?;
                Arc::get_mut(&mut layer.mlp.c_fc2)
                    .unwrap()
                    .merge_weights()?;
                Arc::get_mut(&mut layer.mlp.c_proj)
                    .unwrap()
                    .merge_weights()?;
            }
        }

        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
            kv_cache: EitherCache::Full(pipeline::Cache::new(cfg.num_hidden_layers, true)),
            device: normal_loading_metadata.real_device,
            xlora_classifier: xlora_config.map(|xlora_config| {
                XLoraClassifier::new(xlora_config, count, lora_config.len(), vb, false).unwrap()
            }),
            dtype,
            mapper,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_kv_heads: cfg.num_key_value_heads,
                num_attn_heads: cfg.num_attention_heads,
                sliding_window: None,
                k_head_dim: cfg.hidden_size / cfg.num_attention_heads,
                v_head_dim: cfg.hidden_size / cfg.num_attention_heads,
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
        })
    }
}

impl IsqModel for XLoraLlama {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        let mut tensors = Vec::new();
        tensors.push((Arc::get_mut(&mut self.lm_head).unwrap().quant_inner(), None));
        for (i, layer) in self.blocks.iter_mut().enumerate() {
            tensors.push((
                Arc::get_mut(&mut layer.attn.q_proj).unwrap().quant_inner(),
                Some(i),
            ));
            tensors.push((
                Arc::get_mut(&mut layer.attn.k_proj).unwrap().quant_inner(),
                Some(i),
            ));
            tensors.push((
                Arc::get_mut(&mut layer.attn.v_proj).unwrap().quant_inner(),
                Some(i),
            ));
            tensors.push((
                Arc::get_mut(&mut layer.attn.o_proj).unwrap().quant_inner(),
                Some(i),
            ));
            tensors.push((
                Arc::get_mut(&mut layer.mlp.c_fc1).unwrap().quant_inner(),
                Some(i),
            ));
            tensors.push((
                Arc::get_mut(&mut layer.mlp.c_fc2).unwrap().quant_inner(),
                Some(i),
            ));
            tensors.push((
                Arc::get_mut(&mut layer.mlp.c_proj).unwrap().quant_inner(),
                Some(i),
            ));
        }
        (tensors, &*self.mapper)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        panic!("Cannot generate UQFF for an adapter model.")
    }
}

impl NormalModel for XLoraLlama {
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
    fn cache(&self) -> &super::EitherCache {
        &self.kv_cache
    }
    fn cache_mut(&mut self) -> &mut super::EitherCache {
        &mut self.kv_cache
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn is_xlora(&self) -> bool {
        true
    }
    fn max_seq_len(&self) -> usize {
        self.blocks[0].attn.max_seq_len
    }
    fn config(&self) -> &ModelConfigMetadata {
        &self.cfg
    }
}

impl ScalingsMaker for XLoraLlama {
    fn dtype(&self) -> DType {
        self.dtype
    }
    fn get_cache(&self) -> &pipeline::EitherCache {
        &self.kv_cache
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

impl AnyMoeBaseModelMixin for XLoraLlama {}
