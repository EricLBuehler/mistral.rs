#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{collections::HashMap, sync::Arc};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::SdpaParams,
    layers::{Activation, RotaryEmbedding, Sdpa},
    lora::{linear, LinearLayerLike, LoraConfig, Ordering},
    paged_attention::ModelConfigMetadata,
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, NormalLoadingMetadata,
    },
    utils::progress::NiceProgressBar,
};
/// Phi model.
/// https://huggingface.co/microsoft/phi-2
/// There is an alternative implementation of the phi model in mixformers.rs.
/// This corresponds to the model update made with the following commit:
/// https://huggingface.co/microsoft/phi-2/commit/cb2f4533604d8b67de604e7df03bfe6f3ca22869
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, LayerNorm};
use mistralrs_quant::{QuantMethod, ShardedVarBuilder};
use tqdm::Iter;
use tracing::info;

use crate::{
    device_map::{DeviceMappedMask, DeviceMapper},
    layers::{embedding, layer_norm, CausalMasker},
    models::phi2::Config,
    pipeline::{extract_logits, NormalModel},
};

use super::{classifier::XLoraClassifier, Cache, NonGranularState, ScalingsMaker, XLoraConfig};

#[derive(Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    fc1: Arc<dyn LinearLayerLike + Send + Sync>,
    fc2: Arc<dyn LinearLayerLike + Send + Sync>,
    act: Activation,
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
        let fc1 = linear(
            cfg.hidden_size,
            cfg.intermediate_size,
            mapper.set_device(layer_idx, vb.pp("fc1"), loading_isq),
            mapper.set_device(layer_idx, vb.pp("fc1"), false),
            lora_config,
            count,
            ord,
            preload_adapters,
        )?;
        let fc2 = linear(
            cfg.intermediate_size,
            cfg.hidden_size,
            mapper.set_device(layer_idx, vb.pp("fc2"), loading_isq),
            mapper.set_device(layer_idx, vb.pp("fc2"), false),
            lora_config,
            count,
            ord,
            preload_adapters,
        )?;
        Ok(Self {
            fc1,
            fc2,
            // This does not match the mixformers implementation where Gelu is used rather than
            // GeluNew.
            act: cfg.hidden_act,
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
        if let Some(t) = self.fc1.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        let mut res = self.fc2.lora_forward(
            &self
                .fc1
                .lora_forward(
                    &xs,
                    scalings.clone(),
                    global_scaling_weight,
                    is_scaling_pass,
                )?
                .apply(&self.act)?,
            scalings,
            global_scaling_weight,
            is_scaling_pass,
        )?;
        if self.fc1.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

struct Attention {
    q_proj: Arc<dyn LinearLayerLike + Send + Sync>,
    k_proj: Arc<dyn LinearLayerLike + Send + Sync>,
    v_proj: Arc<dyn LinearLayerLike + Send + Sync>,
    dense: Arc<dyn LinearLayerLike + Send + Sync>,
    q_layernorm: Option<LayerNorm>,
    k_layernorm: Option<LayerNorm>,
    rotary_emb: Arc<RotaryEmbedding>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sdpa_params: SdpaParams,
}

impl Attention {
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
        rope: Arc<RotaryEmbedding>,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads();
        let head_dim = cfg.head_dim();
        let q_proj = linear(
            cfg.hidden_size,
            num_heads * head_dim,
            mapper.set_device(layer_idx, vb.pp("q_proj"), loading_isq),
            mapper.set_device(layer_idx, vb.pp("q_proj"), false),
            lora_config,
            count,
            ord,
            preload_adapters,
        )?;
        let k_proj = linear(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            mapper.set_device(layer_idx, vb.pp("k_proj"), loading_isq),
            mapper.set_device(layer_idx, vb.pp("k_proj"), false),
            lora_config,
            count,
            ord,
            preload_adapters,
        )?;
        let v_proj = linear(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            mapper.set_device(layer_idx, vb.pp("v_proj"), loading_isq),
            mapper.set_device(layer_idx, vb.pp("v_proj"), false),
            lora_config,
            count,
            ord,
            preload_adapters,
        )?;
        let dense = linear(
            num_heads * head_dim,
            cfg.hidden_size,
            mapper.set_device(layer_idx, vb.pp("dense"), loading_isq),
            mapper.set_device(layer_idx, vb.pp("dense"), false),
            lora_config,
            count,
            ord,
            preload_adapters,
        )?;
        let (q_layernorm, k_layernorm) = if cfg.qk_layernorm {
            let q_layernorm = layer_norm(head_dim, cfg.layer_norm_eps, vb.pp("q_layernorm"))?;
            let k_layernorm = layer_norm(head_dim, cfg.layer_norm_eps, vb.pp("k_layernorm"))?;
            (Some(q_layernorm), Some(k_layernorm))
        } else {
            (None, None)
        };
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            dense,
            q_layernorm,
            k_layernorm,
            rotary_emb: rope,
            num_heads,
            num_kv_heads,
            head_dim,
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
        mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut Option<(Tensor, Tensor)>,
        scalings: Option<Tensor>,
        global_scaling_weight: f64,
        is_scaling_pass: Option<f64>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let (b_size, seq_len, _n_embd) = xs.dims3()?;
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

        let q = match &self.q_layernorm {
            None => q,
            Some(ln) => q.apply(ln)?,
        };
        let k = match &self.k_layernorm {
            None => k,
            Some(ln) => k.apply(ln)?,
        };

        let (q, k, v) = if seq_len != 1 {
            let q = q
                .reshape((b_size, seq_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?;
            let k = k
                .reshape((b_size, seq_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_size, seq_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        } else {
            let q = q.reshape((b_size, self.num_heads, seq_len, self.head_dim))?;
            let k = k.reshape((b_size, self.num_kv_heads, seq_len, self.head_dim))?;
            let v = v.reshape((b_size, self.num_kv_heads, seq_len, self.head_dim))?;
            (q, k, v)
        };

        let (q, k) = self.rotary_emb.forward(&q, &k, seqlen_offsets)?;

        let (k, v) = Cache::update_kv_cache(kv_cache, k, v)?;

        let attn_output =
            Sdpa.run_attention(&q, &k, &v, mask, Some(flash_params), &self.sdpa_params)?;

        let mut attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_size, seq_len, ()))?;
        if let Some(t) = self.q_proj.quantized_act_type() {
            attn_output = attn_output.to_dtype(t)?;
        }
        let mut res = self.dense.lora_forward(
            &attn_output,
            scalings,
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
    input_layernorm: LayerNorm,
}

impl DecoderLayer {
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
        rope: Arc<RotaryEmbedding>,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Self> {
        let self_attn = Attention::new(
            cfg,
            vb.pp("self_attn"),
            lora_config,
            count,
            ord,
            mapper,
            layer_idx,
            loading_isq,
            rope,
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
        let input_layernorm = layer_norm(
            cfg.hidden_size,
            cfg.layer_norm_eps,
            mapper.set_device(layer_idx, vb.pp("input_layernorm"), false),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut Option<(Tensor, Tensor)>,
        scalings: Option<Tensor>,
        global_scaling_weight: f64,
        is_scaling_pass: Option<f64>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = xs.apply(&self.input_layernorm)?;
        let attn_outputs = self.self_attn.forward(
            &xs,
            mask,
            seqlen_offsets,
            kv_cache,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
            flash_params,
        )?;
        let feed_forward_hidden_states =
            self.mlp
                .forward(&xs, scalings, global_scaling_weight, is_scaling_pass)?;
        attn_outputs + feed_forward_hidden_states + residual
    }
}

pub struct Model {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    final_layernorm: LayerNorm,
    lm_head: Arc<dyn LinearLayerLike + Send + Sync>,
    cache: EitherCache,
    device: Device,
    max_seq_len: usize,
    xlora_classifier: Option<XLoraClassifier>,
    dtype: DType,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    cfg: ModelConfigMetadata,
}

impl Model {
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

        let embed_tokens = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), false),
            &cfg.quantization_config,
        )?;
        let final_layernorm = layer_norm(
            cfg.hidden_size,
            cfg.layer_norm_eps,
            mapper.set_nm_device(vb_m.pp("final_layernorm"), false),
        )?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_m = vb_m.pp("layers");
        let mut ropes = HashMap::new();
        for layer_idx in 0..cfg.num_hidden_layers {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            // Alternative rope scalings are not supported
            ropes.insert(
                device.location(),
                Arc::new(RotaryEmbedding::new_partial(
                    cfg.rope_theta,
                    (cfg.partial_rotary_factor * cfg.head_dim() as f64) as usize,
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
                cfg,
                vb_m.pp(layer_idx),
                lora_config,
                &mut count,
                &xlora_ordering,
                &*mapper,
                layer_idx,
                normal_loading_metadata.loading_isq,
                rotary_emb,
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
                Arc::get_mut(&mut layer.self_attn.dense)
                    .unwrap()
                    .merge_weights()?;
                Arc::get_mut(&mut layer.self_attn.q_proj)
                    .unwrap()
                    .merge_weights()?;
                Arc::get_mut(&mut layer.self_attn.v_proj)
                    .unwrap()
                    .merge_weights()?;

                Arc::get_mut(&mut layer.mlp.fc1).unwrap().merge_weights()?;
                Arc::get_mut(&mut layer.mlp.fc2).unwrap().merge_weights()?;
            }
        }
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
        Ok(Self {
            embed_tokens,
            layers,
            final_layernorm,
            lm_head,
            cache: EitherCache::Full(Cache::new(cfg.num_hidden_layers, true)),
            device: normal_loading_metadata.real_device,
            max_seq_len: cfg.max_position_embeddings,
            dtype: vb.dtype(),
            xlora_classifier: xlora_config.map(|xlora_config| {
                XLoraClassifier::new(xlora_config, count, lora_config.len(), vb, false).unwrap()
            }),
            mapper,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_kv_heads: cfg.num_key_value_heads(),
                num_attn_heads: cfg.num_attention_heads,
                sliding_window: None,
                k_head_dim: cfg.head_dim(),
                v_head_dim: cfg.head_dim(),
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
        let mut xs = input_ids.apply(&self.embed_tokens)?;
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
        let mask = CausalMasker.make_causal_mask_matrix(
            input_ids,
            &*cache,
            xs.dtype(),
            self.cfg.num_attn_heads,
        )?;
        let mask = DeviceMappedMask::new(mask, &*self.mapper)?;
        for (i, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;
            xs = layer.forward(
                &xs,
                mask.as_ref().map(|m| m.get(xs.device())),
                seqlen_offsets,
                &mut cache[i],
                scalings.clone(),
                self.xlora_classifier
                    .as_ref()
                    .map(|classifier| classifier.get_global_scaling_weight())
                    .unwrap_or(1.0),
                is_scaling_pass,
                flash_params,
            )?;
        }
        let xs = xs.to_device(&self.device)?;
        xs.apply(&self.final_layernorm)
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

impl IsqModel for Model {
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
                Arc::get_mut(&mut layer.self_attn.dense)
                    .unwrap()
                    .quant_inner(),
                Some(i),
            ));
            tensors.push((
                Arc::get_mut(&mut layer.mlp.fc1).unwrap().quant_inner(),
                Some(i),
            ));
            tensors.push((
                Arc::get_mut(&mut layer.mlp.fc2).unwrap().quant_inner(),
                Some(i),
            ));
        }
        (tensors, &*self.mapper)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        panic!("Cannot generate UQFF for an adapter model.")
    }
}

impl NormalModel for Model {
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

impl ScalingsMaker for Model {
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

impl AnyMoeBaseModelMixin for Model {}
