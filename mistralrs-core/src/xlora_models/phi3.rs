#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

// This implementation is based on:
// https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/modeling_phi3.py
use crate::{
    layers::ScaledDotProductAttention,
    lora::{linear_no_bias, LinearLayerLike, LoraConfig, Ordering},
    pipeline::{IsqModel, NormalLoadingMetadata},
};
use candle_core::{quantized::QMatMul, DType, Device, Module, Result, Tensor, D};
use candle_nn::VarBuilder;
use std::{collections::HashMap, sync::Arc};
use tqdm::Iter;
use tracing::info;

use crate::{
    device_map::DeviceMapper,
    layers::{repeat_kv, CausalMasker, PhiRotaryEmbedding, QLinear, RmsNorm},
    models::phi3::Config,
    pipeline::{extract_logits, NormalModel},
};

use crate::pipeline::Cache;

use super::{classifier::XLoraClassifier, NonGranularState, ScalingsMaker, XLoraConfig};

#[derive(Debug, Clone)]
struct Attention {
    qkv_proj: Arc<dyn LinearLayerLike + Send + Sync>,
    o_proj: Arc<dyn LinearLayerLike + Send + Sync>,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<PhiRotaryEmbedding>,
    use_flash_attn: bool,
    sliding_window: Option<usize>,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: Arc<PhiRotaryEmbedding>,
        cfg: &Config,
        vb: VarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        count: &mut usize,
        ord: &Ordering,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        preload_adapters: &Option<HashMap<String, (VarBuilder, LoraConfig)>>,
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();
        let op_size = num_heads * head_dim + 2 * num_kv_heads * head_dim;
        let qkv_proj = linear_no_bias(
            cfg.hidden_size,
            op_size,
            mapper.set_device(layer_idx, vb.pp("qkv_proj"), loading_isq),
            mapper.set_device(layer_idx, vb.pp("qkv_proj"), false),
            lora_config,
            count,
            ord,
            preload_adapters,
        )?;
        let o_proj = linear_no_bias(
            num_heads * head_dim,
            cfg.hidden_size,
            mapper.set_device(layer_idx, vb.pp("o_proj"), loading_isq),
            mapper.set_device(layer_idx, vb.pp("o_proj"), false),
            lora_config,
            count,
            ord,
            preload_adapters,
        )?;
        Ok(Self {
            qkv_proj,
            o_proj,
            rotary_emb,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
            use_flash_attn: cfg.use_flash_attn,
            sliding_window: cfg.sliding_window,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        position_ids: &[usize],
        kv_cache: &mut Option<(Tensor, Tensor)>,
        scalings: Option<Tensor>,
        global_scaling_weight: f64,
        is_scaling_pass: Option<f64>,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let original_dtype = xs.dtype();
        let mut xs = xs.clone();
        if self.qkv_proj.is_quant() {
            xs = xs.to_dtype(DType::F32)?;
        }
        let mut qkv = self.qkv_proj.lora_forward(
            &xs,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?;
        if self.qkv_proj.is_quant() {
            qkv = qkv.to_dtype(original_dtype)?;
        }
        let query_pos = self.num_heads * self.head_dim;
        let q = qkv.narrow(D::Minus1, 0, query_pos)?;
        let k = qkv.narrow(D::Minus1, query_pos, self.num_kv_heads * self.head_dim)?;
        let v = qkv.narrow(
            D::Minus1,
            query_pos + self.num_kv_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
        )?;

        let q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = self
            .rotary_emb
            .forward(&q, &k, seqlen_offsets, position_ids)?;

        let (k, v, attn_mask) = Cache::update_kv_cache_sliding_window(
            kv_cache,
            k,
            v,
            attention_mask,
            self.sliding_window,
            true,
        )?;

        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let mut attn_output = ScaledDotProductAttention.run_attention(
            &q,
            &k,
            &v,
            self.num_heads,
            self.head_dim,
            attn_mask.as_ref(),
            self.use_flash_attn,
            b_sz,
            q_len,
        )?;

        if self.qkv_proj.is_quant() {
            attn_output = attn_output.to_dtype(DType::F32)?;
        }
        let mut res = self.o_proj.lora_forward(
            &attn_output.transpose(1, 2)?.reshape((b_sz, q_len, ()))?,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?;
        if self.qkv_proj.is_quant() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    gate_up_proj: Arc<dyn LinearLayerLike + Send + Sync>,
    down_proj: Arc<dyn LinearLayerLike + Send + Sync>,
    act_fn: candle_nn::Activation,
    i_size: usize,
}

impl Mlp {
    #[allow(clippy::too_many_arguments)]
    fn new(
        cfg: &Config,
        vb: VarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        count: &mut usize,
        ord: &Ordering,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        preload_adapters: &Option<HashMap<String, (VarBuilder, LoraConfig)>>,
    ) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let gate_up_proj = linear_no_bias(
            hidden_size,
            2 * i_size,
            mapper.set_device(layer_idx, vb.pp("gate_up_proj"), loading_isq),
            mapper.set_device(layer_idx, vb.pp("gate_up_proj"), false),
            lora_config,
            count,
            ord,
            preload_adapters,
        )?;
        let down_proj = linear_no_bias(
            i_size,
            hidden_size,
            mapper.set_device(layer_idx, vb.pp("down_proj"), loading_isq),
            mapper.set_device(layer_idx, vb.pp("down_proj"), false),
            lora_config,
            count,
            ord,
            preload_adapters,
        )?;
        Ok(Self {
            gate_up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
            i_size,
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
        if self.gate_up_proj.is_quant() {
            xs = xs.to_dtype(DType::F32)?;
        }
        let up_states = self.gate_up_proj.lora_forward(
            &xs,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?;
        let gate = up_states.narrow(D::Minus1, 0, self.i_size)?;
        let up_states = up_states.narrow(D::Minus1, self.i_size, self.i_size)?;
        let up_states = (up_states * gate.apply(&self.act_fn))?;
        let mut res = self.down_proj.lora_forward(
            &up_states,
            scalings,
            global_scaling_weight,
            is_scaling_pass,
        )?;
        if self.gate_up_proj.is_quant() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: Arc<PhiRotaryEmbedding>,
        cfg: &Config,
        vb: VarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        count: &mut usize,
        ord: &Ordering,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        preload_adapters: &Option<HashMap<String, (VarBuilder, LoraConfig)>>,
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
        let mlp = Mlp::new(
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
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        position_ids: &[usize],
        kv_cache: &mut Option<(Tensor, Tensor)>,
        scalings: Option<Tensor>,
        global_scaling_weight: f64,
        is_scaling_pass: Option<f64>,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(
            &xs,
            attention_mask,
            seqlen_offsets,
            position_ids,
            kv_cache,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self.mlp.forward(
            &xs.apply(&self.post_attention_layernorm)?,
            scalings,
            global_scaling_weight,
            is_scaling_pass,
        );
        residual + xs
    }
}

pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: QLinear,
    dtype: DType,
    pub device: Device,
    pub cache: Cache,
    pub max_seq_len: usize,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    xlora_classifier: Option<XLoraClassifier>,
    sliding_window: Option<usize>,
}

impl Model {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cfg: &Config,
        vb: VarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        _is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (VarBuilder, LoraConfig)>>,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let mapper = normal_loading_metadata
            .mapper
            .into_mapper(cfg.num_hidden_layers, &normal_loading_metadata.real_device)?;
        let embed_tokens = candle_nn::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), false),
        )?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        let mut count = 0;
        for layer_idx in 0..cfg.num_hidden_layers {
            let rotary_emb = Arc::new(PhiRotaryEmbedding::new(
                vb.dtype(),
                cfg.clone(),
                mapper
                    .device_for(layer_idx, false)
                    .unwrap_or(&normal_loading_metadata.real_device),
            )?);
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
                Arc::get_mut(&mut layer.self_attn.qkv_proj)
                    .unwrap()
                    .merge_weights()?;
                Arc::get_mut(&mut layer.self_attn.o_proj)
                    .unwrap()
                    .merge_weights()?;

                Arc::get_mut(&mut layer.mlp.down_proj)
                    .unwrap()
                    .merge_weights()?;
                Arc::get_mut(&mut layer.mlp.gate_up_proj)
                    .unwrap()
                    .merge_weights()?;
            }
        }
        let norm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;
        let lm_head = candle_nn::linear_no_bias(
            cfg.hidden_size,
            cfg.vocab_size,
            mapper.set_nm_device(vb.pp("lm_head"), normal_loading_metadata.loading_isq),
        )?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head: QLinear::from_linear(lm_head),
            device: normal_loading_metadata.real_device,
            dtype: vb.dtype(),
            cache: Cache::new(cfg.num_hidden_layers, true),
            max_seq_len: cfg.max_position_embeddings,
            mapper,
            xlora_classifier: xlora_config.map(|xlora_config| {
                XLoraClassifier::new(xlora_config, count, lora_config.len(), vb, false).unwrap()
            }),
            sliding_window: cfg.sliding_window,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn inner_forward(
        &mut self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        position_ids: &[usize],
        scalings: Option<Tensor>,
        is_full_pass: bool,
        no_kv_cache: bool,
        is_scaling_pass: Option<f64>,
    ) -> Result<Tensor> {
        let mut xs = self.embed_tokens.forward(input_ids)?;
        let mut cache = if is_full_pass {
            if no_kv_cache {
                let mut new_cache = Vec::new();
                for _ in 0..self.cache.xlora_lock().len() {
                    new_cache.push(None);
                }

                self.cache.xlora_lock().clone_from(&new_cache);
            }
            self.cache.xlora_lock()
        } else {
            self.cache.lock()
        };
        let attention_mask = CausalMasker.make_causal_mask_with_sliding_window_as_attn_bias(
            input_ids,
            &cache,
            self.sliding_window,
            xs.dtype(),
            self.layers[0].self_attn.num_heads,
        )?;
        let past_key_values_length = CausalMasker.calculate_past_kv_len(&cache)?;
        let position_ids = position_ids
            .iter()
            .map(|p| *p + past_key_values_length)
            .collect::<Vec<_>>();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            xs = self.mapper.map(xs, i)?;
            xs = layer.forward(
                &xs,
                attention_mask
                    .as_ref()
                    .map(|m| m.to_device(xs.device()).unwrap())
                    .as_ref(),
                seqlen_offsets,
                &position_ids,
                &mut cache[i],
                scalings.clone(),
                self.xlora_classifier
                    .as_ref()
                    .map(|classifier| classifier.get_global_scaling_weight())
                    .unwrap_or(1.0),
                is_scaling_pass,
            )?
        }
        let xs = xs.to_device(&self.device)?;
        xs.apply(&self.norm)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        input_ids_full: &Tensor,
        seqlen_offsets: &[usize],
        seqlen_offsets_full: &[usize],
        start_offsets_kernel: Tensor,
        start_offsets_kernel_full: Tensor,
        no_kv_cache: bool,
        non_granular_state: &Option<NonGranularState>,
        context_lens: Vec<(usize, usize)>,
        position_ids: Vec<usize>,
    ) -> Result<Tensor> {
        if self.xlora_classifier.is_some() {
            let scalings = self.get_scalings(
                input_ids,
                input_ids_full,
                seqlen_offsets,
                seqlen_offsets_full,
                &start_offsets_kernel,
                &start_offsets_kernel_full,
                no_kv_cache,
                non_granular_state,
                &position_ids,
            )?;

            if no_kv_cache {
                let mut res = self
                    .inner_forward(
                        input_ids_full,
                        seqlen_offsets_full,
                        &position_ids,
                        Some(scalings),
                        true,
                        no_kv_cache,
                        None,
                    )?
                    .contiguous()?;
                if self.lm_head.is_quant() {
                    res = res.to_dtype(DType::F32)?;
                }
                extract_logits(&res.apply(&self.lm_head)?, context_lens)
            } else {
                // is_full_pass=true is ok because no_kv_cache=false
                let mut res = self
                    .inner_forward(
                        input_ids,
                        seqlen_offsets,
                        &position_ids,
                        Some(scalings),
                        true,
                        no_kv_cache,
                        None,
                    )?
                    .contiguous()?;
                if self.lm_head.is_quant() {
                    res = res.to_dtype(DType::F32)?;
                }
                extract_logits(&res.apply(&self.lm_head)?, context_lens)
            }
        } else {
            let mut res = self
                .inner_forward(
                    input_ids,
                    seqlen_offsets,
                    &position_ids,
                    None,
                    false,
                    no_kv_cache,
                    None,
                )?
                .contiguous()?;
            if self.lm_head.is_quant() {
                res = res.to_dtype(DType::F32)?;
            }
            extract_logits(&res.apply(&self.lm_head)?, context_lens)
        }
    }
}

impl IsqModel for Model {
    fn get_tensors(&mut self) -> (Vec<(&mut QMatMul, Option<usize>)>, &dyn DeviceMapper) {
        let mut tensors = Vec::new();
        tensors.push((self.lm_head.inner(), None));
        for (i, layer) in self.layers.iter_mut().enumerate() {
            tensors.push((
                Arc::get_mut(&mut layer.self_attn.qkv_proj).unwrap().inner(),
                Some(i),
            ));
            tensors.push((
                Arc::get_mut(&mut layer.self_attn.o_proj).unwrap().inner(),
                Some(i),
            ));
            tensors.push((
                Arc::get_mut(&mut layer.mlp.down_proj).unwrap().inner(),
                Some(i),
            ));
            tensors.push((
                Arc::get_mut(&mut layer.mlp.gate_up_proj).unwrap().inner(),
                Some(i),
            ));
        }
        (tensors, &*self.mapper)
    }
}

impl NormalModel for Model {
    fn forward(
        &mut self,
        _input_ids: &Tensor,
        _seqlen_offsets: &[usize],
        _start_offsets_kernel: Tensor,
        _context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
    ) -> Result<Tensor> {
        unreachable!()
    }
    fn xlora_forward(
        &mut self,
        input_ids: &Tensor,
        input_ids_full: &Tensor,
        seqlen_offsets: &[usize],
        seqlen_offsets_full: &[usize],
        start_offsets_kernel: Tensor,
        start_offsets_kernel_full: Tensor,
        no_kv_cache: bool,
        non_granular_state: &Option<crate::xlora_models::NonGranularState>,
        context_lens: Vec<(usize, usize)>,
        position_ids: Vec<usize>,
    ) -> Result<Tensor> {
        self.forward(
            input_ids,
            input_ids_full,
            seqlen_offsets,
            seqlen_offsets_full,
            start_offsets_kernel,
            start_offsets_kernel_full,
            no_kv_cache,
            non_granular_state,
            context_lens,
            position_ids,
        )
    }
    fn cache(&self) -> &Cache {
        &self.cache
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
    fn activate_adapters(&mut self, adapter_names: Vec<String>) -> Result<usize> {
        let mut sum = 0;
        for layer in self.layers.iter_mut() {
            sum += Arc::get_mut(&mut layer.self_attn.qkv_proj)
                .unwrap()
                .activate(&adapter_names)?;
            sum += Arc::get_mut(&mut layer.self_attn.o_proj)
                .unwrap()
                .activate(&adapter_names)?;

            sum += Arc::get_mut(&mut layer.mlp.down_proj)
                .unwrap()
                .activate(&adapter_names)?;
            sum += Arc::get_mut(&mut layer.mlp.gate_up_proj)
                .unwrap()
                .activate(&adapter_names)?;
        }
        Ok(sum)
    }
}

impl ScalingsMaker for Model {
    fn dtype(&self) -> DType {
        self.dtype
    }
    fn get_cache(&self) -> &Cache {
        &self.cache
    }
    fn get_classifier(&self) -> &XLoraClassifier {
        self.xlora_classifier.as_ref().unwrap()
    }
    fn forward(
        &mut self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        _start_offsets_kernel: Tensor,
        scalings: Tensor,
        is_full_pass: bool,
        no_kv_cache: bool,
        is_scaling_pass: Option<f64>,
        context_lens: &[usize],
    ) -> Result<Tensor> {
        // NOTE(EricLBuehler): hacky yes, but passing the context lens to start the position ids calculation works
        self.inner_forward(
            input_ids,
            seqlen_offsets,
            context_lens,
            Some(scalings),
            is_full_pass,
            no_kv_cache,
            is_scaling_pass,
        )
    }
}
