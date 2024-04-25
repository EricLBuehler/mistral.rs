#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::Arc;

/// Phi model.
/// https://huggingface.co/microsoft/phi-2
/// There is an alternative implementation of the phi model in mixformers.rs.
/// This corresponds to the model update made with the following commit:
/// https://huggingface.co/microsoft/phi-2/commit/cb2f4533604d8b67de604e7df03bfe6f3ca22869
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{
    embedding, layer_norm, Activation, Embedding, LayerNorm, Linear, RotaryEmbedding, VarBuilder,
};
use mistralrs_lora::{linear, LinearLayerLike, LoraConfig, Ordering};

use crate::{
    device_map::DeviceMapper,
    models::{flash_attn, phi2::Config, repeat_kv},
    pipeline::{extract_logits, NormalModel},
    DeviceMapMetadata,
};

use super::{classifier::XLoraClassifier, Cache, NonGranularState, ScalingsMaker, XLoraConfig};

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    fc1: Arc<dyn LinearLayerLike + Send + Sync>,
    fc2: Arc<dyn LinearLayerLike + Send + Sync>,
    act: Activation,
}

impl MLP {
    fn new(
        cfg: &Config,
        vb: VarBuilder,
        lora_config: &[(String, LoraConfig)],
        count: &mut usize,
        ord: &Ordering,
    ) -> Result<Self> {
        let fc1 = linear(
            cfg.hidden_size,
            cfg.intermediate_size,
            vb.pp("fc1"),
            lora_config,
            count,
            ord,
        )?;
        let fc2 = linear(
            cfg.intermediate_size,
            cfg.hidden_size,
            vb.pp("fc2"),
            lora_config,
            count,
            ord,
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
        self.fc2.lora_forward(
            &self
                .fc1
                .lora_forward(xs, scalings.clone(), global_scaling_weight, is_scaling_pass)?
                .apply(&self.act)?,
            scalings,
            global_scaling_weight,
            is_scaling_pass,
        )
    }
}

#[derive(Clone)]
struct Attention {
    q_proj: Arc<dyn LinearLayerLike + Send + Sync>,
    k_proj: Arc<dyn LinearLayerLike + Send + Sync>,
    v_proj: Arc<dyn LinearLayerLike + Send + Sync>,
    dense: Arc<dyn LinearLayerLike + Send + Sync>,
    q_layernorm: Option<LayerNorm>,
    k_layernorm: Option<LayerNorm>,
    rotary_emb: RotaryEmbedding,
    softmax_scale: f64,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    use_flash_attn: bool,
}

fn get_mask(size: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<_> = (0..size)
        .flat_map(|i| (0..size).map(move |j| u8::from(j > i)))
        .collect();
    Tensor::from_slice(&mask, (size, size), device)
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

impl Attention {
    fn new(
        cfg: &Config,
        vb: VarBuilder,
        lora_config: &[(String, LoraConfig)],
        count: &mut usize,
        ord: &Ordering,
        is_gptx: bool,
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads();
        let head_dim = cfg.head_dim();
        let q_proj = linear(
            cfg.hidden_size,
            num_heads * head_dim,
            vb.pp("q_proj"),
            lora_config,
            count,
            ord,
        )?;
        let k_proj = linear(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            vb.pp("k_proj"),
            lora_config,
            count,
            ord,
        )?;
        let v_proj = linear(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            vb.pp("v_proj"),
            lora_config,
            count,
            ord,
        )?;
        let dense = linear(
            num_heads * head_dim,
            cfg.hidden_size,
            vb.pp("dense"),
            lora_config,
            count,
            ord,
        )?;
        // Alternative rope scalings are not supported.
        let rotary_emb = RotaryEmbedding::new_partial(
            cfg.rope_theta,
            cfg.head_dim(),
            (cfg.partial_rotary_factor * cfg.head_dim() as f64) as usize,
            cfg.max_position_embeddings,
            vb.device(),
            is_gptx,
            vb.dtype(),
        )?;
        let (q_layernorm, k_layernorm) = if cfg.qk_layernorm {
            let q_layernorm = layer_norm(head_dim, cfg.layer_norm_eps, vb.pp("q_layernorm"))?;
            let k_layernorm = layer_norm(head_dim, cfg.layer_norm_eps, vb.pp("k_layernorm"))?;
            (Some(q_layernorm), Some(k_layernorm))
        } else {
            (None, None)
        };
        let softmax_scale = 1f64 / (head_dim as f64).sqrt();
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            dense,
            q_layernorm,
            k_layernorm,
            rotary_emb,
            softmax_scale,
            num_heads,
            num_kv_heads,
            head_dim,
            use_flash_attn: cfg.use_flash_attn,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &mut self,
        xs: &Tensor,
        mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        kv_cache: &mut Option<(Tensor, Tensor)>,
        scalings: Option<Tensor>,
        global_scaling_weight: f64,
        is_scaling_pass: Option<f64>,
    ) -> Result<Tensor> {
        let (b_size, seq_len, _n_embd) = xs.dims3()?;
        let q = self.q_proj.lora_forward(
            xs,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?;
        let k = self.k_proj.lora_forward(
            xs,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?;
        let v = self.v_proj.lora_forward(
            xs,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?;

        let q = match &self.q_layernorm {
            None => q,
            Some(ln) => q.apply(ln)?,
        };
        let k = match &self.k_layernorm {
            None => k,
            Some(ln) => k.apply(ln)?,
        };

        let mut q = q.reshape((b_size * seq_len, self.num_heads, self.head_dim))?;
        let mut k = k.reshape((b_size * seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v
            .reshape((b_size, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        self.rotary_emb.forward(
            seqlen_offsets,
            &start_offsets_kernel,
            &mut q,
            &mut k,
            b_size,
        )?;

        if q.rank() == 3 {
            q = q
                .reshape((b_size, seq_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            k = k
                .reshape((b_size, seq_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
        }

        let (k, v) = match &*kv_cache {
            None => (k, v),
            Some((prev_k, prev_v)) => {
                let k = candle_nn::ops::kvconcat(prev_k, &k, 2)?;
                let v = candle_nn::ops::kvconcat(prev_v, &v, 2)?;
                (k, v)
            }
        };
        *kv_cache = Some((k.clone(), v.clone()));

        let k = repeat_kv(k, self.num_heads / self.num_kv_heads)?.contiguous()?;
        let v = repeat_kv(v, self.num_heads / self.num_kv_heads)?.contiguous()?;

        let attn_output = if self.use_flash_attn {
            // flash-attn expects (b_sz, seq_len, nheads, head_dim)
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;
            flash_attn(&q, &k, &v, self.softmax_scale as f32, seq_len > 1)?.transpose(1, 2)?
        } else {
            let attn_weights = (q
                .to_dtype(DType::F32)?
                .contiguous()?
                .matmul(&k.to_dtype(DType::F32)?.t()?)?
                * self.softmax_scale)?;
            let attn_weights = match mask {
                None => attn_weights,
                Some(mask) => masked_fill(
                    &attn_weights,
                    &mask.broadcast_left((b_size, self.num_heads))?,
                    f32::NEG_INFINITY,
                )?,
            };
            let attn_weights =
                candle_nn::ops::softmax_last_dim(&attn_weights)?.to_dtype(v.dtype())?;
            attn_weights.matmul(&v)?
        };

        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_size, seq_len, ()))?;
        self.dense.lora_forward(
            &attn_output,
            scalings,
            global_scaling_weight,
            is_scaling_pass,
        )
    }
}

#[derive(Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: LayerNorm,
}

impl DecoderLayer {
    fn new(
        cfg: &Config,
        vb: VarBuilder,
        lora_config: &[(String, LoraConfig)],
        count: &mut usize,
        ord: &Ordering,
        is_gptx: bool,
    ) -> Result<Self> {
        let self_attn = Attention::new(cfg, vb.pp("self_attn"), lora_config, count, ord, is_gptx)?;
        let mlp = MLP::new(cfg, vb.pp("mlp"), lora_config, count, ord)?;
        let input_layernorm = layer_norm(
            cfg.hidden_size,
            cfg.layer_norm_eps,
            vb.pp("input_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &mut self,
        xs: &Tensor,
        mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        kv_cache: &mut Option<(Tensor, Tensor)>,
        scalings: Option<Tensor>,
        global_scaling_weight: f64,
        is_scaling_pass: Option<f64>,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = xs.apply(&self.input_layernorm)?;
        let attn_outputs = self.self_attn.forward(
            &xs,
            mask,
            seqlen_offsets,
            start_offsets_kernel,
            kv_cache,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
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
    lm_head: Linear,
    pub cache: Cache,
    pub device: Device,
    pub max_seq_len: usize,
    xlora_classifier: Option<XLoraClassifier>,
    dtype: DType,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
}

impl Model {
    pub fn new(
        cfg: &Config,
        vb: VarBuilder,
        lora_config: &[(String, LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        is_gptx: bool,
        mapper: DeviceMapMetadata,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let final_layernorm = layer_norm(
            cfg.hidden_size,
            cfg.layer_norm_eps,
            vb_m.pp("final_layernorm"),
        )?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_m = vb_m.pp("layers");
        let mut count = 0;
        let mapper = mapper.into_mapper(cfg.num_hidden_layers, vb.device())?;
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(
                cfg,
                mapper.set_device(layer_idx, vb_m.pp(layer_idx)),
                lora_config,
                &mut count,
                &xlora_ordering,
                is_gptx,
            )?;
            layers.push(layer)
        }
        let lm_head = candle_nn::linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        Ok(Self {
            embed_tokens,
            layers,
            final_layernorm,
            lm_head,
            cache: Cache::new(cfg.num_hidden_layers, true),
            device: vb.device().clone(),
            max_seq_len: cfg.max_position_embeddings,
            dtype: vb.dtype(),
            xlora_classifier: xlora_config.map(|xlora_config| {
                XLoraClassifier::new(xlora_config, count, lora_config.len(), vb, false).unwrap()
            }),
            mapper,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn inner_forward(
        &mut self,
        xs: &Tensor,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        scalings: Option<Tensor>,
        is_full_pass: bool,
        no_kv_cache: bool,
        is_scaling_pass: Option<f64>,
    ) -> Result<Tensor> {
        let (_b_size, seq_len) = xs.dims2()?;
        let mut xs = xs.apply(&self.embed_tokens)?;
        let mask = if seq_len <= 1 {
            None
        } else {
            Some(get_mask(seq_len, xs.device())?)
        };
        let mut cache = if is_full_pass {
            if no_kv_cache {
                let mut new_cache = Vec::new();
                for _ in 0..self.cache.xlora_lock().len() {
                    new_cache.push(None);
                }

                *self.cache.xlora_lock() = new_cache.clone();
            }
            self.cache.xlora_lock()
        } else {
            self.cache.lock()
        };
        for (i, layer) in self.layers.iter_mut().enumerate() {
            xs = self.mapper.map(xs, i)?;
            xs = layer.forward(
                &xs,
                mask.as_ref(),
                seqlen_offsets,
                start_offsets_kernel.clone(),
                &mut cache[i],
                scalings.clone(),
                self.xlora_classifier
                    .as_ref()
                    .map(|classifier| classifier.get_global_scaling_weight())
                    .unwrap_or(1.0),
                is_scaling_pass,
            )?;
        }
        let xs = xs.to_device(&self.device)?;
        xs.apply(&self.final_layernorm)?
            .narrow(1, seq_len - 1, 1)?
            .apply(&self.lm_head)?
            .squeeze(1)
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
        context_lens: Vec<usize>,
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
            )?;

            if no_kv_cache {
                extract_logits(
                    &self
                        .inner_forward(
                            input_ids_full,
                            seqlen_offsets_full,
                            start_offsets_kernel_full,
                            Some(scalings),
                            true,
                            no_kv_cache,
                            None,
                        )?
                        .contiguous()?
                        .apply(&self.lm_head)?,
                    context_lens,
                )
            } else {
                // is_full_pass=true is ok because no_kv_cache=false
                extract_logits(
                    &self
                        .inner_forward(
                            input_ids,
                            seqlen_offsets,
                            start_offsets_kernel,
                            Some(scalings),
                            true,
                            no_kv_cache,
                            None,
                        )?
                        .contiguous()?
                        .apply(&self.lm_head)?,
                    context_lens,
                )
            }
        } else {
            extract_logits(
                &self
                    .inner_forward(
                        input_ids,
                        seqlen_offsets,
                        start_offsets_kernel,
                        None,
                        false,
                        no_kv_cache,
                        None,
                    )?
                    .contiguous()?
                    .apply(&self.lm_head)?,
                context_lens,
            )
        }
    }
}

impl NormalModel for Model {
    fn forward(
        &mut self,
        _input_ids: &Tensor,
        _seqlen_offsets: &[usize],
        _start_offsets_kernel: Tensor,
        _context_lens: Vec<usize>,
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
        context_lens: Vec<usize>,
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
        start_offsets_kernel: Tensor,
        scalings: Tensor,
        is_full_pass: bool,
        no_kv_cache: bool,
        is_scaling_pass: Option<f64>,
    ) -> Result<Tensor> {
        self.inner_forward(
            input_ids,
            seqlen_offsets,
            start_offsets_kernel,
            Some(scalings),
            is_full_pass,
            no_kv_cache,
            is_scaling_pass,
        )
    }
}
