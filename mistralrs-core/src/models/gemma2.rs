#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{collections::HashMap, sync::Arc};

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::Linear;
use mistralrs_quant::{
    ColumnParallelLayer, QuantMethod, QuantMethodConfig, QuantizedConfig, RowParallelLayer,
    ShardedVarBuilder, UnquantLinear,
};

use crate::{
    amoe::{AnyMoeBaseModelMixin, AnyMoeConfig, AnyMoeExpertType, MlpLayer, MoeMlp},
    attention::SdpaParams,
    device_map::DeviceMapper,
    get_delta_from_lora_ab,
    layers::{embedding, Activation, CausalMasker, MatMul, Mlp, RmsNorm, RotaryEmbedding, Sdpa},
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        extract_logits,
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, NormalCache, NormalLoadingMetadata, NormalModel,
    },
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct Config {
    pub attention_bias: bool,
    pub head_dim: usize,
    // The code gemma configs include both hidden_act and hidden_activation.
    pub hidden_act: Option<Activation>,
    pub hidden_activation: Option<Activation>,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub vocab_size: usize,
    pub sliding_window: usize,
    pub attn_logit_softcapping: Option<f64>,
    pub final_logit_softcapping: Option<f64>,
    pub query_pre_attn_scalar: usize,
    pub max_position_embeddings: usize,
    pub quantization_config: Option<QuantizedConfig>,
    pub use_flash_attn: bool,
    #[allow(dead_code)]
    pub tie_word_embeddings: bool,
}

impl Config {
    pub fn hidden_act(&self) -> Result<Activation> {
        match (self.hidden_act, self.hidden_activation) {
            (None, Some(act)) | (Some(act), None) => Ok(act),
            (Some(act), Some(_)) => {
                // If both are set just use hidden_act
                Ok(act)
            }
            (None, None) => candle_core::bail!("none of hidden_act and hidden_activation are set"),
        }
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
    rotary_emb: Arc<RotaryEmbedding>,
    use_sliding_window: bool,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

impl Attention {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        layer_idx: usize,
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
        );
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
        let sliding_window = if layer_idx % 2 == 0 {
            // ^ Order is SWA, global, SWA
            Some(cfg.sliding_window)
        } else {
            None
        };
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: num_heads / comm.world_size(),
            num_kv_heads: (num_kv_heads / comm.world_size()).max(1),
            head_dim,
            rotary_emb,
            use_sliding_window: layer_idx % 2 == 0, // Order is SWA, global, SWA
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: mistralrs_quant::compute_n_kv_groups(
                    cfg.num_key_value_heads,
                    cfg.num_attention_heads,
                    comm,
                ),
                use_flash_attn: cfg.use_flash_attn,
                softcap: cfg.attn_logit_softcapping.map(|x| x as f32),
                softmax_scale: 1.0 / (cfg.query_pre_attn_scalar as f32).sqrt(),
                sliding_window,
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        sliding_attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let original_dtype = xs.dtype();
        let mut xs = xs.clone();
        if let Some(t) = self.q_proj.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        let mut q = MatMul.qmethod_matmul(&xs, &*self.q_proj)?;
        let mut k = MatMul.qmethod_matmul(&xs, &*self.k_proj)?;
        let mut v = MatMul.qmethod_matmul(&xs, &*self.v_proj)?;
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

        let mask = if self.use_sliding_window {
            sliding_attention_mask
        } else {
            attention_mask
        };

        let mut attn_output = match &self.paged_attn {
            Some(paged_attn) => match metadata {
                Some(((key_cache, value_cache), input_metadata)) => paged_attn.forward(
                    &q,
                    &k,
                    &v,
                    attention_mask,
                    Some(key_cache),
                    Some(value_cache),
                    input_metadata,
                    &self.sdpa_params,
                    Some(flash_params),
                )?,
                None => {
                    // If we don't have metadata, we are most likely generating an imatrix so we don't want to populate that.
                    // Generating the dummy metadata with the assumption that we are not generating text (only processing prompts).
                    let input_metadata = PagedAttentionInputMetadata::dummy(q.device())?;
                    // Sanity check.
                    assert!(attention_mask.is_some());
                    paged_attn.forward(
                        &q,
                        &k,
                        &v,
                        attention_mask,
                        None,
                        None,
                        &input_metadata,
                        &self.sdpa_params,
                        Some(flash_params),
                    )?
                }
            },
            None => {
                // self.sliding_window is None if !self.use_sliding_window
                let (k, v) = kv_cache.append(&k, &v)?;

                Sdpa.run_attention(&q, &k, &v, mask, Some(flash_params), &self.sdpa_params)?
            }
        };

        if let Some(t) = self.q_proj.quantized_act_type() {
            attn_output = attn_output.to_dtype(t)?;
        }
        attn_output = if attention_mask.is_some() {
            attn_output.transpose(1, 2)?.reshape((b_sz, q_len, ()))?
        } else {
            attn_output.reshape((b_sz, q_len, ()))?
        };
        let mut res = MatMul.qmethod_matmul(&attn_output, &*self.o_proj)?;
        if self.q_proj.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: Box<dyn MlpLayer>,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
}

impl DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let self_attn = Attention::new(
            rotary_emb,
            cfg,
            layer_idx,
            mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq),
            paged_attn,
            comm,
        )?;
        let mlp = Mlp::new(
            mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq),
            cfg.hidden_size,
            cfg.intermediate_size,
            &cfg.quantization_config,
            cfg.hidden_act()?,
            comm,
        )?;
        let input_layernorm = RmsNorm::new_gemma(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("input_layernorm"), false),
        )?;
        let post_attention_layernorm = RmsNorm::new_gemma(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("post_attention_layernorm"), false),
        )?;
        let pre_feedforward_layernorm = RmsNorm::new_gemma(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("pre_feedforward_layernorm"), false),
        )?;
        let post_feedforward_layernorm = RmsNorm::new_gemma(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("post_feedforward_layernorm"), false),
        )?;
        Ok(Self {
            self_attn,
            mlp: Box::new(mlp),
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        sliding_attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .self_attn
            .forward(
                &xs,
                attention_mask,
                sliding_attention_mask,
                seqlen_offsets,
                kv_cache,
                metadata,
                flash_params,
            )?
            .apply(&self.post_attention_layernorm)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .mlp
            .forward(&xs.apply(&self.pre_feedforward_layernorm)?)?
            .apply(&self.post_feedforward_layernorm)?;
        residual + xs
    }
}

pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Arc<dyn QuantMethod>,
    hidden_size: usize,
    device: Device,
    cache: EitherCache,
    max_seq_len: usize,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    sliding_window: usize,
    final_logit_softcapping: Option<f64>,
    cfg: ModelConfigMetadata,
}

impl Model {
    pub fn new(
        cfg: &Config,
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

        let vb_m = vb.pp("model");
        let embed_tokens = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), false),
            &cfg.quantization_config,
        )?;
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
                    vb_m.dtype(),
                )?),
            );
        }
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
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
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => {
                    Some(PagedAttention::new(cfg.head_dim, device, None)?)
                }
            };
            let comm = mapper.get_comm_for(layer_idx)?;
            let layer = DecoderLayer::new(
                rotary_emb.clone(),
                cfg,
                vb_l.pp(layer_idx),
                &*mapper,
                layer_idx,
                normal_loading_metadata.loading_isq,
                paged_attn,
                &comm,
            )?;
            layers.push(layer)
        }
        let norm = RmsNorm::new_gemma(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;
        let lm_head = mapper.cast_nm_device(
            embed_tokens.embeddings(),
            normal_loading_metadata.loading_isq,
        )?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head: Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
                Linear::new(lm_head, None),
            ))?),
            device: normal_loading_metadata.real_device,
            hidden_size: cfg.hidden_size,
            cache: EitherCache::Normal(NormalCache::new_sliding(
                cfg.num_hidden_layers,
                cfg.max_position_embeddings,
                Some(cfg.sliding_window),
            )),
            max_seq_len: cfg.max_position_embeddings,
            sliding_window: cfg.sliding_window,
            final_logit_softcapping: cfg.final_logit_softcapping,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_attn_heads: cfg.num_attention_heads / mapper.get_comm_for(0)?.world_size(),
                num_kv_heads: (cfg.num_key_value_heads / mapper.get_comm_for(0)?.world_size())
                    .max(1),
                sliding_window: None,
                k_head_dim: cfg.head_dim,
                v_head_dim: cfg.head_dim,
            },
            mapper,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let xs = self.embed_tokens.forward(input_ids)?;
        let mut xs = (xs * (self.hidden_size as f64).sqrt())?;
        let cache = &mut self.cache.normal().0;
        let attention_mask = CausalMasker.make_causal_mask_matrix(
            input_ids,
            &*cache,
            xs.dtype(),
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
            xs.dtype(),
            self.cfg.num_attn_heads,
        )?;
        // PagedAttention prompt chunking
        let sliding_attention_mask = sliding_attention_mask.filter(|_| {
            metadata
                .as_ref()
                .map(|(_, meta)| meta.is_first_prompt_chunk)
                .unwrap_or(true)
        });
        for (i, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;
            xs = layer.forward(
                &xs,
                attention_mask
                    .as_ref()
                    .map(|m| m.to_device(xs.device()).unwrap())
                    .as_ref(),
                sliding_attention_mask
                    .as_ref()
                    .map(|m| m.to_device(xs.device()).unwrap())
                    .as_ref(),
                seqlen_offsets,
                &mut cache[i],
                metadata
                    .as_ref()
                    .map(|(kv_cache, metadata)| (kv_cache[i].clone(), *metadata)),
                flash_params,
            )?;
        }
        let xs = xs.to_device(&self.device)?;
        let mut xs = xs.apply(&self.norm)?;
        if let Some(t) = self.lm_head.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }

        let mut xs = MatMul.qmethod_matmul(&xs, &*self.lm_head)?;

        if let Some(final_logit_softcapping) = self.final_logit_softcapping {
            xs = (xs / final_logit_softcapping)?;
            xs = xs.tanh()?;
            xs = (xs * final_logit_softcapping)?;
        }

        extract_logits(&xs, context_lens)
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
        tensors.push((&mut self.lm_head, None));
        for (i, layer) in self.layers.iter_mut().enumerate() {
            tensors.push((&mut layer.self_attn.q_proj, Some(i)));
            tensors.push((&mut layer.self_attn.k_proj, Some(i)));
            tensors.push((&mut layer.self_attn.v_proj, Some(i)));
            tensors.push((&mut layer.self_attn.o_proj, Some(i)));
            tensors.extend(
                layer
                    .mlp
                    .get_isq_layers()
                    .into_iter()
                    .map(|m| (m, Some(i)))
                    .collect::<Vec<_>>(),
            );
        }
        (tensors, &*self.mapper)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        let uvb_m = uvb.pp("model");
        uvb_m.pp("embed_tokens").add(&self.embed_tokens);
        uvb_m.pp("norm").add(&self.norm.undo_gemma().unwrap());

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb_m.pp("layers").pp(layer_idx);
            uvb_l
                .pp("input_layernorm")
                .add(&layer.input_layernorm.undo_gemma().unwrap());
            uvb_l
                .pp("post_attention_layernorm")
                .add(&layer.post_attention_layernorm.undo_gemma().unwrap());
            uvb_l
                .pp("pre_feedforward_layernorm")
                .add(&layer.pre_feedforward_layernorm.undo_gemma().unwrap());
            uvb_l
                .pp("post_feedforward_layernorm")
                .add(&layer.post_feedforward_layernorm.undo_gemma().unwrap());
        }

        uvb.to_safetensors()
    }

    fn imatrix_names(&self) -> candle_core::Result<Vec<Option<String>>> {
        // NOTE: dependant on the exact implementation in get_layers!
        let mut names = Vec::new();
        // lm_head
        names.push(None);
        for i in 0..self.layers.len() {
            names.push(Some(format!("blk.{i}.attn_q.weight")));
            names.push(Some(format!("blk.{i}.attn_k.weight")));
            names.push(Some(format!("blk.{i}.attn_v.weight")));
            names.push(Some(format!("blk.{i}.attn_output.weight")));
            names.push(Some(format!("blk.{i}.ffn_gate.weight")));
            names.push(Some(format!("blk.{i}.ffn_up.weight")));
            names.push(Some(format!("blk.{i}.ffn_down.weight")));
        }
        Ok(names)
    }
}

impl NormalModel for Model {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        self.forward(
            input_ids,
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
        false
    }
    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    fn config(&self) -> &ModelConfigMetadata {
        &self.cfg
    }
}

impl AnyMoeBaseModelMixin for Model {
    fn get_mlps(&self) -> Vec<&dyn MlpLayer> {
        let mut mlps = Vec::new();
        for layer in &self.layers {
            mlps.push(&*layer.mlp);
        }
        mlps
    }
    fn get_mlps_mut(&mut self) -> Vec<&mut Box<dyn MlpLayer>> {
        let mut mlps = Vec::new();
        for layer in &mut self.layers {
            mlps.push(&mut layer.mlp);
        }
        mlps
    }
    fn create_anymoe_layers(
        &mut self,
        additional_vbs: Vec<ShardedVarBuilder>,
        config: AnyMoeConfig,
        (prefix, mlp): (String, String),
        mut layers: Vec<usize>,
        expert_type: AnyMoeExpertType,
        gate_vb: Option<ShardedVarBuilder>,
    ) -> Result<()> {
        let mut experts: Vec<Vec<Box<dyn MlpLayer>>> = Vec::new();
        if layers.is_empty() {
            layers = (0..self.layers.len()).collect::<Vec<_>>();
        }
        for _ in 0..layers.len() {
            experts.push(Vec::new());
        }
        for vb in additional_vbs {
            let vb = vb.pp(&prefix);
            for (layer, row) in experts.iter_mut().enumerate() {
                if !layers.contains(&layer) {
                    continue;
                }

                let intermediate_size = self.layers[layer].mlp.get_params()[1];
                let hidden_size = self.layers[layer].mlp.get_params()[0];
                match expert_type {
                    AnyMoeExpertType::FineTuned => {
                        let (dtype, device) = self.layers[layer].mlp.dtype_device();
                        row.push(Box::new(Mlp::replicate(
                            self.layers[layer].mlp.get_params(),
                            vb.pp(layer).pp(&mlp).set_dtype(dtype).set_device(device),
                            self.layers[layer].mlp.hidden_act(),
                            &self.mapper.get_comm_for(layer)?,
                        )?));
                    }
                    AnyMoeExpertType::LoraAdapter {
                        rank,
                        alpha,
                        ref target_modules,
                    } => {
                        let vb_mlp = vb.pp(layer).pp(&mlp);

                        let gate_proj_delta = if target_modules.contains(&"gate_proj".to_string()) {
                            Some(get_delta_from_lora_ab!(
                                vb_mlp,
                                rank,
                                alpha,
                                (hidden_size, intermediate_size),
                                "gate_proj"
                            ))
                        } else {
                            None
                        };
                        let up_proj_delta = if target_modules.contains(&"up_proj".to_string()) {
                            Some(get_delta_from_lora_ab!(
                                vb_mlp,
                                rank,
                                alpha,
                                (hidden_size, intermediate_size),
                                "up_proj"
                            ))
                        } else {
                            None
                        };
                        let down_proj_delta = if target_modules.contains(&"down_proj".to_string()) {
                            Some(get_delta_from_lora_ab!(
                                vb_mlp,
                                rank,
                                alpha,
                                (intermediate_size, hidden_size),
                                "down_proj"
                            ))
                        } else {
                            None
                        };

                        row.push(self.layers[layer].mlp.new_added_delta(vec![
                            gate_proj_delta,
                            up_proj_delta,
                            down_proj_delta,
                        ])?);
                    }
                }
            }
        }
        for (layer, expert) in layers.into_iter().zip(experts) {
            let mut experts_all = vec![self.layers[layer].mlp.clone()];
            experts_all.extend(expert);
            let (dtype, device) = self.layers[layer].mlp.dtype_device();
            self.layers[layer].mlp = Box::new(MoeMlp::new(
                experts_all,
                config.clone(),
                dtype,
                &device,
                layer,
                gate_vb.as_ref(),
            )?);
        }
        Ok(())
    }
    fn amoe_supported(&self) -> bool {
        true
    }
}
