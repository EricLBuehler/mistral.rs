#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

// This implementation is based on:
// https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/modeling_phi3.py
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::VarBuilder;
use mistralrs_quant::{QuantMethod, QuantMethodConfig, QuantizedConfig, UnquantLinear};
use std::{collections::HashMap, sync::Arc};

use crate::{
    amoe::{AnyMoeConfig, AnyMoeExpertType, AnyMoeTrainableLayer, MlpLayer, MoeMlp},
    attention::SdpaParams,
    device_map::DeviceMapper,
    get_delta_from_lora_ab,
    layers::{
        Activation, MatMul, PhiRopeConfig, PhiRopeScalingConfig, PhiRotaryEmbedding, RmsNorm, Sdpa,
    },
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, KvCache, NormalCache, NormalLoadingMetadata,
    },
    serde_default_fn,
    transformer::{self, AutoAnyMoeBaseModelMixin, AutoIsqModel, DecoderLayer, ModelWrapper},
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

serde_default_fn!(bool, word_emb_default, false);

// https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/config.json
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, Default)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub rope_scaling: Option<PhiRopeScalingConfig>,
    pub max_position_embeddings: usize,
    pub use_flash_attn: bool,
    pub sliding_window: Option<usize>,
    pub original_max_position_embeddings: usize,
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default = "word_emb_default")]
    pub tie_word_embeddings: bool,
}

impl From<Config> for PhiRopeConfig {
    fn from(val: Config) -> Self {
        PhiRopeConfig {
            rope_scaling: val.rope_scaling,
            max_position_embeddings: val.max_position_embeddings,
            original_max_position_embeddings: val.original_max_position_embeddings,
            rope_theta: val.rope_theta,
            head_dim: val.hidden_size / val.num_attention_heads,
        }
    }
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

struct Attention {
    qkv_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<PhiRotaryEmbedding>,
    sliding_window: Option<usize>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

impl Attention {
    fn new(
        rotary_emb: Arc<PhiRotaryEmbedding>,
        cfg: &Config,
        vb: VarBuilder,
        paged_attn: Option<PagedAttention>,
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();
        let op_size = num_heads * head_dim + 2 * num_kv_heads * head_dim;

        let qkv_proj = mistralrs_quant::linear_no_bias(
            cfg.hidden_size,
            op_size,
            &cfg.quantization_config,
            vb.pp("qkv_proj"),
        )?;

        let o_proj = mistralrs_quant::linear_no_bias(
            num_heads * head_dim,
            cfg.hidden_size,
            &cfg.quantization_config,
            vb.pp("o_proj"),
        )?;

        Ok(Self {
            qkv_proj,
            o_proj,
            rotary_emb,
            num_heads,
            num_kv_heads,
            head_dim,
            sliding_window: cfg.sliding_window,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: num_heads / num_kv_heads,
                use_flash_attn: cfg.use_flash_attn,
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window: cfg.sliding_window,
            },
        })
    }
}

impl transformer::Attention for Attention {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        _sliding_attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        _start_offsets_kernel: Option<Tensor>,
        position_ids: Option<&[usize]>,
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &mut PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let original_dtype = xs.dtype();
        let mut xs = xs.clone();
        if let Some(t) = self.qkv_proj.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        let mut qkv = MatMul.qmethod_matmul(&xs, &*self.qkv_proj)?;
        if self.qkv_proj.quantized_act_type().is_some() {
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

        let (q, k) = self
            .rotary_emb
            .forward(&q, &k, seqlen_offsets, position_ids.unwrap())?;

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
                    None,
                )?,
                None => {
                    // If we don't have metadata, we are most likely generating an imatrix so we don't want to populate that.
                    // Generating the dummy metadata with the assumption that we are not generating text (only processing prompts).
                    let mut input_metadata = PagedAttentionInputMetadata::dummy(q.device())?;
                    // Sanity check.
                    assert!(attention_mask.is_some());
                    paged_attn.forward(
                        &q,
                        &k,
                        &v,
                        attention_mask,
                        None,
                        None,
                        &mut input_metadata,
                        None,
                    )?
                }
            },
            _ => {
                let (k, v, attn_mask) =
                    kv_cache.append_sliding_window(&k, &v, attention_mask, self.sliding_window)?;

                Sdpa.run_attention(
                    &q,
                    &k,
                    &v,
                    attn_mask.as_ref(),
                    Some(flash_params),
                    &self.sdpa_params,
                )?
            }
        };

        if let Some(t) = self.qkv_proj.quantized_act_type() {
            attn_output = attn_output.to_dtype(t)?;
        }
        attn_output = if attention_mask.is_some() {
            attn_output.transpose(1, 2)?.reshape((b_sz, q_len, ()))?
        } else {
            attn_output.reshape((b_sz, q_len, ()))?
        };
        let mut res = MatMul.qmethod_matmul(&attn_output, &*self.o_proj)?;
        if self.qkv_proj.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }

    fn get_tensors(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        vec![&mut self.qkv_proj, &mut self.o_proj]
    }
}

#[derive(Clone)]
struct Mlp {
    gate_up_proj: Arc<dyn QuantMethod>,
    down_proj: Arc<dyn QuantMethod>,
    act_fn: Activation,
    i_size: usize,
    params: Vec<usize>,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;

        let gate_up_proj = mistralrs_quant::linear_no_bias(
            hidden_size,
            2 * i_size,
            &cfg.quantization_config,
            vb.pp("gate_up_proj"),
        )?;

        let down_proj = mistralrs_quant::linear_no_bias(
            i_size,
            hidden_size,
            &cfg.quantization_config,
            vb.pp("down_proj"),
        )?;

        Ok(Self {
            gate_up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
            i_size,
            params: vec![hidden_size, i_size],
        })
    }
}

impl AnyMoeTrainableLayer for Mlp {}

impl MlpLayer for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let original_dtype = xs.dtype();
        let mut xs = xs.clone();
        if let Some(t) = self.gate_up_proj.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        let up_states = MatMul.qmethod_matmul(&xs, &*self.gate_up_proj)?;
        let gate = up_states.narrow(D::Minus1, 0, self.i_size)?;
        let up_states = up_states.narrow(D::Minus1, self.i_size, self.i_size)?;
        let up_states = (up_states * gate.apply(&self.act_fn))?;
        let mut res = MatMul.qmethod_matmul(&up_states, &*self.down_proj)?;
        if self.gate_up_proj.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
    fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        vec![&mut self.gate_up_proj, &mut self.down_proj]
    }
    fn clone(&self) -> Box<dyn MlpLayer> {
        Box::new(Clone::clone(self))
    }
    fn get_params(&self) -> &[usize] {
        &self.params
    }
    // gate_up, down
    fn new_added_delta(&self, deltas: Vec<Option<Tensor>>) -> Result<Box<dyn MlpLayer>> {
        let new_gate_up = if let Some(ref delta) = deltas[0] {
            self.gate_up_proj.add_delta_w(delta)?
        } else {
            self.gate_up_proj.clone()
        };
        let new_down = if let Some(ref delta) = deltas[1] {
            self.down_proj.add_delta_w(delta)?
        } else {
            self.down_proj.clone()
        };

        Ok(Box::new(Self {
            gate_up_proj: new_gate_up,
            down_proj: new_down,
            act_fn: self.act_fn,
            i_size: self.i_size,
            params: self.params.clone(),
        }))
    }

    fn dtype_device(&self) -> (DType, Device) {
        self.gate_up_proj.dtype_and_device()
    }
}

pub struct Model {
    model: transformer::Model,
}

impl Model {
    fn new_layer(
        rotary_emb: Arc<PhiRotaryEmbedding>,
        cfg: &Config,
        vb: VarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        paged_attn: Option<PagedAttention>,
    ) -> Result<DecoderLayer> {
        let self_attn = Attention::new(
            rotary_emb,
            cfg,
            mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq),
            paged_attn,
        )?;
        let mlp = Mlp::new(cfg, mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq))?;
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
        Ok(DecoderLayer {
            self_attn: Box::new(self_attn),
            mlp: Box::new(mlp),
            input_layernorm,
            post_attention_layernorm: None,
            // This is correct! See `DecoderLayer::forward` for explanation.
            pre_feedforward_layernorm: Some(post_attention_layernorm),
            post_feedforward_layernorm: None,
        })
    }

    pub fn new(
        cfg: &Config,
        vb: VarBuilder,
        _is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        if let Some(ref quant_cfg) = &cfg.quantization_config {
            tracing::info!(
                "Using {} quantization: {}.",
                quant_cfg.quant_method.to_string(),
                quant_cfg.get_bits_name(&vb)
            );
        }
        let mapper = normal_loading_metadata.mapper;
        let vb_m = vb.pp("model");

        let embed_tokens = candle_nn::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), false),
        )?;
        let mut ropes = HashMap::new();
        for layer_idx in 0..cfg.num_hidden_layers {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            ropes.insert(
                device.location(),
                Arc::new(PhiRotaryEmbedding::new(vb.dtype(), cfg.clone(), device)?),
            );
        }
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in
            NiceProgressBar::<_, 'b'>(0..cfg.num_hidden_layers, "Loading repeating layers")
        {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let rotary_emb = ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => Some(PagedAttention::new(
                    cfg.num_attention_heads,
                    cfg.head_dim(),
                    (1.0 / (cfg.head_dim() as f64).sqrt()) as f32,
                    Some(cfg.num_key_value_heads),
                    cfg.sliding_window,
                    device,
                    None,
                )?),
            };
            let layer = Self::new_layer(
                rotary_emb.clone(),
                cfg,
                vb_l.pp(layer_idx),
                &*mapper,
                layer_idx,
                normal_loading_metadata.loading_isq,
                paged_attn,
            )?;
            layers.push(layer)
        }
        let norm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;
        let lm_head = if !cfg.tie_word_embeddings {
            mistralrs_quant::linear_no_bias(
                cfg.hidden_size,
                cfg.vocab_size,
                &None,
                mapper.set_nm_device(vb.pp("lm_head"), normal_loading_metadata.loading_isq),
            )?
        } else {
            Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
                candle_nn::Linear::new(
                    mapper.cast_nm_device(
                        embed_tokens.embeddings(),
                        normal_loading_metadata.loading_isq,
                    )?,
                    None,
                ),
            ))?)
        };
        Ok(Self {
            model: transformer::Model {
                embed_tokens,
                layers,
                norm,
                lm_head,
                device: normal_loading_metadata.real_device,
                hidden_size: None,
                cache: EitherCache::Normal(NormalCache::new(
                    cfg.num_hidden_layers,
                    cfg.max_position_embeddings,
                )),
                max_seq_len: cfg.max_position_embeddings,
                mapper,
                use_two_attention_masks: false,
                sliding_window: cfg.sliding_window,
                final_logit_softcapping: None,
                cfg: ModelConfigMetadata {
                    num_layers: cfg.num_hidden_layers,
                    hidden_size: cfg.hidden_size,
                    num_kv_heads: cfg.num_key_value_heads,
                    num_attn_heads: cfg.num_attention_heads,
                    sliding_window: cfg.sliding_window,
                    head_dim: None,
                },
            },
        })
    }
}

impl ModelWrapper for Model {
    fn get_model(&self) -> &transformer::Model {
        &self.model
    }

    fn get_model_mut(&mut self) -> &mut transformer::Model {
        &mut self.model
    }
}

impl AutoIsqModel for Model {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        let uvb_m = uvb.pp("model");
        uvb_m.pp("embed_tokens").add(&self.model.embed_tokens);
        uvb_m.pp("norm").add(&self.model.norm);

        for (layer_idx, layer) in self.model.layers.iter().enumerate() {
            let uvb_l = uvb_m.pp("layers").pp(layer_idx);
            uvb_l.pp("input_layernorm").add(&layer.input_layernorm);
            uvb_l
                .pp("post_attention_layernorm")
                .add(layer.pre_feedforward_layernorm.as_ref().unwrap());
        }

        uvb.to_safetensors()
    }

    fn imatrix_names(&self) -> candle_core::Result<Vec<Option<String>>> {
        // NOTE: dependant on the exact implementation in get_layers!
        let mut names = Vec::new();
        // lm_head
        names.push(None);
        for i in 0..self.model.layers.len() {
            names.push(Some(format!("blk.{i}.attn_qkv.weight")));
            names.push(Some(format!("blk.{i}.attn_output.weight")));
            names.push(Some(format!("blk.{i}.ffn_gate.weight")));
            names.push(Some(format!("blk.{i}.ffn_up.weight")));
            names.push(Some(format!("blk.{i}.ffn_down.weight")));
        }
        Ok(names)
    }
}

impl AutoAnyMoeBaseModelMixin for Model {
    fn create_anymoe_layers(
        &mut self,
        additional_vbs: Vec<VarBuilder>,
        config: AnyMoeConfig,
        (prefix, mlp): (String, String),
        mut layers: Vec<usize>,
        expert_type: AnyMoeExpertType,
        gate_vb: Option<VarBuilder>,
    ) -> Result<()> {
        let mut experts: Vec<Vec<Box<dyn MlpLayer>>> = Vec::new();
        if layers.is_empty() {
            layers = (0..self.model.layers.len()).collect::<Vec<_>>();
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

                let intermediate_size = self.model.layers[layer].mlp.get_params()[1];
                let hidden_size = self.model.layers[layer].mlp.get_params()[0];
                match expert_type {
                    AnyMoeExpertType::FineTuned => {
                        let (dtype, device) = self.model.layers[layer].mlp.dtype_device();
                        row.push(Box::new(Mlp::new(
                            &Config {
                                intermediate_size: self.model.layers[layer].mlp.get_params()[1],
                                hidden_size: self.model.layers[layer].mlp.get_params()[0],
                                ..Default::default()
                            },
                            vb.pp(layer).pp(&mlp).set_dtype(dtype).set_device(device),
                        )?));
                    }
                    AnyMoeExpertType::LoraAdapter {
                        rank,
                        alpha,
                        ref target_modules,
                    } => {
                        let vb_mlp = vb.pp(layer).pp(&mlp);

                        let gate_up_proj_delta =
                            if target_modules.contains(&"gate_up_proj".to_string()) {
                                Some(get_delta_from_lora_ab!(
                                    vb_mlp,
                                    rank,
                                    alpha,
                                    (hidden_size, 2 * intermediate_size),
                                    "gate_up_proj"
                                ))
                            } else {
                                None
                            };
                        let down_proj_delta = if target_modules.contains(&"down_proj".to_string()) {
                            Some(get_delta_from_lora_ab!(
                                vb_mlp,
                                rank,
                                alpha,
                                (hidden_size, intermediate_size),
                                "down_proj"
                            ))
                        } else {
                            None
                        };

                        row.push(
                            self.model.layers[layer]
                                .mlp
                                .new_added_delta(vec![gate_up_proj_delta, down_proj_delta])?,
                        );
                    }
                }
            }
        }
        for (layer, expert) in layers.into_iter().zip(experts) {
            let mut experts_all = vec![self.model.layers[layer].mlp.clone()];
            experts_all.extend(expert);
            let (dtype, device) = self.model.layers[layer].mlp.dtype_device();
            self.model.layers[layer].mlp = Box::new(MoeMlp::new(
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
}
