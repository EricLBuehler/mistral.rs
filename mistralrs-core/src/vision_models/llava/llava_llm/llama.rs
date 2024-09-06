//LLaMA without fused RoPE
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::too_many_arguments
)]

use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{embedding, linear_no_bias as linear, Embedding, Module, VarBuilder};
use mistralrs_quant::{QuantMethod, QuantMethodConfig, UnquantLinear};

use crate::{
    amoe::{AnyMoeBaseModelMixin, AnyMoeTrainableLayer, MlpLayer, MoeMlp},
    attention::SdpaParams,
    device_map::DeviceMapper,
    get_delta_from_lora_ab,
    layers::{CausalMasker, MatMul, RmsNorm, Sdpa},
    layers_masker::PastKvLenCache,
    models::llama::Config,
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        extract_logits,
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        IsqModel, NormalLoadingMetadata, NormalModel,
    },
    utils::progress::NiceProgressBar,
    AnyMoeConfig, AnyMoeExpertType,
};

use super::{LLaVALLM, OrdinaryRoPE};

struct CausalSelfAttention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

impl CausalSelfAttention {
    fn forward(
        &self,
        x: &Tensor,
        attention_mask: &Option<Tensor>,
        seqlen_offsets: &[usize],
        _start_offsets_kernel: Tensor,
        block_idx: usize,
        kv_cache: &mut crate::pipeline::LayerCaches,
        rope_parameter: (&Tensor, &Tensor),
        metadata: Option<((Tensor, Tensor), &mut PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let original_dtype = x.dtype();
        let mut x = x.clone();
        if let Some(t) = self.q_proj.quantized_act_type() {
            x = x.to_dtype(t)?;
        }
        let mut q = MatMul.qmethod_matmul(&x, &*self.q_proj)?;
        let mut k = MatMul.qmethod_matmul(&x, &*self.k_proj)?;
        let mut v = MatMul.qmethod_matmul(&x, &*self.v_proj)?;
        if self.q_proj.quantized_act_type().is_some() {
            q = q.to_dtype(original_dtype)?;
            k = k.to_dtype(original_dtype)?;
            v = v.to_dtype(original_dtype)?;
        }

        let mut q = q
            .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let mut k = k
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        q = OrdinaryRoPE::forward(&q, seqlen_offsets[0], rope_parameter.0, rope_parameter.1)?;
        k = OrdinaryRoPE::forward(&k, seqlen_offsets[0], rope_parameter.0, rope_parameter.1)?;
        let v = v
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;

        let mut y = match &self.paged_attn {
            Some(paged_attn) => {
                let ((key_cache, value_cache), input_metadata) = metadata.unwrap();
                paged_attn.forward(
                    &q,
                    &k,
                    &v,
                    attention_mask.clone().as_ref(),
                    Some(key_cache),
                    Some(value_cache),
                    input_metadata,
                    None,
                )?
            }
            None => {
                let (k, v) =
                    crate::pipeline::Cache::update_kv_cache(&mut kv_cache[block_idx], k, v, false)?;

                Sdpa.run_attention(
                    &q,
                    &k,
                    &v,
                    attention_mask.clone().as_ref(),
                    Some(flash_params),
                    &self.sdpa_params,
                )?
            }
        };

        if let Some(t) = self.q_proj.quantized_act_type() {
            y = y.to_dtype(t)?;
        }
        y = if attention_mask.is_some() {
            y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?
        } else {
            y.reshape((b_sz, seq_len, ()))?
        };
        let mut res = MatMul.qmethod_matmul(&y, &*self.o_proj)?;
        if self.q_proj.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }

    fn load(vb: VarBuilder, cfg: &Config, paged_attn: Option<PagedAttention>) -> Result<Self> {
        let size_in = cfg.hidden_size;
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
        let q_proj = mistralrs_quant::linear_no_bias(
            size_in,
            size_q,
            &cfg.quantization_config,
            vb.pp("q_proj"),
        )?;
        let k_proj = mistralrs_quant::linear_no_bias(
            size_in,
            size_kv,
            &cfg.quantization_config,
            vb.pp("k_proj"),
        )?;
        let v_proj = mistralrs_quant::linear_no_bias(
            size_in,
            size_kv,
            &cfg.quantization_config,
            vb.pp("v_proj"),
        )?;
        let o_proj = mistralrs_quant::linear_no_bias(
            size_q,
            size_in,
            &cfg.quantization_config,
            vb.pp("o_proj"),
        )?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            max_seq_len: cfg.max_position_embeddings,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
                use_flash_attn: cfg.use_flash_attn,
                softcap: None,
                softmax_scale: 1.0 / ((cfg.hidden_size / cfg.num_attention_heads) as f32).sqrt(),
                sliding_window: None,
            },
        })
    }
}
#[derive(Clone)]
struct Mlp {
    c_fc1: Arc<dyn QuantMethod>,
    c_fc2: Arc<dyn QuantMethod>,
    c_proj: Arc<dyn QuantMethod>,
    params: Vec<usize>,
}

impl Mlp {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let h_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let c_fc1 = mistralrs_quant::linear_no_bias(
            h_size,
            i_size,
            &cfg.quantization_config,
            vb.pp("gate_proj"),
        )?;
        let c_fc2 = mistralrs_quant::linear_no_bias(
            h_size,
            i_size,
            &cfg.quantization_config,
            vb.pp("up_proj"),
        )?;
        let c_proj = mistralrs_quant::linear_no_bias(
            i_size,
            h_size,
            &cfg.quantization_config,
            vb.pp("down_proj"),
        )?;
        Ok(Self {
            c_fc1,
            c_fc2,
            c_proj,
            params: vec![h_size, i_size],
        })
    }
}

impl AnyMoeTrainableLayer for Mlp {}

impl MlpLayer for Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let original_dtype = x.dtype();
        let mut x = x.clone();
        if let Some(t) = self.c_fc1.quantized_act_type() {
            x = x.to_dtype(t)?;
        }
        let x = (candle_nn::ops::silu(&MatMul.qmethod_matmul(&x, &*self.c_fc1)?)?
            * MatMul.qmethod_matmul(&x, &*self.c_fc2)?)?;
        let mut res = MatMul.qmethod_matmul(&x, &*self.c_proj)?;
        if self.c_fc1.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
    fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        vec![&mut self.c_fc1, &mut self.c_fc2, &mut self.c_proj]
    }
    fn clone(&self) -> Box<dyn MlpLayer> {
        Box::new(Clone::clone(self))
    }
    fn get_params(&self) -> &[usize] {
        &self.params
    }
    // c_fc1, c_fc2, c_proj
    fn new_added_delta(&self, deltas: Vec<Option<Tensor>>) -> Result<Box<dyn MlpLayer>> {
        let new_c_fc1 = if let Some(ref delta) = deltas[0] {
            self.c_fc1.add_delta_w(delta)?
        } else {
            self.c_fc1.clone()
        };
        let new_c_fc2 = if let Some(ref delta) = deltas[1] {
            self.c_fc2.add_delta_w(delta)?
        } else {
            self.c_fc2.clone()
        };
        let new_c_proj = if let Some(ref delta) = deltas[2] {
            self.c_proj.add_delta_w(delta)?
        } else {
            self.c_proj.clone()
        };

        Ok(Box::new(Self {
            c_fc1: new_c_fc1,
            c_fc2: new_c_fc2,
            c_proj: new_c_proj,
            params: self.params.clone(),
        }))
    }

    fn dtype_device(&self) -> (DType, Device) {
        self.c_fc1.dtype_and_device()
    }
}

struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    mlp: Box<dyn MlpLayer>,
}

impl Block {
    fn forward(
        &self,
        x: &Tensor,
        attention_mask: &Option<Tensor>,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        block_idx: usize,
        kv_cache: &mut crate::pipeline::LayerCaches,
        rope_parameters: (&Tensor, &Tensor),
        metadata: Option<((Tensor, Tensor), &mut PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let residual = x;
        let mut x = self.rms_1.forward(x)?;
        x = (self.attn.forward(
            &x,
            attention_mask,
            seqlen_offsets,
            start_offsets_kernel,
            block_idx,
            kv_cache,
            rope_parameters,
            metadata,
            flash_params,
        )? + residual)?;
        let residual = &x;
        x = (self.mlp.forward(&self.rms_2.forward(&x)?)? + residual)?;
        Ok(x)
    }

    fn load(
        vb: VarBuilder,
        cfg: &Config,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        paged_attn: Option<PagedAttention>,
    ) -> Result<Self> {
        let attn = CausalSelfAttention::load(
            mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq),
            cfg,
            paged_attn,
        )?;
        let mlp = Mlp::load(mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq), cfg)?;
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
            mlp: Box::new(mlp),
        })
    }
}

pub struct Llama {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: Arc<dyn QuantMethod>,
    kv_cache: crate::pipeline::Cache,
    device: Device,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    rope_parameters: (Tensor, Tensor),
    cfg: ModelConfigMetadata,
}

impl Llama {
    pub fn forward_input(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &mut PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let x = self.wte.forward(input_ids)?;
        self.forward_input_embed(
            input_ids,
            x,
            seqlen_offsets,
            start_offsets_kernel,
            context_lens,
            metadata,
            flash_params,
        )
    }

    pub fn new(
        cfg: &Config,
        vb: VarBuilder,
        _is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let mapper = normal_loading_metadata.mapper;
        let wte = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb.pp("model.embed_tokens"), false),
        )?;
        let lm_head = linear(
            cfg.hidden_size,
            cfg.vocab_size,
            mapper.set_nm_device(vb.pp("lm_head"), normal_loading_metadata.loading_isq),
        )?;
        let ln_f = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb.pp("model.norm"), false),
        )?;
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;

        let blocks: Vec<_> =
            NiceProgressBar::<_, 'b'>(0..cfg.num_hidden_layers, "Loading repeating layers")
                .into_iter()
                .map(|i| {
                    let paged_attn = match &attention_mechanism {
                        AttentionImplementation::Eager => None,
                        AttentionImplementation::PagedAttention => Some(
                            PagedAttention::new(
                                cfg.num_attention_heads,
                                head_dim,
                                (1.0 / (head_dim as f64).sqrt()) as f32,
                                Some(cfg.num_key_value_heads),
                                None,
                                &normal_loading_metadata.real_device,
                                None,
                            )
                            .expect("Failed to create PagedAttention"),
                        ),
                    };
                    Block::load(
                        vb.pp(format!("model.layers.{i}")),
                        cfg,
                        &*mapper,
                        i,
                        normal_loading_metadata.loading_isq,
                        paged_attn,
                    )
                    .expect("Failed to load block.")
                })
                .collect();
        let rope_parameters = OrdinaryRoPE::create_parameters(
            head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            vb.dtype(),
            &normal_loading_metadata.real_device,
        )?;
        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head: Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(lm_head))?),
            kv_cache: crate::pipeline::Cache::new(cfg.num_hidden_layers, false),
            device: normal_loading_metadata.real_device,
            mapper,
            rope_parameters,
            cfg: ModelConfigMetadata {
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_kv_heads: cfg.num_key_value_heads,
                num_attn_heads: cfg.num_attention_heads,
                sliding_window: None,
                head_dim: None,
            },
        })
    }
}

impl IsqModel for Llama {
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
}

impl LLaVALLM for Llama {
    fn embed(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.wte.forward(input_ids)
    }
    fn forward_input_embed(
        &self,
        input_ids: &Tensor,
        input_embed: Tensor,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        context_lens: Vec<(usize, usize)>,
        mut metadata: Option<(Vec<(Tensor, Tensor)>, &mut PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let mut x = input_embed;
        let mut cache = self.kv_cache.lock();
        let mask = CausalMasker.make_causal_mask_as_attn_bias(
            input_ids,
            metadata
                .as_ref()
                .map(|(_, _)| &seqlen_offsets as &dyn PastKvLenCache)
                .unwrap_or(&*cache as &dyn PastKvLenCache),
            x.dtype(),
            self.blocks[0].attn.num_attention_heads,
        )?;
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = self.mapper.map(x, block_idx)?;
            x = block.forward(
                &x,
                &mask.clone().map(|m| m.to_device(x.device()).unwrap()),
                seqlen_offsets,
                start_offsets_kernel.clone(),
                block_idx,
                &mut cache,
                (&self.rope_parameters.0, &self.rope_parameters.1),
                metadata
                    .as_mut()
                    .map(|(kv_cache, metadata)| (kv_cache[block_idx].clone(), &mut **metadata)),
                flash_params,
            )?;
        }
        x = x.to_device(&self.device)?;
        x = self.ln_f.forward(&x)?;
        if let Some(t) = self.lm_head.quantized_act_type() {
            x = x.to_dtype(t)?;
        }
        let xs = MatMul.qmethod_matmul(&x, &*self.lm_head)?;
        extract_logits(&xs, context_lens)
    }
}

impl NormalModel for Llama {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &mut PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        self.forward_input(
            input_ids,
            seqlen_offsets,
            start_offsets_kernel,
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
        _start_offsets_kernel: Tensor,
        _start_offsets_kernel_full: Tensor,
        _no_kv_cache: bool,
        _non_granular_state: &Option<crate::xlora_models::NonGranularState>,
        _context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        _flash_params: &FlashParams,
        _flash_params_full: &FlashParams,
    ) -> Result<Tensor> {
        unimplemented!()
    }
    fn cache(&self) -> &crate::pipeline::Cache {
        &self.kv_cache
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

impl AnyMoeBaseModelMixin for Llama {
    fn get_mlps(&self) -> Vec<&dyn MlpLayer> {
        let mut mlps = Vec::new();
        for layer in &self.blocks {
            mlps.push(&*layer.mlp);
        }
        mlps
    }
    fn get_mlps_mut(&mut self) -> Vec<&mut Box<dyn MlpLayer>> {
        let mut mlps = Vec::new();
        for layer in &mut self.blocks {
            mlps.push(&mut layer.mlp);
        }
        mlps
    }
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
            layers = (0..self.blocks.len()).collect::<Vec<_>>();
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

                let intermediate_size = self.blocks[layer].mlp.get_params()[1];
                let hidden_size = self.blocks[layer].mlp.get_params()[0];
                match expert_type {
                    AnyMoeExpertType::FineTuned => {
                        let (dtype, device) = self.blocks[layer].mlp.dtype_device();
                        row.push(Box::new(Mlp::load(
                            vb.pp(layer).pp(&mlp).set_dtype(dtype).set_device(device),
                            &Config {
                                intermediate_size: self.blocks[layer].mlp.get_params()[1],
                                hidden_size: self.blocks[layer].mlp.get_params()[0],
                                ..Default::default()
                            },
                        )?));
                    }
                    AnyMoeExpertType::LoraAdapter {
                        rank,
                        alpha,
                        ref target_modules,
                    } => {
                        let vb_mlp = vb.pp(layer).pp(&mlp);

                        let c_fc1_delta = if target_modules.contains(&"c_fc1".to_string()) {
                            Some(get_delta_from_lora_ab!(
                                vb_mlp,
                                rank,
                                alpha,
                                (hidden_size, intermediate_size),
                                "c_fc1"
                            ))
                        } else {
                            None
                        };
                        let c_fc2_delta = if target_modules.contains(&"c_fc2".to_string()) {
                            Some(get_delta_from_lora_ab!(
                                vb_mlp,
                                rank,
                                alpha,
                                (hidden_size, intermediate_size),
                                "c_fc2"
                            ))
                        } else {
                            None
                        };
                        let c_proj_delta = if target_modules.contains(&"c_proj".to_string()) {
                            Some(get_delta_from_lora_ab!(
                                vb_mlp,
                                rank,
                                alpha,
                                (intermediate_size, hidden_size),
                                "c_proj"
                            ))
                        } else {
                            None
                        };

                        row.push(self.blocks[layer].mlp.new_added_delta(vec![
                            c_fc1_delta,
                            c_fc2_delta,
                            c_proj_delta,
                        ])?);
                    }
                }
            }
        }
        for (layer, expert) in layers.into_iter().zip(experts) {
            let mut experts_all = vec![self.blocks[layer].mlp.clone()];
            experts_all.extend(expert);
            let (dtype, device) = self.blocks[layer].mlp.dtype_device();
            self.blocks[layer].mlp = Box::new(MoeMlp::new(
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
