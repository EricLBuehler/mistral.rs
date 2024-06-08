#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::HashMap;

use crate::lora::{
    get_lora_cfg, AdapterSwapper, LinearLayerLike, LoraConfig, Merge, Ordering, QLoraLinear,
};
use candle_core::quantized::QMatMul;
use candle_core::quantized::{ggml_file, gguf_file};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, Module, RotaryEmbedding, VarBuilder};
use tqdm::Iter;
use tracing::info;

use crate::device_map::DeviceMapper;
use crate::layers::{repeat_kv, CausalMasker, MatMul, QRmsNorm, ScaledDotProductAttention};
use crate::pipeline::{extract_logits, Cache};
use crate::DeviceMapMetadata;

use super::classifier::XLoraClassifier;
use super::{verify_sanity_adapters, NonGranularState, ScalingsMaker, XLoraConfig};
use crate::models::quantized_llama::PropsGGUF;
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;

const MAX_SEQ_LEN: u32 = 4096;
const SUPPORTED_LAYERS: [&str; 7] = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "mlp.gate_proj",
];

#[derive(Debug)]
struct Mlp {
    feed_forward_w1: QLoraLinear,
    feed_forward_w2: QLoraLinear,
    feed_forward_w3: QLoraLinear,
}

impl Mlp {
    fn forward(
        &self,
        xs: &Tensor,
        scalings: Option<Tensor>,
        global_scaling_weight: f64,
        is_scaling_pass: Option<f64>,
    ) -> Result<Tensor> {
        let w1 = self.feed_forward_w1.lora_forward(
            xs,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?;
        let w3 = self.feed_forward_w3.lora_forward(
            xs,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?;
        self.feed_forward_w2.lora_forward(
            &(candle_nn::ops::silu(&w1)? * w3)?,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )
    }
}

#[derive(Debug)]
enum MlpOrMoe {
    Mlp(Mlp),
    MoE {
        n_expert_used: usize,
        feed_forward_gate_inp: QMatMul,
        experts: Vec<Mlp>,
    },
}

impl MlpOrMoe {
    fn forward(
        &self,
        xs: &Tensor,
        scalings: Option<Tensor>,
        global_scaling_weight: f64,
        is_scaling_pass: Option<f64>,
    ) -> Result<Tensor> {
        match self {
            Self::MoE {
                feed_forward_gate_inp,
                experts,
                n_expert_used,
            } => {
                let (b_size, seq_len, hidden_dim) = xs.dims3()?;
                let xs = xs.reshape(((), hidden_dim))?;
                let router_logits = MatMul.qmatmul(&xs, feed_forward_gate_inp)?;
                let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

                // In order to extract topk, we extract the data from the tensor and manipulate it
                // directly. Maybe we will want to use some custom ops instead at some point.
                let routing_weights = routing_weights.to_dtype(DType::F32)?.to_vec2::<f32>()?;

                // routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
                // top_x contains the row indexes to evaluate for each expert.
                let mut top_x = vec![vec![]; experts.len()];
                let mut selected_rws = vec![vec![]; experts.len()];
                for (row_idx, rw) in routing_weights.iter().enumerate() {
                    let mut dst = (0..rw.len() as u32).collect::<Vec<u32>>();
                    dst.sort_by(|&i, &j| rw[j as usize].total_cmp(&rw[i as usize]));
                    let mut sum_routing_weights = 0f32;
                    for &expert_idx in dst.iter().take(*n_expert_used) {
                        let expert_idx = expert_idx as usize;
                        let routing_weight = rw[expert_idx];
                        sum_routing_weights += routing_weight;
                        top_x[expert_idx].push(row_idx as u32);
                    }
                    for &expert_idx in dst.iter().take(*n_expert_used) {
                        let expert_idx = expert_idx as usize;
                        let routing_weight = rw[expert_idx];
                        selected_rws[expert_idx].push(routing_weight / sum_routing_weights)
                    }
                }

                // routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
                // expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

                let mut ys = xs.zeros_like()?;
                for (expert_idx, expert_layer) in experts.iter().enumerate() {
                    let top_x = &top_x[expert_idx];
                    if top_x.is_empty() {
                        continue;
                    }
                    let top_x = Tensor::new(top_x.as_slice(), xs.device())?;
                    let selected_rws =
                        Tensor::new(selected_rws[expert_idx].as_slice(), xs.device())?
                            .reshape(((), 1))?;
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
                    let current_hidden_states =
                        current_hidden_states.broadcast_mul(&selected_rws)?;
                    ys = ys.index_add(&top_x, &current_hidden_states, 0)?;
                }

                let ys = ys.reshape((b_size, seq_len, hidden_dim))?;
                Ok(ys)
            }
            Self::Mlp(mlp) => {
                mlp.forward(xs, scalings.clone(), global_scaling_weight, is_scaling_pass)
            }
        }
    }
}

#[derive(Debug)]
struct LayerWeights {
    attention_wq: QLoraLinear,
    attention_wk: QLoraLinear,
    attention_wv: QLoraLinear,
    attention_wo: QLoraLinear,
    attention_norm: QRmsNorm,
    mlp_or_moe: MlpOrMoe,
    ffn_norm: QRmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rotary: RotaryEmbedding,
}

impl LayerWeights {
    #[allow(clippy::too_many_arguments)]
    fn forward_attn(
        &mut self,
        x: &Tensor,
        mask: &Option<Tensor>,
        start_offsets: &[usize],
        start_offsets_kernel: Tensor,
        kv_cache: &mut Option<(Tensor, Tensor)>,
        scalings: Option<Tensor>,
        global_scaling_weight: f64,
        is_scaling_pass: Option<f64>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, n_embd) = x.dims3()?;
        let q = self.attention_wq.lora_forward(
            x,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?;
        let k = self.attention_wk.lora_forward(
            x,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?;
        let v = self.attention_wv.lora_forward(
            x,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?;

        let mut q = q.reshape((b_sz * seq_len, self.n_head, self.head_dim))?;
        let mut k = k.reshape((b_sz * seq_len, self.n_kv_head, self.head_dim))?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;

        self.rotary
            .forward(start_offsets, &start_offsets_kernel, &mut q, &mut k, b_sz)?;

        if q.rank() == 3 {
            q = q
                .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            k = k
                .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
                .transpose(1, 2)?;
        }

        let (k, v) = Cache::update_kv_cache(kv_cache, k, v, false)?;

        let k = repeat_kv(k, self.n_head / self.n_kv_head)?.contiguous()?;
        let v = repeat_kv(v, self.n_head / self.n_kv_head)?.contiguous()?;

        let y = ScaledDotProductAttention.run_attention(
            &q,
            &k,
            &v,
            self.n_head,
            self.head_dim,
            mask.clone().as_ref(),
            false,
            b_sz,
            seq_len,
        )?;

        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, n_embd])?;
        let y = self.attention_wo.lora_forward(
            &y,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?;
        Ok(y)
    }
}

pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: QRmsNorm,
    output: QMatMul,
    pub device: Device,
    pub cache: Cache,
    xlora_classifier: Option<XLoraClassifier>,
    pub max_seq_len: usize,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
}

impl ModelConfig::FromAdapterGGML for ModelWeights {
    fn from_ggml(
        mut ct: ggml_file::Content,
        gqa: usize,
        lora_config: &[((String, String), LoraConfig)],
        vb: &VarBuilder,
        ordering: &Ordering,
        xlora_config: Option<XLoraConfig>,
        preload_adapters: &Option<HashMap<String, (VarBuilder, LoraConfig)>>,
    ) -> Result<Self> {
        let head_dim = (ct.hparams.n_embd / ct.hparams.n_head) as usize;
        let rotary = RotaryEmbedding::new_partial(
            10000.,
            head_dim,
            ct.hparams.n_rot as usize,
            MAX_SEQ_LEN as usize,
            &ct.device,
            false,
            DType::F32,
        )?;
        let tok_embeddings = ct.remove("tok_embeddings.weight")?;
        let tok_embeddings = tok_embeddings.dequantize(&ct.device)?;
        let norm = QRmsNorm::new(ct.remove("norm.weight")?, 1e-5)?;
        let output = ct.remove("output.weight")?;
        let mut layers = Vec::with_capacity(ct.hparams.n_layer as usize);
        let mut count = 0;
        for layer_idx in 0..ct.hparams.n_layer {
            let prefix = format!("layers.{layer_idx}");
            let attention_wq = ct.remove(&format!("{prefix}.attention.wq.weight"))?;
            let attention_wk = ct.remove(&format!("{prefix}.attention.wk.weight"))?;
            let attention_wv = ct.remove(&format!("{prefix}.attention.wv.weight"))?;
            let attention_wo = ct.remove(&format!("{prefix}.attention.wo.weight"))?;
            let mlp_or_moe = {
                let feed_forward_w1 = ct.remove(&format!("{prefix}.feed_forward.w1.weight"))?;
                let feed_forward_w2 = ct.remove(&format!("{prefix}.feed_forward.w2.weight"))?;
                let feed_forward_w3 = ct.remove(&format!("{prefix}.feed_forward.w3.weight"))?;
                let cfg_w1 = get_lora_cfg(&feed_forward_w1);
                let cfg_w2 = get_lora_cfg(&feed_forward_w2);
                let cfg_w3 = get_lora_cfg(&feed_forward_w3);
                MlpOrMoe::Mlp(Mlp {
                    feed_forward_w1: QLoraLinear::new(
                        QMatMul::from_qtensor(feed_forward_w1)?,
                        &cfg_w1,
                        lora_config,
                        vb,
                        ordering,
                        format!("model.layers.{layer_idx}.mlp.gate_proj"),
                        &mut count,
                        preload_adapters,
                    )?,
                    feed_forward_w2: QLoraLinear::new(
                        QMatMul::from_qtensor(feed_forward_w2)?,
                        &cfg_w2,
                        lora_config,
                        vb,
                        ordering,
                        format!("model.layers.{layer_idx}.mlp.down_proj"),
                        &mut count,
                        preload_adapters,
                    )?,
                    feed_forward_w3: QLoraLinear::new(
                        QMatMul::from_qtensor(feed_forward_w3)?,
                        &cfg_w3,
                        lora_config,
                        vb,
                        ordering,
                        format!("model.layers.{layer_idx}.mlp.up_proj"),
                        &mut count,
                        preload_adapters,
                    )?,
                })
            };
            let attention_norm = ct.remove(&format!("{prefix}.attention_norm.weight"))?;
            let ffn_norm = ct.remove(&format!("{prefix}.ffn_norm.weight"))?;
            let cfgq = get_lora_cfg(&attention_wq);
            let cfgk = get_lora_cfg(&attention_wk);
            let cfgv = get_lora_cfg(&attention_wv);
            let cfgo = get_lora_cfg(&attention_wo);
            layers.push(LayerWeights {
                attention_wq: QLoraLinear::new(
                    QMatMul::from_qtensor(attention_wq)?,
                    &cfgq,
                    lora_config,
                    vb,
                    ordering,
                    format!("model.layers.{layer_idx}.self_attn.q_proj"),
                    &mut count,
                    preload_adapters,
                )?,
                attention_wk: QLoraLinear::new(
                    QMatMul::from_qtensor(attention_wk)?,
                    &cfgk,
                    lora_config,
                    vb,
                    ordering,
                    format!("model.layers.{layer_idx}.self_attn.k_proj"),
                    &mut count,
                    preload_adapters,
                )?,
                attention_wv: QLoraLinear::new(
                    QMatMul::from_qtensor(attention_wv)?,
                    &cfgv,
                    lora_config,
                    vb,
                    ordering,
                    format!("model.layers.{layer_idx}.self_attn.v_proj"),
                    &mut count,
                    preload_adapters,
                )?,
                attention_wo: QLoraLinear::new(
                    QMatMul::from_qtensor(attention_wo)?,
                    &cfgo,
                    lora_config,
                    vb,
                    ordering,
                    format!("model.layers.{layer_idx}.self_attn.o_proj"),
                    &mut count,
                    preload_adapters,
                )?,
                attention_norm: QRmsNorm::new(attention_norm, 1e-5)?,
                mlp_or_moe,
                ffn_norm: QRmsNorm::new(ffn_norm, 1e-5)?,
                n_head: ct.hparams.n_head as usize,
                n_kv_head: ct.hparams.n_head as usize / gqa,
                head_dim: (ct.hparams.n_embd / ct.hparams.n_head) as usize,
                rotary: rotary.clone(),
            })
        }
        if xlora_config.is_none() && preload_adapters.is_none() {
            // We are now a LoRA model so we must merge the weights
            info!("Merging LoRA adapters.");
            for layer in layers.iter_mut().tqdm() {
                layer.attention_wk.merge_weights()?;
                layer.attention_wo.merge_weights()?;
                layer.attention_wq.merge_weights()?;
                layer.attention_wv.merge_weights()?;
                match &mut layer.mlp_or_moe {
                    MlpOrMoe::Mlp(ref mut m) => {
                        m.feed_forward_w1.merge_weights()?;
                        m.feed_forward_w2.merge_weights()?;
                        m.feed_forward_w3.merge_weights()?;
                    }
                    MlpOrMoe::MoE {
                        n_expert_used: _,
                        feed_forward_gate_inp: _,
                        experts,
                    } => {
                        for expert in experts {
                            expert.feed_forward_w1.merge_weights()?;
                            expert.feed_forward_w2.merge_weights()?;
                            expert.feed_forward_w3.merge_weights()?;
                        }
                    }
                }
            }
        }
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, ct.hparams.n_embd as usize),
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
            device: ct.device.clone(),
            cache: Cache::new(ct.hparams.n_layer as usize, true),
            xlora_classifier: xlora_config.map(|xlora_config| {
                XLoraClassifier::new(xlora_config, count, lora_config.len(), vb.clone(), true)
                    .unwrap()
            }),
            max_seq_len: MAX_SEQ_LEN as usize, // Cannot determine from ggml.
            mapper: None,
        })
    }
}

impl ModelConfig::FromAdapterGGUF for ModelWeights {
    #[allow(clippy::too_many_arguments)]
    fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
        lora_config: &[((String, String), LoraConfig)],
        vb: &VarBuilder,
        ordering: &Ordering,
        xlora_config: Option<XLoraConfig>,
        mapper: DeviceMapMetadata,
        preload_adapters: &Option<HashMap<String, (VarBuilder, LoraConfig)>>,
    ) -> Result<Self> {
        verify_sanity_adapters(ordering, &SUPPORTED_LAYERS)?;

        // Parameter extraction from metadata.
        let metadata = ContentMetadata {
            path_prefix: "llama",
            metadata: &ct.metadata,
        };
        let PropsGGUF {
            n_expert,
            n_expert_used,
            head_count,
            head_count_kv,
            block_count,
            embedding_length,
            rope_dim,
            rms_norm_eps,
            max_seq_len,
            rope_freq_base,
        } = PropsGGUF::try_from(metadata).or_else(|err| candle_core::bail!("{err}"))?;

        let head_dim = embedding_length / head_count;

        let tok_embeddings = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings.dequantize(device)?;
        let norm = QRmsNorm::new(
            ct.tensor(reader, "output_norm.weight", device)?,
            rms_norm_eps,
        )?;
        let output = ct.tensor(reader, "output.weight", device)?;
        let mut layers = Vec::with_capacity(block_count);
        let mut count = 0;
        let mapper = mapper.into_mapper(block_count, device)?;
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let device = mapper.device_for(layer_idx, false).unwrap_or(device);
            let rotary = RotaryEmbedding::new_partial(
                rope_freq_base,
                head_dim,
                rope_dim,
                max_seq_len,
                device,
                false,
                DType::F32,
            )?;

            let attention_wq = ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?;
            let attention_wk = ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?;
            let attention_wv = ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?;
            let attention_wo =
                ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?;
            let mlp_or_moe = if n_expert <= 1 {
                let feed_forward_w1 =
                    ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?;
                let feed_forward_w2 =
                    ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?;
                let feed_forward_w3 =
                    ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?;
                let cfg_w1 = get_lora_cfg(&feed_forward_w1);
                let cfg_w2 = get_lora_cfg(&feed_forward_w2);
                let cfg_w3 = get_lora_cfg(&feed_forward_w3);
                MlpOrMoe::Mlp(Mlp {
                    feed_forward_w1: QLoraLinear::new(
                        QMatMul::from_qtensor(feed_forward_w1)?,
                        &cfg_w1,
                        lora_config,
                        vb,
                        ordering,
                        format!("model.layers.{layer_idx}.mlp.gate_proj"),
                        &mut count,
                        preload_adapters,
                    )?,
                    feed_forward_w2: QLoraLinear::new(
                        QMatMul::from_qtensor(feed_forward_w2)?,
                        &cfg_w2,
                        lora_config,
                        vb,
                        ordering,
                        format!("model.layers.{layer_idx}.mlp.down_proj"),
                        &mut count,
                        preload_adapters,
                    )?,
                    feed_forward_w3: QLoraLinear::new(
                        QMatMul::from_qtensor(feed_forward_w3)?,
                        &cfg_w3,
                        lora_config,
                        vb,
                        ordering,
                        format!("model.layers.{layer_idx}.mlp.up_proj"),
                        &mut count,
                        preload_adapters,
                    )?,
                })
            } else {
                let feed_forward_gate_inp =
                    ct.tensor(reader, &format!("{prefix}.ffn_gate_inp.weight"), device)?;
                let mut experts = Vec::with_capacity(n_expert);
                for i in 0..n_expert {
                    let feed_forward_w1 =
                        ct.tensor(reader, &format!("{prefix}.ffn_gate.{i}.weight"), device)?;
                    let feed_forward_w2 =
                        ct.tensor(reader, &format!("{prefix}.ffn_down.{i}.weight"), device)?;
                    let feed_forward_w3 =
                        ct.tensor(reader, &format!("{prefix}.ffn_up.{i}.weight"), device)?;
                    let cfg_w1 = get_lora_cfg(&feed_forward_w1);
                    let cfg_w2 = get_lora_cfg(&feed_forward_w2);
                    let cfg_w3 = get_lora_cfg(&feed_forward_w3);
                    experts.push(Mlp {
                        feed_forward_w1: QLoraLinear::new(
                            QMatMul::from_qtensor(feed_forward_w1)?,
                            &cfg_w1,
                            lora_config,
                            vb,
                            ordering,
                            format!("model.layers.{layer_idx}.mlp.gate_proj.{i}"),
                            &mut count,
                            preload_adapters,
                        )?,
                        feed_forward_w2: QLoraLinear::new(
                            QMatMul::from_qtensor(feed_forward_w2)?,
                            &cfg_w2,
                            lora_config,
                            vb,
                            ordering,
                            format!("model.layers.{layer_idx}.mlp.down_proj.{i}"),
                            &mut count,
                            preload_adapters,
                        )?,
                        feed_forward_w3: QLoraLinear::new(
                            QMatMul::from_qtensor(feed_forward_w3)?,
                            &cfg_w3,
                            lora_config,
                            vb,
                            ordering,
                            format!("model.layers.{layer_idx}.mlp.up_proj.{i}"),
                            &mut count,
                            preload_adapters,
                        )?,
                    })
                }
                MlpOrMoe::MoE {
                    n_expert_used,
                    feed_forward_gate_inp: QMatMul::from_qtensor(feed_forward_gate_inp)?,
                    experts,
                }
            };
            let attention_norm =
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?;
            let ffn_norm = ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?;
            let cfgq = get_lora_cfg(&attention_wq);
            let cfgk = get_lora_cfg(&attention_wk);
            let cfgv = get_lora_cfg(&attention_wv);
            let cfgo = get_lora_cfg(&attention_wo);
            layers.push(LayerWeights {
                attention_wq: QLoraLinear::new(
                    QMatMul::from_qtensor(attention_wq)?,
                    &cfgq,
                    lora_config,
                    vb,
                    ordering,
                    format!("model.layers.{layer_idx}.self_attn.q_proj"),
                    &mut count,
                    preload_adapters,
                )?,
                attention_wk: QLoraLinear::new(
                    QMatMul::from_qtensor(attention_wk)?,
                    &cfgk,
                    lora_config,
                    vb,
                    ordering,
                    format!("model.layers.{layer_idx}.self_attn.k_proj"),
                    &mut count,
                    preload_adapters,
                )?,
                attention_wv: QLoraLinear::new(
                    QMatMul::from_qtensor(attention_wv)?,
                    &cfgv,
                    lora_config,
                    vb,
                    ordering,
                    format!("model.layers.{layer_idx}.self_attn.v_proj"),
                    &mut count,
                    preload_adapters,
                )?,
                attention_wo: QLoraLinear::new(
                    QMatMul::from_qtensor(attention_wo)?,
                    &cfgo,
                    lora_config,
                    vb,
                    ordering,
                    format!("model.layers.{layer_idx}.self_attn.o_proj"),
                    &mut count,
                    preload_adapters,
                )?,
                attention_norm: QRmsNorm::new(attention_norm, rms_norm_eps)?,
                mlp_or_moe,
                ffn_norm: QRmsNorm::new(ffn_norm, rms_norm_eps)?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim: embedding_length / head_count,
                rotary: rotary.clone(),
            })
        }
        if xlora_config.is_none() && preload_adapters.is_none() {
            // We are now a LoRA model so we must merge the weights
            info!("Merging LoRA adapters.");
            for layer in layers.iter_mut().tqdm() {
                layer.attention_wk.merge_weights()?;
                layer.attention_wo.merge_weights()?;
                layer.attention_wq.merge_weights()?;
                layer.attention_wv.merge_weights()?;
                match &mut layer.mlp_or_moe {
                    MlpOrMoe::Mlp(ref mut m) => {
                        m.feed_forward_w1.merge_weights()?;
                        m.feed_forward_w2.merge_weights()?;
                        m.feed_forward_w3.merge_weights()?;
                    }
                    MlpOrMoe::MoE {
                        n_expert_used: _,
                        feed_forward_gate_inp: _,
                        experts,
                    } => {
                        for expert in experts {
                            expert.feed_forward_w1.merge_weights()?;
                            expert.feed_forward_w2.merge_weights()?;
                            expert.feed_forward_w3.merge_weights()?;
                        }
                    }
                }
            }
        }
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
            device: device.clone(),
            cache: Cache::new(block_count, true),
            xlora_classifier: xlora_config.map(|xlora_config| {
                XLoraClassifier::new(xlora_config, count, lora_config.len(), vb.clone(), true)
                    .unwrap()
            }),
            max_seq_len,
            mapper: Some(mapper),
        })
    }
}

impl ModelWeights {
    pub fn activate_adapters(&mut self, adapter_names: Vec<String>) -> Result<usize> {
        let mut sum = 0;
        for layer in self.layers.iter_mut() {
            sum += layer.attention_wk.activate(&adapter_names)?;
            sum += layer.attention_wo.activate(&adapter_names)?;
            sum += layer.attention_wq.activate(&adapter_names)?;
            sum += layer.attention_wv.activate(&adapter_names)?;
            match &mut layer.mlp_or_moe {
                MlpOrMoe::Mlp(ref mut m) => {
                    sum += m.feed_forward_w1.activate(&adapter_names)?;
                    sum += m.feed_forward_w2.activate(&adapter_names)?;
                    sum += m.feed_forward_w3.activate(&adapter_names)?;
                }
                MlpOrMoe::MoE {
                    n_expert_used: _,
                    feed_forward_gate_inp: _,
                    experts,
                } => {
                    for expert in experts {
                        sum += expert.feed_forward_w1.activate(&adapter_names)?;
                        sum += expert.feed_forward_w2.activate(&adapter_names)?;
                        sum += expert.feed_forward_w3.activate(&adapter_names)?;
                    }
                }
            }
        }
        Ok(sum)
    }

    #[allow(clippy::too_many_arguments)]
    fn inner_forward(
        &mut self,
        x: &Tensor,
        start_offsets: &[usize],
        start_offsets_kernel: Tensor,
        scalings: Option<Tensor>,
        is_full_pass: bool,
        no_kv_cache: bool,
        is_scaling_pass: Option<f64>,
    ) -> Result<Tensor> {
        let mut layer_in = self.tok_embeddings.forward(x)?;
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
        let mask = CausalMasker.make_causal_mask_as_attn_bias(
            x,
            &cache,
            x.dtype(),
            self.layers[0].n_head,
        )?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            if let Some(ref mapper) = self.mapper {
                layer_in = mapper.map(layer_in, i)?;
            }
            let x = layer_in;
            let residual = &x;
            let x = layer.attention_norm.forward(&x)?;
            let attn = layer.forward_attn(
                &x,
                &mask.as_ref().map(|m| m.to_device(x.device()).unwrap()),
                start_offsets,
                start_offsets_kernel.clone(),
                &mut cache[i],
                scalings.clone(),
                self.xlora_classifier
                    .as_ref()
                    .map(|classifier| classifier.get_global_scaling_weight())
                    .unwrap_or(1.0),
                is_scaling_pass,
            )?;
            let x = (attn + residual)?;

            // MLP
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp_or_moe.forward(
                &x,
                scalings.clone(),
                self.xlora_classifier
                    .as_ref()
                    .map(|classifier| classifier.get_global_scaling_weight())
                    .unwrap_or(1.0),
                is_scaling_pass,
            )?;
            let x = (x + residual)?;
            layer_in = x;
        }
        let layer_in = layer_in.to_device(&self.device)?;
        self.norm.forward(&layer_in)
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
                &vec![usize::MAX; context_lens.len()],
            )?;

            if no_kv_cache {
                extract_logits(
                    &MatMul.qmatmul(
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
                            .contiguous()?,
                        &self.output,
                    )?,
                    context_lens,
                )
            } else {
                // is_full_pass=true is ok because no_kv_cache=false
                extract_logits(
                    &MatMul.qmatmul(
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
                            .contiguous()?,
                        &self.output,
                    )?,
                    context_lens,
                )
            }
        } else {
            extract_logits(
                &MatMul.qmatmul(
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
                        .contiguous()?,
                    &self.output,
                )?,
                context_lens,
            )
        }
    }
}

impl ScalingsMaker for ModelWeights {
    fn dtype(&self) -> DType {
        DType::F32 // for dummy scalings
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
        _context_lens: &[usize],
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
