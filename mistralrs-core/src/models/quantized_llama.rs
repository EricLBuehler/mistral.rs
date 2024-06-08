#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::quantized::{ggml_file, gguf_file};
use candle_core::quantized::{QMatMul, QTensor};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, Module, RotaryEmbedding};

use crate::device_map::DeviceMapper;
use crate::layers::{repeat_kv, CausalMasker, MatMul, QRmsNorm, ScaledDotProductAttention};
use crate::pipeline::{extract_logits, Cache};
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;
use crate::DeviceMapMetadata;

const MAX_SEQ_LEN: u32 = 4096;

#[derive(Debug, Clone)]
struct Mlp {
    feed_forward_w1: QMatMul,
    feed_forward_w2: QMatMul,
    feed_forward_w3: QMatMul,
}

impl Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = MatMul.qmatmul(xs, &self.feed_forward_w1)?;
        let w3 = MatMul.qmatmul(xs, &self.feed_forward_w3)?;
        let y = &(candle_nn::ops::silu(&w1)? * w3)?;
        MatMul.qmatmul(y, &self.feed_forward_w2)
    }
}

#[derive(Debug, Clone)]
enum MlpOrMoe {
    Mlp(Mlp),
    MoE {
        n_expert_used: usize,
        feed_forward_gate_inp: QMatMul,
        experts: Vec<Mlp>,
    },
}

impl MlpOrMoe {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::MoE {
                feed_forward_gate_inp,
                experts,
                n_expert_used,
            } => {
                let (b_size, seq_len, hidden_dim) = xs.dims3()?;
                let xs = xs.reshape(((), hidden_dim))?;
                let router_logits = feed_forward_gate_inp.forward(&xs)?;
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
                    let current_hidden_states = expert_layer.forward(&current_state)?;
                    let current_hidden_states =
                        current_hidden_states.broadcast_mul(&selected_rws)?;
                    ys = ys.index_add(&top_x, &current_hidden_states, 0)?;
                }

                let ys = ys.reshape((b_size, seq_len, hidden_dim))?;
                Ok(ys)
            }
            Self::Mlp(mlp) => mlp.forward(xs),
        }
    }
}

#[derive(Debug, Clone)]
struct LayerWeights {
    attention_wq: QMatMul,
    attention_wk: QMatMul,
    attention_wv: QMatMul,
    attention_wo: QMatMul,
    attention_norm: QRmsNorm,
    mlp_or_moe: MlpOrMoe,
    ffn_norm: QRmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rotary: RotaryEmbedding,
}

impl LayerWeights {
    fn forward_attn(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        start_offsets: &[usize],
        start_offsets_kernel: Tensor,
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, n_embd) = x.dims3()?;

        let q = MatMul.qmatmul(x, &self.attention_wq)?;
        let k = MatMul.qmatmul(x, &self.attention_wk)?;
        let v = MatMul.qmatmul(x, &self.attention_wv)?;

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

        let k = repeat_kv(k, self.n_head / self.n_kv_head)?;
        let v = repeat_kv(v, self.n_head / self.n_kv_head)?;

        let y = ScaledDotProductAttention.run_attention(
            &q,
            &k,
            &v,
            self.n_head,
            self.head_dim,
            mask,
            false,
            b_sz,
            seq_len,
        )?;

        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, n_embd])?;
        let y = MatMul.qmatmul(&y, &self.attention_wo)?;
        Ok(y)
    }
}

#[derive(Debug)]
pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: QRmsNorm,
    output: QMatMul,
    pub device: Device,
    pub cache: Cache,
    pub max_seq_len: usize,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
}

impl ModelConfig::FromGGML for ModelWeights {
    fn from_ggml(mut ct: ggml_file::Content, gqa: usize) -> Result<Self> {
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
                MlpOrMoe::Mlp(Mlp {
                    feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1)?,
                    feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2)?,
                    feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3)?,
                })
            };
            let attention_norm = ct.remove(&format!("{prefix}.attention_norm.weight"))?;
            let ffn_norm = ct.remove(&format!("{prefix}.ffn_norm.weight"))?;
            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_norm: QRmsNorm::new(attention_norm, 1e-5)?,
                mlp_or_moe,
                ffn_norm: QRmsNorm::new(ffn_norm, 1e-5)?,
                n_head: ct.hparams.n_head as usize,
                n_kv_head: ct.hparams.n_head as usize / gqa,
                head_dim: (ct.hparams.n_embd / ct.hparams.n_head) as usize,
                rotary: rotary.clone(),
            })
        }
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, ct.hparams.n_embd as usize),
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
            device: ct.device.clone(),
            cache: Cache::new(ct.hparams.n_layer as usize, false),
            max_seq_len: MAX_SEQ_LEN as usize, // Cannot determine from ggml.
            mapper: None,
        })
    }
}

// llama `llm` fields:
// https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#llm
// NOTE: Types here do not match spec
pub(crate) struct PropsGGUF {
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub head_count: usize,
    pub head_count_kv: usize,
    pub block_count: usize,
    pub embedding_length: usize,
    pub rope_dim: usize,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
    pub rope_freq_base: f32,
}

impl TryFrom<ContentMetadata<'_>> for PropsGGUF {
    type Error = anyhow::Error;

    fn try_from(c: ContentMetadata) -> std::result::Result<Self, Self::Error> {
        c.verify_arch("llama")?;

        let required = [
            "attention.head_count",
            "attention.head_count_kv",
            "block_count",
            "embedding_length",
            "rope.dimension_count",
            "attention.layer_norm_rms_epsilon",
        ];
        c.has_required_keys(&required)?;

        // NOTE: Values are not aligned with GGUFv3 types
        // TODO: Normalize value types to spec
        let props = Self {
            n_expert: c.get_value::<u32>("expert_count").ok().unwrap_or(0) as usize,
            n_expert_used: c.get_value::<u32>("expert_used_count").ok().unwrap_or(0) as usize,
            head_count: c.get_value::<u32>("attention.head_count")? as usize,
            head_count_kv: c.get_value::<u32>("attention.head_count_kv")? as usize,
            block_count: c.get_value::<u32>("block_count")? as usize,
            embedding_length: c.get_value::<u32>("embedding_length")? as usize,
            rope_dim: c.get_value::<u32>("rope.dimension_count")? as usize,
            // Strangely this value is generally 1e-6 in GGUF file but used to be 1e-5 by default.
            rms_norm_eps: c.get_value("attention.layer_norm_rms_epsilon")?,
            max_seq_len: c
                .get_value::<u64>("context_length")
                .ok()
                .unwrap_or(MAX_SEQ_LEN as u64) as usize,
            rope_freq_base: c.get_value("rope.freq_base").ok().unwrap_or(10_000_f32),
        };

        Ok(props)
    }
}

impl ModelConfig::FromGGUF for ModelWeights {
    fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
        mapper: DeviceMapMetadata,
    ) -> Result<Self> {
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
                MlpOrMoe::Mlp(Mlp {
                    feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1)?,
                    feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2)?,
                    feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3)?,
                })
            } else {
                let feed_forward_gate_inp =
                    ct.tensor(reader, &format!("{prefix}.ffn_gate_inp.weight"), device)?;
                let mut experts = Vec::with_capacity(n_expert);
                match ct.tensor(reader, &format!("{prefix}.ffn_gate_exps.weight"), device) {
                    Ok(feed_forward_gate_exps) => {
                        let feed_forward_down_exps =
                            ct.tensor(reader, &format!("{prefix}.ffn_down_exps.weight"), device)?;
                        let feed_forward_up_exps =
                            ct.tensor(reader, &format!("{prefix}.ffn_up_exps.weight"), device)?;

                        let dequant_ffn_gate = feed_forward_gate_exps
                            .dequantize(device)?
                            .chunk(n_expert, 0)?;
                        let dequant_ffn_down = feed_forward_down_exps
                            .dequantize(device)?
                            .chunk(n_expert, 0)?;
                        let dequant_ffn_up = feed_forward_up_exps
                            .dequantize(device)?
                            .chunk(n_expert, 0)?;

                        assert_eq!(dequant_ffn_up.len(), dequant_ffn_down.len());
                        assert_eq!(dequant_ffn_gate.len(), dequant_ffn_down.len());
                        assert_eq!(dequant_ffn_gate.len(), n_expert);

                        let gate_type = feed_forward_gate_exps.dtype();
                        let down_type = feed_forward_down_exps.dtype();
                        let up_type = feed_forward_up_exps.dtype();

                        for (ff_w1, (ff_w2, ff_w3)) in dequant_ffn_gate
                            .into_iter()
                            .zip(dequant_ffn_down.into_iter().zip(dequant_ffn_up))
                        {
                            experts.push(Mlp {
                                feed_forward_w1: QMatMul::from_qtensor(QTensor::quantize(
                                    &ff_w1, gate_type,
                                )?)?,
                                feed_forward_w2: QMatMul::from_qtensor(QTensor::quantize(
                                    &ff_w2, down_type,
                                )?)?,
                                feed_forward_w3: QMatMul::from_qtensor(QTensor::quantize(
                                    &ff_w3, up_type,
                                )?)?,
                            })
                        }
                    }
                    Err(_) => {
                        for i in 0..n_expert {
                            let feed_forward_w1 = ct.tensor(
                                reader,
                                &format!("{prefix}.ffn_gate.{i}.weight"),
                                device,
                            )?;
                            let feed_forward_w2 = ct.tensor(
                                reader,
                                &format!("{prefix}.ffn_down.{i}.weight"),
                                device,
                            )?;
                            let feed_forward_w3 =
                                ct.tensor(reader, &format!("{prefix}.ffn_up.{i}.weight"), device)?;
                            experts.push(Mlp {
                                feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1)?,
                                feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2)?,
                                feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3)?,
                            })
                        }
                    }
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
            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_norm: QRmsNorm::new(attention_norm, rms_norm_eps)?,
                mlp_or_moe,
                ffn_norm: QRmsNorm::new(ffn_norm, rms_norm_eps)?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                rotary: rotary.clone(),
            })
        }
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
            device: device.clone(),
            cache: Cache::new(block_count, false),
            max_seq_len,
            mapper: Some(mapper),
        })
    }
}

impl ModelWeights {
    pub fn forward(
        &mut self,
        x: &Tensor,
        start_offsets: &[usize],
        start_offsets_kernel: Tensor,
        context_lens: Vec<(usize, usize)>,
    ) -> Result<Tensor> {
        let mut layer_in = self.tok_embeddings.forward(x)?;
        let mut cache = self.cache.lock();
        let mask = CausalMasker.make_causal_mask_as_attn_bias(
            x,
            &cache,
            DType::F32,
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
                mask.as_ref()
                    .map(|m| m.to_device(x.device()).unwrap())
                    .as_ref(),
                start_offsets,
                start_offsets_kernel.clone(),
                &mut cache[i],
            )?;
            let x = (attn + residual)?;

            // MLP
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp_or_moe.forward(&x)?;
            let x = (x + residual)?;
            layer_in = x;
        }
        let layer_in = layer_in.to_device(&self.device)?;
        let x = self.norm.forward(&layer_in)?;
        extract_logits(
            &MatMul.qmatmul(&x.contiguous()?, &self.output)?,
            context_lens,
        )
    }
}
