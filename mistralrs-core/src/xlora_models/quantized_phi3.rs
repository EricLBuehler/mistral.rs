#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::HashMap;

use crate::device_map::DeviceMapper;
use crate::layers::repeat_kv;
use crate::layers::CausalMasker;
use crate::layers::MatMul;
use crate::layers::RmsNorm;
use crate::layers::ScaledDotProductAttention;
use crate::lora::get_lora_cfg;
use crate::lora::AdapterSwapper;
use crate::lora::LinearLayerLike;
use crate::lora::LoraConfig;
use crate::lora::Merge;
use crate::lora::Ordering;
use crate::lora::QLoraLinear;
use crate::pipeline::extract_logits;
use crate::DeviceMapMetadata;
use candle_core::quantized::gguf_file;
use candle_core::quantized::QMatMul;
use candle_core::quantized::QTensor;
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::Embedding;
use candle_nn::VarBuilder;
use tqdm::Iter;
use tracing::info;

use super::classifier::XLoraClassifier;
use super::verify_sanity_adapters;
use super::Cache;
use super::NonGranularState;
use super::ScalingsMaker;
use super::XLoraConfig;
use crate::models::quantized_phi3::PropsGGUF;
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;

const SUPPORTED_LAYERS: [&str; 4] = [
    "self_attn.qkv_proj",
    "self_attn.o_proj",
    "mlp.gate_up_proj",
    "mlp.down_proj",
];

#[derive(Debug)]
struct Mlp {
    ffn_up: QLoraLinear,
    ffn_down: QLoraLinear,
    i_size: usize,
}

impl Mlp {
    fn forward(
        &self,
        xs: &Tensor,
        scalings: Option<Tensor>,
        global_scaling_weight: f64,
        is_scaling_pass: Option<f64>,
    ) -> Result<Tensor> {
        let up_states = self.ffn_up.lora_forward(
            xs,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?;
        let gate = up_states.narrow(D::Minus1, 0, self.i_size)?;
        let up_states = up_states.narrow(D::Minus1, self.i_size, self.i_size)?;
        let up_states = (up_states * gate.silu()?)?;
        self.ffn_down
            .lora_forward(&up_states, scalings, global_scaling_weight, is_scaling_pass)
    }
}

fn rms_norm(w: QTensor, eps: f64) -> Result<RmsNorm> {
    let w = w.dequantize(&w.device())?;
    let rms = RmsNorm::from_w(w, eps)?;
    Ok(rms)
}

#[derive(Debug)]
struct LayerWeights {
    attn_qkv: QLoraLinear,
    attn_output: QLoraLinear,
    attn_norm: RmsNorm,
    ffn_norm: RmsNorm,
    mlp: Mlp,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
    sliding_window: usize,
}

impl LayerWeights {
    fn apply_rotary_emb(&self, xs: &Tensor, seqlen_offsets: &[usize]) -> Result<Tensor> {
        let (_b_sz, _h, seq_len, _n_embd) = xs.dims4()?;
        let mut outputs = Vec::new();
        for (i, offset) in seqlen_offsets.iter().enumerate() {
            let cos = self.cos.narrow(0, *offset, seq_len)?;
            let sin = self.sin.narrow(0, *offset, seq_len)?;
            outputs.push(candle_nn::rotary_emb::rope(
                &xs.i(i)?.unsqueeze(0)?.contiguous()?,
                &cos,
                &sin,
            )?);
        }
        Tensor::cat(&outputs, 0)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_attn(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut Option<(Tensor, Tensor)>,
        scalings: Option<Tensor>,
        global_scaling_weight: f64,
        is_scaling_pass: Option<f64>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, n_embd) = x.dims3()?;
        let qkv = self.attn_qkv.lora_forward(
            x,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?;

        let query_pos = self.n_head * self.head_dim;
        let q = qkv.narrow(D::Minus1, 0, query_pos)?;
        let k = qkv.narrow(D::Minus1, query_pos, self.n_kv_head * self.head_dim)?;
        let v = qkv.narrow(
            D::Minus1,
            query_pos + self.n_kv_head * self.head_dim,
            self.n_kv_head * self.head_dim,
        )?;

        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;

        let q = self.apply_rotary_emb(&q, seqlen_offsets)?.contiguous()?;
        let k = self.apply_rotary_emb(&k, seqlen_offsets)?;

        let (k, v, attn_mask) = Cache::update_kv_cache_sliding_window(
            kv_cache,
            k,
            v,
            mask,
            Some(self.sliding_window),
            true,
        )?;

        let k = repeat_kv(k, self.n_head / self.n_kv_head)?;
        let v = repeat_kv(v, self.n_head / self.n_kv_head)?;

        let y = ScaledDotProductAttention.run_attention(
            &q,
            &k,
            &v,
            self.n_head,
            self.head_dim,
            attn_mask.as_ref(),
            false,
            b_sz,
            seq_len,
        )?;

        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, n_embd])?;
        let y =
            self.attn_output
                .lora_forward(&y, scalings, global_scaling_weight, is_scaling_pass)?;
        Ok(y)
    }
}

pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    output_norm: RmsNorm,
    output: QMatMul,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
    pub device: Device,
    pub cache: Cache,
    pub max_seq_len: usize,
    xlora_classifier: Option<XLoraClassifier>,
}

fn precomput_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    device: &Device,
    context_window: usize,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, context_window as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((context_window, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let cos = idx_theta.cos()?;
    let sin = idx_theta.sin()?;
    Ok((cos, sin))
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
            path_prefix: "phi3",
            metadata: &ct.metadata,
        };
        let PropsGGUF {
            head_count,
            head_count_kv,
            block_count,
            embedding_length,
            i_size,
            rope_dim,
            rms_eps,
            context_window,
        } = PropsGGUF::try_from(metadata).or_else(|err| candle_core::bail!("{err}"))?;

        let (cos, sin) = precomput_freqs_cis(rope_dim, 10_000., device, context_window)?;

        let tok_embeddings = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings.dequantize(device)?;
        let output_norm = rms_norm(ct.tensor(reader, "output_norm.weight", device)?, rms_eps)?;
        let output = QMatMul::from_qtensor(ct.tensor(reader, "output.weight", device)?)?;
        let mut layers = Vec::with_capacity(block_count);
        let mapper = mapper.into_mapper(block_count, device)?;
        let mut count = 0;
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let device = mapper.device_for(layer_idx, false).unwrap_or(device);
            let ffn_up = ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?;
            let ffn_down = ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?;
            let cfg_up = get_lora_cfg(&ffn_up);
            let cfg_down = get_lora_cfg(&ffn_down);
            let mlp = Mlp {
                ffn_up: QLoraLinear::new(
                    QMatMul::from_qtensor(ffn_up)?,
                    &cfg_up,
                    lora_config,
                    vb,
                    ordering,
                    format!("{prefix}.mlp.gate_up_proj"),
                    &mut count,
                    preload_adapters,
                )?,
                ffn_down: QLoraLinear::new(
                    QMatMul::from_qtensor(ffn_down)?,
                    &cfg_down,
                    lora_config,
                    vb,
                    ordering,
                    format!("{prefix}.mlp.down_proj"),
                    &mut count,
                    preload_adapters,
                )?,
                i_size,
            };
            let attn_norm = rms_norm(
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?,
                rms_eps,
            )?;
            let ffn_norm = rms_norm(
                ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?,
                rms_eps,
            )?;
            let qkv = ct.tensor(reader, &format!("{prefix}.attn_qkv.weight"), device)?;
            let output = ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?;
            let cfg_qkv = get_lora_cfg(&qkv);
            let cfg_out = get_lora_cfg(&output);
            layers.push(LayerWeights {
                attn_qkv: QLoraLinear::new(
                    QMatMul::from_qtensor(qkv)?,
                    &cfg_qkv,
                    lora_config,
                    vb,
                    ordering,
                    format!("{prefix}.self_attn.qkv_proj"),
                    &mut count,
                    preload_adapters,
                )?,
                attn_output: QLoraLinear::new(
                    QMatMul::from_qtensor(output)?,
                    &cfg_out,
                    lora_config,
                    vb,
                    ordering,
                    format!("{prefix}.self_attn.o_proj"),
                    &mut count,
                    preload_adapters,
                )?,
                attn_norm,
                ffn_norm,
                mlp,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim: embedding_length / head_count,
                cos: cos.to_device(device)?,
                sin: sin.to_device(device)?,
                sliding_window: context_window,
            })
        }
        if xlora_config.is_none() {
            // We are now a LoRA model so we must merge the weights
            info!("Merging LoRA adapters.");
            for layer in layers.iter_mut().tqdm() {
                layer.attn_qkv.merge_weights()?;
                layer.attn_output.merge_weights()?;
                layer.mlp.ffn_down.merge_weights()?;
                layer.mlp.ffn_up.merge_weights()?;
            }
        }
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            output_norm,
            output,
            mapper: Some(mapper),
            device: device.clone(),
            cache: Cache::new(block_count, true),
            max_seq_len: context_window,
            xlora_classifier: xlora_config.map(|xlora_config| {
                XLoraClassifier::new(xlora_config, count, lora_config.len(), vb.clone(), true)
                    .unwrap()
            }),
        })
    }
}

impl ModelWeights {
    pub fn activate_adapters(&mut self, adapter_names: Vec<String>) -> Result<usize> {
        let mut sum = 0;
        for layer in self.layers.iter_mut() {
            sum += layer.attn_qkv.activate(&adapter_names)?;
            sum += layer.attn_output.activate(&adapter_names)?;
            sum += layer.mlp.ffn_down.activate(&adapter_names)?;
            sum += layer.mlp.ffn_up.activate(&adapter_names)?;
        }
        Ok(sum)
    }

    pub fn inner_forward(
        &mut self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        scalings: Option<Tensor>,
        is_full_pass: bool,
        no_kv_cache: bool,
        is_scaling_pass: Option<f64>,
    ) -> Result<Tensor> {
        let (_b_sz, seq_len) = input_ids.dims2()?;
        let mut xs = self.tok_embeddings.forward(input_ids)?;
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
        let mask = CausalMasker.make_causal_mask_with_sliding_window_as_attn_bias(
            input_ids,
            &cache,
            Some(self.max_seq_len),
            xs.dtype(),
            self.layers[0].n_head,
        )?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            if let Some(ref mapper) = self.mapper {
                xs = mapper.map(xs, i)?;
            }
            let residual = &xs;
            let ys = xs.apply(&layer.attn_norm)?;
            let ys = layer.forward_attn(
                &ys,
                mask.as_ref()
                    .map(|m| m.to_device(xs.device()).unwrap())
                    .as_ref(),
                seqlen_offsets,
                &mut cache[i],
                scalings.clone(),
                self.xlora_classifier
                    .as_ref()
                    .map(|classifier| classifier.get_global_scaling_weight())
                    .unwrap_or(1.0),
                is_scaling_pass,
            )?;
            let ys = (ys + residual)?;
            let residual = &ys;
            let ys = ys.apply(&layer.ffn_norm)?;
            let ys = layer.mlp.forward(
                &ys,
                scalings.clone(),
                self.xlora_classifier
                    .as_ref()
                    .map(|classifier| classifier.get_global_scaling_weight())
                    .unwrap_or(1.0),
                is_scaling_pass,
            )?;
            xs = (ys + residual)?
        }
        let xs = xs.to_device(&self.device)?;
        let xs = xs.apply(&self.output_norm)?.i((.., seq_len - 1, ..))?;
        self.output.forward(&xs)
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
                        .inner_forward(input_ids, seqlen_offsets, None, false, no_kv_cache, None)?
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
        _start_offsets_kernel: Tensor,
        scalings: Tensor,
        is_full_pass: bool,
        no_kv_cache: bool,
        is_scaling_pass: Option<f64>,
        _context_lens: &[usize],
    ) -> Result<Tensor> {
        self.inner_forward(
            input_ids,
            seqlen_offsets,
            Some(scalings),
            is_full_pass,
            no_kv_cache,
            is_scaling_pass,
        )
    }
}
