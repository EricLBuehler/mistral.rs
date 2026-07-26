//! ERNIE-4.5-0.3B dense decoder: the PaddleOCR-VL language model.
//!
//! Mirrors the transformers `PaddleOCR*` text classes. Hidden states are `[batch, seq, hidden]` and
//! attention tensors `[batch, heads, seq, head_dim]`, driven by the engine KV cache / paged metadata.
//!
//! The parity-critical piece is the 3D **chunked** mrope: cos/sin are built per position-axis, then for
//! each channel chunk we pick axis `i % 3` (Qwen2.5-VL scheme), NOT the interleaved Qwen3-VL scheme.

use std::sync::Arc;

use super::config::TextConfig;
use crate::attention::{AttentionMask, Sdpa, SdpaParams};
use crate::device_map::DeviceMapper;
use crate::paged_attention::{AttentionImplementation, PagedAttention};
use crate::pipeline::text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata};
use crate::pipeline::KvCache as EngineKvCache;
use crate::utils::unvarbuilder::UnVarBuilder;
use candle_core::{DType, Device, Result, Tensor, D};
use mistralrs_quant::{
    ColumnParallelLayer, QuantMethod, ReplicatedLayer, RowParallelLayer, ShardedVarBuilder,
};

/// RMSNorm (`PaddleOCRRMSNorm`, eps from config): upcast f32, `x*rsqrt(mean(x^2)+eps)`, `* weight`.
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn load(vb: ShardedVarBuilder, dim: usize, eps: f64) -> Result<Self> {
        Ok(Self {
            weight: vb.get(dim, "weight")?,
            eps,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        // variance = mean(x^2) over the last dim, keepdim for broadcast.
        let var = x.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = x.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        // weight is applied AFTER casting back to the input dtype (moot at f32, kept for bf16 later).
        normed.to_dtype(dtype)?.broadcast_mul(&self.weight)
    }
}

/// `rotate_half` (neox): split the last dim in half, return `cat(-x2, x1)`.
fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let hd = x.dim(D::Minus1)?;
    let x1 = x.narrow(D::Minus1, 0, hd / 2)?;
    let x2 = x.narrow(D::Minus1, hd / 2, hd / 2)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

/// `inv_freq[i] = 1 / theta^(2i/head_dim)`, `i in 0..head_dim/2`, shape `[half]`. Computed in f32 to
/// mirror torch; precomputed at load so the decode hot path never re-allocates it.
fn rope_inv_freq(head_dim: usize, theta: f64, dev: &Device) -> Result<Tensor> {
    let half = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1f32 / (theta as f32).powf((2 * i) as f32 / head_dim as f32))
        .collect();
    Tensor::from_vec(inv_freq, half, dev)
}

/// Build the full-`head_dim` cos/sin tables from 3D `position_ids`.
///
/// `position_ids`: `[3, batch, seq]` (one row per t/h/w axis), `inv_freq`: `[head_dim/2]`. Returns
/// `(cos, sin)` each `[3, batch, seq, head_dim]`, mirrors `PaddleOCRRotaryEmbedding.forward` (inv_freq
/// outer pos, `cat(freqs, freqs)`, cos/sin).
fn rope_tables(position_ids: &Tensor, inv_freq: &Tensor) -> Result<(Tensor, Tensor)> {
    let (three, batch, seq) = position_ids.dims3()?;
    let half = inv_freq.dim(0)?;
    let pos = position_ids
        .to_dtype(DType::F32)?
        .reshape((three, batch, seq, 1))?;
    let freqs = pos.broadcast_mul(&inv_freq.reshape((1, 1, 1, half))?)?; // [3,batch,seq,half]
    let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?; // [3,batch,seq,head_dim]
    Ok((emb.cos()?, emb.sin()?))
}

/// Chunked-select of a cos/sin table `[3, batch, seq, head_dim]` down to `[batch, seq, head_dim]`.
///
/// `apply_multimodal_rotary_pos_emb`: split the last dim into sections `[16,24,24,16,24,24]`; for
/// chunk index `i` take axis-plane `i % 3`, then concat. This is the parity-critical mrope wiring.
fn mrope_select(table: &Tensor, sections_doubled: &[usize]) -> Result<Tensor> {
    let mut parts = Vec::with_capacity(sections_doubled.len());
    let mut offset = 0;
    for (i, &s) in sections_doubled.iter().enumerate() {
        let plane = i % 3;
        let chunk = table
            .narrow(D::Minus1, offset, s)? // [3, batch, seq, s]
            .narrow(0, plane, 1)? // [1, batch, seq, s]
            .squeeze(0)?; // [batch, seq, s]
        parts.push(chunk);
        offset += s;
    }
    Tensor::cat(&parts, D::Minus1) // [batch, seq, head_dim]
}

/// Apply rope to `x` `[batch, heads, seq, head_dim]` with `cos`/`sin` `[batch, seq, head_dim]`
/// (broadcast over the head axis).
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let cos = cos.unsqueeze(1)?; // [batch, 1, seq, head_dim]
    let sin = sin.unsqueeze(1)?;
    let a = x.broadcast_mul(&cos)?;
    let b = rotate_half(x)?.broadcast_mul(&sin)?;
    a + b
}

/// GQA self-attention with 3D chunked mrope, causal, scale head_dim^-0.5, no biases.
struct Attention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

impl Attention {
    fn load(
        vb: ShardedVarBuilder,
        cfg: &TextConfig,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let h = cfg.hidden_size;
        let (nh, nkv, hd) = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
        );
        let q_proj = ColumnParallelLayer::new(
            h,
            nh * hd,
            &cfg.quantization_config,
            false,
            comm,
            mapper.set_device(layer_idx, vb.pp("q_proj"), loading_isq),
        )?;
        let kv_shard = mistralrs_quant::compute_kv_shard(nkv, hd, comm)?;
        let k_proj = ColumnParallelLayer::new_with_shard(
            h,
            nkv * hd,
            &cfg.quantization_config,
            false,
            comm,
            kv_shard,
            mapper.set_device(layer_idx, vb.pp("k_proj"), loading_isq),
        )?;
        let v_proj = ColumnParallelLayer::new_with_shard(
            h,
            nkv * hd,
            &cfg.quantization_config,
            false,
            comm,
            kv_shard,
            mapper.set_device(layer_idx, vb.pp("v_proj"), loading_isq),
        )?;
        let o_proj = RowParallelLayer::new(
            nh * hd,
            h,
            &cfg.quantization_config,
            false,
            comm,
            mapper.set_device(layer_idx, vb.pp("o_proj"), loading_isq),
        )?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: nh / comm.world_size(),
            num_kv_heads: (nkv / comm.world_size()).max(1),
            head_dim: hd,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: mistralrs_quant::compute_n_kv_groups(nkv, nh, comm)?,
                softcap: None,
                softmax_scale: (hd as f32).powf(-0.5),
                sliding_window: None,
                sinks: None,
            },
        })
    }

    /// Engine attention for a `[batch, q_len, hidden]` block. With `paged_attn` the K/V ride the paged
    /// cache (`metadata` selects this layer's slot); otherwise the engine `NormalCache` slot is appended
    /// and `Sdpa` runs over the growing K/V (unit test asserts this matches full recompute). Returns
    /// `[batch, q_len, num_heads*head_dim]` for `o_proj`; the `q_len==1` decode reshape skips the
    /// transpose because it is a memory no-op.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: &AttentionMask,
        kv_cache: &mut EngineKvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: Option<&FlashParams>,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = x.dims3()?;
        let hd = self.head_dim;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;
        let (q, k, v) = if q_len != 1 {
            let q = q
                .reshape((b_sz, q_len, self.num_heads, hd))?
                .transpose(1, 2)?;
            let k = k
                .reshape((b_sz, q_len, self.num_kv_heads, hd))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, q_len, self.num_kv_heads, hd))?
                .transpose(1, 2)?;
            (q, k, v)
        } else {
            let q = q.reshape((b_sz, self.num_heads, q_len, hd))?;
            let k = k.reshape((b_sz, self.num_kv_heads, q_len, hd))?;
            let v = v.reshape((b_sz, self.num_kv_heads, q_len, hd))?;
            (q, k, v)
        };

        let q = apply_rope(&q, cos, sin)?.contiguous()?;
        let k = apply_rope(&k, cos, sin)?.contiguous()?;
        let v = v.contiguous()?;

        let attn = match &self.paged_attn {
            Some(paged_attn) => match metadata {
                Some(((key_cache, value_cache), input_metadata)) => paged_attn.forward(
                    &q,
                    &k,
                    &v,
                    mask,
                    Some(key_cache),
                    Some(value_cache),
                    input_metadata,
                    &self.sdpa_params,
                    flash_params,
                )?,
                // No metadata: imatrix-style prompt pass with no cache to populate.
                None => {
                    let input_metadata = PagedAttentionInputMetadata::dummy(q.device())?;
                    paged_attn.forward(
                        &q,
                        &k,
                        &v,
                        mask,
                        None,
                        None,
                        &input_metadata,
                        &self.sdpa_params,
                        flash_params,
                    )?
                }
            },
            None => {
                let (k, v) = kv_cache.append(&k, &v)?;
                // cache may store f16 (cpu_kv_f16); cast back to q's dtype for the compute-dtype attention.
                let k = k.contiguous()?.to_dtype(q.dtype())?;
                let v = v.contiguous()?.to_dtype(q.dtype())?;
                Sdpa.run_attention(&q, &k, &v, mask, flash_params, &self.sdpa_params)?
            }
        };

        // decode (mask None) returns [batch, seq, heads*hd] already; prefill needs the head transpose.
        let attn = if matches!(mask, AttentionMask::None) {
            attn.reshape((b_sz, q_len, ()))?
        } else {
            attn.transpose(1, 2)?.reshape((b_sz, q_len, ()))?
        };
        self.o_proj.forward(&attn)
    }
}

/// SwiGLU MLP: `down(silu(gate(x)) * up(x))`, no biases.
struct Mlp {
    gate_proj: Arc<dyn QuantMethod>,
    up_proj: Arc<dyn QuantMethod>,
    down_proj: Arc<dyn QuantMethod>,
}

impl Mlp {
    fn load(
        vb: ShardedVarBuilder,
        cfg: &TextConfig,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        let gate_proj = ColumnParallelLayer::new(
            h,
            i,
            &cfg.quantization_config,
            false,
            comm,
            mapper.set_device(layer_idx, vb.pp("gate_proj"), loading_isq),
        )?;
        let up_proj = ColumnParallelLayer::new(
            h,
            i,
            &cfg.quantization_config,
            false,
            comm,
            mapper.set_device(layer_idx, vb.pp("up_proj"), loading_isq),
        )?;
        let down_proj = RowParallelLayer::new(
            i,
            h,
            &cfg.quantization_config,
            false,
            comm,
            mapper.set_device(layer_idx, vb.pp("down_proj"), loading_isq),
        )?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

/// One pre-norm decoder layer: `h += attn(ln1(h)); h += mlp(ln2(h))`.
struct DecoderLayer {
    input_layernorm: RmsNorm,
    self_attn: Attention,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
}

impl DecoderLayer {
    fn load(
        vb: ShardedVarBuilder,
        cfg: &TextConfig,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        Ok(Self {
            input_layernorm: RmsNorm::load(
                mapper.set_device(layer_idx, vb.pp("input_layernorm"), false),
                cfg.hidden_size,
                cfg.rms_norm_eps,
            )?,
            self_attn: Attention::load(
                vb.pp("self_attn"),
                cfg,
                mapper,
                layer_idx,
                loading_isq,
                paged_attn,
                comm,
            )?,
            post_attention_layernorm: RmsNorm::load(
                mapper.set_device(layer_idx, vb.pp("post_attention_layernorm"), false),
                cfg.hidden_size,
                cfg.rms_norm_eps,
            )?,
            mlp: Mlp::load(vb.pp("mlp"), cfg, mapper, layer_idx, loading_isq, comm)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: &AttentionMask,
        kv_cache: &mut EngineKvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: Option<&FlashParams>,
    ) -> Result<Tensor> {
        let h = (x + self.self_attn.forward(
            &self.input_layernorm.forward(x)?,
            cos,
            sin,
            mask,
            kv_cache,
            metadata,
            flash_params,
        )?)?;
        let out = (&h
            + self
                .mlp
                .forward(&self.post_attention_layernorm.forward(&h)?)?)?;
        Ok(out)
    }
}

/// Output of a text forward pass.
pub struct TextOutput {
    /// `lm_head(norm(last_layer_out))` (== reference `logits`).
    pub logits: Tensor,
}

/// ERNIE text decoder + untied lm_head. Embeddings are NOT loaded here: this stage teacher-forces
/// the reference merged input embeds; the embed/scatter path is wired separately.
pub struct ErnieTextModel {
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Arc<dyn QuantMethod>,
    inv_freq: Tensor,
    cfg: TextConfig,
}

impl ErnieTextModel {
    /// `vb` is the checkpoint root (keys `model.layers.*`, `model.norm.*`, `lm_head.*`).
    pub fn load(
        vb: ShardedVarBuilder,
        cfg: &TextConfig,
        mapper: &dyn DeviceMapper,
        loading_isq: bool,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let vm = vb.pp("model");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let comm = mapper.get_comm_for(i)?;
            let device = mapper.device_for(i, false).unwrap_or(vb.device());
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => {
                    Some(PagedAttention::new(cfg.head_dim, device, None)?)
                }
            };
            layers.push(DecoderLayer::load(
                vm.pp("layers").pp(i),
                cfg,
                mapper,
                i,
                loading_isq,
                paged_attn,
                &comm,
            )?);
        }
        let norm = RmsNorm::load(
            mapper.set_nm_device(vm.pp("norm"), false),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;
        let lm_head = ReplicatedLayer::new(
            cfg.hidden_size,
            cfg.vocab_size,
            &cfg.quantization_config,
            false,
            mapper.set_nm_device(vb.pp("lm_head"), loading_isq),
        )?;
        let inv_freq = rope_inv_freq(cfg.head_dim, cfg.rope_theta, vb.device())?;
        Ok(Self {
            layers,
            norm,
            lm_head,
            inv_freq,
            cfg: cfg.clone(),
        })
    }

    /// Engine forward the `MultimodalModel` trait calls; `mask` is the engine causal mask
    /// (`Custom` on prefill, `None` on single-token decode) built by the caller. With `paged` set, each
    /// layer reads its `(key_cache, value_cache)` slot and paged metadata; otherwise it drives the
    /// per-layer `NormalCache` slots (`caches[i]`, `EngineKvCache::append`) and matches a single-shot run
    /// over the whole sequence (unit test `engine_cache_matches_full_recompute`).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        inputs_embeds: &Tensor,
        position_ids: &Tensor,
        caches: &mut [EngineKvCache],
        mask: &AttentionMask,
        paged: Option<(&[(Tensor, Tensor)], &PagedAttentionInputMetadata)>,
        flash_params: Option<&FlashParams>,
    ) -> Result<TextOutput> {
        let dev = inputs_embeds.device();

        let inv_freq = self.inv_freq.to_device(dev)?;
        let (cos_t, sin_t) = rope_tables(position_ids, &inv_freq)?;
        let sections_doubled: Vec<usize> =
            [self.cfg.mrope_section, self.cfg.mrope_section].concat();
        let cos = mrope_select(&cos_t, &sections_doubled)?;
        let sin = mrope_select(&sin_t, &sections_doubled)?;

        // rope tables are built in f32; cast to the activation dtype (bf16 on the GPU path).
        let dtype = inputs_embeds.dtype();
        let (cos, sin) = (cos.to_dtype(dtype)?, sin.to_dtype(dtype)?);

        let mut h = inputs_embeds.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            let metadata = paged.map(|(kv, meta)| (kv[i].clone(), meta));
            h = layer.forward(&h, &cos, &sin, mask, &mut caches[i], metadata, flash_params)?;
        }
        let normed = self.norm.forward(&h)?;
        let logits = self.lm_head.forward(&normed)?;
        Ok(TextOutput { logits })
    }

    /// Non-quantized residuals (RMSNorm weights) keyed with the checkpoint's `model.*` paths. The
    /// q/k/v/o, MLP, and lm_head projections are ISQ-quantized so they are excluded here.
    pub fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        let uvb_m = uvb.pp("model");
        uvb_m
            .pp("norm")
            .add_tensor("weight", self.norm.weight.clone());
        for (i, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb_m.pp("layers").pp(i);
            uvb_l
                .pp("input_layernorm")
                .add_tensor("weight", layer.input_layernorm.weight.clone());
            uvb_l
                .pp("post_attention_layernorm")
                .add_tensor("weight", layer.post_attention_layernorm.weight.clone());
        }
        uvb.to_safetensors()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::CausalMasker;
    use crate::layers_masker::{CausalMaskConfig, PastKvLenCache};
    use candle_core::Var;
    use candle_nn::VarMap;
    use mistralrs_quant::ShardedSafeTensors;
    use rand::{rngs::StdRng, SeedableRng};
    use rand_distr::{Distribution, Normal};

    const TEST_SEED: u64 = 0x5EED_5EED;

    fn randn_vec(rng: &mut StdRng, mean: f32, std: f32, n: usize) -> Vec<f32> {
        let normal = Normal::new(mean, std).unwrap();
        (0..n).map(|_| normal.sample(rng)).collect()
    }

    // Engine causal mask the same way `MultimodalModel::forward` builds it: `[1, n_new]` ids over a
    // `[offset]` past-length cache. `None` for single-token decode, `Custom` otherwise.
    fn causal_mask(n_new: usize, offset: usize, dev: &Device) -> AttentionMask {
        let ids = Tensor::zeros((1, n_new), DType::U32, dev).unwrap();
        let offsets = [offset];
        let offsets_slice = offsets.as_slice();
        CausalMasker
            .make_causal_mask(
                &ids,
                &offsets_slice as &dyn PastKvLenCache,
                DType::F32,
                &CausalMaskConfig::default(),
            )
            .unwrap()
    }

    fn tiny_cfg() -> TextConfig {
        TextConfig {
            hidden_size: 12,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 6,
            intermediate_size: 16,
            vocab_size: 10,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            mrope_section: [1, 1, 1],
            quantization_config: None,
        }
    }

    fn max_abs(a: &Tensor, b: &Tensor) -> f32 {
        a.sub(b)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    // Deterministic f32 model (fixed-seed rng) whose keys match `ErnieTextModel::load`; norm weights
    // centered at 1 so the outputs are non-degenerate (else the equality assert below passes vacuously).
    fn tiny_model(cfg: &TextConfig, dev: &Device, rng: &mut StdRng) -> ErnieTextModel {
        let (lin, nrm) = ((0.0f32, 0.3f32), (1.0f32, 0.05f32)); // (mean, stdev): linears / norm weights
        let (h, nh, nkv, hd, inter, vocab) = (
            cfg.hidden_size,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            cfg.intermediate_size,
            cfg.vocab_size,
        );
        let vm = VarMap::new();
        {
            let mut data = vm.data().lock().unwrap();
            let mut put = |name: String, shape: Vec<usize>, (mean, std): (f32, f32)| {
                let n: usize = shape.iter().product();
                let t = Tensor::from_vec(randn_vec(rng, mean, std, n), shape, dev).unwrap();
                data.insert(name, Var::from_tensor(&t).unwrap());
            };
            for i in 0..cfg.num_hidden_layers {
                let p = format!("model.layers.{i}");
                put(format!("{p}.input_layernorm.weight"), vec![h], nrm);
                put(
                    format!("{p}.self_attn.q_proj.weight"),
                    vec![nh * hd, h],
                    lin,
                );
                put(
                    format!("{p}.self_attn.k_proj.weight"),
                    vec![nkv * hd, h],
                    lin,
                );
                put(
                    format!("{p}.self_attn.v_proj.weight"),
                    vec![nkv * hd, h],
                    lin,
                );
                put(
                    format!("{p}.self_attn.o_proj.weight"),
                    vec![h, nh * hd],
                    lin,
                );
                put(format!("{p}.post_attention_layernorm.weight"), vec![h], nrm);
                put(format!("{p}.mlp.gate_proj.weight"), vec![inter, h], lin);
                put(format!("{p}.mlp.up_proj.weight"), vec![inter, h], lin);
                put(format!("{p}.mlp.down_proj.weight"), vec![h, inter], lin);
            }
            put("model.norm.weight".to_string(), vec![h], nrm);
            put("lm_head.weight".to_string(), vec![vocab, h], lin);
        }
        let vb = ShardedSafeTensors::wrap(vm, DType::F32, dev.clone());
        let mapper = crate::device_map::DeviceMapSetting::dummy()
            .into_mapper(cfg.num_hidden_layers, dev, None, std::slice::from_ref(dev))
            .unwrap();
        ErnieTextModel::load(vb, cfg, &*mapper, false, AttentionImplementation::Eager).unwrap()
    }

    // Incremental engine decode (prefill + one-token-per-step, driving `KvCache::append`) must
    // reproduce a single-shot `forward` over the whole sequence bit-for-bit. Both round stored
    // K/V through the same cache dtype, so they agree at 1e-5 even on cpu_kv_f16; a real cache-mapping
    // bug diverges on the logit scale (>1e-2, asserted below).
    #[test]
    fn engine_cache_matches_full_recompute() {
        const TOL: f32 = 1e-5;
        let dev = Device::Cpu;
        let mut rng = StdRng::seed_from_u64(TEST_SEED); // candle CPU rng is unseedable; seed init here
        let cfg = tiny_cfg();
        let model = tiny_model(&cfg, &dev, &mut rng);

        let seq = 5usize;
        let embeds = Tensor::from_vec(
            randn_vec(&mut rng, 0.0, 1.0, seq * cfg.hidden_size),
            (1, seq, cfg.hidden_size),
            &dev,
        )
        .unwrap();
        // distinct t/h/w rows so the 3-axis chunked mrope actually matters.
        let pos = Tensor::from_vec(
            vec![0i64, 1, 2, 3, 4, 0, 0, 1, 1, 2, 0, 1, 0, 1, 0],
            (3, 1, seq),
            &dev,
        )
        .unwrap();

        let mut ref_caches: Vec<EngineKvCache> = (0..cfg.num_hidden_layers)
            .map(|_| EngineKvCache::new_normal(2, 64, 512))
            .collect();
        let full = model
            .forward(
                &embeds,
                &pos,
                &mut ref_caches,
                &causal_mask(seq, 0, &dev),
                None,
                None,
            )
            .unwrap()
            .logits; // [1, seq, vocab]
        let hi = full
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let lo = full
            .flatten_all()
            .unwrap()
            .min(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(hi - lo > 1e-2, "degenerate logits, test would be vacuous");

        let prefill = 2usize;
        let mut caches: Vec<EngineKvCache> = (0..cfg.num_hidden_layers)
            .map(|_| EngineKvCache::new_normal(2, 64, 512))
            .collect();

        let out = model
            .forward(
                &embeds.narrow(1, 0, prefill).unwrap(),
                &pos.narrow(2, 0, prefill).unwrap(),
                &mut caches,
                &causal_mask(prefill, 0, &dev),
                None,
                None,
            )
            .unwrap()
            .logits;
        for i in 0..prefill {
            let d = max_abs(
                &out.narrow(1, i, 1).unwrap(),
                &full.narrow(1, i, 1).unwrap(),
            );
            assert!(d < TOL, "prefill row {i} diff {d} (tol {TOL})");
        }

        for t in prefill..seq {
            let out = model
                .forward(
                    &embeds.narrow(1, t, 1).unwrap(),
                    &pos.narrow(2, t, 1).unwrap(),
                    &mut caches,
                    &causal_mask(1, t, &dev),
                    None,
                    None,
                )
                .unwrap()
                .logits;
            let d = max_abs(&out, &full.narrow(1, t, 1).unwrap());
            assert!(d < TOL, "decode step {t} diff {d} (tol {TOL})");
        }
    }

    // ISQ residuals must expose the RMSNorm weights (kept full-precision) and exclude the
    // quantized projections/lm_head, so a UQFF write pairs the residual set with the quant layers.
    #[test]
    fn residual_tensors_excludes_quantized_projections() {
        let dev = Device::Cpu;
        let mut rng = StdRng::seed_from_u64(TEST_SEED);
        let cfg = tiny_cfg();
        let model = tiny_model(&cfg, &dev, &mut rng);
        let names: std::collections::HashSet<String> = model
            .residual_tensors()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        assert!(names.contains("model.norm.weight"));
        assert!(names.contains("model.layers.0.input_layernorm.weight"));
        assert!(names.contains("model.layers.1.post_attention_layernorm.weight"));
        assert!(!names.contains("model.layers.0.self_attn.q_proj.weight"));
        assert!(!names.contains("lm_head.weight"));
    }

    // A batch-of-2 prefill must reproduce each sequence's logits from a batch-1 run of that sequence
    // alone. A batching bug (shared positions, a mask that leaks across rows, or an attention reshape
    // that mixes the batch axis) surfaces here as cross-sequence contamination.
    #[test]
    fn batch_forward_matches_separate_sequences() {
        const TOL: f32 = 1e-5;
        let dev = Device::Cpu;
        let mut rng = StdRng::seed_from_u64(TEST_SEED);
        let cfg = tiny_cfg();
        let model = tiny_model(&cfg, &dev, &mut rng);

        let seq = 5usize;
        let h = cfg.hidden_size;
        let embeds0 =
            Tensor::from_vec(randn_vec(&mut rng, 0.0, 1.0, seq * h), (1, seq, h), &dev).unwrap();
        let embeds1 =
            Tensor::from_vec(randn_vec(&mut rng, 0.0, 1.0, seq * h), (1, seq, h), &dev).unwrap();
        // distinct t/h/w mrope rows per sequence so the positions actually diverge across the batch.
        let pos0 = Tensor::from_vec(
            vec![0i64, 1, 2, 3, 4, 0, 0, 1, 1, 2, 0, 1, 0, 1, 0],
            (3, 1, seq),
            &dev,
        )
        .unwrap();
        let pos1 = Tensor::from_vec(
            vec![0i64, 1, 2, 3, 4, 0, 1, 1, 2, 2, 1, 0, 1, 0, 1],
            (3, 1, seq),
            &dev,
        )
        .unwrap();

        let run = |embeds: &Tensor, pos: &Tensor, batch: usize| {
            let ids = Tensor::zeros((batch, seq), DType::U32, &dev).unwrap();
            let offsets = vec![0usize; batch];
            let offsets_slice = offsets.as_slice();
            let mask = CausalMasker
                .make_causal_mask(
                    &ids,
                    &offsets_slice as &dyn PastKvLenCache,
                    DType::F32,
                    &CausalMaskConfig::default(),
                )
                .unwrap();
            let mut caches: Vec<EngineKvCache> = (0..cfg.num_hidden_layers)
                .map(|_| EngineKvCache::new_normal(2, 64, 512))
                .collect();
            model
                .forward(embeds, pos, &mut caches, &mask, None, None)
                .unwrap()
                .logits
        };

        let sep0 = run(&embeds0, &pos0, 1); // [1, seq, vocab]
        let sep1 = run(&embeds1, &pos1, 1);
        // guard: distinct inputs must yield distinct logits, else the equality assert is vacuous.
        assert!(
            max_abs(&sep0, &sep1) > 1e-2,
            "sequences too similar, contamination test would be vacuous"
        );

        let embeds = Tensor::cat(&[&embeds0, &embeds1], 0).unwrap(); // [2, seq, hidden]
        let pos = Tensor::cat(&[&pos0, &pos1], 1).unwrap(); // [3, 2, seq]
        let batched = run(&embeds, &pos, 2); // [2, seq, vocab]

        let d0 = max_abs(&batched.narrow(0, 0, 1).unwrap(), &sep0);
        let d1 = max_abs(&batched.narrow(0, 1, 1).unwrap(), &sep1);
        assert!(d0 < TOL, "batch row 0 diff {d0} (tol {TOL})");
        assert!(d1 < TOL, "batch row 1 diff {d1} (tol {TOL})");
    }
}
