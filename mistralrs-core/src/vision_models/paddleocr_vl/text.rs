//! ERNIE-4.5-0.3B dense decoder ported to candle.
//!
//! Faithful to the native transformers 5.13 `PaddleOCR*` text classes. Everything runs at
//! f32/CPU for parity vs the HF reference. Batch is fixed to 1 (one layout crop = one VLM call),
//! so hidden states are `[seq, hidden]` and attention tensors are `[heads, seq, head_dim]`, the
//! leading batch axis is dropped.
//!
//! The single highest parity risk is the 3D **chunked** mrope: cos/sin are built per
//! position-axis, then for each 128-dim channel chunk we pick axis `i % 3` (Qwen2.5-VL scheme),
//! NOT the interleaved Qwen3-VL scheme.

use super::config::TextConfig;
use crate::layers::linear_no_bias;
use crate::pipeline::KvCache as EngineKvCache;
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Linear, Module};
use mistralrs_quant::ShardedVarBuilder;

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

/// Build the full-`head_dim` cos/sin tables from 3D `position_ids`.
///
/// `position_ids`: `[3, seq]` (one row per t/h/w axis). Returns `(cos, sin)` each `[3, seq, head_dim]`,
/// mirrors `PaddleOCRRotaryEmbedding.forward` (inv_freq outer pos, `cat(freqs, freqs)`, cos/sin).
fn rope_tables(
    position_ids: &Tensor,
    head_dim: usize,
    theta: f64,
    dev: &Device,
) -> Result<(Tensor, Tensor)> {
    let (three, seq) = position_ids.dims2()?;
    let half = head_dim / 2;
    // inv_freq[i] = 1 / theta^(2i/head_dim), i in 0..half. Computed in f32 to mirror torch.
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1f32 / (theta as f32).powf((2 * i) as f32 / head_dim as f32))
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, (1, 1, half), dev)?; // [1,1,half]
    let pos = position_ids
        .to_dtype(DType::F32)?
        .reshape((three, seq, 1))?; // [3,seq,1]
    let freqs = pos.broadcast_mul(&inv_freq)?; // [3,seq,half]
    let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?; // [3,seq,head_dim]
    Ok((emb.cos()?, emb.sin()?))
}

/// Chunked-select of a cos/sin table `[3, seq, head_dim]` down to `[seq, head_dim]`.
///
/// `apply_multimodal_rotary_pos_emb`: split the last dim into sections `[16,24,24,16,24,24]`; for
/// chunk index `i` take axis-plane `i % 3`, then concat. This is the parity-critical mrope wiring.
fn mrope_select(table: &Tensor, sections_doubled: &[usize]) -> Result<Tensor> {
    let mut parts = Vec::with_capacity(sections_doubled.len());
    let mut offset = 0;
    for (i, &s) in sections_doubled.iter().enumerate() {
        let plane = i % 3;
        let chunk = table
            .narrow(D::Minus1, offset, s)? // [3, seq, s]
            .narrow(0, plane, 1)? // [1, seq, s]
            .squeeze(0)?; // [seq, s]
        parts.push(chunk);
        offset += s;
    }
    Tensor::cat(&parts, D::Minus1) // [seq, head_dim]
}

/// Apply rope to `x` `[heads, seq, head_dim]` with `cos`/`sin` `[seq, head_dim]` (broadcast over heads).
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let cos = cos.unsqueeze(0)?; // [1, seq, head_dim]
    let sin = sin.unsqueeze(0)?;
    let a = x.broadcast_mul(&cos)?;
    let b = rotate_half(x)?.broadcast_mul(&sin)?;
    a + b
}

/// `repeat_kv`: `[num_kv, seq, hd] -> [num_kv*n_rep, seq, hd]`; kv head j feeds out-heads j*n_rep..+n_rep.
fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (nkv, seq, hd) = x.dims3()?;
    x.unsqueeze(1)? // [nkv, 1, seq, hd]
        .broadcast_as((nkv, n_rep, seq, hd))?
        .contiguous()?
        .reshape((nkv * n_rep, seq, hd))
}

/// GQA self-attention with 3D chunked mrope, causal, scale head_dim^-0.5, no biases.
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Attention {
    fn load(vb: ShardedVarBuilder, cfg: &TextConfig) -> Result<Self> {
        let h = cfg.hidden_size;
        let (nh, nkv, hd) = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
        );
        Ok(Self {
            q_proj: linear_no_bias(h, nh * hd, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(h, nkv * hd, vb.pp("k_proj"))?,
            v_proj: linear_no_bias(h, nkv * hd, vb.pp("v_proj"))?,
            o_proj: linear_no_bias(nh * hd, h, vb.pp("o_proj"))?,
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
            scale: (hd as f64).powf(-0.5),
        })
    }

    /// `x`: `[seq, hidden]`; `cos`/`sin`: `[seq, head_dim]`; `mask`: additive causal `[seq, seq]`.
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let seq = x.dim(0)?;
        let hd = self.head_dim;
        // project, then [seq, heads*hd] -> [heads, seq, hd]
        let q = self
            .q_proj
            .forward(x)?
            .reshape((seq, self.num_heads, hd))?
            .transpose(0, 1)?
            .contiguous()?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((seq, self.num_kv_heads, hd))?
            .transpose(0, 1)?
            .contiguous()?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((seq, self.num_kv_heads, hd))?
            .transpose(0, 1)?
            .contiguous()?;

        let q = apply_rope(&q, cos, sin)?;
        let k = apply_rope(&k, cos, sin)?;

        let n_rep = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(&k, n_rep)?; // [heads, seq, hd]
        let v = repeat_kv(&v, n_rep)?;

        // scores = (q @ k^T) * scale + mask ; softmax over keys ; @ v
        let k_t = k.transpose(1, 2)?.contiguous()?; // [heads, hd, seq]
        let scores = q.matmul(&k_t)?.affine(self.scale, 0.0)?; // [heads, seq, seq]
        let scores = scores.broadcast_add(mask)?;
        let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let ctx = probs.matmul(&v)?; // [heads, seq, hd]

        let ctx = ctx
            .transpose(0, 1)? // [seq, heads, hd]
            .contiguous()?
            .reshape((seq, self.num_heads * hd))?;
        self.o_proj.forward(&ctx)
    }

    /// KV-cache decode: `x`/`cos`/`sin` cover only the `n_new` new tokens; `cache` holds the prior
    /// post-RoPE K and V (`[num_kv, offset, head_dim]`); `mask` is `[n_new, offset+n_new]`. Unifies
    /// prefill (offset 0, mask == plain causal) and single-token decode (n_new 1, mask all-zero).
    /// Numerically identical to the full-recompute `forward`: same RoPE math + attention, K/V reused.
    fn forward_cached(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
        cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let n_new = x.dim(0)?;
        let hd = self.head_dim;
        let q = self
            .q_proj
            .forward(x)?
            .reshape((n_new, self.num_heads, hd))?
            .transpose(0, 1)?
            .contiguous()?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((n_new, self.num_kv_heads, hd))?
            .transpose(0, 1)?
            .contiguous()?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((n_new, self.num_kv_heads, hd))?
            .transpose(0, 1)?
            .contiguous()?;

        let q = apply_rope(&q, cos, sin)?;
        let k = apply_rope(&k, cos, sin)?;

        // append the new K/V to the cache (K stored post-RoPE, V raw), then store back
        let (k, v) = match cache.take() {
            Some((pk, pv)) => (Tensor::cat(&[&pk, &k], 1)?, Tensor::cat(&[&pv, &v], 1)?),
            None => (k, v),
        };
        *cache = Some((k.clone(), v.clone()));

        let n_rep = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(&k, n_rep)?; // [heads, offset+n_new, hd]
        let v = repeat_kv(&v, n_rep)?;

        let k_t = k.transpose(1, 2)?.contiguous()?;
        let scores = q.matmul(&k_t)?.affine(self.scale, 0.0)?; // [heads, n_new, offset+n_new]
        let scores = scores.broadcast_add(mask)?;
        let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let ctx = probs.matmul(&v)?; // [heads, n_new, hd]

        let ctx = ctx
            .transpose(0, 1)?
            .contiguous()?
            .reshape((n_new, self.num_heads * hd))?;
        self.o_proj.forward(&ctx)
    }

    /// Same as `forward_cached`, but driving the engine's `NormalCache` slot (`KvCache::append`)
    /// instead of the hand-rolled `Option<(K,V)>`. The engine cache stores `[batch, heads, seq, hd]`
    /// and appends on dim 2, so we carry a batch axis of 1 through the append then drop it; the
    /// growing concat is the same op as `forward_cached`'s `cat(dim=1)` (unit test asserts equality).
    fn forward_engine(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
        kv_cache: &mut EngineKvCache,
    ) -> Result<Tensor> {
        let n_new = x.dim(0)?;
        let hd = self.head_dim;
        let q = self
            .q_proj
            .forward(x)?
            .reshape((n_new, self.num_heads, hd))?
            .transpose(0, 1)?
            .contiguous()?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((n_new, self.num_kv_heads, hd))?
            .transpose(0, 1)?
            .contiguous()?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((n_new, self.num_kv_heads, hd))?
            .transpose(0, 1)?
            .contiguous()?;

        let q = apply_rope(&q, cos, sin)?;
        let k = apply_rope(&k, cos, sin)?;

        // append K (post-RoPE) / V (raw) to the engine cache with a batch axis, then drop it.
        let (k, v) = kv_cache.append(&k.unsqueeze(0)?, &v.unsqueeze(0)?)?;
        // cache may store f16 (cpu_kv_f16); cast back to q's dtype so attention runs in the compute dtype.
        let k = k.squeeze(0)?.contiguous()?.to_dtype(q.dtype())?; // [num_kv, offset+n_new, hd]
        let v = v.squeeze(0)?.contiguous()?.to_dtype(q.dtype())?;

        let n_rep = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(&k, n_rep)?;
        let v = repeat_kv(&v, n_rep)?;

        let k_t = k.transpose(1, 2)?.contiguous()?;
        let scores = q.matmul(&k_t)?.affine(self.scale, 0.0)?;
        let scores = scores.broadcast_add(mask)?;
        let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let ctx = probs.matmul(&v)?;

        let ctx = ctx
            .transpose(0, 1)?
            .contiguous()?
            .reshape((n_new, self.num_heads * hd))?;
        self.o_proj.forward(&ctx)
    }
}

/// SwiGLU MLP: `down(silu(gate(x)) * up(x))`, no biases.
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn load(vb: ShardedVarBuilder, cfg: &TextConfig) -> Result<Self> {
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        Ok(Self {
            gate_proj: linear_no_bias(h, i, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(h, i, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(i, h, vb.pp("down_proj"))?,
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
    fn load(vb: ShardedVarBuilder, cfg: &TextConfig) -> Result<Self> {
        Ok(Self {
            input_layernorm: RmsNorm::load(
                vb.pp("input_layernorm"),
                cfg.hidden_size,
                cfg.rms_norm_eps,
            )?,
            self_attn: Attention::load(vb.pp("self_attn"), cfg)?,
            post_attention_layernorm: RmsNorm::load(
                vb.pp("post_attention_layernorm"),
                cfg.hidden_size,
                cfg.rms_norm_eps,
            )?,
            mlp: Mlp::load(vb.pp("mlp"), cfg)?,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let h = (x + self
            .self_attn
            .forward(&self.input_layernorm.forward(x)?, cos, sin, mask)?)?;
        let out = (&h
            + self
                .mlp
                .forward(&self.post_attention_layernorm.forward(&h)?)?)?;
        Ok(out)
    }

    fn forward_cached(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
        cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let h = (x + self.self_attn.forward_cached(
            &self.input_layernorm.forward(x)?,
            cos,
            sin,
            mask,
            cache,
        )?)?;
        let out = (&h
            + self
                .mlp
                .forward(&self.post_attention_layernorm.forward(&h)?)?)?;
        Ok(out)
    }

    fn forward_engine(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
        kv_cache: &mut EngineKvCache,
    ) -> Result<Tensor> {
        let h = (x + self.self_attn.forward_engine(
            &self.input_layernorm.forward(x)?,
            cos,
            sin,
            mask,
            kv_cache,
        )?)?;
        let out = (&h
            + self
                .mlp
                .forward(&self.post_attention_layernorm.forward(&h)?)?)?;
        Ok(out)
    }
}

/// Output of a text forward pass, for parity checks against the reference tensors.
pub struct TextOutput {
    /// Output of decoder layer 0 (== reference `lm_hidden_0`).
    pub layer0_out: Tensor,
    /// Final `norm(last_layer_out)` (== reference `lm_hidden_last`, the lm_head input).
    pub last_hidden: Tensor,
    /// `lm_head(last_hidden)` (== reference `logits`).
    pub logits: Tensor,
}

/// ERNIE text decoder + untied lm_head. Embeddings are NOT loaded here: this stage teacher-forces
/// the reference merged input embeds; the embed/scatter path is wired separately.
pub struct ErnieTextModel {
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    cfg: TextConfig,
}

impl ErnieTextModel {
    /// `vb` is the checkpoint root (keys `model.layers.*`, `model.norm.*`, `lm_head.*`).
    pub fn load(vb: ShardedVarBuilder, cfg: &TextConfig) -> Result<Self> {
        let vm = vb.pp("model");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::load(vm.pp("layers").pp(i), cfg)?);
        }
        let norm = RmsNorm::load(vm.pp("norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let lm_head = linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        Ok(Self {
            layers,
            norm,
            lm_head,
            cfg: cfg.clone(),
        })
    }

    /// `inputs_embeds`: `[seq, hidden]`; `position_ids`: `[3, seq]` (t/h/w rows).
    pub fn forward(&self, inputs_embeds: &Tensor, position_ids: &Tensor) -> Result<TextOutput> {
        let dev = inputs_embeds.device();
        let seq = inputs_embeds.dim(0)?;

        let (cos_t, sin_t) =
            rope_tables(position_ids, self.cfg.head_dim, self.cfg.rope_theta, dev)?;
        let sections_doubled: Vec<usize> =
            [self.cfg.mrope_section, self.cfg.mrope_section].concat();
        let cos = mrope_select(&cos_t, &sections_doubled)?; // [seq, head_dim]
        let sin = mrope_select(&sin_t, &sections_doubled)?;

        let mask = causal_mask(seq, dev)?;

        // rope tables + mask are built in f32; cast to the activation dtype (bf16 on the GPU path).
        let dtype = inputs_embeds.dtype();
        let (cos, sin, mask) = (
            cos.to_dtype(dtype)?,
            sin.to_dtype(dtype)?,
            mask.to_dtype(dtype)?,
        );

        let mut h = inputs_embeds.clone();
        let mut layer0_out = None;
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &cos, &sin, &mask)?;
            if i == 0 {
                layer0_out = Some(h.clone());
            }
        }
        let last_hidden = self.norm.forward(&h)?;
        let logits = self.lm_head.forward(&last_hidden)?;
        Ok(TextOutput {
            layer0_out: layer0_out.unwrap(),
            last_hidden,
            logits,
        })
    }

    /// Fresh per-layer KV cache (all `None`), one slot per decoder layer.
    pub fn empty_cache(&self) -> KvCache {
        vec![None; self.layers.len()]
    }

    /// KV-cache forward for `n_new` tokens starting at absolute position `offset`. Prefill passes
    /// the whole prompt with `offset == 0`; decode passes one token per step with the growing offset.
    /// `position_ids` is `[3, n_new]` (the new tokens' mrope positions). Numerically identical to
    /// `forward` on the same total sequence, the cache just avoids recomputing prior K/V.
    pub fn forward_cached(
        &self,
        inputs_embeds: &Tensor,
        position_ids: &Tensor,
        cache: &mut KvCache,
        offset: usize,
    ) -> Result<TextOutput> {
        let dev = inputs_embeds.device();
        let n_new = inputs_embeds.dim(0)?;

        let (cos_t, sin_t) =
            rope_tables(position_ids, self.cfg.head_dim, self.cfg.rope_theta, dev)?;
        let sections_doubled: Vec<usize> =
            [self.cfg.mrope_section, self.cfg.mrope_section].concat();
        let cos = mrope_select(&cos_t, &sections_doubled)?; // [n_new, head_dim]
        let sin = mrope_select(&sin_t, &sections_doubled)?;

        let mask = causal_mask_offset(n_new, offset, dev)?;

        // rope tables + mask are built in f32; cast to the activation dtype (bf16 on the GPU path).
        let dtype = inputs_embeds.dtype();
        let (cos, sin, mask) = (
            cos.to_dtype(dtype)?,
            sin.to_dtype(dtype)?,
            mask.to_dtype(dtype)?,
        );

        let mut h = inputs_embeds.clone();
        let mut layer0_out = None;
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward_cached(&h, &cos, &sin, &mask, &mut cache[i])?;
            if i == 0 {
                layer0_out = Some(h.clone());
            }
        }
        let last_hidden = self.norm.forward(&h)?;
        let logits = self.lm_head.forward(&last_hidden)?;
        Ok(TextOutput {
            layer0_out: layer0_out.unwrap(),
            last_hidden,
            logits,
        })
    }

    /// Engine-cache twin of `forward_cached`: drives the engine's per-layer `NormalCache` slots
    /// (`caches[i]`, `KvCache::append`) instead of the hand-rolled `KvCache`. This is the path the
    /// `MultimodalModel` trait `forward` calls; `offset` is `ctx.seqlen_offsets()[0]` (past length).
    /// Numerically identical to `forward_cached`/`forward` (unit test `engine_cache_matches_*`).
    pub fn forward_engine(
        &self,
        inputs_embeds: &Tensor,
        position_ids: &Tensor,
        caches: &mut [EngineKvCache],
        offset: usize,
    ) -> Result<TextOutput> {
        let dev = inputs_embeds.device();
        let n_new = inputs_embeds.dim(0)?;

        let (cos_t, sin_t) =
            rope_tables(position_ids, self.cfg.head_dim, self.cfg.rope_theta, dev)?;
        let sections_doubled: Vec<usize> =
            [self.cfg.mrope_section, self.cfg.mrope_section].concat();
        let cos = mrope_select(&cos_t, &sections_doubled)?;
        let sin = mrope_select(&sin_t, &sections_doubled)?;

        let mask = causal_mask_offset(n_new, offset, dev)?;

        // rope tables + mask are built in f32; cast to the activation dtype (bf16 on the GPU path).
        let dtype = inputs_embeds.dtype();
        let (cos, sin, mask) = (
            cos.to_dtype(dtype)?,
            sin.to_dtype(dtype)?,
            mask.to_dtype(dtype)?,
        );

        let mut h = inputs_embeds.clone();
        let mut layer0_out = None;
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward_engine(&h, &cos, &sin, &mask, &mut caches[i])?;
            if i == 0 {
                layer0_out = Some(h.clone());
            }
        }
        let last_hidden = self.norm.forward(&h)?;
        let logits = self.lm_head.forward(&last_hidden)?;
        Ok(TextOutput {
            layer0_out: layer0_out.unwrap(),
            last_hidden,
            logits,
        })
    }
}

/// Per-layer KV cache: `cache[layer]` is `Some((k, v))` with K post-RoPE and V raw, each
/// `[num_kv_heads, cached_len, head_dim]`. `None` before the first (prefill) step.
pub type KvCache = Vec<Option<(Tensor, Tensor)>>;

/// Additive causal mask `[seq, seq]`: 0 on/below the diagonal, `-inf` above.
fn causal_mask(seq: usize, dev: &Device) -> Result<Tensor> {
    let mut data = vec![0f32; seq * seq];
    for i in 0..seq {
        for j in (i + 1)..seq {
            data[i * seq + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(data, (seq, seq), dev)
}

/// Additive causal mask `[n_new, offset+n_new]` for cached decode: new query `i` sits at absolute
/// position `offset+i` and may attend to keys `0..=offset+i`. Reduces to `causal_mask` when offset 0.
fn causal_mask_offset(n_new: usize, offset: usize, dev: &Device) -> Result<Tensor> {
    let total = offset + n_new;
    let mut data = vec![0f32; n_new * total];
    for i in 0..n_new {
        for j in (offset + i + 1)..total {
            data[i * total + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(data, (n_new, total), dev)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Var;
    use candle_nn::VarMap;
    use mistralrs_quant::ShardedSafeTensors;
    use rand::{rngs::StdRng, SeedableRng};
    use rand_distr::{Distribution, Normal};

    const TEST_SEED: u64 = 0x5EED_5EED;
    // engine-cache-vs-recompute bound on the f16 CPU KV path (cpu_kv_f16 rounds stored K/V); the f32
    // path is bit-exact at 1e-5. Covers the measured f16 storage band with margin, well under the
    // >1e-2 logit scale a real cache-mapping bug would produce.
    const KV_F16_MATCH_TOL: f32 = 5e-3;

    fn randn_vec(rng: &mut StdRng, mean: f32, std: f32, n: usize) -> Vec<f32> {
        let normal = Normal::new(mean, std).unwrap();
        (0..n).map(|_| normal.sample(rng)).collect()
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
        ErnieTextModel::load(vb, cfg).unwrap()
    }

    // The engine-cache `forward_engine` (prefill + one-token-per-step decode driving `KvCache::append`)
    // must reproduce the full-recompute `forward` bit-for-bit: this is what proves the cache mapping
    // preserves the reference-verified attention math once wired to the engine.
    #[test]
    fn engine_cache_matches_full_recompute() {
        // engine-cache decode == full-recompute is bit-exact on the f32 CPU KV path; cpu_kv_f16
        // (avx2/f16c or aarch64, on by default incl. CI runners) rounds stored K/V, so on that path
        // the two differ by the f16 storage band. Read the resolved cache dtype (process-global
        // OnceLock, set once) and assert the matching bound -- no MISTRALRS_CPU_KV_F32 set_var, so no
        // ordering race in the shared unfiltered test binary. A real cache-mapping bug diverges on the
        // logit scale (>1e-2, asserted below), above both bounds.
        let tol: f32 = if crate::kv_cache::cpu_kv_f16() {
            KV_F16_MATCH_TOL
        } else {
            1e-5
        };
        let dev = Device::Cpu;
        let mut rng = StdRng::seed_from_u64(TEST_SEED); // candle CPU rng is unseedable; seed init here
        let cfg = tiny_cfg();
        let model = tiny_model(&cfg, &dev, &mut rng);

        let seq = 5usize;
        let embeds = Tensor::from_vec(
            randn_vec(&mut rng, 0.0, 1.0, seq * cfg.hidden_size),
            (seq, cfg.hidden_size),
            &dev,
        )
        .unwrap();
        // distinct t/h/w rows so the 3-axis chunked mrope actually matters.
        let pos = Tensor::from_vec(
            vec![0i64, 1, 2, 3, 4, 0, 0, 1, 1, 2, 0, 1, 0, 1, 0],
            (3, seq),
            &dev,
        )
        .unwrap();

        let full = model.forward(&embeds, &pos).unwrap().logits; // [seq, vocab]
        let hi = full
            .max(0)
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let lo = full
            .min(0)
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
            .forward_engine(
                &embeds.narrow(0, 0, prefill).unwrap(),
                &pos.narrow(1, 0, prefill).unwrap(),
                &mut caches,
                0,
            )
            .unwrap()
            .logits;
        for i in 0..prefill {
            let d = max_abs(
                &out.narrow(0, i, 1).unwrap(),
                &full.narrow(0, i, 1).unwrap(),
            );
            assert!(d < tol, "prefill row {i} diff {d} (tol {tol})");
        }

        for t in prefill..seq {
            let out = model
                .forward_engine(
                    &embeds.narrow(0, t, 1).unwrap(),
                    &pos.narrow(1, t, 1).unwrap(),
                    &mut caches,
                    t,
                )
                .unwrap()
                .logits;
            let d = max_abs(&out, &full.narrow(0, t, 1).unwrap());
            assert!(d < tol, "decode step {t} diff {d} (tol {tol})");
        }
    }
}
