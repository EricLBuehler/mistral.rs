//! SigLIP/NaViT native-resolution vision tower ported to candle.
//!
//! Faithful to the native transformers 5.13 `PaddleOCR*` vision classes on the VLM inference
//! path (`interpolate_pos_encoding=True, use_rope=True, return_pooler_output=False,
//! window_size=-1`). f32/CPU for parity.
//!
//! Two position signals are BOTH applied: (1) a learned 27x27 position table, bilinearly
//! interpolated to the native grid and ADDED pre-encoder; (2) 2D axial RoPE inside every
//! attention layer. The pooling `head.*` is vestigial and not loaded.
//!
//! Batch/packing: one layout crop = one image = K=1, so patches are a single `[N, hidden]`
//! sequence with full (non-causal) attention. Multi-image block-diagonal masking via
//! `cu_seqlens` is deferred, not exercised by the OCR path.

use super::config::VisionConfig;
use crate::attention::{AttentionMask, SdpaParams};
use crate::layers::{layer_norm, linear, Sdpa};
use crate::pipeline::text_models_inputs_processor::FlashParams;
use candle_core::{Device, Result, Tensor, D};
use candle_nn::{LayerNorm, Linear, Module};
use mistralrs_quant::ShardedVarBuilder;

/// `rotate_half` (neox): split the last dim in half, return `cat(-x2, x1)`. Same as text.rs.
fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let hd = x.dim(D::Minus1)?;
    let x1 = x.narrow(D::Minus1, 0, hd / 2)?;
    let x2 = x.narrow(D::Minus1, hd / 2, hd / 2)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

/// Apply rope to `x` `[heads, seq, head_dim]` with `cos`/`sin` `[seq, head_dim]` (broadcast heads).
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let cos = cos.unsqueeze(0)?; // [1, seq, head_dim]
    let sin = sin.unsqueeze(0)?;
    x.broadcast_mul(&cos)? + rotate_half(x)?.broadcast_mul(&sin)?
}

/// 2D axial RoPE cos/sin tables `[N=h*w, head_dim]` (`apply_rotary_pos_emb_vision`).
///
/// `SigLIPRotaryEmbedding(dim=head_dim/2, theta=rope_theta)` -> 18 inv-freqs over `arange(0,36,2)/36`.
/// Per patch (t=1 so within-frame index = flat index): `h_id=j//w`, `w_id=j%w`; the low half of the
/// rope block gets the height freqs, the high half the width freqs; the block is then repeated to
/// fill head_dim (neox), so dim `d` pairs with `d + head_dim/2` and shares the same angle.
fn vision_rope(
    h: usize,
    w: usize,
    head_dim: usize,
    theta: f64,
    dev: &Device,
) -> Result<(Tensor, Tensor)> {
    let rope_dim = head_dim / 2; // 36
    let half = rope_dim / 2; // 18 freqs (height block || width block)
    let inv_freq: Vec<f32> = (0..half)
        .map(|k| 1f32 / (theta as f32).powf((2 * k) as f32 / rope_dim as f32))
        .collect();
    let n = h * w;
    let mut cos = vec![0f32; n * head_dim];
    let mut sin = vec![0f32; n * head_dim];
    for j in 0..n {
        let h_id = (j / w) as f32;
        let w_id = (j % w) as f32;
        for k in 0..half {
            // freqs36[k] = height freq; freqs36[half+k] = width freq.
            let ah = h_id * inv_freq[k];
            let aw = w_id * inv_freq[k];
            let base = j * head_dim;
            // low block [0..rope_dim) and its neox mirror [rope_dim..head_dim) share the angle.
            cos[base + k] = ah.cos();
            cos[base + half + k] = aw.cos();
            cos[base + rope_dim + k] = ah.cos();
            cos[base + rope_dim + half + k] = aw.cos();
            sin[base + k] = ah.sin();
            sin[base + half + k] = aw.sin();
            sin[base + rope_dim + k] = ah.sin();
            sin[base + rope_dim + half + k] = aw.sin();
        }
    }
    Ok((
        Tensor::from_vec(cos, (n, head_dim), dev)?,
        Tensor::from_vec(sin, (n, head_dim), dev)?,
    ))
}

/// Patch embed: `Conv2d(3->hidden, k=patch, s=patch, valid)`. Because kernel==stride==patch and the
/// input is already patchified `[N, 3, patch, patch]`, the conv is exactly a per-patch flatten
/// (channel-outer, row-major) + matmul with the weight reshaped to `[hidden, 3*patch*patch]`.
struct PatchEmbed {
    weight: Tensor, // [hidden, 3*patch*patch]
    bias: Tensor,   // [hidden]
}

impl PatchEmbed {
    fn load(vb: ShardedVarBuilder, cfg: &VisionConfig) -> Result<Self> {
        let flat = cfg.num_channels * cfg.patch_size * cfg.patch_size;
        let w = vb.get(
            (
                cfg.hidden_size,
                cfg.num_channels,
                cfg.patch_size,
                cfg.patch_size,
            ),
            "weight",
        )?;
        Ok(Self {
            weight: w.reshape((cfg.hidden_size, flat))?,
            bias: vb.get(cfg.hidden_size, "bias")?,
        })
    }

    /// `pixel_values`: `[N, 3, patch, patch]` -> `[N, hidden]`.
    fn forward(&self, pv: &Tensor) -> Result<Tensor> {
        let n = pv.dim(0)?;
        let flat = pv.dim(1)? * pv.dim(2)? * pv.dim(3)?;
        // pixel_values are f32 from preprocess; cast to the weight dtype (bf16 on the GPU path).
        let x = pv.reshape((n, flat))?.to_dtype(self.weight.dtype())?;
        x.matmul(&self.weight.t()?)?.broadcast_add(&self.bias)
    }
}

/// Non-causal 16-head attention with 2D axial RoPE, biased q/k/v/out. `out_proj` name (not `o_proj`).
struct VisionAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl VisionAttention {
    fn load(vb: ShardedVarBuilder, cfg: &VisionConfig) -> Result<Self> {
        let h = cfg.hidden_size;
        Ok(Self {
            q_proj: linear(h, h, vb.pp("q_proj"))?,
            k_proj: linear(h, h, vb.pp("k_proj"))?,
            v_proj: linear(h, h, vb.pp("v_proj"))?,
            out_proj: linear(h, h, vb.pp("out_proj"))?,
            num_heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim,
            scale: (cfg.head_dim as f64).powf(-0.5),
        })
    }

    /// `x`: `[N, hidden]`; `cos`/`sin`: `[N, head_dim]`. Full (non-causal, single-image) attention.
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let n = x.dim(0)?;
        let hd = self.head_dim;
        let reshape = |t: Tensor| -> Result<Tensor> {
            t.reshape((n, self.num_heads, hd))?
                .transpose(0, 1)?
                .contiguous()
        };
        let q = apply_rope(&reshape(self.q_proj.forward(x)?)?, cos, sin)?;
        let k = apply_rope(&reshape(self.k_proj.forward(x)?)?, cos, sin)?;
        let v = reshape(self.v_proj.forward(x)?)?; // [heads, N, hd]

        // K=1 single-image path: full non-causal attention (no cu_seqlens mask). Routed through the
        // engine Sdpa so GPU builds hit the fused/flash kernel and CPU the fused flash path, instead
        // of materializing the N-by-N score matrix. Bidirectional, so FlashParams carries causal=false.
        // Multi-image block-diagonal masking is deferred until the packed path is exercised.
        let sdpa_params = SdpaParams {
            n_kv_groups: 1,
            sliding_window: None,
            softcap: None,
            softmax_scale: self.scale as f32,
            sinks: None,
        };
        let flash_params = FlashParams::empty(false);
        let ctx = Sdpa.run_attention(
            &q.unsqueeze(0)?, // [1, heads, N, hd]
            &k.unsqueeze(0)?,
            &v.unsqueeze(0)?,
            &AttentionMask::None,
            Some(&flash_params),
            &sdpa_params,
        )?; // [1, heads, N, hd]

        let ctx = ctx
            .squeeze(0)?
            .transpose(0, 1)?
            .contiguous()?
            .reshape((n, self.num_heads * hd))?;
        self.out_proj.forward(&ctx)
    }
}

/// Vision MLP: `fc2(gelu_pytorch_tanh(fc1(x)))`. candle `.gelu()` is the tanh approximation.
struct VisionMlp {
    fc1: Linear,
    fc2: Linear,
}

impl VisionMlp {
    fn load(vb: ShardedVarBuilder, cfg: &VisionConfig) -> Result<Self> {
        Ok(Self {
            fc1: linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("fc1"))?,
            fc2: linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("fc2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&self.fc1.forward(x)?.gelu()?)
    }
}

/// Pre-norm encoder block: `x += attn(ln1(x)); x += mlp(ln2(x))`. LayerNorms eps 1e-6.
struct EncoderLayer {
    layer_norm1: LayerNorm,
    self_attn: VisionAttention,
    layer_norm2: LayerNorm,
    mlp: VisionMlp,
}

impl EncoderLayer {
    fn load(vb: ShardedVarBuilder, cfg: &VisionConfig) -> Result<Self> {
        let eps = cfg.layer_norm_eps;
        Ok(Self {
            layer_norm1: layer_norm(cfg.hidden_size, eps, vb.pp("layer_norm1"))?,
            self_attn: VisionAttention::load(vb.pp("self_attn"), cfg)?,
            layer_norm2: layer_norm(cfg.hidden_size, eps, vb.pp("layer_norm2"))?,
            mlp: VisionMlp::load(vb.pp("mlp"), cfg)?,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let h = (x + self
            .self_attn
            .forward(&self.layer_norm1.forward(x)?, cos, sin)?)?;
        &h + self.mlp.forward(&self.layer_norm2.forward(&h)?)?
    }
}

/// Intermediate activations captured for stage-by-stage parity vs the reference.
pub struct VisionOutput {
    /// Conv patch-embed output (== reference `patch_embed_out`, squeezed to `[N, hidden]`).
    pub patch_embed: Tensor,
    /// After adding the interpolated learned position embedding (== reference `vision_embeds`).
    pub embeds: Tensor,
    /// After encoder layer 0 (== reference `vision_block0`).
    pub block0: Tensor,
    /// After the final encoder layer (== reference `vision_block26`).
    pub block_last: Tensor,
    /// After `post_layernorm` (== reference `vision_post_ln`), the connector's input.
    pub post_ln: Tensor,
}

/// The SigLIP/NaViT vision tower: patch embed -> +interp-pos -> 27 pre-norm blocks -> post_layernorm.
pub struct VisionModel {
    patch_embed: PatchEmbed,
    position_embedding: Tensor, // [num_positions=729, hidden]
    layers: Vec<EncoderLayer>,
    post_layernorm: LayerNorm,
    cfg: VisionConfig,
}

impl VisionModel {
    /// `vb` is the vision-tower root (`visual.vision_model.*`).
    pub fn load(vb: ShardedVarBuilder, cfg: &VisionConfig) -> Result<Self> {
        let emb = vb.pp("embeddings");
        let patch_embed = PatchEmbed::load(emb.pp("patch_embedding"), cfg)?;
        let position_embedding = emb
            .pp("position_embedding")
            .get((cfg.num_positions, cfg.hidden_size), "weight")?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let enc = vb.pp("encoder").pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(EncoderLayer::load(enc.pp(i), cfg)?);
        }
        let post_layernorm =
            layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("post_layernorm"))?;
        Ok(Self {
            patch_embed,
            position_embedding,
            layers,
            post_layernorm,
            cfg: cfg.clone(),
        })
    }

    /// Bilinearly interpolate the learned 27x27 position table to `(h, w)` -> `[h*w, hidden]`.
    ///
    /// `align_corners=True`: the native transformers 5.13 tower interpolates via
    /// `get_vision_bilinear_indices_and_weights`, which samples with `linspace(0, side-1, n)`,
    /// i.e. align_corners=True, NOT the `F.interpolate(align_corners=False)` of the custom_code
    /// repo file. align_corners=False is off by ~42 vs the reference; align_corners=True
    /// reconstructs the reference to <1e-4. candle's CPU bilinear uses the same `scale*o`
    /// (= linspace) formula for align_corners=True.
    fn interpolate_pos(&self, h: usize, w: usize) -> Result<Tensor> {
        let g = self.cfg.pos_grid;
        let d = self.cfg.hidden_size;
        let p = self
            .position_embedding
            .reshape((1, g, g, d))?
            .permute((0, 3, 1, 2))? // [1, d, 27, 27]
            .contiguous()?;
        let up = p.upsample_bilinear2d(h, w, true)?; // [1, d, h, w]
        up.permute((0, 2, 3, 1))?.contiguous()?.reshape((h * w, d)) // row-major (row, col)
    }

    /// `pixel_values`: `[N, 3, patch, patch]`; grid `(t, h, w)` with `N = t*h*w`. Still images: t=1.
    pub fn forward(
        &self,
        pixel_values: &Tensor,
        t: usize,
        h: usize,
        w: usize,
    ) -> Result<VisionOutput> {
        assert_eq!(
            t, 1,
            "video temporal grid (t>1) not in scope for the OCR path"
        );
        let dev = pixel_values.device();
        let patch_embed = self.patch_embed.forward(pixel_values)?; // [N, hidden]
        let pos = self.interpolate_pos(h, w)?; // [h*w, hidden] == [N, hidden] for t=1
        let embeds = (&patch_embed + &pos)?;

        let (cos, sin) = vision_rope(h, w, self.cfg.head_dim, self.cfg.rope_theta, dev)?;
        let (cos, sin) = (cos.to_dtype(embeds.dtype())?, sin.to_dtype(embeds.dtype())?);

        let mut x = embeds.clone();
        let mut block0 = None;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, &cos, &sin)?;
            if i == 0 {
                block0 = Some(x.clone());
            }
        }
        let block_last = x.clone();
        let post_ln = self.post_layernorm.forward(&x)?;
        Ok(VisionOutput {
            patch_embed,
            embeds,
            block0: block0.unwrap(),
            block_last,
            post_ln,
        })
    }
}
