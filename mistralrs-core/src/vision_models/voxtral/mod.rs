#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Module, Result, Tensor};
use mistralrs_quant::{QuantMethod, ShardedVarBuilder};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::SdpaParams,
    device_map::{DeviceMappedMask, DeviceMapper},
    layers::{embedding, CausalMasker, MatMul, RmsNorm, RotaryEmbedding, Sdpa},
    layers_masker::PastKvLenCache,
    paged_attention::{AttentionImplementation, ModelConfigMetadata},
    pipeline::{
        extract_logits,
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, NormalCache, NormalLoadingMetadata, VisionModel,
    },
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

mod adapter;
mod audio_processing;
pub mod config;
mod encoder;
mod inputs_processor;

pub(crate) use inputs_processor::VoxtralProcessor;

use adapter::VoxtralTemporalAdapter;
use config::VoxtralConfig;
use encoder::VoxtralEncoder;

struct DecoderAttention {
    wq: Arc<dyn QuantMethod>,
    wk: Arc<dyn QuantMethod>,
    wv: Arc<dyn QuantMethod>,
    wo: Arc<dyn QuantMethod>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    sdpa_params: SdpaParams,
}

impl DecoderAttention {
    fn new(
        cfg: &VoxtralConfig,
        rotary_emb: Arc<RotaryEmbedding>,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
    ) -> Result<Self> {
        let dim = cfg.dim;
        let num_heads = cfg.n_heads;
        let num_kv_heads = cfg.n_kv_heads;
        let head_dim = cfg.head_dim;

        let vb = mapper.set_device(layer_idx, vb, loading_isq);
        let wq = mistralrs_quant::linear_b(
            dim,
            num_heads * head_dim,
            cfg.use_biases,
            &None,
            vb.pp("wq"),
        )?;
        let wk = mistralrs_quant::linear_b(
            dim,
            num_kv_heads * head_dim,
            cfg.use_biases,
            &None,
            vb.pp("wk"),
        )?;
        let wv = mistralrs_quant::linear_b(
            dim,
            num_kv_heads * head_dim,
            cfg.use_biases,
            &None,
            vb.pp("wv"),
        )?;
        let wo = mistralrs_quant::linear_b(
            num_heads * head_dim,
            dim,
            cfg.use_biases,
            &None,
            vb.pp("wo"),
        )?;

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_emb,
            sdpa_params: SdpaParams {
                n_kv_groups: num_heads / num_kv_heads,
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window: cfg.sliding_window,
                sinks: None,
            },
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let original_dtype = xs.dtype();
        let mut xs = xs.clone();
        if let Some(t) = self.wq.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        let mut q = MatMul.qmethod_matmul(&xs, &*self.wq)?;
        let mut k = MatMul.qmethod_matmul(&xs, &*self.wk)?;
        let mut v = MatMul.qmethod_matmul(&xs, &*self.wv)?;
        if self.wq.quantized_act_type().is_some() {
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

        let (k, v) = kv_cache.append(&k, &v)?;

        let mut attn_output = Sdpa.run_attention(
            &q,
            &k,
            &v,
            attention_mask,
            Some(flash_params),
            &self.sdpa_params,
        )?;

        if let Some(t) = self.wq.quantized_act_type() {
            attn_output = attn_output.to_dtype(t)?;
        }
        attn_output = if attention_mask.is_some() {
            attn_output.transpose(1, 2)?.reshape((b_sz, q_len, ()))?
        } else {
            attn_output.reshape((b_sz, q_len, ()))?
        };
        let mut res = MatMul.qmethod_matmul(&attn_output, &*self.wo)?;
        if self.wq.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

struct DecoderMlp {
    w1: Arc<dyn QuantMethod>, // gate
    w2: Arc<dyn QuantMethod>, // down
    w3: Arc<dyn QuantMethod>, // up
}

impl DecoderMlp {
    fn new(
        cfg: &VoxtralConfig,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
    ) -> Result<Self> {
        let vb = mapper.set_device(layer_idx, vb, loading_isq);
        let w1 =
            mistralrs_quant::linear_b(cfg.dim, cfg.hidden_dim, cfg.use_biases, &None, vb.pp("w1"))?;
        let w2 =
            mistralrs_quant::linear_b(cfg.hidden_dim, cfg.dim, cfg.use_biases, &None, vb.pp("w2"))?;
        let w3 =
            mistralrs_quant::linear_b(cfg.dim, cfg.hidden_dim, cfg.use_biases, &None, vb.pp("w3"))?;
        Ok(Self { w1, w2, w3 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let original_dtype = xs.dtype();
        let mut xs_act = xs.clone();
        if let Some(t) = self.w1.quantized_act_type() {
            xs_act = xs_act.to_dtype(t)?;
        }
        let gate = MatMul.qmethod_matmul(&xs_act, &*self.w1)?;
        let gate = candle_nn::ops::silu(&gate)?;
        let up = MatMul.qmethod_matmul(&xs_act, &*self.w3)?;
        let xs = (gate * up)?;
        let res = MatMul.qmethod_matmul(&xs, &*self.w2)?;
        if self.w1.quantized_act_type().is_some() {
            return res.to_dtype(original_dtype);
        }
        Ok(res)
    }

    fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        vec![&mut self.w1, &mut self.w3, &mut self.w2]
    }
}

/// Adaptive RMS normalization with time conditioning.
/// Applies: `ffn_norm(x) * (1 + ada_norm_mlp(t_cond))`
/// MLP: Linear(dim→t_cond_dim) → GELU → Linear(t_cond_dim→dim)
struct AdaptiveNorm {
    w0: Arc<dyn QuantMethod>,
    w2: Arc<dyn QuantMethod>,
}

impl AdaptiveNorm {
    fn new(dim: usize, t_cond_dim: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let w0 = mistralrs_quant::linear_b(dim, t_cond_dim, false, &None, vb.pp("0"))?;
        let w2 = mistralrs_quant::linear_b(t_cond_dim, dim, false, &None, vb.pp("2"))?;
        Ok(Self { w0, w2 })
    }

    fn forward(&self, t_cond: &Tensor) -> Result<Tensor> {
        let xs = MatMul.qmethod_matmul(t_cond, &*self.w0)?;
        let xs = xs.gelu_erf()?;
        MatMul.qmethod_matmul(&xs, &*self.w2)
    }
}

/// Compute sinusoidal time embedding (no learned parameters).
/// Input: scalar timestep t, model dim.
/// Output: [1, dim] tensor.
/// Sinusoidal time embedding matching `VoxtralRealtimeTimeEmbedding`:
/// `inv_freq[i] = exp(-log(10000) * i / (dim/2))`, output = `cat(cos(t*inv_freq), sin(t*inv_freq))`
fn time_embedding(t: f32, dim: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    let half_dim = dim / 2;
    let log_10000 = (10000f64).ln();
    let freqs: Vec<f32> = (0..half_dim)
        .map(|i| (-(i as f64) * log_10000 / half_dim as f64).exp() as f32)
        .collect();
    let freqs = Tensor::from_vec(freqs, half_dim, device)?;
    let args = (freqs * t as f64)?;
    let cos = args.cos()?;
    let sin = args.sin()?;
    Tensor::cat(&[&cos, &sin], 0)?.unsqueeze(0)?.to_dtype(dtype)
}

struct DecoderLayer {
    attention: DecoderAttention,
    feed_forward: DecoderMlp,
    attention_norm: RmsNorm,
    ffn_norm: RmsNorm,
    ada_norm: Option<AdaptiveNorm>,
}

impl DecoderLayer {
    fn new(
        cfg: &VoxtralConfig,
        rotary_emb: Arc<RotaryEmbedding>,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
    ) -> Result<Self> {
        let attention = DecoderAttention::new(
            cfg,
            rotary_emb,
            vb.pp("attention"),
            mapper,
            layer_idx,
            loading_isq,
        )?;
        let feed_forward =
            DecoderMlp::new(cfg, vb.pp("feed_forward"), mapper, layer_idx, loading_isq)?;
        let attention_norm = RmsNorm::new(
            cfg.dim,
            cfg.norm_eps,
            mapper.set_device(layer_idx, vb.pp("attention_norm"), false),
        )?;
        let ffn_norm = RmsNorm::new(
            cfg.dim,
            cfg.norm_eps,
            mapper.set_device(layer_idx, vb.pp("ffn_norm"), false),
        )?;
        let ada_norm = if cfg.ada_rms_norm_t_cond {
            Some(AdaptiveNorm::new(
                cfg.dim,
                cfg.ada_rms_norm_t_cond_dim,
                mapper.set_device(layer_idx, vb.pp("ada_rms_norm_t_cond"), false),
            )?)
        } else {
            None
        };
        Ok(Self {
            attention,
            feed_forward,
            attention_norm,
            ffn_norm,
            ada_norm,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        t_cond: Option<&Tensor>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.attention_norm.forward(xs)?;
        let xs =
            self.attention
                .forward(&xs, attention_mask, seqlen_offsets, kv_cache, flash_params)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let mut ffn_in = self.ffn_norm.forward(&xs)?;
        // Adaptive scaling: ffn_in = ffn_norm(x) * (1 + ada_norm(t_cond))
        if let (Some(ada_norm), Some(t_cond)) = (&self.ada_norm, t_cond) {
            let scale = ada_norm.forward(t_cond)?;
            ffn_in = ffn_in.broadcast_mul(&(scale + 1.0)?)?;
        }
        let xs = self.feed_forward.forward(&ffn_in)?;
        residual + xs
    }
}

#[derive(Default)]
pub struct VoxtralSpecificArgs {
    pub mel_features: Option<Tensor>,
    /// Number of delay tokens for time conditioning (streaming pad tokens).
    /// Defaults to 0 if not provided.
    pub n_delay_tokens: Option<f32>,
}

pub struct VoxtralModel {
    tok_embeddings: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    output: Arc<dyn QuantMethod>,
    encoder: VoxtralEncoder,
    adapter: VoxtralTemporalAdapter,
    cache: EitherCache,
    device: Device,
    max_seq_len: usize,
    cfg: ModelConfigMetadata,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    sliding_window: Option<usize>,
    num_heads: usize,
    model_dim: usize,
    ada_rms_norm_t_cond: bool,
    dtype: DType,
    /// Precomputed audio embeddings [B, N_audio, dim] stored during prompt phase
    /// and retrieved at each generation step for per-position audio conditioning.
    audio_embeds_cache: Arc<Mutex<Option<Tensor>>>,
}

impl VoxtralModel {
    pub fn new(
        cfg: &VoxtralConfig,
        vb: ShardedVarBuilder,
        _is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        _attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let mapper = normal_loading_metadata.mapper;

        // Encoder: mm_streams_embeddings.embedding_module.whisper_encoder
        let enc_cfg = &cfg.multimodal.whisper_model_args.encoder_args;
        let vb_mm = vb.pp("mm_streams_embeddings").pp("embedding_module");
        let encoder = VoxtralEncoder::new(
            enc_cfg,
            mapper.set_nm_device(vb_mm.pp("whisper_encoder"), false),
        )?;

        // Adapter: mm_streams_embeddings.embedding_module.audio_language_projection
        let ds_cfg = &cfg.multimodal.whisper_model_args.downsample_args;
        let adapter = VoxtralTemporalAdapter::new(
            enc_cfg.dim,
            cfg.dim,
            ds_cfg.downsample_factor,
            mapper.set_nm_device(vb_mm.clone(), false),
        )?;

        // Decoder embeddings: mm_streams_embeddings.embedding_module.tok_embeddings
        let tok_embeddings = embedding(
            cfg.vocab_size,
            cfg.dim,
            mapper.set_nm_device(vb_mm.pp("tok_embeddings"), false),
            &None,
        )?;

        // Decoder layers
        let head_dim = cfg.head_dim;
        let mut ropes = HashMap::new();
        for layer_idx in 0..cfg.n_layers {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            ropes.insert(
                device.location(),
                Arc::new(RotaryEmbedding::new(
                    cfg.rope_theta as f32,
                    head_dim,
                    cfg.model_max_length,
                    device,
                    false, // !is_gptx: consolidated.safetensors stores Q/K in interleaved layout
                    vb.dtype(),
                )?),
            );
        }

        let vb_layers = vb.pp("layers");
        let layers: Vec<DecoderLayer> = NiceProgressBar::<_, 'b'>(
            0..cfg.n_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .par_iter_if_isq(|layer_idx| {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let rotary_emb = ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();
            DecoderLayer::new(
                cfg,
                rotary_emb,
                vb_layers.pp(layer_idx),
                &*mapper,
                layer_idx,
                normal_loading_metadata.loading_isq,
            )
        })?;

        let norm = RmsNorm::new(
            cfg.dim,
            cfg.norm_eps,
            mapper.set_nm_device(vb.pp("norm"), false),
        )?;

        // output (lm_head) — may be tied with tok_embeddings
        let output = if cfg.tied_embeddings {
            mistralrs_quant::linear_b(
                cfg.dim,
                cfg.vocab_size,
                false,
                &None,
                mapper.set_nm_device(
                    vb.pp("mm_streams_embeddings")
                        .pp("embedding_module")
                        .pp("tok_embeddings"), // reuse embeddings weight
                    normal_loading_metadata.loading_isq,
                ),
            )?
        } else {
            mistralrs_quant::linear_b(
                cfg.dim,
                cfg.vocab_size,
                false,
                &None,
                mapper.set_nm_device(vb.pp("output"), normal_loading_metadata.loading_isq),
            )?
        };

        let cfg_meta = ModelConfigMetadata {
            max_seq_len: cfg.model_max_length,
            num_layers: cfg.n_layers,
            hidden_size: cfg.dim,
            num_kv_heads: cfg.n_kv_heads,
            num_attn_heads: cfg.n_heads,
            sliding_window: cfg.sliding_window,
            k_head_dim: cfg.head_dim,
            v_head_dim: cfg.head_dim,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Self {
            tok_embeddings,
            layers,
            norm,
            output,
            encoder,
            adapter,
            cache: EitherCache::Normal(NormalCache::new_sliding(
                cfg.n_layers,
                cfg.model_max_length,
                cfg.sliding_window,
            )),
            device: normal_loading_metadata.real_device,
            max_seq_len: cfg.model_max_length,
            cfg: cfg_meta,
            mapper,
            sliding_window: cfg.sliding_window,
            num_heads: cfg.n_heads,
            model_dim: cfg.dim,
            ada_rms_norm_t_cond: cfg.ada_rms_norm_t_cond,
            dtype: vb.dtype(),
            audio_embeds_cache: Arc::new(Mutex::new(None)),
        })
    }

    fn inner_forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        flash_params: &FlashParams,
        mel_features: Option<&Tensor>,
        n_delay_tokens: f32,
    ) -> Result<Tensor> {
        let text_embeds = self.tok_embeddings.forward(input_ids)?;

        let input_embeds = if let Some(mel) = mel_features {
            // Prompt phase: encode audio, store embeddings for generation steps.
            self.encoder.reset_cache();
            let audio_hidden = self.encoder.forward(mel)?;
            let audio_embeds = self.adapter.forward(&audio_hidden)?;
            let audio_embeds = audio_embeds.to_dtype(text_embeds.dtype())?;

            // Store for per-step conditioning during autoregressive generation.
            *self
                .audio_embeds_cache
                .lock()
                .expect("audio_embeds_cache lock") = Some(audio_embeds.clone());

            // Add audio embeddings to text at overlapping positions (0..min(prompt_len, N_audio)).
            // Audio is left-padded with silence so positions 0..31 contain encoded silence
            // that gets added to the BOS + left_pad token embeddings.
            // Audio extends beyond the prompt; remaining positions are used during generation.
            let text_len = text_embeds.dim(1)?;
            let audio_len = audio_embeds.dim(1)?;
            let overlap = text_len.min(audio_len);
            let text_prefix = text_embeds.narrow(1, 0, overlap)?;
            let audio_prefix = audio_embeds.narrow(1, 0, overlap)?;
            let combined_prefix = (text_prefix + audio_prefix)?;
            if overlap < text_len {
                let text_suffix = text_embeds.narrow(1, overlap, text_len - overlap)?;
                Tensor::cat(&[&combined_prefix, &text_suffix], 1)?
            } else {
                combined_prefix
            }
        } else {
            // Generation phase: add audio embedding at the current position.
            let cache = self
                .audio_embeds_cache
                .lock()
                .expect("audio_embeds_cache lock");
            if let Some(ref audio_embeds) = *cache {
                let audio_len = audio_embeds.dim(1)?;
                let pos = seqlen_offsets[0];
                let seq_len = text_embeds.dim(1)?;
                let end_pos = (pos + seq_len).min(audio_len);
                if pos < end_pos {
                    let n = end_pos - pos;
                    let audio_slice = audio_embeds.narrow(1, pos, n)?;
                    let text_prefix = text_embeds.narrow(1, 0, n)?;
                    let combined = (text_prefix + audio_slice)?;
                    if n < seq_len {
                        let text_suffix = text_embeds.narrow(1, n, seq_len - n)?;
                        Tensor::cat(&[&combined, &text_suffix], 1)?
                    } else {
                        combined
                    }
                } else {
                    // Past audio length: text-only
                    text_embeds
                }
            } else {
                // No audio (text-only mode)
                text_embeds
            }
        };

        let total_len = input_embeds.dim(1)?;
        let b_sz = input_embeds.dim(0)?;

        // Compute time conditioning embedding if adaptive norm is enabled
        let t_cond = if self.ada_rms_norm_t_cond {
            Some(time_embedding(
                n_delay_tokens,
                self.model_dim,
                input_embeds.device(),
                self.dtype,
            )?)
        } else {
            None
        };

        // Create dummy tokens of the full length for mask generation
        let dummy_toks = Tensor::zeros((b_sz, total_len), DType::U32, input_embeds.device())?;

        // EitherCache::normal() returns MutexGuard via interior mutability
        let mut cache = self.cache.normal();
        let attention_mask = CausalMasker.make_sliding_window_causal_mask_matrix(
            &dummy_toks,
            &cache.0 as &dyn PastKvLenCache,
            self.sliding_window,
            input_embeds.dtype(),
            self.num_heads,
        )?;

        let attention_mask = DeviceMappedMask::new(attention_mask, &*self.mapper)?;
        let mut xs = input_embeds;
        for (i, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;
            let t_cond_mapped = t_cond
                .as_ref()
                .map(|tc| tc.to_device(xs.device()))
                .transpose()?;
            xs = layer.forward(
                &xs,
                attention_mask.as_ref().map(|m| m.get(xs.device())),
                seqlen_offsets,
                &mut cache.0[i],
                t_cond_mapped.as_ref(),
                flash_params,
            )?;
        }
        let xs = xs.to_device(&self.device)?;
        let xs = xs.apply(&self.norm)?;

        let mut xs = extract_logits(&xs, context_lens)?;
        if let Some(t) = self.output.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        let logits = MatMul.qmethod_matmul(&xs, &*self.output)?;
        Ok(logits)
    }
}

impl IsqModel for VoxtralModel {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        let mut tensors = Vec::new();
        // lm_head / output
        tensors.push((&mut self.output, None));
        // Decoder layers
        for (i, layer) in self.layers.iter_mut().enumerate() {
            tensors.push((&mut layer.attention.wq, Some(i)));
            tensors.push((&mut layer.attention.wk, Some(i)));
            tensors.push((&mut layer.attention.wv, Some(i)));
            tensors.push((&mut layer.attention.wo, Some(i)));
            tensors.extend(
                layer
                    .feed_forward
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

        let uvb_mm = uvb.pp("mm_streams_embeddings").pp("embedding_module");

        // Embeddings
        uvb_mm.pp("tok_embeddings").add(&self.tok_embeddings);
        // Final norm
        uvb.pp("norm").add(&self.norm);

        // Decoder layer norms and adaptive norm weights
        for (i, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb.pp("layers").pp(i);
            uvb_l.pp("attention_norm").add(&layer.attention_norm);
            uvb_l.pp("ffn_norm").add(&layer.ffn_norm);
            if let Some(ref ada_norm) = layer.ada_norm {
                let uvb_ada = uvb_l.pp("ada_rms_norm_t_cond");
                uvb_ada.pp("0").add(&ada_norm.w0);
                uvb_ada.pp("2").add(&ada_norm.w2);
            }
        }

        // Encoder weights (all non-quantized)
        let uvb_enc = uvb_mm.pp("whisper_encoder");
        uvb_enc
            .pp("conv_layers")
            .pp("0")
            .pp("conv")
            .add(&self.encoder.conv1);
        uvb_enc
            .pp("conv_layers")
            .pp("1")
            .pp("conv")
            .add(&self.encoder.conv2);
        uvb_enc.pp("transformer").pp("norm").add(&self.encoder.norm);
        for (i, layer) in self.encoder.layers.iter().enumerate() {
            let uvb_l = uvb_enc.pp("transformer").pp("layers").pp(i);
            uvb_l.pp("attention_norm").add(&layer.attention_norm);
            uvb_l.pp("ffn_norm").add(&layer.ffn_norm);
            let uvb_attn = uvb_l.pp("attention");
            uvb_attn.pp("wq").add(&layer.attention.wq);
            uvb_attn.pp("wk").add(&layer.attention.wk);
            uvb_attn.pp("wv").add(&layer.attention.wv);
            uvb_attn.pp("wo").add(&layer.attention.wo);
            let uvb_ff = uvb_l.pp("feed_forward");
            uvb_ff.pp("w1").add(&layer.feed_forward.w1);
            uvb_ff.pp("w2").add(&layer.feed_forward.w2);
            uvb_ff.pp("w3").add(&layer.feed_forward.w3);
        }

        // Adapter weights
        let uvb_ada = uvb_mm.pp("audio_language_projection");
        uvb_ada.pp("0").add(&self.adapter.w_in);
        uvb_ada.pp("2").add(&self.adapter.w_out);

        uvb.to_safetensors()
    }

    fn imatrix_names(&self) -> candle_core::Result<Vec<Option<String>>> {
        let mut names = Vec::new();
        // output / lm_head
        names.push(None);
        for i in 0..self.layers.len() {
            names.push(Some(format!("blk.{i}.attn_q.weight")));
            names.push(Some(format!("blk.{i}.attn_k.weight")));
            names.push(Some(format!("blk.{i}.attn_v.weight")));
            names.push(Some(format!("blk.{i}.attn_output.weight")));
            // w1=gate, w3=up, w2=down (matches get_isq_layers order)
            names.push(Some(format!("blk.{i}.ffn_gate.weight")));
            names.push(Some(format!("blk.{i}.ffn_up.weight")));
            names.push(Some(format!("blk.{i}.ffn_down.weight")));
        }
        Ok(names)
    }
}

impl VisionModel for VoxtralModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        _pixel_values: Option<Tensor>,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        model_specific_args: Box<dyn Any>,
        _metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> candle_core::Result<Tensor> {
        let args = model_specific_args
            .downcast::<VoxtralSpecificArgs>()
            .expect("Downcast to VoxtralSpecificArgs failed");

        self.inner_forward(
            input_ids,
            seqlen_offsets,
            context_lens,
            flash_params,
            args.mel_features.as_ref(),
            args.n_delay_tokens.unwrap_or(0.0),
        )
    }

    fn default_model_specific_args(&self, _input_ids: &Tensor) -> Box<dyn Any> {
        Box::new(VoxtralSpecificArgs::default())
    }

    fn reset_model_specific_state(&self) {
        *self
            .audio_embeds_cache
            .lock()
            .expect("audio_embeds_cache lock") = None;
        self.encoder.reset_cache();
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

    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    fn config(&self) -> &ModelConfigMetadata {
        &self.cfg
    }
}

impl AnyMoeBaseModelMixin for VoxtralModel {}
