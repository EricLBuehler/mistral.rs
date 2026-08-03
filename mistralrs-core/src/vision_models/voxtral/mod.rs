#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::layers_masker::CausalMaskConfig;
use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Module, Result, Tensor};
use mistralrs_quant::{QuantMethod, ShardedVarBuilder};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::{AttentionMask, SdpaParams},
    device_map::{DeviceMappedMask, DeviceMapper},
    layers::{embedding, CausalMasker, RmsNorm, RotaryEmbedding, Sdpa},
    layers_masker::PastKvLenCache,
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        text_models_inputs_processor::PagedAttentionInputMetadata, EitherCache, IsqModel, KvCache,
        ModelForwardContext, MultimodalModel, NormalCache, NormalLoadingMetadata,
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

const AUDIO_ENCODER_KERNEL_SIZE: usize = 3;
const AUDIO_ENCODER_STRIDE: usize = 2;
const AUDIO_ENCODER_LEFT_PADDING: usize = 1;

struct DecoderAttention {
    wq: Arc<dyn QuantMethod>,
    wk: Arc<dyn QuantMethod>,
    wv: Arc<dyn QuantMethod>,
    wo: Arc<dyn QuantMethod>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
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
        paged_attn: Option<PagedAttention>,
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
            paged_attn,
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
        attention_mask: &AttentionMask,
        ctx: &mut ModelForwardContext<'_>,
        kv_cache: &mut KvCache,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let q = self.wq.forward(xs)?;
        let k = self.wk.forward(xs)?;
        let v = self.wv.forward(xs)?;
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

        let positions = ctx
            .text_positions(q.device(), q.dim(2)?)?
            .ok_or_else(|| candle_core::Error::msg("missing RoPE positions"))?
            .clone();
        let (q, k) = self.rotary_emb.forward(&q, &k, &positions)?;

        let metadata = ctx.paged_layer(layer_idx);
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
                    Some(ctx.flash_params()),
                )?,
                None => {
                    let input_metadata = PagedAttentionInputMetadata::dummy(q.device())?;
                    if matches!(attention_mask, AttentionMask::None) {
                        candle_core::bail!("Voxtral paged attention requires metadata for decode");
                    }
                    paged_attn.forward(
                        &q,
                        &k,
                        &v,
                        attention_mask,
                        None,
                        None,
                        &input_metadata,
                        &self.sdpa_params,
                        Some(ctx.flash_params()),
                    )?
                }
            },
            None => {
                let (k, v) = kv_cache.append(&k, &v)?;
                Sdpa.run_attention(
                    &q,
                    &k,
                    &v,
                    attention_mask,
                    Some(ctx.flash_params()),
                    &self.sdpa_params,
                )?
            }
        };

        attn_output = if !matches!(attention_mask, AttentionMask::None) {
            attn_output.transpose(1, 2)?.reshape((b_sz, q_len, ()))?
        } else {
            attn_output.reshape((b_sz, q_len, ()))?
        };
        let res = self.wo.forward(&attn_output)?;
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
        let gate = self.w1.forward(xs)?;
        let up = self.w3.forward(xs)?;
        let xs = crate::ops::mul_and_act(&gate, &up, crate::layers::Activation::Silu)?;
        let res = self.w2.forward(&xs)?;
        Ok(res)
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
        let xs = self.w0.forward(t_cond)?;
        let xs = xs.gelu_erf()?;
        self.w2.forward(&xs)
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
        paged_attn: Option<PagedAttention>,
    ) -> Result<Self> {
        let attention = DecoderAttention::new(
            cfg,
            rotary_emb,
            vb.pp("attention"),
            mapper,
            layer_idx,
            loading_isq,
            paged_attn,
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
        attention_mask: &AttentionMask,
        ctx: &mut ModelForwardContext<'_>,
        kv_cache: &mut KvCache,
        t_cond: Option<&Tensor>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.attention_norm.forward(xs)?;
        let xs = self
            .attention
            .forward(&xs, attention_mask, ctx, kv_cache, layer_idx)?;
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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct VoxtralAudioCacheKey {
    pub(super) sequence_id: usize,
    pub(super) hashes: Vec<u64>,
}

#[derive(Clone, Debug)]
pub(super) struct VoxtralAudioRequest {
    pub(super) logical_index: usize,
    pub(super) key: VoxtralAudioCacheKey,
    pub(super) mel_index: Option<usize>,
}

#[derive(Default)]
pub struct VoxtralSpecificArgs {
    pub mel_features: Option<Tensor>,
    pub(super) mel_lengths: Vec<usize>,
    pub(super) audio_requests: Vec<VoxtralAudioRequest>,
    /// Number of delay tokens for time conditioning (streaming pad tokens).
    /// Defaults to 0 if not provided.
    pub n_delay_tokens: Option<f32>,
}

pub struct VoxtralModel {
    tok_embeddings: Arc<dyn QuantMethod>,
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
    #[allow(dead_code)]
    num_heads: usize,
    model_dim: usize,
    ada_rms_norm_t_cond: bool,
    dtype: DType,
    adapter_downsample_factor: usize,
    audio_embeds_cache: Arc<Mutex<HashMap<VoxtralAudioCacheKey, Tensor>>>,
}

fn encoder_output_len(mel_frames: usize) -> Result<usize> {
    let padded_frames = mel_frames
        .checked_add(AUDIO_ENCODER_LEFT_PADDING)
        .ok_or_else(|| candle_core::Error::msg("Voxtral mel length overflow"))?;
    if padded_frames < AUDIO_ENCODER_KERNEL_SIZE {
        candle_core::bail!("Voxtral mel input is too short for the audio encoder");
    }
    Ok((padded_frames - AUDIO_ENCODER_KERNEL_SIZE) / AUDIO_ENCODER_STRIDE + 1)
}

fn adapter_output_len(encoder_tokens: usize, downsample_factor: usize) -> Result<usize> {
    if downsample_factor == 0 {
        candle_core::bail!("Voxtral adapter downsample factor cannot be zero");
    }
    Ok(encoder_tokens / downsample_factor)
}

fn validate_audio_request_layout(
    logical_count: usize,
    mel_count: usize,
    requests: &[VoxtralAudioRequest],
) -> Result<()> {
    let mut logical_indices = HashSet::with_capacity(requests.len());
    let mut keys = HashSet::with_capacity(requests.len());
    let mut mel_indices = vec![false; mel_count];
    for request in requests {
        if request.logical_index >= logical_count {
            candle_core::bail!(
                "Voxtral audio request index {} exceeds logical batch size {logical_count}",
                request.logical_index
            );
        }
        if !logical_indices.insert(request.logical_index) {
            candle_core::bail!(
                "Voxtral audio request index {} is duplicated",
                request.logical_index
            );
        }
        if !keys.insert(request.key.clone()) {
            candle_core::bail!(
                "Voxtral audio cache key is duplicated for sequence {}",
                request.key.sequence_id
            );
        }
        if request.key.hashes.is_empty() {
            candle_core::bail!(
                "Voxtral audio request {} is missing audio hashes",
                request.key.sequence_id
            );
        }
        if let Some(mel_index) = request.mel_index {
            let Some(seen) = mel_indices.get_mut(mel_index) else {
                candle_core::bail!(
                    "Voxtral mel index {mel_index} exceeds mel batch size {mel_count}"
                );
            };
            if *seen {
                candle_core::bail!("Voxtral mel index {mel_index} is duplicated");
            }
            *seen = true;
        }
    }
    if let Some(missing) = mel_indices.iter().position(|seen| !seen) {
        candle_core::bail!("Voxtral mel batch row {missing} has no request metadata");
    }
    Ok(())
}

fn add_audio_to_segment(segment: &Tensor, audio: &Tensor, offset: usize) -> Result<Tensor> {
    let (segment_batch, segment_len, segment_dim) = segment.dims3()?;
    let (audio_batch, audio_len, audio_dim) = audio.dims3()?;
    if segment_batch != 1 || audio_batch != 1 {
        candle_core::bail!("Voxtral audio conditioning requires single-request segments");
    }
    if segment_dim != audio_dim {
        candle_core::bail!(
            "Voxtral audio/text dimension mismatch: audio={audio_dim}, text={segment_dim}"
        );
    }
    let overlap = segment_len.min(audio_len.saturating_sub(offset));
    if overlap == 0 {
        return Ok(segment.clone());
    }

    let audio = audio
        .narrow(1, offset, overlap)?
        .to_device(segment.device())?
        .to_dtype(segment.dtype())?;
    let combined = (segment.narrow(1, 0, overlap)? + audio)?;
    if overlap == segment_len {
        Ok(combined)
    } else {
        Tensor::cat(
            &[
                &combined,
                &segment.narrow(1, overlap, segment_len - overlap)?,
            ],
            1,
        )
    }
}

fn condition_audio_embeddings(
    text_embeds: &Tensor,
    seqlen_offsets: &[usize],
    query_lens: Option<&[usize]>,
    requests: &[VoxtralAudioRequest],
    audio_cache: &HashMap<VoxtralAudioCacheKey, Tensor>,
) -> Result<Tensor> {
    let (physical_batch, physical_tokens, _) = text_embeds.dims3()?;
    let logical_count = query_lens.map_or(physical_batch, <[usize]>::len);
    if seqlen_offsets.len() != logical_count {
        candle_core::bail!(
            "Voxtral offset count {} does not match logical batch size {logical_count}",
            seqlen_offsets.len()
        );
    }

    let mut audio_by_logical = (0..logical_count)
        .map(|_| None)
        .collect::<Vec<Option<Tensor>>>();
    for request in requests {
        let audio = audio_cache.get(&request.key).ok_or_else(|| {
            candle_core::Error::msg(format!(
                "missing Voxtral audio state for sequence {} and hashes {:?}",
                request.key.sequence_id, request.key.hashes
            ))
        })?;
        audio_by_logical[request.logical_index] = Some(audio.clone());
    }

    if let Some(query_lens) = query_lens {
        if physical_batch != 1 {
            candle_core::bail!(
                "Voxtral packed prefill requires a flat physical batch, received {physical_batch}"
            );
        }
        let logical_tokens = query_lens.iter().sum::<usize>();
        if logical_tokens != physical_tokens {
            candle_core::bail!(
                "Voxtral packed query lengths total {logical_tokens}, expected {physical_tokens}"
            );
        }
        if query_lens.contains(&0) {
            candle_core::bail!("Voxtral packed query lengths cannot be empty");
        }

        let mut cursor = 0;
        let mut segments = Vec::with_capacity(logical_count);
        for (logical_index, &query_len) in query_lens.iter().enumerate() {
            let segment = text_embeds.narrow(1, cursor, query_len)?;
            let segment = match &audio_by_logical[logical_index] {
                Some(audio) => {
                    add_audio_to_segment(&segment, audio, seqlen_offsets[logical_index])?
                }
                None => segment,
            };
            segments.push(segment);
            cursor += query_len;
        }
        return Tensor::cat(&segments, 1);
    }

    if physical_batch != logical_count {
        candle_core::bail!(
            "Voxtral physical batch size {physical_batch} does not match logical batch size {logical_count}"
        );
    }
    let mut rows = Vec::with_capacity(physical_batch);
    for (logical_index, audio) in audio_by_logical.iter().enumerate() {
        let row = text_embeds.narrow(0, logical_index, 1)?;
        rows.push(match audio {
            Some(audio) => add_audio_to_segment(&row, audio, seqlen_offsets[logical_index])?,
            None => row,
        });
    }
    Tensor::cat(&rows, 0)
}

fn reset_audio_cache(cache: &mut HashMap<VoxtralAudioCacheKey, Tensor>, sequence_ids: &[usize]) {
    if sequence_ids.is_empty() {
        cache.clear();
        return;
    }
    let sequence_ids = sequence_ids.iter().copied().collect::<HashSet<_>>();
    cache.retain(|key, _| !sequence_ids.contains(&key.sequence_id));
}

impl VoxtralModel {
    pub fn new(
        cfg: &VoxtralConfig,
        vb: ShardedVarBuilder,
        _is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
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
            mapper.set_nm_device(
                vb_mm.pp("tok_embeddings"),
                normal_loading_metadata.loading_isq,
            ),
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
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => {
                    Some(PagedAttention::new(head_dim, device, None)?)
                }
            };
            DecoderLayer::new(
                cfg,
                rotary_emb,
                vb_layers.pp(layer_idx),
                &*mapper,
                layer_idx,
                normal_loading_metadata.loading_isq,
                paged_attn,
            )
        })?;

        let norm = RmsNorm::new(
            cfg.dim,
            cfg.norm_eps,
            mapper.set_nm_device(vb.pp("norm"), false),
        )?;

        // output (lm_head), may be tied with tok_embeddings
        let output = if cfg.tied_embeddings {
            tok_embeddings.clone()
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
            adapter_downsample_factor: ds_cfg.downsample_factor,
            audio_embeds_cache: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    fn cache_audio_embeddings(
        &self,
        mel_features: &Tensor,
        mel_lengths: &[usize],
        requests: &[VoxtralAudioRequest],
    ) -> Result<()> {
        let (mel_count, padded_frames, _) = mel_features.dims3()?;
        if mel_lengths.len() != mel_count {
            candle_core::bail!(
                "Voxtral mel length count {} does not match mel batch size {mel_count}",
                mel_lengths.len()
            );
        }
        for &frames in mel_lengths {
            if frames == 0 || frames > padded_frames {
                candle_core::bail!(
                    "Voxtral mel length {frames} is outside padded length {padded_frames}"
                );
            }
        }

        self.encoder.reset_cache();
        let audio_hidden = self.encoder.forward(mel_features)?;
        let (_, padded_encoder_tokens, _) = audio_hidden.dims3()?;

        let mut request_by_mel = (0..mel_count)
            .map(|_| None)
            .collect::<Vec<Option<&VoxtralAudioRequest>>>();
        for request in requests {
            if let Some(mel_index) = request.mel_index {
                request_by_mel[mel_index] = Some(request);
            }
        }

        let mut encoded = Vec::with_capacity(mel_count);
        for (mel_index, request) in request_by_mel.into_iter().enumerate() {
            let request = request.expect("validated mel request layout");
            let encoder_tokens = encoder_output_len(mel_lengths[mel_index])?;
            let audio_tokens = adapter_output_len(encoder_tokens, self.adapter_downsample_factor)?;
            if encoder_tokens == 0 || encoder_tokens > padded_encoder_tokens {
                candle_core::bail!(
                    "Voxtral encoder length {encoder_tokens} is outside padded length {padded_encoder_tokens}"
                );
            }
            let audio = audio_hidden
                .narrow(0, mel_index, 1)?
                .narrow(1, 0, encoder_tokens)?;
            let audio = self.adapter.forward(&audio)?.to_dtype(self.dtype)?;
            if audio.dim(1)? != audio_tokens {
                candle_core::bail!(
                    "Voxtral adapter produced {} tokens, expected {audio_tokens}",
                    audio.dim(1)?
                );
            }
            encoded.push((request.key.clone(), audio));
        }
        self.audio_embeds_cache
            .lock()
            .expect("audio_embeds_cache lock poisoned")
            .extend(encoded);
        Ok(())
    }

    fn inner_forward(
        &self,
        input_ids: &Tensor,
        ctx: &mut ModelForwardContext<'_>,
        mel_features: Option<&Tensor>,
        mel_lengths: &[usize],
        audio_requests: &[VoxtralAudioRequest],
        n_delay_tokens: f32,
    ) -> Result<Tensor> {
        let text_embeds = self
            .tok_embeddings
            .embedding_forward(input_ids, self.dtype)?;
        let query_lens = if ctx.flash_params().packed {
            Some(
                ctx.paged_input_metadata()
                    .and_then(|metadata| metadata.query_lens.as_deref())
                    .ok_or_else(|| {
                        candle_core::Error::msg(
                            "Voxtral packed prefill requires logical query lengths",
                        )
                    })?,
            )
        } else {
            None
        };
        let logical_count = query_lens.map_or(text_embeds.dim(0)?, <[usize]>::len);
        let mel_count = match mel_features {
            Some(mel) => mel.dim(0)?,
            None => 0,
        };
        validate_audio_request_layout(logical_count, mel_count, audio_requests)?;
        if let Some(mel_features) = mel_features {
            self.cache_audio_embeddings(mel_features, mel_lengths, audio_requests)?;
        } else if !mel_lengths.is_empty() {
            candle_core::bail!("Voxtral mel lengths were provided without mel features");
        }

        let input_embeds = {
            let cache = self
                .audio_embeds_cache
                .lock()
                .expect("audio_embeds_cache lock poisoned");
            condition_audio_embeddings(
                &text_embeds,
                ctx.seqlen_offsets(),
                query_lens,
                audio_requests,
                &cache,
            )?
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
        let mask_cache = ctx.mask_cache(&cache.0);
        let attention_mask = CausalMasker.make_causal_mask(
            &dummy_toks,
            &mask_cache as &dyn PastKvLenCache,
            input_embeds.dtype(),
            &CausalMaskConfig {
                sliding_window: self.sliding_window,
                ..Default::default()
            },
        )?;
        let attention_mask = if ctx.is_first_prompt_chunk() {
            attention_mask
        } else {
            AttentionMask::None
        };

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
                &attention_mask.get(xs.device()),
                ctx,
                &mut cache.0[i],
                t_cond_mapped.as_ref(),
                i,
            )?;
        }
        let xs = xs.to_device(&self.device)?;
        let xs = xs.apply(&self.norm)?;

        let xs = ctx.logits(&xs)?;
        let logits = self.output.forward(&xs)?;
        Ok(logits)
    }
}

impl IsqModel for VoxtralModel {
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
}

impl crate::speculative::SpeculativeTargetMixin for VoxtralModel {}

impl crate::block_diffusion::BlockDiffusionMixin for VoxtralModel {}

impl MultimodalModel for VoxtralModel {
    fn requires_uniform_completion_batch(&self) -> bool {
        true
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        _pixel_values: Option<Tensor>,
        model_specific_args: Box<dyn Any>,
        ctx: &mut ModelForwardContext<'_>,
    ) -> candle_core::Result<Tensor> {
        let args = model_specific_args
            .downcast::<VoxtralSpecificArgs>()
            .expect("Downcast to VoxtralSpecificArgs failed");

        self.inner_forward(
            input_ids,
            ctx,
            args.mel_features.as_ref(),
            &args.mel_lengths,
            &args.audio_requests,
            args.n_delay_tokens.unwrap_or(0.0),
        )
    }

    fn default_model_specific_args(&self, _input_ids: &Tensor) -> Box<dyn Any> {
        Box::new(VoxtralSpecificArgs::default())
    }

    fn reset_model_specific_state(&self) {
        reset_audio_cache(
            &mut self
                .audio_embeds_cache
                .lock()
                .expect("audio_embeds_cache lock poisoned"),
            &[],
        );
        self.encoder.reset_cache();
    }

    fn reset_model_specific_state_for_sequences(&self, sequence_ids: &[usize]) {
        reset_audio_cache(
            &mut self
                .audio_embeds_cache
                .lock()
                .expect("audio_embeds_cache lock poisoned"),
            sequence_ids,
        );
        self.encoder.reset_cache();
    }

    fn cache(&self) -> &EitherCache {
        &self.cache
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

    fn supports_packed_prefill(&self) -> bool {
        true
    }

    fn supports_mixed_media_batches(&self) -> bool {
        true
    }
}

impl AnyMoeBaseModelMixin for VoxtralModel {}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{Device, Tensor};

    use super::{
        adapter_output_len, condition_audio_embeddings, encoder_output_len, reset_audio_cache,
        validate_audio_request_layout, VoxtralAudioCacheKey, VoxtralAudioRequest,
    };

    fn key(sequence_id: usize, hashes: &[u64]) -> VoxtralAudioCacheKey {
        VoxtralAudioCacheKey {
            sequence_id,
            hashes: hashes.to_vec(),
        }
    }

    #[test]
    fn audio_cache_keys_are_request_and_hash_order_scoped() {
        assert_ne!(key(1, &[10, 20]), key(2, &[10, 20]));
        assert_ne!(key(1, &[10, 20]), key(1, &[20, 10]));
    }

    #[test]
    fn audio_lengths_follow_causal_conv_and_adapter_shapes() -> candle_core::Result<()> {
        assert_eq!(encoder_output_len(8)?, 4);
        assert_eq!(encoder_output_len(9)?, 4);
        assert_eq!(encoder_output_len(10)?, 5);
        assert_eq!(adapter_output_len(encoder_output_len(8)?, 4)?, 1);
        assert_eq!(adapter_output_len(encoder_output_len(9)?, 4)?, 1);
        assert_eq!(adapter_output_len(encoder_output_len(10)?, 4)?, 1);
        assert_eq!(adapter_output_len(encoder_output_len(16)?, 4)?, 2);
        Ok(())
    }

    #[test]
    fn conditioning_isolates_reordered_requests_and_row_offsets() -> candle_core::Result<()> {
        let first_key = key(1, &[10]);
        let second_key = key(2, &[20]);
        let mut cache = HashMap::new();
        cache.insert(
            first_key.clone(),
            Tensor::from_vec(vec![10f32, 11., 12.], (1, 3, 1), &Device::Cpu)?,
        );
        cache.insert(
            second_key.clone(),
            Tensor::from_vec(vec![20f32, 21., 22.], (1, 3, 1), &Device::Cpu)?,
        );
        let requests = vec![
            VoxtralAudioRequest {
                logical_index: 1,
                key: second_key,
                mel_index: None,
            },
            VoxtralAudioRequest {
                logical_index: 0,
                key: first_key,
                mel_index: None,
            },
        ];
        validate_audio_request_layout(2, 0, &requests)?;

        let text = Tensor::zeros((2, 1, 1), candle_core::DType::F32, &Device::Cpu)?;
        let conditioned = condition_audio_embeddings(&text, &[1, 2], None, &requests, &cache)?;
        assert_eq!(conditioned.flatten_all()?.to_vec1::<f32>()?, vec![11., 22.]);
        Ok(())
    }

    #[test]
    fn packed_conditioning_respects_logical_query_ranges() -> candle_core::Result<()> {
        let first_key = key(1, &[10]);
        let second_key = key(2, &[20]);
        let cache = HashMap::from([
            (
                first_key.clone(),
                Tensor::from_vec(vec![10f32, 11., 12.], (1, 3, 1), &Device::Cpu)?,
            ),
            (
                second_key.clone(),
                Tensor::from_vec(vec![20f32, 21., 22.], (1, 3, 1), &Device::Cpu)?,
            ),
        ]);
        let requests = vec![
            VoxtralAudioRequest {
                logical_index: 1,
                key: second_key,
                mel_index: None,
            },
            VoxtralAudioRequest {
                logical_index: 0,
                key: first_key,
                mel_index: None,
            },
        ];
        validate_audio_request_layout(2, 0, &requests)?;

        let text = Tensor::zeros((1, 3, 1), candle_core::DType::F32, &Device::Cpu)?;
        let conditioned =
            condition_audio_embeddings(&text, &[0, 1], Some(&[2, 1]), &requests, &cache)?;
        assert_eq!(
            conditioned.flatten_all()?.to_vec1::<f32>()?,
            vec![10., 11., 21.]
        );
        Ok(())
    }

    #[test]
    fn audio_request_layout_rejects_mel_cardinality_errors() {
        let duplicate = vec![
            VoxtralAudioRequest {
                logical_index: 0,
                key: key(1, &[10]),
                mel_index: Some(0),
            },
            VoxtralAudioRequest {
                logical_index: 0,
                key: key(2, &[20]),
                mel_index: Some(1),
            },
        ];
        assert!(validate_audio_request_layout(2, 2, &duplicate)
            .unwrap_err()
            .to_string()
            .contains("duplicated"));

        let missing = vec![VoxtralAudioRequest {
            logical_index: 0,
            key: key(1, &[10]),
            mel_index: Some(0),
        }];
        assert!(validate_audio_request_layout(2, 2, &missing)
            .unwrap_err()
            .to_string()
            .contains("row 1 has no request metadata"));
    }

    #[test]
    fn sequence_scoped_reset_preserves_other_audio_requests() -> candle_core::Result<()> {
        let first_key = key(1, &[10]);
        let first_replacement_key = key(1, &[11]);
        let second_key = key(2, &[20]);
        let mut cache = HashMap::from([
            (
                first_key.clone(),
                Tensor::zeros((1, 1, 1), candle_core::DType::F32, &Device::Cpu)?,
            ),
            (
                first_replacement_key.clone(),
                Tensor::zeros((1, 1, 1), candle_core::DType::F32, &Device::Cpu)?,
            ),
            (
                second_key.clone(),
                Tensor::zeros((1, 1, 1), candle_core::DType::F32, &Device::Cpu)?,
            ),
        ]);

        reset_audio_cache(&mut cache, &[1]);
        assert!(!cache.contains_key(&first_key));
        assert!(!cache.contains_key(&first_replacement_key));
        assert!(cache.contains_key(&second_key));

        reset_audio_cache(&mut cache, &[]);
        assert!(cache.is_empty());
        Ok(())
    }
}
