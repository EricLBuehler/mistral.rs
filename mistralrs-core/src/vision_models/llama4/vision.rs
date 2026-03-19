#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{collections::HashMap, sync::Arc};

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{LayerNorm, LayerNormConfig, Linear, Module};
use indicatif::MultiProgress;
use mistralrs_quant::{ColumnParallelLayer, QuantMethod, RowParallelLayer, ShardedVarBuilder};

use crate::{
    attention::SdpaParams,
    layers::{layer_norm, linear_no_bias, Activation, Sdpa},
    ops::RepeatInterleaveOp,
    pipeline::{text_models_inputs_processor::FlashParams, IsqModel},
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

use super::config::VisionConfig;

struct Llama4UnfoldConvolution {
    linear: Linear,
    kernel_size: usize,
    patch_size: usize,
}

impl Llama4UnfoldConvolution {
    fn new(cfg: &VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let kernel_size = cfg.patch_size;
        let linear = linear_no_bias(
            cfg.num_channels * kernel_size * kernel_size,
            cfg.hidden_size,
            vb.pp("linear"),
        )?;
        Ok(Self {
            linear,
            kernel_size,
            patch_size: cfg.patch_size,
        })
    }

    fn unfold(&self, xs: &Tensor) -> Result<Tensor> {
        // In original code
        let kernel_size = (self.kernel_size, self.kernel_size);
        let stride = (self.patch_size, self.patch_size);
        let padding = (0, 0);
        let dilation = (1, 1);
        let (bs, c, h, w) = xs.dims4()?;

        let h_out = (h + 2 * padding.0 - dilation.0 * (kernel_size.0 - 1) - 1) / stride.0 + 1;
        let w_out = (w + 2 * padding.1 - dilation.1 * (kernel_size.1 - 1) - 1) / stride.1 + 1;

        // Extract blocks
        let mut blocks = Vec::new();
        for i in (0..h - kernel_size.0 * dilation.0 + 1).step_by(stride.0) {
            for j in (0..w - kernel_size.1 * dilation.1 + 1).step_by(stride.1) {
                let mut block = Vec::new();
                for di in 0..kernel_size.0 {
                    for dj in 0..kernel_size.1 {
                        let h_idx = i + di * dilation.0;
                        let w_idx = j + dj * dilation.0;
                        // Get the block for all channels and add to our list
                        block.push(xs.i((.., .., h_idx, w_idx))?);
                    }
                }

                // Stack the channel-blocks
                // (b, k*k, c)
                let mut block = Tensor::stack(&block, 1)?;
                block = block.permute((0, 2, 1))?;
                blocks.push(block);
            }
        }

        // (b, c, k*k, l)
        let mut result = Tensor::stack(&blocks, D::Minus1)?;
        // (b, c*k*k, l)
        result = result.reshape((bs, c * kernel_size.0 * kernel_size.1, h_out * w_out))?;
        Ok(result)
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // let hidden_states = {
        //     let mut patches = hidden_states
        //         .unfold(2, self.kernel_size, self.patch_size)?
        //         .unfold(3, self.kernel_size, self.patch_size)?;
        //     patches = patches.contiguous()?.permute((0, 2, 3, 1, 4, 5))?;
        //     let b = patches.dim(0)?;
        //     let out_h = patches.dim(1)?;
        //     let out_w = patches.dim(2)?;
        //     let c = patches.dim(3)?;
        //     patches.reshape((b, out_h * out_w, c * self.kernel_size * self.kernel_size))?
        // };

        let mut hidden_states = self.unfold(hidden_states)?;
        hidden_states = hidden_states.transpose(1, 2)?;
        self.linear.forward(&hidden_states)
    }
}

struct Llama4VisionAttention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    sdpa_params: SdpaParams,
    head_dim: usize,
    freqs: Llama4VisionRotaryEmbedding,
}

impl Llama4VisionAttention {
    fn new(
        cfg: &VisionConfig,
        vb: ShardedVarBuilder,
        freqs: Llama4VisionRotaryEmbedding,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        Ok(Self {
            q_proj: ColumnParallelLayer::new(
                cfg.hidden_size,
                cfg.num_attention_heads * head_dim,
                &None,
                true,
                comm,
                vb.pp("q_proj"),
            )?,
            k_proj: ColumnParallelLayer::new(
                cfg.hidden_size,
                cfg.num_attention_heads * head_dim,
                &None,
                true,
                comm,
                vb.pp("k_proj"),
            )?,
            v_proj: ColumnParallelLayer::new(
                cfg.hidden_size,
                cfg.num_attention_heads * head_dim,
                &None,
                true,
                comm,
                vb.pp("v_proj"),
            )?,
            o_proj: RowParallelLayer::new(
                cfg.hidden_size,
                cfg.num_attention_heads * head_dim,
                &None,
                true,
                comm,
                vb.pp("o_proj"),
            )?,
            sdpa_params: SdpaParams {
                n_kv_groups: 1,
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window: None,
                sinks: None,
            },
            head_dim,
            freqs,
        })
    }

    fn forward(&self, hidden_state: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let mut hidden_state = hidden_state.clone();
        let original_dtype = hidden_state.dtype();
        if let Some(t) = self.q_proj.quantized_act_type() {
            hidden_state = hidden_state.to_dtype(t)?;
        }
        let mut q = self.q_proj.forward(&hidden_state)?;
        let mut k = self.k_proj.forward(&hidden_state)?;
        let mut v = self.v_proj.forward(&hidden_state)?;
        if self.q_proj.quantized_act_type().is_some() {
            q = q.to_dtype(original_dtype)?;
            k = k.to_dtype(original_dtype)?;
            v = v.to_dtype(original_dtype)?;
        }

        // Should be same, no caching...
        let (bs, q_sq, _) = q.dims3()?;
        let (_, k_sq, _) = k.dims3()?;

        q = q
            .reshape((bs, q_sq, (), self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        k = k
            .reshape((bs, k_sq, (), self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        v = v
            .reshape((bs, k_sq, (), self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Apply rope
        {
            q = candle_nn::rotary_emb::rope_i(&q, &self.freqs.cos, &self.freqs.sin)?;
            k = candle_nn::rotary_emb::rope_i(&k, &self.freqs.cos, &self.freqs.sin)?;
        }

        let flash_params = FlashParams {
            max_q: 0,
            max_k: 0,
            cumulative_seqlens_q: HashMap::new(),
            cumulative_seqlens_k: HashMap::new(),
            causal: false,
        };

        let mut attn_output = Sdpa
            .run_attention(
                &q,
                &k,
                &v,
                attention_mask,
                Some(&flash_params),
                &self.sdpa_params,
            )?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((bs, q_sq, ()))?
            .to_dtype(q.dtype())?;

        if let Some(t) = self.q_proj.quantized_act_type() {
            attn_output = attn_output.to_dtype(t)?;
        }
        let mut res = self.o_proj.forward(&attn_output)?;
        if self.q_proj.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

struct Llama4Mlp {
    act: Activation,
    fc1: Arc<dyn QuantMethod>,
    fc2: Arc<dyn QuantMethod>,
}

impl Llama4Mlp {
    fn new(
        cfg: &VisionConfig,
        vb: ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        Ok(Self {
            act: cfg.hidden_act,
            fc1: ColumnParallelLayer::new(
                cfg.hidden_size,
                cfg.intermediate_size,
                &None,
                true,
                comm,
                vb.pp("fc1"),
            )?,
            fc2: RowParallelLayer::new(
                cfg.intermediate_size,
                cfg.hidden_size,
                &None,
                true,
                comm,
                vb.pp("fc2"),
            )?,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let original_dtype = hidden_states.dtype();
        let mut hidden_states = hidden_states.clone();
        if let Some(t) = self.fc1.quantized_act_type() {
            hidden_states = hidden_states.to_dtype(t)?;
        }
        hidden_states = self.fc1.forward(&hidden_states)?;
        hidden_states = self.act.forward(&hidden_states)?;
        hidden_states = self.fc2.forward(&hidden_states)?;
        if self.fc1.quantized_act_type().is_some() {
            hidden_states = hidden_states.to_dtype(original_dtype)?;
        }
        Ok(hidden_states)
    }
}

struct Llama4VisionEncoderLayer {
    self_attn: Llama4VisionAttention,
    mlp: Llama4Mlp,
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
}

impl Llama4VisionEncoderLayer {
    fn new(
        cfg: &VisionConfig,
        vb: ShardedVarBuilder,
        freqs: Llama4VisionRotaryEmbedding,
        real_dev: &Device,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let self_attn = Llama4VisionAttention::new(cfg, vb.pp("self_attn"), freqs, comm)?;
        let mlp = Llama4Mlp::new(cfg, vb.pp("mlp"), comm)?;

        let input_layernorm = layer_norm(
            cfg.hidden_size,
            cfg.norm_eps,
            vb.pp("input_layernorm").set_device(real_dev.clone()),
        )?;
        let post_attention_layernorm = layer_norm(
            cfg.hidden_size,
            cfg.norm_eps,
            vb.pp("post_attention_layernorm")
                .set_device(real_dev.clone()),
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(&self, hidden_state: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        // Self attn
        let residual = hidden_state;
        let mut hidden_state = self.input_layernorm.forward(hidden_state)?;

        hidden_state = self.self_attn.forward(&hidden_state, attention_mask)?;
        hidden_state = (residual + hidden_state)?;

        // FF
        let residual = hidden_state.clone();
        hidden_state = self.post_attention_layernorm.forward(&hidden_state)?;

        hidden_state = self.mlp.forward(&hidden_state)?;
        residual + hidden_state
    }
}

struct Llama4VisionEncoder {
    layers: Vec<Llama4VisionEncoderLayer>,
}

impl Llama4VisionEncoder {
    fn new(
        cfg: &VisionConfig,
        num_layers: usize,
        vb: ShardedVarBuilder,
        freqs: Llama4VisionRotaryEmbedding,
        real_dev: &Device,
        comm: &Arc<mistralrs_quant::Comm>,
        multi_progress: &Arc<MultiProgress>,
    ) -> Result<Self> {
        let layers_vb = vb.pp("layers");
        let layers = NiceProgressBar::<_, 'b'>(
            0..num_layers,
            "Loading vision repeating layers",
            multi_progress,
        )
        .par_iter_if_isq(|i| {
            Llama4VisionEncoderLayer::new(cfg, layers_vb.pp(i), freqs.clone(), real_dev, comm)
        })?;
        Ok(Self { layers })
    }

    fn forward_with_states(
        &self,
        hidden_state: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut hidden_state = hidden_state.clone();
        for layer in self.layers.iter() {
            hidden_state = layer.forward(&hidden_state, attention_mask)?;
        }
        Ok(hidden_state)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb_t = UnVarBuilder::new();

        for (i, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb_t.pp("layers").pp(i);
            uvb_l.pp("input_layernorm").add(&layer.input_layernorm);
            uvb_l
                .pp("post_attention_layernorm")
                .add(&layer.post_attention_layernorm);
        }

        uvb_t.to_safetensors()
    }
}

struct Llama4VisionPixelShuffleMLP {
    act: Activation,
    fc1: Arc<dyn QuantMethod>,
    fc2: Arc<dyn QuantMethod>,
}

impl Llama4VisionPixelShuffleMLP {
    fn new(
        cfg: &VisionConfig,
        vb: ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        Ok(Self {
            act: Activation::Gelu,
            fc1: ColumnParallelLayer::new(
                cfg.intermediate_size,
                cfg.projector_input_dim,
                &None,
                false,
                comm,
                vb.pp("fc1"),
            )?,
            fc2: RowParallelLayer::new(
                cfg.projector_input_dim,
                cfg.projector_output_dim,
                &None,
                false,
                comm,
                vb.pp("fc2"),
            )?,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let original_dtype = hidden_states.dtype();
        let mut hidden_states = hidden_states.clone();
        if let Some(t) = self.fc1.quantized_act_type() {
            hidden_states = hidden_states.to_dtype(t)?;
        }
        hidden_states = self.act.forward(
            &self
                .fc2
                .forward(&self.act.forward(&self.fc1.forward(&hidden_states)?)?)?,
        )?;
        if self.fc1.quantized_act_type().is_some() {
            hidden_states = hidden_states.to_dtype(original_dtype)?;
        }
        Ok(hidden_states)
    }
}

struct Llama4VisionPixelShuffle {
    mlp: Llama4VisionPixelShuffleMLP,
    pixel_shuffle_ratio: f32,
}

impl Llama4VisionPixelShuffle {
    fn new(
        cfg: &VisionConfig,
        vb: ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let mlp = Llama4VisionPixelShuffleMLP::new(cfg, vb.pp("mlp"), comm)?;
        Ok(Self {
            mlp,
            pixel_shuffle_ratio: cfg.pixel_shuffle_ratio,
        })
    }

    fn pixel_shuffle(&self, xs: &Tensor) -> Result<Tensor> {
        let (bs, num_patches, _c) = xs.dims3()?;
        let patch_size = (num_patches as f32).sqrt() as usize;

        let mut xs = xs.reshape((bs, patch_size, patch_size, ()))?;
        let (_bs, h, w, c) = xs.dims4()?;

        xs = xs.reshape((
            bs,
            h,
            (w as f32 * self.pixel_shuffle_ratio) as usize,
            (c as f32 / self.pixel_shuffle_ratio) as usize,
        ))?;
        xs = xs.permute((0, 2, 1, 3))?.contiguous()?;

        xs = xs.reshape((
            bs,
            (h as f32 * self.pixel_shuffle_ratio) as usize,
            (w as f32 * self.pixel_shuffle_ratio) as usize,
            (c as f32 / self.pixel_shuffle_ratio.powi(2)) as usize,
        ))?;
        xs = xs.permute((0, 2, 1, 3))?.contiguous()?;

        xs.reshape((bs, (), xs.dim(D::Minus1)?))
    }

    fn forward(&self, encoded_patches: &Tensor) -> Result<Tensor> {
        let encoded_patches = self.pixel_shuffle(encoded_patches)?;
        self.mlp.forward(&encoded_patches)
    }
}

#[derive(Clone)]
struct Llama4VisionRotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl Llama4VisionRotaryEmbedding {
    fn new(cfg: &VisionConfig, device: &Device, dtype: DType) -> Result<Self> {
        let idx = cfg.image_size / cfg.patch_size;
        let mut img_idx =
            Tensor::arange(0f32, idx.pow(2) as f32, device)?.reshape((idx.pow(2), 1))?;
        img_idx = Tensor::cat(&[&img_idx, &img_idx.narrow(0, 0, 1)?], 0)?;
        // Insert ID_CLS_TOKEN in the bottom right
        img_idx = img_idx.slice_assign(
            &[
                img_idx.dim(0)? - 1..img_idx.dim(0)?,
                img_idx.dim(1)? - 1..img_idx.dim(1)?,
            ],
            &Tensor::new(-2f32, device)?.reshape((1, 1))?,
        )?;
        let img_ids_flat = img_idx.flatten_all()?.to_vec1::<f32>()?;
        // frequencies_x = img_idx % idx
        // get the coordinates of the 2d matrix along x
        let frequencies_x = {
            let frequencies_x = img_ids_flat
                .iter()
                .map(|x| x % idx as f32)
                .collect::<Vec<_>>();
            Tensor::from_vec(frequencies_x, img_idx.shape().clone(), device)?
        };
        // frequencies_y = img_idx // idx
        // get the coordinates of the 2d matrix along y
        let frequencies_y = {
            let frequencies_y = img_ids_flat
                .iter()
                .map(|x| x / idx as f32)
                .collect::<Vec<_>>();
            Tensor::from_vec(frequencies_y, img_idx.shape().clone(), device)?
        };
        let rope_freq = {
            let freq_dim = cfg.hidden_size / cfg.num_attention_heads / 2;
            let freqs: Vec<_> = (0..freq_dim)
                .step_by(2)
                .take(freq_dim / 2)
                .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / freq_dim as f32))
                .collect();
            let freqs_len = freqs.len();
            Tensor::from_vec(freqs, freqs_len, device)?
        };
        let freqs_x = (frequencies_x + 1.)?
            .unsqueeze(D::Minus1)?
            .broadcast_mul(&rope_freq.unsqueeze(0)?.unsqueeze(0)?)?
            .repeat_interleave(2, D::Minus1)?;
        let freqs_y = (frequencies_y + 1.)?
            .unsqueeze(D::Minus1)?
            .broadcast_mul(&rope_freq.unsqueeze(0)?.unsqueeze(0)?)?
            .repeat_interleave(2, D::Minus1)?;
        let mut freqs = {
            let freqs = Tensor::cat(&[freqs_x, freqs_y], D::Minus1)?.contiguous()?;
            // This implements [..., ::2]
            let indices_every_two = Tensor::new(
                (0..freqs.dim(D::Minus1)?)
                    .step_by(2)
                    .map(|x| x as u32)
                    .collect::<Vec<_>>(),
                device,
            )?;
            freqs.index_select(&indices_every_two, D::Minus1)?
        };
        freqs = freqs.squeeze(1)?;
        freqs = freqs.lt(0.)?.where_cond(&freqs.zeros_like()?, &freqs)?;

        Ok(Self {
            cos: freqs.cos()?.to_dtype(dtype)?,
            sin: freqs.sin()?.to_dtype(dtype)?,
        })
    }
}

pub(super) struct Llama4VisionModel {
    patch_embedding: Llama4UnfoldConvolution,
    class_embedding: Tensor,
    positional_embedding_vlm: Tensor,
    layernorm_pre: LayerNorm,
    layernorm_post: LayerNorm,
    model: Llama4VisionEncoder,
    vision_adapter: Llama4VisionPixelShuffle,
}

impl Llama4VisionModel {
    pub(super) fn new(
        cfg: &VisionConfig,
        vb: ShardedVarBuilder,
        real_dev: &Device,
        comm: &Arc<mistralrs_quant::Comm>,
        multi_progress: &Arc<MultiProgress>,
    ) -> Result<Self> {
        let patch_embedding = Llama4UnfoldConvolution::new(
            cfg,
            vb.pp("patch_embedding").set_device(real_dev.clone()),
        )?;

        let class_embedding = vb
            .get((cfg.hidden_size,), "class_embedding")?
            .to_device(real_dev)?;
        let num_patches = cfg.num_patches();
        let positional_embedding_vlm = vb
            .get((num_patches, cfg.hidden_size), "positional_embedding_vlm")?
            .to_device(real_dev)?;

        // layer norms
        let layernorm_pre = layer_norm(
            cfg.hidden_size,
            LayerNormConfig::default(),
            vb.pp("layernorm_pre").set_device(real_dev.clone()),
        )?;
        let layernorm_post = layer_norm(
            cfg.hidden_size,
            LayerNormConfig::default(),
            vb.pp("layernorm_post").set_device(real_dev.clone()),
        )?;

        let rotary_embedding = Llama4VisionRotaryEmbedding::new(cfg, real_dev, vb.dtype())?;
        let model = Llama4VisionEncoder::new(
            cfg,
            cfg.num_hidden_layers,
            vb.pp("model"),
            rotary_embedding,
            real_dev,
            comm,
            multi_progress,
        )?;

        let vision_adapter = Llama4VisionPixelShuffle::new(cfg, vb.pp("vision_adapter"), comm)?;

        assert_eq!(cfg.vision_feature_layer, -1);

        Ok(Self {
            patch_embedding,
            class_embedding,
            positional_embedding_vlm,
            layernorm_post,
            layernorm_pre,
            model,
            vision_adapter,
        })
    }

    pub(super) fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let pixel_values = pixel_values.to_dtype(self.class_embedding.dtype())?;

        let (bs_times_num_tiles, _num_channels, _height, _width) = pixel_values.dims4()?;
        let num_concurrent_media = 1;

        // Patch embedding
        let mut hidden_state = self.patch_embedding.forward(&pixel_values)?;
        let (_, mut num_patches, hidden_dim) = hidden_state.dims3()?;

        // Add cls token
        hidden_state = hidden_state.reshape((
            bs_times_num_tiles * num_concurrent_media,
            num_patches,
            hidden_dim,
        ))?;
        let class_embedding =
            self.class_embedding
                .expand((hidden_state.dim(0)?, 1, hidden_state.dim(D::Minus1)?))?;
        hidden_state = Tensor::cat(&[hidden_state, class_embedding], 1)?;
        num_patches += 1;

        // Position embeddings
        hidden_state = hidden_state.reshape((
            bs_times_num_tiles * num_concurrent_media,
            num_patches,
            hidden_dim,
        ))?;
        hidden_state = hidden_state.broadcast_add(&self.positional_embedding_vlm)?;

        hidden_state = self.layernorm_pre.forward(&hidden_state)?;

        hidden_state = hidden_state.reshape((bs_times_num_tiles, (), hidden_dim))?;

        // Apply encoder
        hidden_state =
            hidden_state.reshape((bs_times_num_tiles * num_concurrent_media, (), hidden_dim))?;
        hidden_state = self.model.forward_with_states(&hidden_state, None)?;

        hidden_state = self.layernorm_post.forward(&hidden_state)?;

        hidden_state = hidden_state.narrow(1, 0, hidden_state.dim(1)? - 1)?;

        self.vision_adapter.forward(&hidden_state)
    }

    pub fn get_isq_layers(&mut self) -> Vec<&mut std::sync::Arc<dyn mistralrs_quant::QuantMethod>> {
        let mut layers = Vec::new();
        for layer in &mut self.model.layers {
            layers.push(&mut layer.self_attn.q_proj);
            layers.push(&mut layer.self_attn.k_proj);
            layers.push(&mut layer.self_attn.v_proj);
            layers.push(&mut layer.self_attn.o_proj);

            layers.push(&mut layer.mlp.fc1);
            layers.push(&mut layer.mlp.fc2);
        }
        layers.push(&mut self.vision_adapter.mlp.fc1);
        layers.push(&mut self.vision_adapter.mlp.fc2);
        layers
    }
}

impl IsqModel for Llama4VisionModel {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(
            &mut std::sync::Arc<dyn mistralrs_quant::QuantMethod>,
            Option<usize>,
        )>,
        &dyn crate::device_map::DeviceMapper,
    ) {
        unreachable!("Llama4Vision model cannot be quantized.");
    }
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        uvb.pp("patch_embedding")
            .pp("linear")
            .add(&self.patch_embedding.linear);
        uvb.add_tensor("class_embedding", self.class_embedding.clone());
        uvb.add_tensor(
            "positional_embedding_vlm",
            self.positional_embedding_vlm.clone(),
        );

        uvb.pp("layernorm_pre").add(&self.layernorm_pre);
        uvb.pp("layernorm_post").add(&self.layernorm_post);

        uvb.pp("model").extend(self.model.residual_tensors());

        uvb.to_safetensors()
    }
}
