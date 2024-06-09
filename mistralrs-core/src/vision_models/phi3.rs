#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

// This implementation is based on:
// https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/modeling_phi3.py
use candle_core::{
    quantized::QMatMul, shape::ShapeWithOneHole, DType, Device, IndexOp, Module, Result, Shape,
    Tensor, D,
};
use candle_nn::{linear_b, linear_no_bias, VarBuilder};
use either::Either;
use std::{any::Any, collections::HashMap, fmt::Debug, sync::Arc};

use crate::{
    device_map::DeviceMapper,
    layers::{
        repeat_kv, CausalMasker, FusedBiasLinear, MatMul, Nonzero, PhiRopeConfig,
        PhiRotaryEmbedding, RmsNorm, ScaledDotProductAttention,
    },
    pipeline::{
        extract_logits, Cache, IsqModel, NormalLoadingMetadata, Phi3RopeScaling, VisionModel,
    },
    serde_default_fn,
    vision_models::clip::{Activation, ClipConfig, ClipVisionTransformer},
};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct EmbedLayerConfig {
    hd_transform_order: Option<String>,
    projection_cls: Option<String>,
    use_hd_transform: Option<bool>,
    with_learnable_separator: Option<bool>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ImageProcessorConfig {
    image_dim_out: usize,
    name: String,
    num_img_tokens: usize,
    layer_idx: Option<isize>,
    type_feature: Option<String>,
}

serde_default_fn!(bool, d_flash_attn, false);

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_act: candle_nn::Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub rope_scaling: Option<HashMap<String, Phi3RopeScaling>>,
    pub max_position_embeddings: usize,
    #[serde(default = "d_flash_attn")]
    pub use_flash_attn: bool,
    pub sliding_window: Option<usize>,
    pub original_max_position_embeddings: usize,
    pub embd_layer: EmbedLayerConfig,
    pub img_processor: ImageProcessorConfig,
}

impl From<Config> for PhiRopeConfig {
    fn from(val: Config) -> Self {
        PhiRopeConfig {
            rope_scaling: val.rope_scaling,
            max_position_embeddings: val.max_position_embeddings,
            original_max_position_embeddings: val.original_max_position_embeddings,
            rope_theta: val.rope_theta,
            head_dim: val.hidden_size / val.num_attention_heads,
        }
    }
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

trait ModuleWithMetadata: Module + Debug + Send + Sync {
    fn device(&self) -> &Device;
    fn dtype(&self) -> DType;
}

impl ModuleWithMetadata for FusedBiasLinear {
    fn device(&self) -> &Device {
        self.w.device()
    }
    fn dtype(&self) -> DType {
        self.w.dtype()
    }
}

impl ModuleWithMetadata for candle_nn::Activation {
    fn device(&self) -> &Device {
        unreachable!()
    }
    fn dtype(&self) -> DType {
        unreachable!()
    }
}

#[derive(Debug)]
struct BigShapeWithOneHole((usize, usize, usize, usize, usize, ()));

fn hole_size(el_count: usize, prod_d: usize, s: &dyn std::fmt::Debug) -> Result<usize> {
    if prod_d == 0 {
        candle_core::bail!("cannot reshape tensor of {el_count} elements to {s:?}")
    }
    if el_count % prod_d != 0 {
        candle_core::bail!("cannot reshape tensor with {el_count} elements to {s:?}")
    }
    Ok(el_count / prod_d)
}

impl ShapeWithOneHole for BigShapeWithOneHole {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let (d1, d2, d3, d4, d5, ()) = self.0;
        let d = hole_size(el_count, d1 * d2 * d3 * d4 * d5, &self)?;
        Ok((d1, d2, d3, d4, d5, d).into())
    }
}

// =================== BASE LAYERS ===================

#[derive(Debug, Clone)]
struct Attention {
    qkv_proj: QMatMul,
    o_proj: QMatMul,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<PhiRotaryEmbedding>,
    use_flash_attn: bool,
    sliding_window: Option<usize>,
}

impl Attention {
    fn new(rotary_emb: Arc<PhiRotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();
        let op_size = num_heads * head_dim + 2 * num_kv_heads * head_dim;
        let qkv_proj = linear_no_bias(cfg.hidden_size, op_size, vb.pp("qkv_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?;
        Ok(Self {
            qkv_proj: QMatMul::Tensor(qkv_proj.weight().clone()),
            o_proj: QMatMul::Tensor(o_proj.weight().clone()),
            rotary_emb,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
            use_flash_attn: cfg.use_flash_attn,
            sliding_window: cfg.sliding_window,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        position_ids: &[usize],
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let original_dtype = xs.dtype();
        let mut xs = xs.clone();
        if matches!(self.qkv_proj, QMatMul::QTensor(_)) {
            xs = xs.to_dtype(DType::F32)?;
        }
        let mut qkv = MatMul.qmatmul(&xs, &self.qkv_proj)?;
        if matches!(self.qkv_proj, QMatMul::QTensor(_)) {
            qkv = qkv.to_dtype(original_dtype)?;
        }
        let query_pos = self.num_heads * self.head_dim;
        let q = qkv.narrow(D::Minus1, 0, query_pos)?;
        let k = qkv.narrow(D::Minus1, query_pos, self.num_kv_heads * self.head_dim)?;
        let v = qkv.narrow(
            D::Minus1,
            query_pos + self.num_kv_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
        )?;

        let q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = self
            .rotary_emb
            .forward(&q, &k, seqlen_offsets, position_ids)?;

        let (k, v, attn_mask) = Cache::update_kv_cache_sliding_window(
            kv_cache,
            k,
            v,
            attention_mask,
            self.sliding_window,
            true,
        )?;

        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let mut attn_output = ScaledDotProductAttention.run_attention(
            &q,
            &k,
            &v,
            self.num_heads,
            self.head_dim,
            attn_mask.as_ref(),
            self.use_flash_attn,
            b_sz,
            q_len,
        )?;

        if matches!(self.qkv_proj, QMatMul::QTensor(_)) {
            attn_output = attn_output.to_dtype(DType::F32)?;
        }
        let mut res = MatMul.qmatmul(
            &attn_output.transpose(1, 2)?.reshape((b_sz, q_len, ()))?,
            &self.o_proj,
        )?;
        if matches!(self.qkv_proj, QMatMul::QTensor(_)) {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    gate_up_proj: QMatMul,
    down_proj: QMatMul,
    act_fn: candle_nn::Activation,
    i_size: usize,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let gate_up_proj = linear_no_bias(hidden_size, 2 * i_size, vb.pp("gate_up_proj"))?;
        let down_proj = linear_no_bias(i_size, hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_up_proj: QMatMul::Tensor(gate_up_proj.weight().clone()),
            down_proj: QMatMul::Tensor(down_proj.weight().clone()),
            act_fn: cfg.hidden_act,
            i_size,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let original_dtype = xs.dtype();
        let mut xs = xs.clone();
        if matches!(self.gate_up_proj, QMatMul::QTensor(_)) {
            xs = xs.to_dtype(DType::F32)?;
        }
        let up_states = MatMul.qmatmul(&xs, &self.gate_up_proj)?;
        let gate = up_states.narrow(D::Minus1, 0, self.i_size)?;
        let up_states = up_states.narrow(D::Minus1, self.i_size, self.i_size)?;
        let up_states = (up_states * gate.apply(&self.act_fn))?;
        let mut res = MatMul.qmatmul(&up_states, &self.down_proj)?;
        if matches!(self.gate_up_proj, QMatMul::QTensor(_)) {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(
        rotary_emb: Arc<PhiRotaryEmbedding>,
        cfg: &Config,
        vb: VarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
    ) -> Result<Self> {
        let self_attn = Attention::new(
            rotary_emb,
            cfg,
            mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq),
        )?;
        let mlp = Mlp::new(cfg, mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq))?;
        let input_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("input_layernorm"), false),
        )?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("post_attention_layernorm"), false),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        position_ids: &[usize],
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs =
            self.self_attn
                .forward(&xs, attention_mask, seqlen_offsets, position_ids, kv_cache)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.post_attention_layernorm)?.apply(&self.mlp)?;
        residual + xs
    }
}

// =================== ============= ===================

// =================== VISION LAYERS ===================

const MAX_INPUT_ID: f64 = 1e9;

#[derive(Debug)]
struct EmbeddingLayers(Vec<Box<dyn ModuleWithMetadata>>);

impl Module for EmbeddingLayers {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in &self.0 {
            xs = layer.forward(&xs)?;
        }
        Ok(xs)
    }
}

#[derive(Debug)]
pub struct ImageEmbedding {
    wte: candle_nn::Embedding,
    image_dim_out: usize,
    num_img_tokens: usize,
    glb_gn: Option<Tensor>,
    sub_gn: Option<Tensor>,
    layers: EmbeddingLayers,
    type_feature: String,
    layer_idx: isize,
    image_processor: ClipVisionTransformer,
    hd_transform_order: String,
    use_hd_transform: bool,
    vocab_size: usize,
}

impl ImageEmbedding {
    fn new(
        config: &Config,
        wte: candle_nn::Embedding,
        embed_config: &EmbedLayerConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        if config.img_processor.name != "clip_vision_model" {
            candle_core::bail!(
                "img_processor=`{}` nor supported.",
                config.img_processor.name
            );
        }
        let image_dim_out = config.img_processor.image_dim_out;
        let num_img_tokens = config.img_processor.num_img_tokens;

        // CLIP image processor here...
        let image_processor = ClipVisionTransformer::new(
            vb.pp("img_processor.vision_model"),
            &ClipConfig {
                hidden_act: Activation::QuickGelu,
                hidden_size: 1024,
                image_size: 336,
                intermediate_size: 4096,
                layer_norm_eps: 1e-05,
                num_attention_heads: 16,
                num_channels: 3,
                num_hidden_layers: 24,
                patch_size: 14,
                projection_dim: 768,
            },
        )?;

        // High dim transform
        let use_hd_transform = embed_config.use_hd_transform.unwrap_or(false);
        let with_learnable_separator = embed_config.with_learnable_separator.unwrap_or(false);
        let hd_transform_order = embed_config
            .hd_transform_order
            .clone()
            .unwrap_or("glb_sub".to_string());
        assert_eq!(use_hd_transform, with_learnable_separator);
        let (glb_gn, sub_gn) = if with_learnable_separator {
            let glb_gn = vb.get((1, 1, image_dim_out * 4), "glb_GN")?;
            let sub_gn = vb.get((1, 1, 1, image_dim_out * 4), "sub_GN")?;
            (Some(glb_gn), Some(sub_gn))
        } else {
            (None, None)
        };

        // Inner projection
        let projection_cls = embed_config
            .projection_cls
            .clone()
            .unwrap_or("linear".to_string());
        let layers: Vec<Box<dyn ModuleWithMetadata>> =
            match (projection_cls.as_str(), use_hd_transform) {
                ("linear", _) => {
                    vec![Box::new(TryInto::<FusedBiasLinear>::try_into(linear_b(
                        image_dim_out,
                        hidden_size,
                        true,
                        vb.pp("img_projection"),
                    )?)?)]
                }
                ("mlp", true) => {
                    let dim_proj = hidden_size;
                    vec![
                        Box::new(TryInto::<FusedBiasLinear>::try_into(linear_b(
                            image_dim_out * 4,
                            dim_proj,
                            true,
                            vb.pp("img_projection.0"),
                        )?)?),
                        Box::new(candle_nn::Activation::Gelu),
                        Box::new(TryInto::<FusedBiasLinear>::try_into(linear_b(
                            dim_proj,
                            dim_proj,
                            true,
                            vb.pp("img_projection.2"),
                        )?)?),
                    ]
                }
                ("mlp", false) => {
                    let dim_proj = hidden_size;
                    vec![
                        Box::new(TryInto::<FusedBiasLinear>::try_into(linear_b(
                            image_dim_out,
                            dim_proj,
                            true,
                            vb.pp("img_projection.0"),
                        )?)?),
                        Box::new(candle_nn::Activation::Gelu),
                        Box::new(TryInto::<FusedBiasLinear>::try_into(linear_b(
                            dim_proj,
                            dim_proj,
                            true,
                            vb.pp("img_projection.2"),
                        )?)?),
                    ]
                }
                _ => {
                    candle_core::bail!("projection_cls=`{projection_cls}` not implemented.");
                }
            };

        let layer_idx = config.img_processor.layer_idx.unwrap_or(-2);
        let type_feature = config
            .img_processor
            .type_feature
            .clone()
            .unwrap_or("patch".to_string());

        Ok(Self {
            wte,
            image_dim_out,
            num_img_tokens,
            glb_gn,
            sub_gn,
            layer_idx,
            type_feature,
            image_processor,
            layers: EmbeddingLayers(layers),
            hd_transform_order,
            use_hd_transform,
            vocab_size: config.vocab_size,
        })
    }

    fn get_image_features(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let hidden_states = self
            .image_processor
            .forward_get_hidden_states(&pixel_values.to_dtype(self.wte.embeddings().dtype())?)?;
        let img_feature =
            hidden_states[(hidden_states.len() as isize + self.layer_idx) as usize].clone();
        if self.type_feature == "patch" {
            img_feature.i((.., 1..))
        } else if self.type_feature == "cls_patch" {
            Ok(img_feature)
        } else {
            candle_core::bail!("Unsupported image feature type {}", self.type_feature)
        }
    }

    #[allow(non_snake_case)]
    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: &Tensor,
        image_sizes: Option<Vec<(usize, usize)>>,
    ) -> Result<Tensor> {
        let input_ids = input_ids.reshape(((), input_ids.dim(D::Minus1)?))?;

        let input_ids_lt = input_ids.lt(0.0f64)?;
        let input_ids_gt = input_ids.gt(-MAX_INPUT_ID)?;
        // positions = torch.nonzero((input_ids < 0) & (input_ids > -MAX_INPUT_ID), as_tuple=False)
        let positions = Nonzero.nonzero_and::<u8>(&input_ids_lt, &input_ids_gt)?;

        let target_dev = self.layers.0[0].device();
        let target_dtype = self.layers.0[0].dtype();

        let mut select = false;
        // If some, use hd transform case and it contains num_img_toks
        let mut hd_transform = None;
        let mut image_set_tensor = None;
        if positions.dim(0)? > 0 {
            select = true;
            // input_ids[positions[:, 0], positions[:, 1]]
            if self.use_hd_transform && image_sizes.is_some() {
                assert_eq!(pixel_values.dims().len(), 5);
                let bs = pixel_values.dim(0)?;
                let img_features = self.get_image_features(&pixel_values.flatten(0, 1)?)?;
                let base_feat_dim = (img_features.dims()[1] as f32).sqrt() as usize;
                assert_eq!(base_feat_dim, 24);

                // bs x max_num_crops x (24x24) x C
                let img_features =
                    img_features.reshape((bs, (), base_feat_dim.pow(2), self.image_dim_out))?;
                let C = self.image_dim_out;
                let H = base_feat_dim;

                let mut output_imgs = Vec::new();
                let mut output_len = Vec::new();
                for bs_ in 0..bs {
                    let (h, w) = image_sizes.as_ref().unwrap()[bs_];
                    let h = h / 336;
                    let w = w / 336;
                    let B_ = h * w;

                    // 1 x (24x24) x 1024
                    let global_img_feature = img_features.i((bs_, ..1))?;

                    // 1 x 12 x 12 x 4096
                    let glb_img = global_img_feature
                        .reshape((1, H, H, C))?
                        .reshape((1, H / 2, 2, H / 2, 2, C))?
                        .contiguous()?
                        .permute((0, 1, 3, 2, 4, 5))?
                        .reshape((1, H / 2, H / 2, 4 * C))?
                        .contiguous()?;
                    let temp_glbl_gn = self
                        .sub_gn
                        .as_ref()
                        .expect("Need `sub_gn` if `use_hd_transform`")
                        .repeat((1, H / 2, 1, 1))?;

                    // 1 x 156 x 4096
                    let glb_img =
                        Tensor::cat(&[glb_img, temp_glbl_gn], 2)?.reshape((1, (), 4 * C))?;

                    // (max_num_crops-1) x (12x12) x C
                    let sub_img = img_features.i((bs_, 1..))?;

                    // 16x574x1024
                    // Get rid of padding sub_img
                    let sub_img = sub_img.i(..B_)?;

                    // (num_crops, 12, 2, 12, 2, 1024) -> (num_crops, 12, 12, 2, 2, 1024) -> (num_crops, 12*12, 4*1024)
                    let sub_img = sub_img
                        .reshape((B_, H, H, C))?
                        .reshape((B_, H / 2, 2, H / 2, 2, C))?
                        .contiguous()?
                        .permute((0, 1, 3, 2, 4, 5))?
                        .reshape((B_, (), 4 * C))?
                        .contiguous()?;
                    let sub_img = sub_img
                        .reshape(BigShapeWithOneHole((1usize, h, w, 12usize, 12usize, ())))?
                        .permute((0, 1, 3, 2, 4, 5))?
                        .reshape((1, h * 12, w * 12, 4 * C))?;
                    let temp_sub_gn = self
                        .sub_gn
                        .as_ref()
                        .expect("Need `sub_gn` if `use_hd_transform`")
                        .repeat((1, h * 12, 1, 1))?;

                    let sub_img =
                        Tensor::cat(&[sub_img, temp_sub_gn], 2)?.reshape((1, (), 4 * C))?;

                    // (1, num_img_tokens, 1024*4)

                    match self.hd_transform_order.as_str() {
                        "glb_sub" => {
                            output_imgs.push(Tensor::cat(
                                &[
                                    glb_img,
                                    self.glb_gn
                                        .as_ref()
                                        .expect("Need `glb_gn` if `use_hd_transform`")
                                        .clone(),
                                    sub_img,
                                ],
                                1,
                            )?);
                        }
                        "sub_glb" => {
                            output_imgs.push(Tensor::cat(
                                &[
                                    sub_img,
                                    self.glb_gn
                                        .as_ref()
                                        .expect("Need `glb_gn` if `use_hd_transform`")
                                        .clone(),
                                    glb_img,
                                ],
                                1,
                            )?);
                        }
                        other => {
                            candle_core::bail!("Invalid hd_transform_order=`{other}`");
                        }
                    }

                    let temp_len = (h * w + 1) * 144 + 1 + (h + 1) * 12;
                    assert_eq!(temp_len, output_imgs.last().unwrap().dims()[1]);
                    output_len.push(temp_len);
                }

                hd_transform = Some(output_len);
                let mut image_set_tensor_inner = Vec::new();
                for img in output_imgs {
                    let layerout = self
                        .layers
                        .forward(&img.to_device(target_dev)?.to_dtype(target_dtype)?)?;
                    image_set_tensor_inner.push(layerout);
                }
                image_set_tensor = Some(Either::Left(image_set_tensor_inner));
            } else if pixel_values.dims().len() == 4 {
                let tt = self
                    .get_image_features(pixel_values)?
                    .to_device(target_dev)?
                    .to_dtype(target_dtype)?
                    .reshape(((), self.image_dim_out))?;
                let image_set_tensor_inner = self.layers.forward(&tt)?;
                image_set_tensor = Some(Either::Right(image_set_tensor_inner));
            } else if pixel_values.dims().len() == 3 {
                let tt = pixel_values
                    .to_device(target_dev)?
                    .to_dtype(target_dtype)?
                    .reshape(((), self.image_dim_out))?;
                let image_set_tensor_inner = self.layers.forward(&tt)?;
                image_set_tensor = Some(Either::Right(image_set_tensor_inner));
            } else {
                unreachable!()
            }
        }

        let input_ids = input_ids.clamp(0.0, self.vocab_size as f64)?;
        let mut hidden_states = self.wte.forward(&input_ids)?;
        if select {
            match (hd_transform, image_set_tensor) {
                (Some(output_lens), Some(Either::Left(image_set_tensors))) => {
                    let mut idx = 0;
                    for (i, cnt) in output_lens.into_iter().enumerate() {
                        let img_set_tensor = image_set_tensors[i]
                            .to_device(target_dev)?
                            .to_dtype(target_dtype)?;
                        // hidden_states[positions[idx, 0], positions[idx, 1] : positions[idx, 1] + cnt] = ...
                        let p_0 = positions.i((idx, 0))?.to_scalar::<u32>()? as usize;
                        let p_1 = positions.i((idx, 1))?.to_scalar::<u32>()? as usize;
                        hidden_states = hidden_states.slice_assign(
                            &[&p_0, &(p_1..p_1 + cnt), &(..img_set_tensor.dims()[2])],
                            &img_set_tensor,
                        )?;
                        idx += cnt;
                    }
                }
                (None, Some(Either::Right(image_set_tensor))) => {
                    let mut idx = 0;
                    // Know len(img_embeds) == pixel_values.dim(0) == len(selected_g_values)
                    // https://huggingface.co/microsoft/Phi-3-vision-128k-instruct/blob/dbcdaaacf52c8e40cf8de6d6ffa6ff6860e5f256/image_embedding_phi3_v.py#L259
                    for i in 0..pixel_values.dim(0)? {
                        let cnt = self.num_img_tokens;
                        let img_set_tensor = image_set_tensor
                            .i(i * cnt..(i + 1) * cnt)?
                            .to_device(target_dev)?
                            .to_dtype(target_dtype)?;
                        let p_0 = positions.i((idx, 0))?.to_scalar::<u32>()? as usize;
                        let p_1 = positions.i((idx, 1))?.to_scalar::<u32>()? as usize;
                        // hidden_states[positions[idx, 0], positions[idx, 1] : positions[idx, 1] + cnt] = ...
                        hidden_states = hidden_states.slice_assign(
                            &[&p_0, &(p_1..p_1 + cnt), &(..img_set_tensor.dims()[2])],
                            &img_set_tensor,
                        )?;
                        idx += cnt;
                    }
                }
                _ => unreachable!(),
            }
        }

        Ok(hidden_states)
    }
}

// =================== ============= ===================

#[derive(Debug)]
pub struct Model {
    vision_embed_tokens: ImageEmbedding,
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: QMatMul,
    pub device: Device,
    pub cache: Cache,
    pub max_seq_len: usize,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    sliding_window: Option<usize>,
}

impl Model {
    pub fn new(
        cfg: &Config,
        vb: VarBuilder,
        _is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let mapper = normal_loading_metadata
            .mapper
            .into_mapper(cfg.num_hidden_layers, &normal_loading_metadata.real_device)?;
        let embed_tokens = candle_nn::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), false),
        )?;
        let vision_embed_tokens = ImageEmbedding::new(
            cfg,
            embed_tokens.clone(),
            &cfg.embd_layer,
            mapper.set_nm_device(vb_m.pp("vision_embed_tokens"), false),
        )?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let rotary_emb = Arc::new(PhiRotaryEmbedding::new(
                vb.dtype(),
                cfg.clone(),
                mapper
                    .device_for(layer_idx, false)
                    .unwrap_or(&normal_loading_metadata.real_device),
            )?);
            let layer = DecoderLayer::new(
                rotary_emb.clone(),
                cfg,
                vb_l.pp(layer_idx),
                &*mapper,
                layer_idx,
                normal_loading_metadata.loading_isq,
            )?;
            layers.push(layer)
        }
        let norm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;
        let lm_head = linear_no_bias(
            cfg.hidden_size,
            cfg.vocab_size,
            mapper.set_nm_device(vb.pp("lm_head"), normal_loading_metadata.loading_isq),
        )?;
        Ok(Self {
            vision_embed_tokens,
            layers,
            norm,
            lm_head: QMatMul::Tensor(lm_head.weight().clone()),
            device: normal_loading_metadata.real_device,
            cache: Cache::new(cfg.num_hidden_layers, false),
            max_seq_len: cfg.max_position_embeddings,
            mapper,
            sliding_window: cfg.sliding_window,
            embed_tokens,
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        seqlen_offsets: &[usize],
        position_ids: &[usize],
        context_lens: Vec<(usize, usize)>,
        image_sizes: Option<Vec<(usize, usize)>>,
    ) -> Result<Tensor> {
        let mut xs = if let Some(ref pixel_values) = pixel_values {
            self.vision_embed_tokens
                .forward(input_ids, pixel_values, image_sizes)?
        } else {
            self.embed_tokens.forward(input_ids)?
        };
        let mut cache = self.cache.lock();
        let attention_mask = CausalMasker.make_causal_mask_with_sliding_window_as_attn_bias(
            input_ids,
            &cache,
            self.sliding_window,
            xs.dtype(),
            self.layers[0].self_attn.num_heads,
        )?;
        let past_key_values_length = CausalMasker.calculate_past_kv_len(&cache)?;
        let position_ids = position_ids
            .iter()
            .map(|p| *p + past_key_values_length)
            .collect::<Vec<_>>();

        for (i, layer) in self.layers.iter_mut().enumerate() {
            xs = self.mapper.map(xs, i)?;
            xs = layer.forward(
                &xs,
                attention_mask
                    .as_ref()
                    .map(|m| m.to_device(xs.device()).unwrap())
                    .as_ref(),
                seqlen_offsets,
                &position_ids,
                &mut cache[i],
            )?
        }
        let xs = xs.to_device(&self.device)?;
        let mut xs = xs.apply(&self.norm)?;
        if matches!(self.lm_head, QMatMul::QTensor(_)) {
            xs = xs.to_dtype(DType::F32)?;
        }
        extract_logits(&MatMul.qmatmul(&xs, &self.lm_head)?, context_lens)
    }
}

impl IsqModel for Model {
    fn get_tensors(&mut self) -> (Vec<(&mut QMatMul, Option<usize>)>, &dyn DeviceMapper) {
        // TODO(EricLBuehler): more?
        let mut tensors = Vec::new();
        tensors.push((&mut self.lm_head, None));
        for (i, layer) in self.layers.iter_mut().enumerate() {
            tensors.push((&mut layer.self_attn.qkv_proj, Some(i)));
            tensors.push((&mut layer.self_attn.o_proj, Some(i)));
            tensors.push((&mut layer.mlp.gate_up_proj, Some(i)));
            tensors.push((&mut layer.mlp.down_proj, Some(i)));
        }
        (tensors, &*self.mapper)
    }
}

pub(crate) struct Phi3VisionSpecificArgs {
    pub image_sizes: Option<Vec<(usize, usize)>>,
}

impl VisionModel for Model {
    fn forward(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        seqlen_offsets: &[usize],
        _start_offsets_kernel: Tensor,
        context_lens: Vec<(usize, usize)>,
        position_ids: Vec<usize>,
        model_specific_args: Box<dyn Any>,
    ) -> Result<Tensor> {
        let Phi3VisionSpecificArgs { image_sizes } = *model_specific_args
            .downcast()
            .expect("Cannot downcast into `Phi3VisionSpecificArgs`");
        self.forward(
            input_ids,
            pixel_values,
            seqlen_offsets,
            &position_ids,
            context_lens,
            image_sizes,
        )
    }
    fn cache(&self) -> &Cache {
        &self.cache
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    fn has_conv2d(&self) -> bool {
        true
    }
}
