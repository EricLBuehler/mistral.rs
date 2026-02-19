use std::{
    fmt::Debug,
    sync::{Arc, LazyLock, Mutex},
};

use candle_core::{shape::ShapeWithOneHole, DType, Device, IndexOp, Result, Shape, Tensor, D};
use candle_nn::Module;
use mistralrs_quant::{NonZeroOp, QuantMethod, ShardedVarBuilder};

use crate::{
    layers::{AvgPool2d, ReflectionPad2d},
    paged_attention::encoder_cache::EncoderCacheManager,
    utils::unvarbuilder::UnVarBuilder,
    vision_models::{
        phi4::config::Phi4MMImgProcessorConfig,
        siglip::{SiglipVisionConfig, SiglipVisionTransformer},
    },
};

use super::{config::Phi4MMImageEmbedConfig, Phi4MMConfig};

pub(super) const IMAGE_SPECIAL_TOKEN_ID: f64 = 200010.;

trait ModuleWithMetadata: Module + Debug + Send + Sync {
    fn device(&self) -> Device;
    fn dtype(&self) -> DType;
}

#[derive(Debug)]
struct QuantMethodWrapper(Arc<dyn QuantMethod>);

impl Module for QuantMethodWrapper {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.0.forward(xs)
    }
}

impl ModuleWithMetadata for QuantMethodWrapper {
    fn device(&self) -> Device {
        self.0.unquant_weight_bias().unwrap().0.device().clone()
    }
    fn dtype(&self) -> DType {
        self.0.unquant_weight_bias().unwrap().0.dtype()
    }
}

impl ModuleWithMetadata for candle_nn::Activation {
    fn device(&self) -> Device {
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
    if !el_count.is_multiple_of(prod_d) {
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

pub(crate) static PHI4_MM_VISION_CFG: LazyLock<SiglipVisionConfig> =
    LazyLock::new(|| SiglipVisionConfig {
        hidden_size: 1152,
        image_size: 448,
        intermediate_size: 4304,
        num_attention_heads: 16,
        num_hidden_layers: 27,
        patch_size: 14,
        ..Default::default()
    });

pub struct ImageEmbedding {
    wte: candle_nn::Embedding,
    image_dim_out: usize,
    num_img_tokens: usize,
    glb_gn: Option<Tensor>,
    sub_gn: Option<Tensor>,
    layers: EmbeddingLayers,
    type_feature: String,
    layer_idx: isize,
    image_processor: SiglipVisionTransformer,
    hd_transform_order: String,
    use_hd_transform: bool,
    tensors: Vec<(String, Tensor)>,
    img_processor_padding: Option<ReflectionPad2d>,
    crop_size: usize,
    image_token_compression: Option<AvgPool2d>,
    base_feat_height_reduction: usize,
    base_feat_height_target: Option<usize>,
}

impl ImageEmbedding {
    pub fn new(
        cfg: &Phi4MMConfig,
        img_embd_config: &Phi4MMImageEmbedConfig,
        wte: candle_nn::Embedding,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let hidden_size = img_embd_config.n_embd.unwrap_or(cfg.hidden_size);

        let siglip_vision_config = &PHI4_MM_VISION_CFG;
        let image_processor =
            SiglipVisionTransformer::new(siglip_vision_config, vb.pp("img_processor"))?;

        let pe_weight = image_processor.embeddings.position_embedding.embeddings();
        let (l, d) = pe_weight.dims2()?;
        let mut m = (l as f64).sqrt() as usize;
        assert_eq!(m.pow(2), l);
        let img_processor_padding = if !m.is_multiple_of(2) {
            m += 1;
            Some(ReflectionPad2d::new((0, 1, 0, 1)))
        } else {
            None
        };
        let image_dim_out = d;
        let num_img_tokens = (m / 2).pow(2);
        let base_feat_height_target = m;

        // High dim transform
        let use_hd_transform = img_embd_config.use_hd_transform.unwrap_or(false);
        let with_learnable_separator = img_embd_config.with_learnable_separator.unwrap_or(false);
        let hd_transform_order = img_embd_config
            .hd_transform_order
            .clone()
            .unwrap_or("glb_sub".to_string());
        let crop_size = img_embd_config.crop_size.unwrap_or(336);

        let (image_token_compression, base_feat_height_reduction, base_feat_height_target) =
            match &img_embd_config.image_token_compression_cls {
                Some(x) if x == "avg_pool_2d" => (
                    Some(AvgPool2d::new(2, 2)),
                    1_usize,
                    Some(base_feat_height_target / 2),
                ),
                None => (None, 2_usize, None),
                _ => candle_core::bail!("Unexpected image_token_compression_cls"),
            };

        assert_eq!(use_hd_transform, with_learnable_separator);
        let (glb_gn, sub_gn) = if with_learnable_separator {
            let glb_gn = vb.get(
                (1, 1, image_dim_out * base_feat_height_reduction.pow(2)),
                "glb_GN",
            )?;
            let sub_gn = vb.get(
                (1, 1, 1, image_dim_out * base_feat_height_reduction.pow(2)),
                "sub_GN",
            )?;
            (Some(glb_gn), Some(sub_gn))
        } else {
            (None, None)
        };

        // Inner projection
        let projection_cls = img_embd_config
            .projection_cls
            .clone()
            .unwrap_or("linear".to_string());

        let mut tensors = Vec::new();
        let layers: Vec<Box<dyn ModuleWithMetadata>> =
            match (projection_cls.as_str(), use_hd_transform) {
                ("linear", _) => {
                    let a = mistralrs_quant::linear_b(
                        image_dim_out,
                        hidden_size,
                        true,
                        &None,
                        vb.pp("img_projection"),
                    )?;
                    let (a_w, a_b) = a.unquant_weight_bias().unwrap();
                    tensors.push(("img_projection.weight".to_string(), a_w));
                    if let Some(b) = a_b {
                        tensors.push(("img_projection.bias".to_string(), b));
                    }
                    vec![Box::new(QuantMethodWrapper(a))]
                }
                ("mlp", true) => {
                    let dim_proj = hidden_size;
                    let a = mistralrs_quant::linear_b(
                        image_dim_out * base_feat_height_reduction.pow(2),
                        dim_proj,
                        true,
                        &None,
                        vb.pp("img_projection.0"),
                    )?;
                    let (a_w, a_b) = a.unquant_weight_bias().unwrap();
                    tensors.push(("img_projection.0.weight".to_string(), a_w));
                    if let Some(b) = a_b {
                        tensors.push(("img_projection.0.bias".to_string(), b));
                    }
                    let b = mistralrs_quant::linear_b(
                        dim_proj,
                        dim_proj,
                        true,
                        &None,
                        vb.pp("img_projection.2"),
                    )?;
                    let (b_w, b_b) = b.unquant_weight_bias().unwrap();
                    tensors.push(("img_projection.2.weight".to_string(), b_w));
                    if let Some(b) = b_b {
                        tensors.push(("img_projection.2.bias".to_string(), b));
                    }
                    vec![
                        Box::new(QuantMethodWrapper(a)),
                        Box::new(candle_nn::Activation::Gelu),
                        Box::new(QuantMethodWrapper(b)),
                    ]
                }
                ("mlp", false) => {
                    let dim_proj = hidden_size;
                    let a = mistralrs_quant::linear_b(
                        image_dim_out,
                        dim_proj,
                        true,
                        &None,
                        vb.pp("img_projection.0"),
                    )?;
                    let (a_w, a_b) = a.unquant_weight_bias().unwrap();
                    tensors.push(("img_projection.0.weight".to_string(), a_w));
                    if let Some(b) = a_b {
                        tensors.push(("img_projection.0.bias".to_string(), b));
                    }
                    let b = mistralrs_quant::linear_b(
                        dim_proj,
                        dim_proj,
                        true,
                        &None,
                        vb.pp("img_projection.2"),
                    )?;
                    let (b_w, b_b) = b.unquant_weight_bias().unwrap();
                    tensors.push(("img_projection.2.weight".to_string(), b_w));
                    if let Some(b) = b_b {
                        tensors.push(("img_projection.2.bias".to_string(), b));
                    }
                    vec![
                        Box::new(QuantMethodWrapper(a)),
                        Box::new(candle_nn::Activation::Gelu),
                        Box::new(QuantMethodWrapper(b)),
                    ]
                }
                _ => {
                    candle_core::bail!("projection_cls=`{projection_cls}` not implemented.");
                }
            };

        let (layer_idx, type_feature) = match &cfg.img_processor {
            Some(Phi4MMImgProcessorConfig {
                layer_idx,
                type_feature,
            }) => (
                layer_idx.unwrap_or(-2),
                type_feature.clone().unwrap_or("patch".to_string()),
            ),

            None => (-2, "patch".to_string()),
        };

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
            tensors,
            img_processor_padding,
            crop_size,
            image_token_compression,
            base_feat_height_reduction,
            base_feat_height_target,
        })
    }

    fn get_image_features(
        &self,
        img_embeds: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        assert!(self.layer_idx < 0);
        let img_feature = self.image_processor.forward_get_hidden_states(
            &img_embeds.to_dtype(self.image_processor.dtype())?,
            attention_mask,
            None,
            self.layer_idx,
        )?;

        if self.type_feature == "patch" {
            let mut patch_feature = img_feature.clone();
            if let Some(image_token_compression) = &self.image_token_compression {
                // reshape to 2D tensor
                let width = (patch_feature.dim(1)? as f64).sqrt() as usize;
                patch_feature =
                    patch_feature.reshape(((), width, width, patch_feature.dim(D::Minus1)?))?;
                // Convert to NCHW
                patch_feature = patch_feature.permute((0, 3, 1, 2))?;
                if let Some(img_processor_padding) = &self.img_processor_padding {
                    patch_feature = patch_feature.apply(img_processor_padding)?;
                }
                patch_feature = image_token_compression.forward(&patch_feature)?;
                // Convert to NHWC
                patch_feature = patch_feature.permute((0, 2, 3, 1))?;
                patch_feature = patch_feature.reshape((
                    (),
                    patch_feature.dim(1)? * patch_feature.dim(2)?,
                    patch_feature.dim(D::Minus1)?,
                ))?;
            } else if let Some(img_processor_padding) = &self.img_processor_padding {
                // reshape to 2D tensor
                let width = (patch_feature.dim(1)? as f64).sqrt() as usize;
                patch_feature =
                    patch_feature.reshape(((), width, width, patch_feature.dim(D::Minus1)?))?;
                // Convert to NCHW
                patch_feature = patch_feature.permute((0, 3, 1, 2))?;
                patch_feature = patch_feature.apply(img_processor_padding)?;
                // Convert to NHWC
                patch_feature = patch_feature.permute((0, 2, 3, 1))?;
                patch_feature = patch_feature.reshape((
                    (),
                    patch_feature.dim(1)? * patch_feature.dim(2)?,
                    patch_feature.dim(D::Minus1)?,
                ))?;
            };
            Ok(patch_feature)
        } else if self.type_feature == "cls_patch" {
            let mut img_feature = img_feature.clone();
            if let Some(image_token_compression) = &self.image_token_compression {
                // reshape to 2D tensor
                let mut patch_feature = img_feature.i((.., 1..))?;
                let cls_feature = img_feature.i((.., 0))?;
                let width = (patch_feature.dim(1)? as f64).sqrt() as usize;
                patch_feature =
                    patch_feature.reshape(((), width, width, patch_feature.dim(D::Minus1)?))?;
                patch_feature = image_token_compression.forward(&patch_feature)?;
                patch_feature = patch_feature.reshape((
                    (),
                    patch_feature.dim(D::Minus2)? * patch_feature.dim(D::Minus1)?,
                ))?;
                img_feature = Tensor::cat(&[cls_feature, patch_feature], 1)?;
            }
            Ok(img_feature)
        } else {
            candle_core::bail!("Unsupported image feature type {}", self.type_feature)
        }
    }

    #[allow(non_snake_case)]
    pub fn forward(
        &self,
        input_ids: &Tensor,
        input_embeds: &Tensor,
        image_attention_mask: Option<&Tensor>,
        image_sizes: Option<Vec<(u32, u32)>>,
        image_hashes: &[u64],
        encoder_cache: &Mutex<EncoderCacheManager>,
    ) -> Result<Tensor> {
        let input_ids = input_ids.reshape(((), input_ids.dim(D::Minus1)?))?;

        let positions = input_ids.eq(IMAGE_SPECIAL_TOKEN_ID)?.nonzero()?;

        let target_dev = self.layers.0[0].device();
        let target_dtype = self.layers.0[0].dtype();

        let mut select = false;
        let mut image_set_tensor = None;
        if positions.dim(0)? > 0 {
            select = true;

            if self.use_hd_transform {
                if let Some(image_sizes_ref) = image_sizes.as_ref() {
                    assert_eq!(input_embeds.dims().len(), 5);
                    let bs = input_embeds.dim(0)?;

                    // Check encoder cache for all images
                    let all_cached = if !image_hashes.is_empty() && image_hashes.len() == bs {
                        let mut guard = encoder_cache.lock().expect("encoder cache lock poisoned");
                        let mut cached_results = Vec::with_capacity(bs);
                        let mut all_hit = true;
                        for &hash in image_hashes {
                            if let Some(cached) = guard.get(hash) {
                                cached_results.push(cached[0].clone());
                            } else {
                                all_hit = false;
                                break;
                            }
                        }
                        if all_hit {
                            Some(cached_results)
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    if let Some(cached_results) = all_cached {
                        image_set_tensor = Some(cached_results);
                    } else {
                        let img_features = match image_attention_mask {
                            Some(attn_mask) => self.get_image_features(
                                &input_embeds.flatten(0, 1)?,
                                Some(&attn_mask.ne(0.)?.flatten(0, 1)?),
                            )?,
                            None => self.get_image_features(input_embeds, None)?,
                        };

                        let base_feat_height_target = self.base_feat_height_target.unwrap();
                        let base_resolution = self.crop_size;
                        let base_feat_height_reduction = self.base_feat_height_reduction;

                        let base_feat_height = (img_features.dim(1)? as f64).sqrt() as usize;
                        let base_feat_width = base_feat_height;

                        assert_eq!(base_feat_height, base_feat_height_target);
                        assert_eq!(base_feat_width, base_feat_height_target);

                        let img_features = img_features.reshape((
                            bs,
                            (),
                            base_feat_height * base_feat_width,
                            self.image_dim_out,
                        ))?;
                        let C = self.image_dim_out;
                        let H = base_feat_height;

                        let mut output_imgs = Vec::new();
                        for (bs_, &(h, w)) in image_sizes_ref.iter().enumerate().take(bs) {
                            let h = h as usize / base_resolution;
                            let w = w as usize / base_resolution;
                            let B_ = h * w;

                            // 1 x (24x24) x 1024
                            let global_img_feature = img_features.i((bs_, ..1))?;

                            // 1 x 12 x 12 x 4096
                            let glb_img = global_img_feature
                                .reshape((1, H, H, C))?
                                .reshape((
                                    1,
                                    H / base_feat_height_reduction,
                                    base_feat_height_reduction,
                                    H / base_feat_height_reduction,
                                    base_feat_height_reduction,
                                    C,
                                ))?
                                .contiguous()?
                                .permute((0, 1, 3, 2, 4, 5))?
                                .reshape((
                                    1,
                                    H / base_feat_height_reduction,
                                    H / base_feat_height_reduction,
                                    base_feat_height_reduction * base_feat_height_reduction * C,
                                ))?
                                .contiguous()?;
                            let temp_glbl_gn = self
                                .sub_gn
                                .as_ref()
                                .expect("Need `sub_gn` if `use_hd_transform`")
                                .repeat((1, H / base_feat_height_reduction, 1, 1))?;

                            // 1 x 156 x 4096
                            let glb_img = Tensor::cat(&[glb_img, temp_glbl_gn], 2)?.reshape((
                                1,
                                (),
                                base_feat_height_reduction * base_feat_height_reduction * C,
                            ))?;

                            // (max_num_crops-1) x (12x12) x C
                            let mut sub_img = img_features.i((bs_, 1..))?;

                            // 16x574x1024
                            // Get rid of padding sub_img
                            sub_img = sub_img.i(..B_)?;

                            // (num_crops, 12, 2, 12, 2, 1024) -> (num_crops, 12, 12, 2, 2, 1024) -> (num_crops, 12*12, 4*1024)
                            sub_img = sub_img
                                .reshape((B_, H, H, C))?
                                .reshape((
                                    B_,
                                    H / base_feat_height_reduction,
                                    base_feat_height_reduction,
                                    H / base_feat_height_reduction,
                                    base_feat_height_reduction,
                                    C,
                                ))?
                                .contiguous()?
                                .permute((0, 1, 3, 2, 4, 5))?
                                .reshape((
                                    B_,
                                    (),
                                    base_feat_height_reduction * base_feat_height_reduction * C,
                                ))?
                                .contiguous()?;
                            sub_img = sub_img
                                .reshape(BigShapeWithOneHole((
                                    1usize,
                                    h,
                                    w,
                                    base_feat_height / base_feat_height_reduction,
                                    base_feat_width / base_feat_height_reduction,
                                    (),
                                )))?
                                .permute((0, 1, 3, 2, 4, 5))?
                                .reshape((
                                    1,
                                    h * base_feat_height / base_feat_height_reduction,
                                    w * base_feat_width / base_feat_height_reduction,
                                    base_feat_height_reduction * base_feat_height_reduction * C,
                                ))?;

                            let (temp_sub_GN, temp_len) = if let Some(image_attention_mask) =
                                image_attention_mask
                            {
                                let h_indices = Tensor::arange_step(
                                    0,
                                    image_attention_mask.dim(2)? as u32,
                                    2,
                                    &target_dev,
                                )?;
                                let w_indices = Tensor::arange_step(
                                    0,
                                    image_attention_mask.dim(3)? as u32,
                                    2,
                                    &target_dev,
                                )?;

                                let reshaped_image_attention_mask = {
                                    let mut selected = image_attention_mask.i((bs_, 1..B_ + 1))?;
                                    selected = selected.index_select(&h_indices, 1)?;
                                    selected = selected.index_select(&w_indices, 2)?;
                                    selected
                                        .reshape((
                                            1,
                                            h,
                                            w,
                                            base_feat_height / base_feat_height_reduction,
                                            base_feat_width / base_feat_height_reduction,
                                        ))?
                                        .permute((0, 1, 3, 2, 4))?
                                        .reshape((
                                            1,
                                            h * base_feat_height / base_feat_height_reduction,
                                            w * base_feat_width / base_feat_height_reduction,
                                        ))?
                                };

                                let useful_height = reshaped_image_attention_mask
                                    .i((0, .., 0))?
                                    .sum_all()?
                                    .to_scalar::<u32>()?;
                                let useful_width = reshaped_image_attention_mask
                                    .i((0, 0, ..))?
                                    .sum_all()?
                                    .to_scalar::<u32>()?;

                                sub_img = sub_img.i((
                                    ..,
                                    ..useful_height as usize,
                                    ..useful_width as usize,
                                ))?;

                                let temp_len = {
                                    let mut selected = image_attention_mask.i((bs_, ..B_ + 1))?;
                                    selected = selected.index_select(&h_indices, 1)?;
                                    selected = selected.index_select(&w_indices, 2)?;
                                    selected.sum_all()?.to_scalar::<u32>()?
                                };
                                let temp_len = temp_len as usize
                                    + useful_height as usize
                                    + 1
                                    + base_feat_height / base_feat_height_reduction;

                                (
                                    self.sub_gn
                                        .as_ref()
                                        .expect("Need `sub_gn` if `use_hd_transform`")
                                        .repeat((1, useful_height as usize, 1, 1))?,
                                    temp_len,
                                )
                            } else {
                                let temp_len = (h * w + 1) * self.num_img_tokens
                                    + 1
                                    + (h + 1) * base_feat_height / base_feat_height_reduction;

                                (
                                    self.sub_gn
                                        .as_ref()
                                        .expect("Need `sub_gn` if `use_hd_transform`")
                                        .repeat((
                                            1,
                                            h * base_feat_height / base_feat_height_reduction,
                                            1,
                                            1,
                                        ))?,
                                    temp_len,
                                )
                            };

                            let sub_img = Tensor::cat(&[sub_img, temp_sub_GN], 2)?.reshape((
                                1,
                                (),
                                base_feat_height_reduction * base_feat_height_reduction * C,
                            ))?;

                            // (1, num_img_tokens, 1024*4)

                            // glb + sub
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

                            // (h*w+1)*144 + 1 + (h+1)*12
                            assert_eq!(temp_len, output_imgs.last().unwrap().dims()[1]);
                        }

                        let mut image_set_tensor_inner = Vec::new();
                        for (idx, img) in output_imgs.into_iter().enumerate() {
                            let layerout = self
                                .layers
                                .forward(&img.to_device(&target_dev)?.to_dtype(target_dtype)?)?;
                            // Cache each image's projected features
                            if idx < image_hashes.len() {
                                let mut guard =
                                    encoder_cache.lock().expect("encoder cache lock poisoned");
                                guard.insert(image_hashes[idx], vec![layerout.clone()]);
                            }
                            image_set_tensor_inner.push(layerout);
                        }
                        image_set_tensor = Some(image_set_tensor_inner);
                    } // close else (cache miss)
                }
            } else {
                unreachable!()
            }
        }

        let mut hidden_states = self.wte.forward(&input_ids)?;
        if select && self.use_hd_transform {
            match image_set_tensor {
                Some(image_set_tensors) => {
                    let merged_img_set_tensor = Tensor::cat(&image_set_tensors, 1)?.squeeze(0)?;

                    let original_shape = hidden_states.shape().clone();
                    let (hs_b, hs_l, hs_d) = hidden_states.dims3()?;
                    let mut hidden_states_flat = hidden_states.reshape(((), hs_d))?;

                    // Get the equiv 0th and 1th rows of the positions_tuple
                    let positions_transposed = positions.to_dtype(DType::F32)?;
                    let positions_transposed_0 = positions_transposed.i((.., 0))?;
                    let positions_transposed_1 = positions_transposed.i((.., 1))?;

                    let mut linear_index = ((positions_transposed_0 * (hs_l * hs_b) as f64)?
                        + positions_transposed_1)?;
                    linear_index = linear_index.to_dtype(DType::U32)?;
                    linear_index = linear_index.unsqueeze(1)?.repeat((1, hs_d))?;

                    let current_vals = hidden_states_flat.gather(&linear_index, 0)?;
                    let delta = merged_img_set_tensor.broadcast_sub(&current_vals)?;

                    hidden_states_flat =
                        hidden_states_flat.scatter_add(&linear_index, &delta, 0)?;

                    hidden_states = hidden_states_flat.reshape(original_shape)?;
                }
                _ => unreachable!(),
            }
        }

        Ok(hidden_states)
    }

    pub fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        if let Some(glb_gn) = self.glb_gn.clone() {
            uvb.add_tensor("glb_GN", glb_gn);
        }
        if let Some(sub_gn) = self.sub_gn.clone() {
            uvb.add_tensor("sub_GN", sub_gn);
        }
        uvb.extend(self.tensors.clone());
        uvb.pp("img_processor.vision_model")
            .extend(self.image_processor.residual_tensors());

        uvb.to_safetensors()
    }
}
