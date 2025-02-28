use std::{fmt::Debug, sync::Arc};

use candle_core::{shape::ShapeWithOneHole, DType, Device, IndexOp, Result, Shape, Tensor, D};
use candle_nn::Module;
use either::Either;
use mistralrs_quant::{QuantMethod, ShardedVarBuilder};

use crate::{
    layers::AvgPool2d, utils::unvarbuilder::UnVarBuilder, vision_models::{
        phi4::config::Phi4MMImgProcessorConfig,
        siglip::{SiglipVisionConfig, SiglipVisionTransformer},
    }
};

use super::Phi4MMConfig;

const MAX_INPUT_ID: f64 = 1e9;

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

/// A simplified ReflectionPad2d layer for padding only the right and bottom by 1.
struct SimpleReflectionPad2d;

impl Module for SimpleReflectionPad2d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        todo!()
    }
}

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
    vocab_size: usize,
    tensors: Vec<(String, Tensor)>,
    img_processor_padding: Option<SimpleReflectionPad2d>,
    crop_size: usize,
    image_token_compression: Option<AvgPool2d>,
    base_feat_height_reduction: Option<usize>,
    base_feat_height_target: usize,
}

impl ImageEmbedding {
    fn new(cfg: &Phi4MMConfig, wte: candle_nn::Embedding, vb: ShardedVarBuilder) -> Result<Self> {
        let img_embd_config = &cfg.embd_layer.image_embd_layer;

        let hidden_size = img_embd_config.n_embd.unwrap_or(cfg.hidden_size);

        let siglip_vision_config = SiglipVisionConfig {
            hidden_size: 1152,
            image_size: 448,
            intermediate_size: 4304,
            num_attention_heads: 16,
            num_hidden_layers: 27,
            patch_size: 14,
            ..Default::default()
        };
        let image_processor =
            SiglipVisionTransformer::new(&siglip_vision_config, vb.pp("img_processor"))?;

        let pe_weight = image_processor.embeddings.position_embedding.embeddings();
        let (l, d) = pe_weight.dims2()?;
        let mut m = (l as f64).sqrt() as usize;
        assert_eq!(m.pow(2), l);
        let img_processor_padding = if m % 2 != 0 {
            m += 1;
            Some(SimpleReflectionPad2d)
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
                    Some(1 as usize),
                    base_feat_height_target / 2,
                ),
                None => (None, None, 2),
                _ => candle_core::bail!("Unexpected image_token_compression_cls"),
            };

        assert_eq!(use_hd_transform, with_learnable_separator);
        let (glb_gn, sub_gn) = if with_learnable_separator {
            let glb_gn = vb.get((1, 1, image_dim_out * 4), "glb_GN")?;
            let sub_gn = vb.get((1, 1, 1, image_dim_out * 4), "sub_GN")?;
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
                        image_dim_out * 4,
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
            vocab_size: cfg.vocab_size,
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
        let (_, hidden_states) =
            self.image_processor
                .forward_get_hidden_states(&img_embeds, attention_mask, None)?;
        assert!(self.layer_idx < 0);
        let img_feature =
            hidden_states[(hidden_states.len() as isize + self.layer_idx) as usize].clone();

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
        let positions = input_ids_lt.bitwise_and(&input_ids_gt)?.nonzero()?;
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
                        .forward(&img.to_device(&target_dev)?.to_dtype(target_dtype)?)?;
                    image_set_tensor_inner.push(layerout);
                }
                image_set_tensor = Some(Either::Left(image_set_tensor_inner));
            } else if pixel_values.dims().len() == 4 {
                let tt = self
                    .get_image_features(pixel_values)?
                    .to_device(&target_dev)?
                    .to_dtype(target_dtype)?
                    .reshape(((), self.image_dim_out))?;
                let image_set_tensor_inner = self.layers.forward(&tt)?;
                image_set_tensor = Some(Either::Right(image_set_tensor_inner));
            } else if pixel_values.dims().len() == 3 {
                let tt = pixel_values
                    .to_device(&target_dev)?
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
                            .to_device(&target_dev)?
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
                    // https://huggingface.co/microsoft/Phi-3.5-vision-instruct/blob/dbcdaaacf52c8e40cf8de6d6ffa6ff6860e5f256/image_embedding_phi3_v.py#L259
                    for i in 0..pixel_values.dim(0)? {
                        let cnt = self.num_img_tokens;
                        let img_set_tensor = image_set_tensor
                            .i(i * cnt..(i + 1) * cnt)?
                            .to_device(&target_dev)?
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

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
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
