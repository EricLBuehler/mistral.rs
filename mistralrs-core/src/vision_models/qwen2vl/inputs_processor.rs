use std::{
    any::Any,
    num::NonZeroUsize,
    sync::{Arc, RwLock},
};

use anyhow::Result;
use candle_core::{Context, Device, IndexOp, Tensor};
use image::{imageops::FilterType, DynamicImage, GenericImageView};
use mistralrs_vision::{
    ApplyTensorTransforms, ApplyTransforms, Normalize, Rescale, TensorTransforms, ToTensorNoNorm,
    Transforms,
};
use tokenizers::Tokenizer;
use tracing::warn;

use crate::{
    pipeline::{
        text_models_inputs_processor::{
            self, get_completion_input, get_prompt_input, PagedAttentionMeta,
        },
        InputProcessorOutput, InputsProcessor, InputsProcessorType, MessagesAction, Processor,
    },
    sequence::Sequence,
    vision_models::{
        image_processor::{ImagePreProcessor, PreprocessedImages},
        preprocessor_config::{PreProcessorConfig, ToFilter},
        ModelInputs,
    },
};

use super::Qwen2VLVisionSpecificArgs;

// Input processor
struct Qwen2VLImageProcessor {
    // To represent uninitialized, we do this. Should always be init by the time this is read.
    merge_size: RwLock<Option<usize>>,
}
// Processor
pub struct Qwen2VLProcessor;

impl Qwen2VLProcessor {
    pub fn new() -> Self {
        Self
    }
}

impl Processor for Qwen2VLProcessor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(Qwen2VLImageProcessor {
            merge_size: RwLock::new(None),
        })
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &["<|image_pad|>", "<|video_pad|>", "<|placeholder|>"]
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::Keep
    }
}

fn replace_first_occurance(text: &str, to_replace: &str, replacement: &str) -> String {
    if let Some(pos) = text.find(to_replace) {
        let mut result = text.to_string();
        result.replace_range(pos..pos + to_replace.len(), replacement);
        result
    } else {
        text.to_string()
    }
}

impl InputsProcessor for Qwen2VLImageProcessor {
    fn get_type(&self) -> InputsProcessorType {
        InputsProcessorType::Vision
    }
    fn process_inputs(
        &self,
        tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut Sequence],
        is_prompt: bool,
        is_xlora: bool,
        device: &Device,
        no_kv_cache: bool,
        last_n_context_len: Option<(usize, usize)>,
        other_config: Option<Arc<dyn Any>>,
        mut paged_attn_metadata: Option<PagedAttentionMeta<'_>>,
        prompt_batchsize: Option<NonZeroUsize>,
    ) -> Box<dyn Iterator<Item = Result<InputProcessorOutput>>> {
        if is_xlora {
            return Box::new(std::iter::once(Err(anyhow::Error::msg(
                "Cannot make inputs for X-LoRA vision model.",
            ))));
        }
        if no_kv_cache {
            return Box::new(std::iter::once(Err(anyhow::Error::msg(
                "Vision model must have kv cache.",
            ))));
        }
        // TODO(EricLBuehler): support this? Would require some handling of image tokens.
        if prompt_batchsize.is_some() {
            warn!("`prompt_batchsize` is set. MLlama does not support prompt batching.");
        }
        let Some(tokenizer) = tokenizer else {
            return Box::new(std::iter::once(Err(anyhow::Error::msg(
                "MLlamaInputProcessor requires a specified tokenizer.",
            ))));
        };

        let text_models_inputs_processor::InnerInputProcessorOutput {
            inputs:
                text_models_inputs_processor::InputMetadata {
                    input,
                    positions,
                    positions_kernel,
                    context_lens,
                    position_ids,
                    paged_attn_meta,
                    flash_meta,
                },
            seq_indices,
        } = if is_prompt {
            get_prompt_input(
                input_seqs
                    .iter()
                    .map(|seq| seq.get_toks().to_vec())
                    .collect::<Vec<_>>(),
                input_seqs,
                device,
                last_n_context_len,
                paged_attn_metadata.as_mut(),
                None, // TODO: evaluate if it is possible to batch this
            )
            .nth(0)
            .unwrap()
            .unwrap()
        } else {
            get_completion_input(
                input_seqs
                    .iter()
                    .map(|seq| seq.get_toks().to_vec())
                    .collect::<Vec<_>>(),
                input_seqs,
                device,
                no_kv_cache,
                last_n_context_len,
                paged_attn_metadata.as_mut(),
                None, // TODO: evaluate if it is possible to batch this
            )
            .nth(0)
            .unwrap()
            .unwrap()
        };
        let config = other_config.expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");

        let has_images = input_seqs
            .iter()
            .all(|seq| seq.images().is_some_and(|images| !images.is_empty()));

        let (pixel_values, image_grid_thw, video_grid_thw) = if has_images {
            let mut pixel_values_accum = Vec::new();
            let mut image_grid_thw_accum = Vec::new();
            let mut video_grid_thw_accum = Vec::new();

            let mut detok_seqs = tokenizer
                .decode_batch(
                    &input_seqs
                        .iter()
                        .map(|seq| seq.get_toks())
                        .collect::<Vec<_>>(),
                    false,
                )
                .expect("Detokenization failed!");

            for seq in input_seqs.iter_mut() {
                let PreprocessedImages {
                    pixel_values,
                    pixel_attention_mask: _,
                    image_sizes: _,
                    num_img_tokens: _,
                    aspect_ratio_ids: _,
                    aspect_ratio_mask: _,
                    num_tiles: _,
                    image_grid_thw,
                    video_grid_thw,
                } = self
                    .preprocess(
                        seq.take_images()
                            .expect("Need to have images by this point."),
                        vec![],
                        config,
                        device,
                        (usize::MAX, usize::MAX), // Don't use it here...
                    )
                    .expect("Preprocessing failed");
                pixel_values_accum.push(pixel_values.unsqueeze(0).unwrap());
                image_grid_thw_accum.push(image_grid_thw.map(|img| img.unsqueeze(0).unwrap()));
                video_grid_thw_accum.push(video_grid_thw.map(|vid| vid.unsqueeze(0).unwrap()));
            }

            let image_grid_thw_accum = if image_grid_thw_accum.iter().any(|img| img.is_none()) {
                None
            } else {
                Some(
                    image_grid_thw_accum
                        .into_iter()
                        .map(|img| img.unwrap())
                        .collect::<Vec<_>>(),
                )
            };

            let video_grid_thw_accum = if video_grid_thw_accum.iter().any(|img| img.is_none()) {
                None
            } else {
                Some(
                    video_grid_thw_accum
                        .into_iter()
                        .map(|img| img.unwrap())
                        .collect::<Vec<_>>(),
                )
            };

            if let Some(ref image_grid_thw_accum) = image_grid_thw_accum {
                let merge_length = self.merge_size.read().unwrap().unwrap().pow(2);
                let mut index = 0;
                for (batch, text) in detok_seqs.iter_mut().enumerate() {
                    while text.contains("<|image_pad|>") {
                        *text = replace_first_occurance(
                            text,
                            "<|image_pad|>",
                            &"<|placeholder|>".repeat(
                                image_grid_thw_accum[batch]
                                    .squeeze(0)
                                    .unwrap()
                                    .i(index)
                                    .unwrap()
                                    .to_vec1::<u32>()
                                    .unwrap()
                                    .iter()
                                    .product::<u32>() as usize
                                    / merge_length,
                            ),
                        );
                        index += 1;
                    }
                    *text = text.replace("<|placeholder|>", "<|image_pad|>");
                }
            }

            if let Some(ref video_grid_thw_accum) = video_grid_thw_accum {
                let merge_length = self.merge_size.read().unwrap().unwrap().pow(2);
                let mut index = 0;
                for (batch, text) in detok_seqs.iter_mut().enumerate() {
                    while text.contains("<|video_pad|>") {
                        *text = replace_first_occurance(
                            text,
                            "<|video_pad|>",
                            &"<|placeholder|>".repeat(
                                video_grid_thw_accum[batch]
                                    .squeeze(0)
                                    .unwrap()
                                    .i(index)
                                    .unwrap()
                                    .to_vec1::<u32>()
                                    .unwrap()
                                    .iter()
                                    .product::<u32>() as usize
                                    / merge_length,
                            ),
                        );
                        index += 1;
                    }
                    *text = text.replace("<|placeholder|>", "<|video_pad|>");
                }
            }

            (
                Some(Tensor::cat(&pixel_values_accum, 0).unwrap()),
                image_grid_thw_accum.map(|img| Tensor::cat(&img, 0).unwrap()),
                video_grid_thw_accum.map(|vid| Tensor::cat(&vid, 0).unwrap()),
            )
        } else {
            (None, None, None)
        };

        let inputs: Box<dyn Any> = Box::new(ModelInputs {
            input_ids: input,
            seqlen_offsets: positions,
            seqlen_offsets_kernel: positions_kernel,
            context_lens,
            position_ids,
            pixel_values,
            model_specific_args: Box::new(Qwen2VLVisionSpecificArgs {
                image_grid_thw,
                video_grid_thw,
            }),
            paged_attn_meta,
            flash_meta,
        });
        Box::new(std::iter::once(Ok(InputProcessorOutput {
            inputs,
            seq_indices,
        })))
    }
}

impl Qwen2VLImageProcessor {
    fn smart_resize(
        &self,
        height: usize,
        width: usize,
        factor: usize,
        min_pixels: usize,
        max_pixels: usize,
    ) -> candle_core::Result<(usize, usize)> {
        if height < factor || width < factor {
            candle_core::bail!(
                "height:{} or width:{} must be larger than factor:{}",
                height,
                width,
                factor
            );
        } else if (height.max(width) as f64 / height.min(width) as f64) > 200.0 {
            candle_core::bail!(
                "absolute aspect ratio must be smaller than 200, got {:.2}",
                height.max(width) as f64 / height.min(width) as f64
            );
        }

        let mut h_bar = (height as f64 / factor as f64).round() as usize * factor;
        let mut w_bar = (width as f64 / factor as f64).round() as usize * factor;

        if h_bar * w_bar > max_pixels {
            let beta = ((height * width) as f64 / max_pixels as f64).sqrt();
            h_bar = ((height as f64 / beta / factor as f64).floor() as usize) * factor;
            w_bar = ((width as f64 / beta / factor as f64).floor() as usize) * factor;
        } else if h_bar * w_bar < min_pixels {
            let beta = (min_pixels as f64 / (height * width) as f64).sqrt();
            h_bar = ((height as f64 * beta / factor as f64).ceil() as usize) * factor;
            w_bar = ((width as f64 * beta / factor as f64).ceil() as usize) * factor;
        }

        Ok((h_bar, w_bar))
    }

    // patches and t,h,w
    fn preprocess_inner(
        &self,
        images: Vec<DynamicImage>,
        config: &PreProcessorConfig,
        device: &Device,
    ) -> candle_core::Result<(Tensor, (u32, u32, u32))> {
        let mut processed_images = Vec::new();
        let (width, height) = images[0].dimensions();
        let mut resized_height_latest = height;
        let mut resized_width_latest = width;

        for mut image in images {
            if config.do_resize.is_some_and(|x| x) {
                let (resized_height, resized_width) = self.smart_resize(
                    height as usize,
                    width as usize,
                    config.patch_size.context("Require `patch_size`.")?
                        * config.merge_size.context("Require `merge_size`")?,
                    config.min_pixels.context("Require `min_pixels`")?,
                    config.max_pixels.context("Require `max_pixels`")?,
                )?;
                resized_height_latest = resized_height as u32;
                resized_width_latest = resized_width as u32;
                image = image.resize_exact(
                    resized_width as u32,
                    resized_height as u32,
                    config
                        .resampling
                        .map(|resample| Some(resample).to_filter())
                        .unwrap_or(Ok(FilterType::CatmullRom))?,
                );
            }

            // In transformers they rescale from [0, 255] to [0, 1]
            // at the end of resize:
            // https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/image_transforms.py#L340
            let to_tensor_rescale = Transforms {
                input: &ToTensorNoNorm,
                inner_transforms: &[],
            };
            let image = image.apply(to_tensor_rescale, device)?;

            let transforms = TensorTransforms {
                inner_transforms: &[
                    &config
                        .do_rescale
                        .is_some_and(|x| x)
                        .then_some(())
                        .map(|_| Rescale {
                            factor: config.rescale_factor,
                        }),
                    &config
                        .do_normalize
                        .is_some_and(|x| x)
                        .then_some(())
                        .map(|_| Normalize {
                            mean: config.image_mean.unwrap_or(Self::DEFAULT_MEAN).to_vec(),
                            std: config.image_std.unwrap_or(Self::DEFAULT_STD).to_vec(),
                        }),
                ],
            };
            let image = <Tensor as ApplyTensorTransforms>::apply(&image, transforms, device)?;

            processed_images.push(image);
        }

        let mut patches = Tensor::stack(&processed_images, 0)?;
        let temporal_patch_size = config
            .temporal_patch_size
            .context("Require `temporal_patch_size")?;
        let patch_size = config.patch_size.context("Require `patch_size")?;
        let merge_size = config.patch_size.context("Require `merge_size")?;
        // Image
        if patches.dim(0)? == 1 {
            patches = patches.repeat((temporal_patch_size, 1, 1, 1))?;
        }
        let channel = patches.dim(1)?;
        let grid_t = patches.dim(0)? / temporal_patch_size;
        let grid_h = resized_height_latest as usize / patch_size;
        let grid_w = resized_width_latest as usize / patch_size;
        patches = patches.reshape(&[
            grid_t,
            temporal_patch_size,
            channel,
            grid_h / merge_size,
            merge_size,
            patch_size,
            grid_w / merge_size,
            merge_size,
            patch_size,
        ])?;
        patches = patches.permute([0, 3, 6, 4, 7, 2, 1, 5, 8])?;
        let flattened_patches = patches.reshape((
            grid_t * grid_h * grid_w,
            channel * temporal_patch_size * patch_size * patch_size,
        ))?;

        Ok((
            flattened_patches,
            (grid_t as u32, grid_h as u32, grid_w as u32),
        ))
    }
}

impl ImagePreProcessor for Qwen2VLImageProcessor {
    const DEFAULT_MEAN: [f64; 3] = [0.48145466, 0.4578275, 0.40821073];
    const DEFAULT_STD: [f64; 3] = [0.26862954, 0.26130258, 0.27577711];

    fn preprocess(
        &self,
        images: Vec<DynamicImage>,
        videos: Vec<Vec<DynamicImage>>,
        config: &PreProcessorConfig,
        device: &Device,
        (_, _): (usize, usize),
    ) -> candle_core::Result<PreprocessedImages> {
        let mut pixel_values = Vec::new();
        let mut vision_grid_thw = Vec::new();

        if !images.is_empty() {
            for image in images {
                let (patches, (t, h, w)) = self.preprocess_inner(vec![image], config, device)?;
                pixel_values.push(patches);
                vision_grid_thw.push(Tensor::new(&[t, h, w], device)?);
            }
            let pixel_values = Tensor::stack(&pixel_values, 0)?;
            let vision_grid_thw = Tensor::stack(&vision_grid_thw, 0)?;
            return Ok(PreprocessedImages {
                pixel_values,
                pixel_attention_mask: None,
                image_sizes: None,
                num_img_tokens: None,
                aspect_ratio_ids: None,
                aspect_ratio_mask: None,
                num_tiles: None,
                image_grid_thw: Some(vision_grid_thw),
                video_grid_thw: None,
            });
        }

        if !videos.is_empty() {
            for images in videos {
                let (patches, (t, h, w)) = self.preprocess_inner(images, config, device)?;
                pixel_values.push(patches);
                vision_grid_thw.push(Tensor::new(&[t, h, w], device)?);
            }
            let pixel_values = Tensor::stack(&pixel_values, 0)?;
            let vision_grid_thw = Tensor::stack(&vision_grid_thw, 0)?;
            return Ok(PreprocessedImages {
                pixel_values,
                pixel_attention_mask: None,
                image_sizes: None,
                num_img_tokens: None,
                aspect_ratio_ids: None,
                aspect_ratio_mask: None,
                num_tiles: None,
                image_grid_thw: None,
                video_grid_thw: Some(vision_grid_thw),
            });
        }
        todo!()
    }
}
