#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{any::Any, num::NonZeroUsize, sync::Arc};

use candle_core::{Device, IndexOp, Result, Tensor};
use image::{imageops::FilterType, DynamicImage, GenericImageView};
use mistralrs_vision::{ApplyTransforms, Normalize, ToTensorNoNorm, Transforms};
use tokenizers::Tokenizer;
use tracing::warn;

use crate::{
    device_map::DeviceMapper,
    pipeline::{
        text_models_inputs_processor::{
            self, get_completion_input, get_prompt_input, PagedAttentionMeta,
        },
        InputProcessorOutput, InputsProcessor, InputsProcessorType, MessagesAction, Processor,
    },
    sequence::Sequence,
};

use crate::vision_models::{
    image_processor::{ImagePreProcessor, PreprocessedImages},
    preprocessor_config::PreProcessorConfig,
    processor_config::ProcessorConfig,
};
pub struct MiniCpmOImageProcessor;

pub struct MiniCpmOProcessor {
    config: ProcessorConfig,
}

impl MiniCpmOProcessor {
    pub fn new(
        config: ProcessorConfig,
        _preprocessor_config: PreProcessorConfig,
        _max_edge: Option<u32>,
    ) -> Self {
        Self { config }
    }
}

impl Processor for MiniCpmOProcessor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        // Default image_seq_len is 169.
        Arc::new(MiniCpmOImageProcessor)
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &["<fake_token_around_image>", "<image>", "<end_of_utterance>"]
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::Keep
    }
}

impl InputsProcessor for MiniCpmOImageProcessor {
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
        return_raw_logits: bool,
        other_config: Option<Arc<dyn Any>>,
        mut paged_attn_metadata: Option<PagedAttentionMeta<'_>>,
        prompt_batchsize: Option<NonZeroUsize>,
        mapper: Option<&dyn DeviceMapper>,
    ) -> Box<dyn Iterator<Item = anyhow::Result<InputProcessorOutput>>> {
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
            warn!("`prompt_batchsize` is set. MiniCpm-O does not support prompt batching.");
        }
        let Some(tokenizer) = tokenizer else {
            return Box::new(std::iter::once(Err(anyhow::Error::msg(
                "MiniCpmOImageProcessor requires a specified tokenizer.",
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
                return_raw_logits,
                paged_attn_metadata.as_mut(),
                None, // TODO: evaluate if it is possible to batch this
                mapper,
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
                return_raw_logits,
                paged_attn_metadata.as_mut(),
                None, // TODO: evaluate if it is possible to batch this
                mapper,
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

        todo!()
    }
}

impl MiniCpmOImageProcessor {
    fn get_sliced_grid(
        &self,
        (w, h): (usize, usize),
        max_slice_nums: usize,
        scale_resolution: usize,
        never_split: bool,
    ) -> Option<(usize, usize)> {
        let log_ratio = ((w / h) as f32).ln();
        let ratio = (w * h) as f32 / (scale_resolution * scale_resolution) as f32;
        let multiple = ratio.ceil().min(max_slice_nums as f32);
        if multiple <= 1. || never_split {
            return None;
        }

        let mut candidate_split_grid_nums = Vec::new();
        for i in [multiple - 1., multiple, multiple + 1.] {
            if i == 1. || i > max_slice_nums as f32 {
                continue;
            }
            candidate_split_grid_nums.push(i);
        }

        let mut candidate_grids = Vec::new();
        for split_grid_nums in candidate_split_grid_nums {
            let mut m = 1.;
            while m <= split_grid_nums {
                if split_grid_nums % m == 0. {
                    candidate_grids.push((m as usize, split_grid_nums as usize / m as usize));
                }
                m += 1.;
            }
        }

        let mut best_grid = (1, 1);
        let mut min_error = f32::INFINITY;
        for grid in candidate_grids {
            let error = (log_ratio - (grid.0 as f32 / grid.1 as f32).ln()).abs();
            if error < min_error {
                best_grid = grid;
                min_error = error;
            }
        }

        Some(best_grid)
    }

    fn ensure_divide(&self, length: usize, patch_size: usize) -> usize {
        ((length as f32 / patch_size as f32).round() * patch_size as f32).max(patch_size as f32)
            as usize
    }

    fn find_best_resize(
        &self,
        (mut w, mut h): (usize, usize),
        scale_resolution: usize,
        patch_size: usize,
        allow_upscale: bool,
    ) -> (usize, usize) {
        if w * h > scale_resolution * scale_resolution || allow_upscale {
            let r = w as f32 / h as f32;
            h = (scale_resolution as f32 / r.sqrt()) as usize;
            w = (scale_resolution as f32 * r) as usize;
        }
        let best_w = self.ensure_divide(w, patch_size);
        let best_h = self.ensure_divide(h, patch_size);
        (best_w, best_h)
    }

    fn get_refine_size(
        &self,
        (w, h): (usize, usize),
        (grid_x, grid_y): (usize, usize),
        scale_resolution: usize,
        patch_size: usize,
        allow_upscale: bool,
    ) -> (usize, usize) {
        let refine_w = self.ensure_divide(w, grid_x);
        let refine_h = self.ensure_divide(h, grid_y);

        let grid_w = refine_h / grid_x;
        let grid_h = refine_w / grid_y;

        let best_grid_size = self.find_best_resize(
            (grid_w, grid_h),
            scale_resolution,
            patch_size,
            allow_upscale,
        );

        (best_grid_size.0 * grid_x, best_grid_size.1 * grid_y)
    }

    fn split_to_patches(
        &self,
        image: &DynamicImage,
        grid: (usize, usize),
    ) -> Vec<Vec<DynamicImage>> {
        let mut patches = Vec::new();
        let (w, h) = image.dimensions();
        let (w, h) = (w as usize, h as usize);
        let grid_x = w / grid.0;
        let grid_y = h / grid.1;
        for i in (0..h).step_by(grid_y) {
            let mut images = Vec::new();
            for j in (0..w).step_by(grid_x) {
                images.push(image.crop_imm(j as u32, i as u32, grid_x as u32, grid_y as u32));
            }
            patches.push(images);
        }
        patches
    }

    fn get_sliced_images(
        &self,
        image: &DynamicImage,
        max_slice_nums: usize,
        scale_resolution: usize,
        patch_size: usize,
    ) -> Vec<DynamicImage> {
        let dims = image.dimensions();
        let (w, h) = (dims.0 as usize, dims.1 as usize);

        let best_grid = self.get_sliced_grid((w, h), max_slice_nums, scale_resolution, false);

        let (source_images, patches) = if let Some(best_grid) = best_grid {
            // Source image, down-sampling and ensure divided by patch_size
            let best_resize = self.find_best_resize((w, h), scale_resolution, patch_size, false);
            let source_image = image.resize_exact(
                best_resize.0 as u32,
                best_resize.1 as u32,
                FilterType::CatmullRom,
            );
            let refine_size =
                self.get_refine_size((w, h), best_grid, scale_resolution, patch_size, true);
            let refine_image = image.resize_exact(
                refine_size.0 as u32,
                refine_size.1 as u32,
                FilterType::CatmullRom,
            );
            let patches = self
                .split_to_patches(&refine_image, best_grid)
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();

            (source_image, patches)
        } else {
            // Don't need to slice, upsample
            let best_size = self.find_best_resize((w, h), scale_resolution, patch_size, true);
            let source_images = image.resize_exact(
                best_size.0 as u32,
                best_size.1 as u32,
                FilterType::CatmullRom,
            );

            (source_images, vec![])
        };

        [vec![source_images], patches].concat()
    }

    /// image: (3, h, w)
    /// output: (3, patch_size, h*w/patch_size)
    fn reshape_by_patch(&self, image: &Tensor, patch_size: usize) -> Result<Tensor> {
        // Equivalent of torch.nn.functional.unfold with kernel_size and stride both 2-tuples
        let (_c, h, w) = image.dims3()?;
        // Kernel size
        let (kh, kw) = (patch_size, patch_size);
        // Stride
        let (sh, sw) = (patch_size, patch_size);

        let out_h = (h - kh) / sh + 1;
        let out_w = (w - kw) / sw + 1;

        let mut patches = Vec::new();
        for i in 0..out_h {
            for j in 0..out_w {
                // [c, kh, kw]
                let patch = image.i((.., i * sh..i * sh + kh, j * sw..j * sw + kw))?;
                // [c*kh*kw]
                patches.push(patch.flatten_all()?);
            }
        }
        // [C*kH*kW, out_h * out_w]
        let mut patches = Tensor::stack(&patches, 1)?;

        patches = patches.reshape((image.dim(0)?, patch_size, patch_size, ()))?;
        patches
            .permute((0, 1, 3, 2))?
            .reshape((image.dim(0)?, patch_size, ()))
    }
}

impl ImagePreProcessor for MiniCpmOImageProcessor {
    #[allow(clippy::excessive_precision)]
    const DEFAULT_MEAN: [f64; 3] = [0.5, 0.5, 0.5];
    #[allow(clippy::excessive_precision)]
    const DEFAULT_STD: [f64; 3] = [0.5, 0.5, 0.5];

    fn preprocess(
        &self,
        images: Vec<DynamicImage>,
        _videos: Vec<Vec<DynamicImage>>,
        config: &PreProcessorConfig,
        device: &Device,
        (_bs, _max_num_images): (usize, usize),
    ) -> Result<PreprocessedImages> {
        let mut pixel_values = Vec::new();
        let mut tgt_sizes = Vec::new();
        let image_sizes = images
            .iter()
            .map(|img| img.dimensions())
            .collect::<Vec<_>>();
        for image in images {
            let max_slice_nums = config.max_slice_nums.unwrap_or(9);
            let scale_resolution = config.scale_resolution.unwrap_or(448);
            let patch_size = config.patch_size.unwrap_or(14);

            let image_patches =
                self.get_sliced_images(&image, max_slice_nums, scale_resolution, patch_size);

            for slice_image in image_patches {
                let (w, h) = slice_image.dimensions();
                let to_tensor_rescale = Transforms {
                    input: &ToTensorNoNorm,
                    inner_transforms: &[&Normalize {
                        mean: config.image_mean.unwrap_or(Self::DEFAULT_MEAN).to_vec(),
                        std: config.image_std.unwrap_or(Self::DEFAULT_STD).to_vec(),
                    }],
                };
                let mut image = slice_image.apply(to_tensor_rescale, device)?;
                image = self.reshape_by_patch(&image, patch_size)?;
                pixel_values.push(image);
                tgt_sizes.push(Tensor::from_vec(
                    vec![h / patch_size as u32, w / patch_size as u32],
                    (1, 2),
                    &Device::Cpu,
                )?);
            }
        }

        let tgt_sizes = Tensor::cat(&tgt_sizes, 0)?.to_device(device)?;
        Ok(PreprocessedImages {
            pixel_values: Tensor::new(0u32, &Device::Cpu)?,
            pixel_attention_mask: None,
            image_sizes: None,
            num_img_tokens: None,
            aspect_ratio_ids: None,
            aspect_ratio_mask: None,
            num_tiles: None,
            image_grid_thw: None,
            video_grid_thw: None,
            rows: None,
            cols: None,
            pixel_values_list: Some(pixel_values),
            tgt_sizes: Some(tgt_sizes),
            image_sizes_all: Some(image_sizes),
        })
    }
}
