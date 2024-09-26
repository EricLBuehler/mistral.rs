use std::{any::Any, collections::HashMap, num::NonZeroUsize, sync::Arc};

use candle_core::{Context, DType, Device, Result, Tensor, D};
use image::{imageops::FilterType, DynamicImage, GenericImageView};
use mistralrs_vision::{
    ApplyTensorTransforms, ApplyTransforms, Normalize, Rescale, TensorTransforms, ToTensor,
    Transforms,
};
use tokenizers::Tokenizer;

use crate::{
    pipeline::{
        text_models_inputs_processor::PagedAttentionMeta, InputProcessorOutput, InputsProcessor,
        InputsProcessorType,
    },
    sequence::Sequence,
    vision_models::{
        image_processor::{ImagePreProcessor, PreprocessedImages},
        preprocessor_config::{PreProcessorConfig, ToFilter},
    },
};

struct MLlamaImageProcessor;

impl InputsProcessor for MLlamaImageProcessor {
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
        paged_attn_metadata: Option<PagedAttentionMeta<'_>>,
        prompt_batchsize: Option<NonZeroUsize>,
    ) -> Box<dyn Iterator<Item = anyhow::Result<InputProcessorOutput>>> {
        todo!()
    }
}

fn argmin<T, I>(iter: I) -> Option<usize>
where
    T: PartialOrd,
    I: Iterator<Item = T>,
{
    iter.enumerate()
        .fold(None, |min, (idx, item)| match min {
            None => Some((idx, item)),
            Some((min_idx, min_item)) => {
                if item < min_item {
                    Some((idx, item))
                } else {
                    Some((min_idx, min_item))
                }
            }
        })
        .map(|(min_idx, _)| min_idx)
}

impl MLlamaImageProcessor {
    // https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/mllama/image_processing_mllama.py#L53
    fn get_all_supported_aspect_ratios(max_image_tiles: usize) -> Vec<(usize, usize)> {
        (1..max_image_tiles + 1)
            .flat_map(|width| {
                (1..max_image_tiles + 1).filter_map(move |height| {
                    if width * height <= max_image_tiles {
                        Some((width, height))
                    } else {
                        None
                    }
                })
            })
            .collect::<Vec<_>>()
    }

    // https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/mllama/image_processing_mllama.py#L132
    fn get_optimal_tiled_canvas(
        image_height: u32,
        image_width: u32,
        max_image_tiles: usize,
        tile_size: usize,
    ) -> Result<(usize, usize)> {
        let possible_tile_arrangements = Self::get_all_supported_aspect_ratios(max_image_tiles);
        let possible_canvas_sizes: (Vec<_>, Vec<_>) = possible_tile_arrangements
            .into_iter()
            .map(|(h, w)| (h * tile_size, w * tile_size))
            .unzip();
        // Get all possible resolution heights/widths
        let (target_heights, target_widths) = possible_canvas_sizes;

        // Get scaling factors to resize the image without distortion
        let scale_h = target_heights
            .iter()
            .map(|h| *h as f32 / image_height as f32)
            .collect::<Vec<_>>();
        let scale_w = target_widths
            .iter()
            .map(|w| *w as f32 / image_width as f32)
            .collect::<Vec<_>>();

        // Get the min scale between width and height
        let scales = scale_h
            .into_iter()
            .zip(scale_w)
            .map(|(scale_h, scale_w)| if scale_w > scale_h { scale_h } else { scale_w })
            .collect::<Vec<_>>();

        // Filter only scales that allow upscaling
        let upscaling_options = scales
            .iter()
            .copied()
            .filter(|scale| *scale >= 1.)
            .collect::<Vec<_>>();
        let selected_scale = if !upscaling_options.is_empty() {
            upscaling_options
                .into_iter()
                .min_by(|x, y| x.partial_cmp(y).expect("No ordering!"))
                .context("No min, upscale")?
        } else {
            // No upscaling possible, get min downscaling (max scale for scales<1)
            let downscaling_options = scales
                .iter()
                .copied()
                .filter(|scale| *scale < 1.)
                .collect::<Vec<_>>();
            downscaling_options
                .into_iter()
                .max_by(|x, y| x.partial_cmp(y).expect("No ordering!"))
                .context("No max, downscale")?
        };

        // Get all resolutions that support this scaling factor
        let chosen_canvas_h = target_heights
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(i, h)| {
                if scales[i] == selected_scale {
                    Some(h)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let chosen_canvas_w = target_widths
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(i, w)| {
                if scales[i] == selected_scale {
                    Some(w)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        assert_eq!(chosen_canvas_h.len(), chosen_canvas_w.len());
        if chosen_canvas_h.len() > 1 {
            let optimal_idx = argmin(
                chosen_canvas_h
                    .iter()
                    .zip(&chosen_canvas_w)
                    .map(|(h, w)| *h * *w),
            )
            .context("No argmin")?;
            Ok((chosen_canvas_h[optimal_idx], chosen_canvas_w[optimal_idx]))
        } else {
            Ok((chosen_canvas_h[0], chosen_canvas_w[0]))
        }
    }

    // https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/mllama/image_processing_mllama.py#L82
    fn get_image_size_fit_to_canvas(
        image_height: u32,
        image_width: u32,
        canvas_height: usize,
        canvas_width: usize,
        tile_size: usize,
    ) -> (usize, usize) {
        let target_width = (image_width as usize).clamp(tile_size, canvas_width);
        let target_height = (image_height as usize).clamp(tile_size, canvas_height);

        let scale_h = (target_height as f32) / (image_height as f32);
        let scale_w = (target_width as f32) / (image_width as f32);

        if scale_w < scale_h {
            (
                target_height.min((image_height as f32 * scale_w).floor() as usize),
                target_width,
            )
        } else {
            (
                target_height,
                target_width.min((image_width as f32 * scale_h).floor() as usize),
            )
        }
    }

    // https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/mllama/image_processing_mllama.py#L796
    /// Resizes an image to fit within a tiled canvas while maintaining its aspect ratio.
    /// The optimal canvas size is calculated based on the maximum number of tiles and the tile size.
    fn resize(
        &self,
        image: DynamicImage,
        size: &HashMap<String, u32>,
        max_image_tiles: usize,
        filter: FilterType,
    ) -> Result<(DynamicImage, (usize, usize))> {
        let image_height = image.height();
        let image_width = image.width();
        let tile_size = size["height"] as usize;

        let (canvas_height, canvas_width) =
            Self::get_optimal_tiled_canvas(image_height, image_width, max_image_tiles, tile_size)?;
        let num_tiles_height = canvas_height / tile_size;
        let num_tiles_width = canvas_width / tile_size;

        let (new_height, new_width) = Self::get_image_size_fit_to_canvas(
            image_height,
            image_width,
            canvas_height,
            canvas_width,
            tile_size,
        );

        Ok((
            image.resize(new_width as u32, new_height as u32, filter),
            (num_tiles_height, num_tiles_width),
        ))
    }

    // https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/mllama/image_processing_mllama.py#L749
    /// Pad an image to the `size` x `aspect_ratio`. For example, if size is {height: 224, width: 224} and aspect ratio is
    /// (1, 2), the image will be padded to 224x448.
    fn pad(
        &self,
        image: &Tensor,
        size: &HashMap<String, u32>,
        aspect_ratio: (usize, usize),
    ) -> Result<Tensor> {
        let (num_tiles_h, num_tiles_w) = aspect_ratio;
        let padded_height = num_tiles_h * size["height"] as usize;
        let padded_width = num_tiles_w * size["width"] as usize;

        // Add padding on bottom and right sides
        mistralrs_vision::pad(image, padded_height, padded_width)
    }

    // https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/mllama/image_processing_mllama.py#L213
    /// Split an image into a specified number of tiles along its width and height dimensions.
    fn split_to_tiles(
        &self,
        image: &Tensor,
        num_tiles_height: usize,
        num_tiles_width: usize,
    ) -> Result<Tensor> {
        let (ch, h, w) = image.dims3()?;
        let tile_height = h / num_tiles_height;
        let tile_width = w / num_tiles_width;

        let mut image = image.reshape((
            ch,
            num_tiles_height,
            tile_height,
            num_tiles_width,
            tile_width,
        ))?;

        // Permute to (num_tiles_height, num_tiles_width, num_channels, tile_height, tile_width)
        image = image.permute((1, 3, 0, 2, 4))?;

        // Reshape into the desired output shape (num_tiles_width * num_tiles_height, num_channels, tile_height, tile_width)
        image
            .reshape((
                num_tiles_width * num_tiles_height,
                ch,
                tile_height,
                tile_width,
            ))?
            .contiguous()
    }

    // https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/mllama/image_processing_mllama.py#L277
    /// Returns
    /// - stacked and packed images
    /// - a list of lists containing the number of tiles for each image in each batch sample.
    /// Padding uses 0
    fn pack_images(
        &self,
        images: Vec<Tensor>,
        max_image_tiles: usize,
        (_bs, max_num_images): (usize, usize),
    ) -> Result<(Tensor, Vec<usize>)> {
        let (_, ch, tile_h, tile_w) = images[0].dims4()?;

        let mut stacked_images = Tensor::zeros(
            (max_num_images, max_image_tiles, ch, tile_h, tile_w),
            images[0].dtype(),
            images[0].device(),
        )?;
        let mut num_sample_tiles = Vec::new();
        for (i, image) in images.into_iter().enumerate() {
            let num_tiles = image.dim(0)?;
            stacked_images =
                stacked_images.slice_assign(&[&i, &(..num_tiles), &.., &.., &..], &image)?;
            num_sample_tiles.push(num_tiles)
        }
        Ok((stacked_images, num_sample_tiles))
    }

    // https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/mllama/image_processing_mllama.py#L354
    /// Convert aspect ratio tuples to unique ids.
    /// Padding uses 0
    fn convert_aspect_ratios_to_ids(
        &self,
        aspect_ratios: Vec<(usize, usize)>,
        max_image_tiles: usize,
        (_bs, max_num_images): (usize, usize),
        device: &Device,
    ) -> Result<Tensor> {
        let supported_aspect_ratios = Self::get_all_supported_aspect_ratios(max_image_tiles);

        let mut aspect_ratios_ids = vec![0i64; max_num_images];
        for (i, (num_tiles_h, num_tiles_w)) in aspect_ratios.iter().enumerate() {
            aspect_ratios_ids[i] = (supported_aspect_ratios
                .iter()
                .position(|(h, w)| *h == *num_tiles_h && *w == *num_tiles_w)
                .context("Could not find aspect ratio")?
                + 1) as i64;
        }

        Tensor::new(aspect_ratios_ids, device)
    }
}

impl ImagePreProcessor for MLlamaImageProcessor {
    const DEFAULT_MEAN: [f64; 3] = [0.5, 0.5, 0.5];
    const DEFAULT_STD: [f64; 3] = [0.5, 0.5, 0.5];

    fn preprocess(
        &self,
        images: Vec<DynamicImage>,
        config: &PreProcessorConfig,
        device: &Device,
        (bs, max_num_images): (usize, usize),
    ) -> Result<PreprocessedImages> {
        let mut sample_images = Vec::new();
        let mut sample_aspect_ratios = Vec::new();
        let max_image_tiles = config
            .max_image_tiles
            .context("`do_resize=false` is not supported, need `max_image_tiles`!")?;

        for mut image in images {
            // Convert to rgb, default to true
            if config.do_convert_rgb.unwrap_or(true) {
                image = DynamicImage::ImageRgb8(image.to_rgb8());
            }

            let size = config
                .size
                .as_ref()
                .context("`do_resize=false` is not supported, need `size`!")?;

            let (image, aspect_ratio) =
                self.resize(image, size, max_image_tiles, config.resampling.to_filter()?)?;

            // In transformers they rescale from [0, 255] to [0, 1]
            // at the end of resize:
            // https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/image_transforms.py#L340
            let to_tensor_rescale = Transforms {
                input: &ToTensor,
                inner_transforms: &[],
            };
            let mut image = image.apply(to_tensor_rescale, device)?;

            image = self.pad(&image, size, aspect_ratio)?;

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
            image = <Tensor as ApplyTensorTransforms>::apply(&image, transforms, device)?;

            let (num_tiles_height, num_tiles_width) = aspect_ratio;
            image = self.split_to_tiles(&image, num_tiles_height, num_tiles_width)?;

            sample_images.push(image);
            sample_aspect_ratios.push((num_tiles_height, num_tiles_width));
        }

        let (images, num_tiles) =
            self.pack_images(sample_images, max_image_tiles, (bs, max_num_images))?;

        let aspect_ratio_ids = self.convert_aspect_ratios_to_ids(
            sample_aspect_ratios,
            max_image_tiles,
            (bs, max_num_images),
            device,
        )?;

        todo!()
    }
}
