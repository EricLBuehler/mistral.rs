#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use crate::paged_attention::block_hash::{MultiModalFeature, MultimodalKind};
use std::{any::Any, sync::Arc};

use candle_core::{Device, Result, Tensor};
use image::{imageops::FilterType, DynamicImage, GenericImage, GenericImageView, Rgba};
use itertools::Itertools;
use mistralrs_vision::{ApplyTransforms, Normalize, ToTensor, Transforms};
use regex_automata::meta::Regex;
use tokenizers::Tokenizer;

use crate::{
    device_map::DeviceMapper,
    pipeline::{
        text_models_inputs_processor::{
            self, get_completion_input, get_prompt_input, PagedAttentionMeta,
        },
        InputProcessorOutput, InputsProcessor, InputsProcessorType, MessagesAction, Processor,
        ProcessorCreator,
    },
    sequence::{build_mm_features_from_ranges, Sequence},
};

use crate::vision_models::{
    image_processor::{ImagePreProcessor, PreprocessedImages},
    multimodal_layout::{
        MultimodalEmbeddingMap, MultimodalEncoderKey, MultimodalItemLayout, PackedMultimodalLayout,
        RequestMultimodalLayout,
    },
    phi3::Phi3VisionSpecificArgs,
    preprocessor_config::PreProcessorConfig,
    processor_config::ProcessorConfig,
    ModelInputs,
};

// Input processor
pub struct Phi3InputsProcessor {
    image_tag_splitter: Regex,
}
// Processor
pub struct Phi3Processor {
    inputs_processor: Arc<Phi3InputsProcessor>,
}

type Phi3PromptTokens = (Vec<i64>, Vec<(usize, usize)>);

fn phi3_image_token_count(image: &DynamicImage, config: &PreProcessorConfig) -> usize {
    let image =
        Phi3InputsProcessor::hd_transform(image, config.num_crops.expect("Need `num_crops`"));
    let h = image.height() as usize / 336;
    let w = image.width() as usize / 336;
    (h * w + 1) * 144 + (h + 1) * 12 + 1
}

fn phi3_prompt_tokens(
    tokenizer: &Tokenizer,
    image_tag_splitter: &Regex,
    detokenized: &str,
    image_token_counts: &[usize],
) -> anyhow::Result<Phi3PromptTokens> {
    let image_ids = image_tag_splitter
        .find_iter(detokenized)
        .map(|image_tag| {
            detokenized[image_tag.range()]
                .split('|')
                .nth(1)
                .and_then(|tag| tag.split('_').nth(1))
                .ok_or_else(|| anyhow::Error::msg("Phi3 image tag is malformed"))?
                .parse::<usize>()
                .map_err(|_| anyhow::Error::msg("Phi3 image id is not an integer"))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    if image_ids.len() != image_token_counts.len() {
        anyhow::bail!(
            "Phi3 prompt has {} image tags but {} images",
            image_ids.len(),
            image_token_counts.len()
        );
    }
    if image_ids.iter().copied().ne(1..=image_token_counts.len()) {
        anyhow::bail!("Phi3 image ids must appear once in ascending order starting at 1");
    }

    let splits = image_tag_splitter
        .split(detokenized)
        .map(|span| &detokenized[span.range()])
        .collect::<Vec<_>>();
    if splits.len() != image_ids.len() + 1 {
        anyhow::bail!("Phi3 prompt image split count is inconsistent");
    }
    let prompt_chunks = tokenizer
        .encode_batch(splits, true)
        .map_err(|error| anyhow::Error::msg(error.to_string()))?
        .into_iter()
        .map(|encoding| {
            encoding
                .get_ids()
                .iter()
                .map(|token| i64::from(*token))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let image_pads = image_ids
        .iter()
        .zip(image_token_counts)
        .map(|(&image_id, &token_count)| vec![-(image_id as i64); token_count])
        .collect::<Vec<_>>();

    let mut ranges = Vec::with_capacity(image_pads.len());
    let mut offset = 0usize;
    for (chunk, pad) in prompt_chunks.iter().zip(&image_pads) {
        offset += chunk.len();
        ranges.push((offset, pad.len()));
        offset += pad.len();
    }
    let mut tokens = Vec::new();
    for part in prompt_chunks.into_iter().interleave(image_pads) {
        tokens.extend(part);
    }
    Ok((tokens, ranges))
}

fn phi3_layout_items(features: &[MultiModalFeature]) -> Result<Vec<MultimodalItemLayout>> {
    features
        .iter()
        .filter(|feature| feature.kind == MultimodalKind::Image)
        .map(|feature| {
            if feature.item_range.len() != 1 || feature.hashes.len() != 1 {
                candle_core::bail!("Phi3 image feature must describe exactly one image");
            }
            let placeholder = feature.offset..feature.end();
            MultimodalItemLayout::new(
                MultimodalEncoderKey {
                    kind: MultimodalKind::Image,
                    hash: feature.hashes[0],
                },
                feature.item_range.start,
                placeholder.clone(),
                feature.attention_policy,
                vec![MultimodalEmbeddingMap::contiguous(placeholder, 0, 0)?],
            )
        })
        .collect()
}

fn phi3_packed_layout(
    input_seqs: &[&mut Sequence],
    query_lens: &[usize],
) -> Result<PackedMultimodalLayout> {
    if input_seqs.len() != query_lens.len() {
        candle_core::bail!("Phi3 packed multimodal metadata length mismatch");
    }
    let requests = input_seqs
        .iter()
        .zip(query_lens)
        .map(|(seq, &query_len)| {
            if query_len != seq.get_toks().len() {
                candle_core::bail!(
                    "Phi3 packed multimodal prefill requires the complete uncached prompt"
                );
            }
            Ok(RequestMultimodalLayout {
                sequence_id: *seq.id(),
                query: 0..query_len,
                items: phi3_layout_items(seq.mm_features())?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    PackedMultimodalLayout::new(&requests)
}

impl ProcessorCreator for Phi3Processor {
    fn new_processor(
        _: Option<ProcessorConfig>,
        _: PreProcessorConfig,
    ) -> Arc<dyn Processor + Send + Sync> {
        Arc::new(Self {
            inputs_processor: Arc::new(Phi3InputsProcessor {
                image_tag_splitter: Regex::new(r"<\|image_\d+\|>")
                    .expect("Failed to compile split regex."),
            }),
        })
    }
}

impl Processor for Phi3Processor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        self.inputs_processor.clone()
    }
    fn get_special_tokens(&self) -> &[&'static str] {
        &[]
    }
    fn template_action(&self) -> MessagesAction {
        MessagesAction::FlattenOnlyText
    }
}

impl InputsProcessor for Phi3InputsProcessor {
    fn get_type(&self) -> InputsProcessorType {
        InputsProcessorType::Vision
    }

    fn prepare_for_paged_prompt_planning(
        &self,
        tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut Sequence],
        _device: &Device,
        other_config: Option<Arc<dyn Any>>,
        mut paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> anyhow::Result<()> {
        let tokenizer = tokenizer
            .ok_or_else(|| anyhow::Error::msg("Phi3InputProcessor requires a tokenizer"))?;
        let config = other_config.expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");

        for seq in input_seqs {
            if seq.multimodal.has_changed_prompt {
                continue;
            }
            let Some(images) = seq.images() else {
                continue;
            };
            if images.is_empty() {
                continue;
            }
            let image_token_counts = images
                .iter()
                .map(|image| phi3_image_token_count(image, config))
                .collect::<Vec<_>>();
            let detokenized = tokenizer
                .decode(seq.get_toks(), false)
                .map_err(|error| anyhow::Error::msg(error.to_string()))?;
            let (tokens, ranges) = phi3_prompt_tokens(
                &tokenizer,
                &self.image_tag_splitter,
                &detokenized,
                &image_token_counts,
            )?;
            let hashes = seq.image_hashes().unwrap_or_default().to_vec();
            if hashes.len() != ranges.len() {
                anyhow::bail!(
                    "Phi3 has {} image hashes but {} image placeholders",
                    hashes.len(),
                    ranges.len()
                );
            }
            if seq.mm_features().is_empty() {
                seq.set_mm_features(build_mm_features_from_ranges(
                    &ranges,
                    &hashes,
                    MultimodalKind::Image,
                ));
            }
            let new_ids = tokens
                .iter()
                .map(|token| *token as i32 as u32)
                .collect::<Vec<_>>();
            let prompt = tokenizer
                .decode(&new_ids, false)
                .map_err(|error| anyhow::Error::msg(error.to_string()))?;
            seq.set_initial_prompt(prompt);
            seq.set_toks_and_reallocate(new_ids, paged_attn_metadata.as_deref_mut());
            seq.multimodal.has_changed_prompt = true;
        }
        Ok(())
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
        sliding_window: Option<usize>,
        other_config: Option<Arc<dyn Any>>,
        mut paged_attn_metadata: Option<PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
    ) -> anyhow::Result<InputProcessorOutput> {
        if is_xlora {
            return Err(anyhow::Error::msg(
                "Cannot make inputs for X-LoRA vision model.",
            ));
        }
        if no_kv_cache {
            return Err(anyhow::Error::msg("Vision model must have kv cache."));
        }
        let Some(tokenizer) = tokenizer else {
            return Err(anyhow::Error::msg(
                "Phi3InputProcessor requires a specified tokenizer.",
            ));
        };

        let config = other_config
            .clone()
            .expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");

        let has_images = input_seqs.iter().any(|seq| seq.has_images());
        if !has_images {
            return text_models_inputs_processor::TextInputsProcessor
                .process_inputs(
                    Some(tokenizer),
                    input_seqs,
                    is_prompt,
                    is_xlora,
                    device,
                    no_kv_cache,
                    last_n_context_len,
                    return_raw_logits,
                    sliding_window,
                    other_config,
                    paged_attn_metadata,
                    mapper,
                )
                .map(|metadata| {
                    let InputProcessorOutput {
                        inputs,
                        seq_indices,
                    } = metadata;

                    let text_models_inputs_processor::ModelInputs {
                        input_ids,
                        input_ids_full: _,
                        seqlen_offsets,
                        seqlen_offsets_full: _,
                        context_lens,
                        position_ids,
                        paged_attn_meta,
                        flash_meta,
                        flash_meta_full: _,
                        recurrent_batch_kind,
                        adapter_leases,
                    } = *inputs
                        .downcast::<text_models_inputs_processor::ModelInputs>()
                        .expect("Downcast failed.");

                    let inputs: Box<dyn Any> = Box::new(ModelInputs {
                        input_ids,
                        seqlen_offsets,
                        context_lens,
                        position_ids,
                        pixel_values: None,
                        model_specific_args: Box::new(Phi3VisionSpecificArgs {
                            image_sizes: None,
                            image_hashes: vec![],
                            packed_layout: None,
                        }),
                        paged_attn_meta,
                        flash_meta,
                        recurrent_batch_kind,
                        adapter_leases,
                    });
                    InputProcessorOutput {
                        inputs,
                        seq_indices,
                    }
                });
        }

        let mut pixel_values_accum = Vec::new();
        let mut image_sizes = Vec::new();
        let mut image_token_counts = vec![Vec::new(); input_seqs.len()];
        for (seq_idx, seq) in input_seqs.iter_mut().enumerate() {
            if !seq.has_images() {
                continue;
            }
            let cached = seq.count_prefix_cached_mm_items();
            let images = seq
                .take_images()
                .expect("Need to have images by this point.");
            for image in images.into_iter().skip(cached) {
                let PreprocessedImages {
                    pixel_values,
                    image_sizes: Some(image_size),
                    num_img_tokens: Some(num_img_tokens),
                    ..
                } = self.preprocess(
                    vec![image],
                    vec![],
                    config,
                    device,
                    (usize::MAX, usize::MAX),
                )?
                else {
                    anyhow::bail!("Phi3 preprocessing omitted required image metadata");
                };
                if pixel_values.dim(0)? != 1 || num_img_tokens.len() != 1 {
                    anyhow::bail!("Phi3 preprocessing must return one output per image");
                }
                pixel_values_accum.push(pixel_values);
                image_sizes.push(image_size);
                image_token_counts[seq_idx].push(num_img_tokens[0]);
            }
        }
        let pixel_values = if pixel_values_accum.is_empty() {
            None
        } else {
            Some(Tensor::cat(&pixel_values_accum, 0)?)
        };
        let image_sizes = (!image_sizes.is_empty()).then_some(image_sizes);

        let mut toks = Vec::new();
        let detokenized = tokenizer
            .decode_batch(
                &input_seqs
                    .iter()
                    .map(|seq| seq.get_toks())
                    .collect::<Vec<_>>(),
                false,
            )
            .map_err(|error| anyhow::Error::msg(error.to_string()))?;

        for ((detokenized, seq), image_token_counts) in detokenized
            .into_iter()
            .zip(input_seqs.iter_mut())
            .zip(image_token_counts)
        {
            if seq.multimodal.has_changed_prompt || image_token_counts.is_empty() {
                toks.push(seq.get_toks().iter().map(|x| *x as i32 as i64).collect());
                continue;
            }
            let (input_ids, image_ranges) = phi3_prompt_tokens(
                &tokenizer,
                &self.image_tag_splitter,
                &detokenized,
                &image_token_counts,
            )?;
            let hashes = seq.image_hashes().unwrap_or_default().to_vec();
            if hashes.len() != image_ranges.len() {
                anyhow::bail!(
                    "Phi3 has {} image hashes but {} image placeholders",
                    hashes.len(),
                    image_ranges.len()
                );
            }
            let new_ids = input_ids
                .iter()
                .map(|token| *token as i32 as u32)
                .collect::<Vec<_>>();
            let new_prompt = tokenizer
                .decode(&new_ids, false)
                .map_err(|error| anyhow::Error::msg(error.to_string()))?;
            seq.set_initial_prompt(new_prompt);
            if seq.mm_features().is_empty() {
                seq.set_mm_features(build_mm_features_from_ranges(
                    &image_ranges,
                    &hashes,
                    MultimodalKind::Image,
                ));
            }
            seq.set_toks_and_reallocate(new_ids, paged_attn_metadata.as_mut());
            seq.multimodal.has_changed_prompt = true;
            toks.push(input_ids);
        }

        let metadata = if is_prompt {
            get_prompt_input(
                toks.iter().map(Vec::as_slice).collect(),
                input_seqs,
                device,
                last_n_context_len,
                return_raw_logits,
                paged_attn_metadata.as_mut(),
                mapper,
                sliding_window,
            )
        } else {
            get_completion_input(
                toks.iter().map(Vec::as_slice).collect(),
                input_seqs,
                device,
                no_kv_cache,
                last_n_context_len,
                return_raw_logits,
                paged_attn_metadata.as_mut(),
                mapper,
                sliding_window,
            )
        };

        let text_models_inputs_processor::InnerInputProcessorOutput {
            inputs:
                text_models_inputs_processor::InputMetadata {
                    input,
                    positions,
                    context_lens,
                    position_ids,
                    paged_attn_meta,
                    flash_meta,
                },
            seq_indices,
        } = metadata?;
        let image_hashes = input_seqs
            .iter()
            .flat_map(|seq| {
                let cached = seq.count_prefix_cached_mm_items();
                seq.image_hashes()
                    .unwrap_or_default()
                    .iter()
                    .copied()
                    .skip(cached)
            })
            .collect::<Vec<_>>();
        if image_hashes.len() != image_sizes.as_ref().map_or(0, Vec::len) {
            anyhow::bail!("Phi3 image metadata does not match the selected media");
        }
        let packed_layout = if is_prompt && flash_meta.packed {
            let query_lens = paged_attn_meta
                .as_ref()
                .and_then(|metadata| metadata.query_lens.as_deref())
                .ok_or_else(|| {
                    anyhow::Error::msg("packed Phi3 prefill requires logical query lengths")
                })?;
            let layout = phi3_packed_layout(input_seqs, query_lens)?;
            if layout.token_count() != input.dim(1)? {
                anyhow::bail!(
                    "Phi3 packed layout has {} tokens but input has {}",
                    layout.token_count(),
                    input.dim(1)?
                );
            }
            Some(layout)
        } else {
            None
        };
        Ok(InputProcessorOutput {
            inputs: Box::new(ModelInputs {
                input_ids: input,
                seqlen_offsets: positions,
                context_lens,
                position_ids,
                pixel_values,
                model_specific_args: Box::new(Phi3VisionSpecificArgs {
                    image_sizes,
                    image_hashes,
                    packed_layout,
                }),
                paged_attn_meta,
                flash_meta,
                recurrent_batch_kind: if is_prompt {
                    crate::pipeline::RecurrentBatchKind::Prefill
                } else {
                    crate::pipeline::RecurrentBatchKind::Decode
                },
                adapter_leases: crate::vision_models::adapter_leases(input_seqs, &seq_indices),
            }),
            seq_indices,
        })
    }
}

impl Phi3InputsProcessor {
    fn pad_image(
        image: &DynamicImage,
        top: u32,
        bottom: u32,
        left: u32,
        right: u32,
        pad_color: Rgba<u8>,
    ) -> DynamicImage {
        // Calculate the new dimensions
        let new_width = image.width() + left + right;
        let new_height = image.height() + top + bottom;

        // Create a new image with the new dimensions and fill it with the pad color
        let mut new_image = DynamicImage::new_rgb8(new_width, new_height);
        for x in 0..new_width {
            for y in 0..new_height {
                new_image.put_pixel(x, y, pad_color);
            }
        }

        // Paste the original image into the center of the new image
        new_image
            .copy_from(image, 0, 0)
            .expect("Failed to copy image");

        new_image
    }

    fn padding_336(img: &DynamicImage) -> DynamicImage {
        let (_width, height) = img.dimensions();
        let tar = ((height as f64 / 336.0).ceil() * 336.0) as u32;
        let top_padding = ((tar as f64 - height as f64 + 1.) / 2.) as u32;
        let bottom_padding = tar - height - top_padding;
        let left_padding = 0u32;
        let right_padding = 0u32;
        Self::pad_image(
            img,
            top_padding,
            bottom_padding,
            left_padding,
            right_padding,
            Rgba([255u8, 255, 255, 255]),
        )
    }

    fn hd_transform(img: &DynamicImage, hd_num: usize) -> DynamicImage {
        let (mut width, mut height) = img.dimensions();
        let mut transposed = false;

        let img = if width < height {
            let img = img.rotate90();
            transposed = true;
            width = img.width();
            height = img.height();
            img
        } else {
            // NOTE: Don't love the clone.
            img.clone()
        };

        let ratio = width as f64 / height as f64;
        let mut scale = 1.0;
        while (scale * (scale / ratio).ceil()) <= hd_num as f64 {
            scale += 1.0;
        }
        scale -= 1.0;

        let new_width = (scale * 336.0) as u32;
        let new_height = (new_width as f64 / ratio) as u32;

        let resized_img = img.resize_exact(new_width, new_height, FilterType::Nearest);
        let padded_img = Self::padding_336(&resized_img);

        if transposed {
            return padded_img.rotate270();
        }

        padded_img
    }
}

fn pad_to_max_num_crops_tensor(image: &Tensor, max_crops: usize) -> Result<Tensor> {
    let (b, _, h, w) = image.dims4()?;
    if b < max_crops {
        let pad = Tensor::zeros((max_crops - b, 3, h, w), image.dtype(), image.device())?;
        Tensor::cat(&[image, &pad], 0)
    } else {
        Ok(image.clone())
    }
}

impl ImagePreProcessor for Phi3InputsProcessor {
    #[allow(clippy::excessive_precision)]
    const DEFAULT_MEAN: [f64; 3] = [0.48145466, 0.4578275, 0.40821073];
    #[allow(clippy::excessive_precision)]
    const DEFAULT_STD: [f64; 3] = [0.26862954, 0.26130258, 0.27577711];

    fn preprocess(
        &self,
        mut images: Vec<DynamicImage>,
        videos: Vec<Vec<DynamicImage>>,
        config: &PreProcessorConfig,
        device: &Device,
        (_, _): (usize, usize),
    ) -> Result<PreprocessedImages> {
        // If no images, will not call this.
        assert!(!images.is_empty());
        assert!(videos.is_empty());

        let mut image_sizes = Vec::new();
        let mut padded_images = Vec::new();
        let mut num_img_tokens = Vec::new();
        // If >1 images, resize them all to the largest, potentially destroying aspect ratio
        let mut max_size = None;
        for image in images.iter() {
            if max_size.is_none() {
                max_size = Some((image.dimensions().0 as usize, image.dimensions().1 as usize))
            } else if max_size.is_some_and(|(x, _)| image.dimensions().0 as usize > x) {
                max_size = Some((image.dimensions().0 as usize, max_size.unwrap().1));
            } else if max_size.is_some_and(|(_, y)| image.dimensions().1 as usize > y) {
                max_size = Some((max_size.unwrap().0, image.dimensions().1 as usize));
            }
        }
        let (max_w, max_h) = max_size.unwrap();
        for image in images.iter_mut() {
            *image = image.resize_exact(max_w as u32, max_h as u32, FilterType::Nearest);
        }

        for image in images.iter_mut() {
            // Convert to rgb, default to true
            if config.do_convert_rgb.unwrap_or(true) {
                *image = DynamicImage::ImageRgb8(image.to_rgb8());
            }

            let hd_image = Self::hd_transform(image, config.num_crops.expect("Need `num_crops`"));

            // Both hd and global have a normalization
            // Transforms for the HD image
            let transforms_hd = Transforms {
                input: &ToTensor,
                inner_transforms: &[&Normalize {
                    mean: config.image_mean.unwrap_or(Self::DEFAULT_MEAN).to_vec(),
                    std: config.image_std.unwrap_or(Self::DEFAULT_STD).to_vec(),
                }],
            };

            // (3,h,w)
            let hd_image = hd_image.apply(transforms_hd, device)?;

            // Resize with bicubic interpolation
            // (3,336,336)
            let global_image = hd_image.unsqueeze(0)?.interpolate2d(336, 336)?;

            let (_, h, w) = hd_image.dims3()?;
            let num_image_tokens = ((h as f32 / 336. * w as f32 / 336. + 1.) * 144.
                + ((h as f32 / 336.) + 1.) * 12.
                + 1.) as usize;

            let hd_image_reshape = hd_image
                .reshape((
                    1,
                    3,
                    (h as f32 / 336.) as usize,
                    336,
                    (w as f32 / 336.) as usize,
                    336,
                ))?
                .permute((0, 2, 4, 1, 3, 5))?
                .reshape(((), 3, 336, 336))?;
            let hd_image_reshape = Tensor::cat(&[global_image, hd_image_reshape], 0)?;
            let image_transformed = pad_to_max_num_crops_tensor(
                &hd_image_reshape,
                config.num_crops.expect("Need `num_crops`") + 1,
            )?;
            image_sizes.push((h, w));
            padded_images.push(image_transformed);
            num_img_tokens.push(num_image_tokens);
        }
        if padded_images.len() > 1 {
            candle_core::bail!("Can only process one image per batch");
        }
        let image_sizes = image_sizes[0];

        Ok(PreprocessedImages {
            pixel_values: Tensor::stack(&padded_images, 0)?,
            image_sizes: Some((image_sizes.0, image_sizes.1)),
            pixel_attention_mask: None,
            num_img_tokens: Some(num_img_tokens),
            aspect_ratio_ids: None,
            aspect_ratio_mask: None,
            num_tiles: None,
            image_grid_thw: None,
            video_grid_thw: None,
            rows: None,
            cols: None,
            pixel_values_list: None,
            tgt_sizes: None,
            image_sizes_all: None,
            num_crops: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, ops::Range};

    use candle_core::{Device, Tensor};

    use super::*;
    use crate::paged_attention::block_hash::MultimodalAttentionPolicy;
    use crate::vision_models::multimodal_layout::{
        MultimodalEncoderOutputs, RequestMultimodalLayout,
    };

    fn feature(item_range: Range<usize>) -> MultiModalFeature {
        MultiModalFeature {
            kind: MultimodalKind::Image,
            item_range,
            hashes: vec![17],
            offset: 1,
            length: 2,
            attention_policy: MultimodalAttentionPolicy::Causal,
            splittable: false,
        }
    }

    #[test]
    fn packed_layout_splices_media_and_preserves_text_request() {
        let layout = PackedMultimodalLayout::new(&[
            RequestMultimodalLayout {
                sequence_id: 1,
                query: 0..4,
                items: phi3_layout_items(&[feature(0..1)]).unwrap(),
            },
            RequestMultimodalLayout {
                sequence_id: 2,
                query: 0..2,
                items: vec![],
            },
        ])
        .unwrap();
        let text = Tensor::from_vec(
            vec![0f32, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
            (1, 6, 2),
            &Device::Cpu,
        )
        .unwrap();
        let outputs: MultimodalEncoderOutputs = HashMap::from([(
            MultimodalEncoderKey {
                kind: MultimodalKind::Image,
                hash: 17,
            },
            vec![Tensor::from_vec(vec![20f32, 21., 30., 31.], (2, 2), &Device::Cpu).unwrap()],
        )]);

        assert_eq!(
            layout
                .splice_embeddings(&text, &outputs)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![0., 1., 20., 21., 30., 31., 6., 7., 8., 9., 10., 11.]
        );
    }

    #[test]
    fn packed_layout_rejects_grouped_image_feature() {
        assert!(phi3_layout_items(&[feature(0..2)]).is_err());
    }

    #[test]
    fn planning_token_count_matches_preprocessor() {
        let processor = Phi3InputsProcessor {
            image_tag_splitter: Regex::new(r"<\|image_\d+\|>").unwrap(),
        };
        let config = PreProcessorConfig {
            num_crops: Some(4),
            ..Default::default()
        };

        for image in [
            DynamicImage::new_rgb8(640, 320),
            DynamicImage::new_rgb8(320, 640),
        ] {
            let expected = phi3_image_token_count(&image, &config);
            let preprocessed = processor
                .preprocess(
                    vec![image],
                    vec![],
                    &config,
                    &Device::Cpu,
                    (usize::MAX, usize::MAX),
                )
                .unwrap();
            assert_eq!(preprocessed.num_img_tokens.unwrap(), vec![expected]);
        }
    }
}
