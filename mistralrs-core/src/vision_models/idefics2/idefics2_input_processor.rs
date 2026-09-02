#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{any::Any, ops::Range, sync::Arc};

use candle_core::{Device, Result, Tensor};
use image::{DynamicImage, GenericImageView};
use indexmap::IndexMap;
use mistralrs_vision::{ApplyTransforms, Normalize, Rescale, ToTensorNoNorm, Transforms};
use tokenizers::Tokenizer;

use crate::{
    device_map::DeviceMapper,
    paged_attention::block_hash::{MultiModalFeature, MultimodalAttentionPolicy, MultimodalKind},
    pipeline::{
        apply_chat_template,
        text_models_inputs_processor::{
            self, get_completion_input, get_prompt_input, PagedAttentionMeta,
        },
        InputProcessorOutput, InputsProcessor, InputsProcessorType, InputsProcessorValidationError,
        MessagesAction, Processor,
    },
    request::ReasoningEffort,
    sequence::{find_image_placeholder_ranges, Sequence},
    vision_models::{
        multimodal_layout::{
            MultimodalEmbeddingMap, MultimodalEncoderKey, MultimodalItemLayout,
            PackedMultimodalLayout, RequestMultimodalLayout,
        },
        ModelInputs,
    },
    MessageContent, Pipeline, Tool,
};

use crate::vision_models::{
    image_processor::{ImagePreProcessor, PreprocessedImages},
    preprocessor_config::{PreProcessorConfig, ToFilter},
    processor_config::ProcessorConfig,
};

// Input processor
pub struct Idefics2ImageProcessor {
    max_edge: Option<u32>,
    image_seq_len: usize,
}
// Processor
pub struct Idefics2Processor {
    config: ProcessorConfig,
    preprocessor_config: PreProcessorConfig,
    fake_image_token: &'static str,
    image_token: &'static str,
    max_edge: Option<u32>,
}

impl Idefics2Processor {
    pub fn new(
        config: ProcessorConfig,
        preprocessor_config: PreProcessorConfig,
        max_edge: Option<u32>,
    ) -> Self {
        Self {
            config,
            preprocessor_config,
            fake_image_token: "<fake_token_around_image>",
            image_token: "<image>",
            max_edge,
        }
    }
}

impl Processor for Idefics2Processor {
    fn process(
        &self,
        pipeline: &dyn Pipeline,
        messages: Vec<IndexMap<String, MessageContent>>,
        add_generation_prompt: bool,
        add_special_tokens: bool,
        enable_thinking: Option<bool>,
        reasoning_effort: Option<ReasoningEffort>,
        tools: Vec<Tool>,
    ) -> anyhow::Result<(Vec<u32>, String)> {
        let mut prompt = apply_chat_template(
            pipeline,
            messages,
            add_generation_prompt,
            enable_thinking,
            reasoning_effort,
            self.template_action(),
            tools,
        )?;

        let mut image_str = format!(
            "{}{}{}",
            self.fake_image_token,
            self.image_token.repeat(
                self.config
                    .image_seq_len
                    .expect("Idefics 2 model needs `image_seq_len`")
            ),
            self.fake_image_token
        );
        if self
            .preprocessor_config
            .do_image_splitting
            .is_some_and(|x| x)
        {
            // 4 patches + 1 original
            image_str = image_str.repeat(5);
        }

        prompt = prompt.replace(self.image_token, &image_str);
        // Deal with any adjacent images.
        prompt = prompt.replace(
            &format!("{}{}", self.fake_image_token, self.fake_image_token),
            self.fake_image_token,
        );

        let Some(tokenizer) = &pipeline.tokenizer() else {
            anyhow::bail!("Idefics2InputProcessor requires a specified tokenizer.",);
        };
        let encoding = tokenizer
            .encode_fast(prompt.clone(), add_special_tokens)
            .map_err(anyhow::Error::msg)?;
        Ok((encoding.get_ids().to_vec(), prompt))
    }

    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(Idefics2ImageProcessor {
            max_edge: self.max_edge,
            image_seq_len: self
                .config
                .image_seq_len
                .expect("Idefics 2 model needs `image_seq_len`"),
        })
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &["<fake_token_around_image>", "<image>", "<end_of_utterance>"]
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::Keep
    }
}

const IMAGE_TOKEN: &str = "<image>";
const SPLIT_IMAGE_COUNT: usize = 5;

fn subimages_per_image(config: &PreProcessorConfig) -> usize {
    if config.do_image_splitting.is_some_and(|enabled| enabled) {
        SPLIT_IMAGE_COUNT
    } else {
        1
    }
}

fn image_token_ranges(tokens: &[u32], image_token_id: u32) -> Vec<Range<usize>> {
    find_image_placeholder_ranges(tokens, image_token_id)
        .into_iter()
        .map(|(start, len)| start..start + len)
        .collect()
}

fn idefics2_mm_features(
    tokens: &[u32],
    image_token_id: u32,
    image_hashes: &[u64],
    subimages_per_image: usize,
    image_seq_len: usize,
) -> anyhow::Result<Vec<MultiModalFeature>> {
    if subimages_per_image == 0 {
        anyhow::bail!("Idefics2 image must have at least one encoder input");
    }
    let ranges = image_token_ranges(tokens, image_token_id);
    if ranges.iter().any(|range| range.len() != image_seq_len) {
        return Err(InputsProcessorValidationError(
            "Idefics2 image placeholder has an unexpected length".to_string(),
        )
        .into());
    }
    let expected_ranges = image_hashes
        .len()
        .checked_mul(subimages_per_image)
        .ok_or_else(|| candle_core::Error::msg("Idefics2 image count overflow"))?;
    if ranges.len() != expected_ranges {
        return Err(InputsProcessorValidationError(format!(
            "Idefics2 has {} image placeholder spans but {} encoder inputs",
            ranges.len(),
            expected_ranges
        ))
        .into());
    }

    image_hashes
        .iter()
        .copied()
        .enumerate()
        .map(|(item_index, hash)| {
            let start_index = item_index * subimages_per_image;
            let end_index = start_index + subimages_per_image;
            let item_ranges = &ranges[start_index..end_index];
            let offset = item_ranges[0].start;
            let end = item_ranges[item_ranges.len() - 1].end;
            Ok(MultiModalFeature {
                kind: MultimodalKind::Image,
                item_range: item_index..item_index + 1,
                hashes: vec![hash],
                offset,
                length: end - offset,
                attention_policy: MultimodalAttentionPolicy::Causal,
                splittable: false,
            })
        })
        .collect()
}

fn idefics2_layout_items(
    tokens: &[u32],
    features: &[MultiModalFeature],
    image_token_id: u32,
    subimages_per_image: usize,
    image_seq_len: usize,
) -> Result<Vec<MultimodalItemLayout>> {
    features
        .iter()
        .filter(|feature| feature.kind == MultimodalKind::Image)
        .map(|feature| {
            if feature.item_range.len() != 1 || feature.hashes.len() != 1 {
                candle_core::bail!("Idefics2 image feature must describe exactly one image");
            }
            let placeholder = feature.offset..feature.end();
            let item_tokens = tokens.get(placeholder.clone()).ok_or_else(|| {
                candle_core::Error::msg("Idefics2 image feature is outside the prompt")
            })?;
            let ranges = image_token_ranges(item_tokens, image_token_id)
                .into_iter()
                .map(|range| range.start + placeholder.start..range.end + placeholder.start)
                .collect::<Vec<_>>();
            if ranges.len() != subimages_per_image
                || ranges.iter().any(|range| range.len() != image_seq_len)
            {
                candle_core::bail!(
                    "Idefics2 image feature does not match its encoder output layout"
                );
            }
            let maps = ranges
                .into_iter()
                .enumerate()
                .map(|(source_output, range)| {
                    MultimodalEmbeddingMap::contiguous(range, 0, source_output)
                })
                .collect::<Result<Vec<_>>>()?;
            MultimodalItemLayout::new(
                MultimodalEncoderKey {
                    kind: MultimodalKind::Image,
                    hash: feature.hashes[0],
                },
                feature.item_range.start,
                placeholder,
                feature.attention_policy,
                maps,
            )
        })
        .collect()
}

fn prompt_query(seq: &Sequence, query_len: usize) -> Result<Range<usize>> {
    if let Some(query) = seq.active_prompt_query_range() {
        if query.len() != query_len {
            candle_core::bail!(
                "Idefics2 active prompt has {} tokens but packed metadata has {query_len}",
                query.len()
            );
        }
        return Ok(query);
    }
    let token_count = seq.prompt_position_source_toks().len();
    let start = token_count.checked_sub(query_len).ok_or_else(|| {
        candle_core::Error::msg("Idefics2 packed query is longer than the prompt")
    })?;
    Ok(start..token_count)
}

fn validate_idefics2_image_spans(
    query: &Range<usize>,
    items: &[MultimodalItemLayout],
) -> Result<()> {
    for item in items {
        let overlaps_query =
            item.placeholder.start < query.end && query.start < item.placeholder.end;
        if overlaps_query
            && (query.start > item.placeholder.start || query.end < item.placeholder.end)
        {
            candle_core::bail!(
                "Idefics2 image item {} must be scheduled as a complete span",
                item.item_index
            );
        }
    }
    Ok(())
}

fn idefics2_packed_layout(
    input_seqs: &[&mut Sequence],
    query_lens: &[usize],
    image_token_id: u32,
    subimages_per_image: usize,
    image_seq_len: usize,
) -> Result<PackedMultimodalLayout> {
    if input_seqs.len() != query_lens.len() {
        candle_core::bail!("Idefics2 packed multimodal metadata length mismatch");
    }
    let requests = input_seqs
        .iter()
        .zip(query_lens)
        .map(|(seq, &query_len)| {
            let tokens = seq.prompt_position_source_toks();
            let query = prompt_query(seq, query_len)?;
            let items = idefics2_layout_items(
                tokens,
                seq.mm_features(),
                image_token_id,
                subimages_per_image,
                image_seq_len,
            )?;
            validate_idefics2_image_spans(&query, &items)?;
            Ok(RequestMultimodalLayout {
                sequence_id: *seq.id(),
                query,
                items,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    PackedMultimodalLayout::new(&requests)
}

fn image_item_selection(
    is_chunked: bool,
    active_item_range: Option<Range<usize>>,
    active_local_range: Option<Range<usize>>,
    cached_items: usize,
    available_items: usize,
    total_items: usize,
) -> Result<Option<(Range<usize>, Range<usize>)>> {
    if available_items > total_items || cached_items > total_items {
        candle_core::bail!("Idefics2 image selection metadata is inconsistent");
    }
    let selection = if is_chunked {
        active_local_range.zip(active_item_range)
    } else {
        let retained_start = total_items - available_items;
        let original_start = cached_items.max(retained_start);
        (original_start < total_items).then_some((
            original_start - retained_start..available_items,
            original_start..total_items,
        ))
    };
    if let Some((local, original)) = &selection {
        if local.start > local.end
            || local.end > available_items
            || original.start > original.end
            || original.end > total_items
            || local.len() != original.len()
        {
            candle_core::bail!("Idefics2 active image range is outside the retained images");
        }
    }
    Ok(selection)
}

impl InputsProcessor for Idefics2ImageProcessor {
    fn get_type(&self) -> InputsProcessorType {
        InputsProcessorType::Vision
    }

    fn prepare_for_paged_prompt_planning(
        &self,
        tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut Sequence],
        _device: &Device,
        other_config: Option<Arc<dyn Any>>,
        _paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> anyhow::Result<()> {
        let tokenizer = tokenizer.ok_or_else(|| {
            anyhow::Error::msg("Idefics2InputProcessor requires a specified tokenizer.")
        })?;
        let config = other_config.expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");
        let image_token_id = tokenizer
            .token_to_id(IMAGE_TOKEN)
            .ok_or_else(|| anyhow::Error::msg("Idefics2 tokenizer is missing the image token"))?;
        let subimages_per_image = subimages_per_image(config);

        for seq in input_seqs.iter_mut() {
            if seq.mm_features().is_empty() {
                if let Some(hashes) = seq.image_hashes().map(<[u64]>::to_vec) {
                    seq.set_mm_features(idefics2_mm_features(
                        seq.get_toks(),
                        image_token_id,
                        &hashes,
                        subimages_per_image,
                        self.image_seq_len,
                    )?);
                }
            }
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
        let tokenizer = tokenizer.ok_or_else(|| {
            anyhow::Error::msg("Idefics2InputProcessor requires a specified tokenizer.")
        })?;
        let config = other_config.expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");
        let image_token_id = tokenizer
            .token_to_id(IMAGE_TOKEN)
            .ok_or_else(|| anyhow::Error::msg("Idefics2 tokenizer is missing the image token"))?;
        let subimages_per_image = subimages_per_image(config);

        if is_prompt {
            for seq in input_seqs.iter_mut() {
                if seq.mm_features().is_empty() {
                    if let Some(hashes) = seq.image_hashes().map(<[u64]>::to_vec) {
                        seq.set_mm_features(idefics2_mm_features(
                            seq.prompt_position_source_toks(),
                            image_token_id,
                            &hashes,
                            subimages_per_image,
                            self.image_seq_len,
                        )?);
                    }
                }
            }
        }

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
        } = if is_prompt {
            get_prompt_input(
                input_seqs
                    .iter()
                    .map(|seq| seq.get_toks())
                    .collect::<Vec<_>>(),
                input_seqs,
                device,
                last_n_context_len,
                return_raw_logits,
                paged_attn_metadata.as_mut(),
                mapper,
                sliding_window,
            )?
        } else {
            get_completion_input(
                input_seqs
                    .iter()
                    .map(|seq| seq.get_toks())
                    .collect::<Vec<_>>(),
                input_seqs,
                device,
                no_kv_cache,
                last_n_context_len,
                return_raw_logits,
                paged_attn_metadata.as_mut(),
                mapper,
                sliding_window,
            )?
        };
        let has_any_images = input_seqs.iter().any(|seq| seq.has_images());
        let has_all_images = input_seqs.iter().all(|seq| seq.has_images());
        if is_prompt && has_any_images && !has_all_images && !flash_meta.packed {
            anyhow::bail!("Idefics2 mixed image and text rows require packed prefill");
        }

        let mut image_hashes = Vec::new();
        let mut subimage_counts = Vec::new();
        let (pixel_values, pixel_attention_mask) = if is_prompt && has_any_images {
            let mut pixel_values_accum = Vec::new();
            let mut pixel_attention_mask_accum = Vec::new();
            for seq in input_seqs.iter_mut() {
                let available_items = seq.images().map_or(0, <[_]>::len);
                let total_items = seq
                    .mm_features()
                    .iter()
                    .filter(|feature| feature.kind == MultimodalKind::Image)
                    .map(|feature| feature.item_range.end)
                    .max()
                    .unwrap_or(available_items);
                let active_item_range = seq.active_multimodal_item_range(MultimodalKind::Image);
                let active_local_range =
                    seq.active_local_multimodal_item_range(MultimodalKind::Image, available_items);
                let image_selection = image_item_selection(
                    seq.is_chunked_prefill_view(),
                    active_item_range,
                    active_local_range,
                    seq.count_prefix_cached_mm_items_by_kind(MultimodalKind::Image),
                    available_items,
                    total_items,
                )?;
                let Some((image_range, original_range)) = image_selection else {
                    continue;
                };
                let selected_hashes = if seq.is_chunked_prefill_view() {
                    seq.image_hashes().unwrap_or_default().to_vec()
                } else {
                    seq.image_hashes()
                        .unwrap_or_default()
                        .get(original_range)
                        .ok_or_else(|| {
                            anyhow::Error::msg(
                                "Idefics2 active image hashes are outside the retained images",
                            )
                        })?
                        .to_vec()
                };
                if selected_hashes.len() != image_range.len() {
                    anyhow::bail!(
                        "Idefics2 selected {} images but has {} image hashes",
                        image_range.len(),
                        selected_hashes.len()
                    );
                }
                let PreprocessedImages {
                    pixel_values,
                    pixel_attention_mask,
                    image_sizes: _,
                    num_img_tokens: _,
                    aspect_ratio_ids: _,
                    aspect_ratio_mask: _,
                    num_tiles: _,
                    image_grid_thw: _,
                    video_grid_thw: _,
                    rows: _,
                    cols: _,
                    pixel_values_list: _,
                    tgt_sizes: _,
                    image_sizes_all: _,
                    num_crops: _,
                } = self.preprocess(
                    seq.multimodal
                        .clone_images_range(image_range)
                        .expect("Need to have images by this point."),
                    vec![],
                    config,
                    device,
                    (usize::MAX, usize::MAX), // Don't use it here...
                )?;
                let expected_subimages = selected_hashes
                    .len()
                    .checked_mul(subimages_per_image)
                    .ok_or_else(|| anyhow::Error::msg("Idefics2 image count overflow"))?;
                if pixel_values.dim(0)? != expected_subimages {
                    anyhow::bail!(
                        "Idefics2 preprocessing produced {} encoder inputs for {} images",
                        pixel_values.dim(0)?,
                        selected_hashes.len()
                    );
                }
                let pixel_attention_mask = pixel_attention_mask.ok_or_else(|| {
                    anyhow::Error::msg("Idefics2 preprocessing omitted its pixel mask")
                })?;
                if flash_meta.packed {
                    pixel_values_accum.push(pixel_values);
                    pixel_attention_mask_accum.push(pixel_attention_mask);
                } else {
                    pixel_values_accum.push(pixel_values.unsqueeze(0)?);
                    pixel_attention_mask_accum.push(pixel_attention_mask.unsqueeze(0)?);
                }
                subimage_counts.extend(std::iter::repeat_n(
                    subimages_per_image,
                    selected_hashes.len(),
                ));
                image_hashes.extend(selected_hashes);
            }
            if pixel_values_accum.is_empty() {
                (None, None)
            } else {
                (
                    Some(Tensor::cat(&pixel_values_accum, 0)?),
                    Some(Tensor::cat(&pixel_attention_mask_accum, 0)?),
                )
            }
        } else {
            (None, None)
        };

        let packed_layout = if is_prompt && flash_meta.packed {
            let query_lens = paged_attn_meta
                .as_ref()
                .and_then(|metadata| metadata.query_lens.as_deref())
                .ok_or_else(|| {
                    anyhow::Error::msg("packed Idefics2 prefill requires logical query lengths")
                })?;
            let layout = idefics2_packed_layout(
                input_seqs,
                query_lens,
                image_token_id,
                subimages_per_image,
                self.image_seq_len,
            )?;
            if layout.token_count() != input.dim(1)? {
                anyhow::bail!(
                    "Idefics2 packed layout has {} tokens but input has {}",
                    layout.token_count(),
                    input.dim(1)?
                );
            }
            Some(layout)
        } else {
            None
        };
        let packed_prefill = flash_meta.packed;

        let inputs: Box<dyn Any> = Box::new(ModelInputs {
            input_ids: input,
            seqlen_offsets: positions,
            context_lens,
            position_ids,
            pixel_values,
            model_specific_args: Box::new(super::Idefics2SpecificArgs {
                pixel_attention_mask,
                image_hashes,
                subimage_counts,
                packed_layout,
                packed_prefill,
            }),
            paged_attn_meta,
            flash_meta,
            recurrent_batch_kind: if is_prompt {
                crate::pipeline::RecurrentBatchKind::Prefill
            } else {
                crate::pipeline::RecurrentBatchKind::Decode
            },
            adapter_leases: crate::vision_models::adapter_leases(input_seqs, &seq_indices),
        });
        Ok(InputProcessorOutput {
            inputs,
            seq_indices,
        })
    }
}

impl ImagePreProcessor for Idefics2ImageProcessor {
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
        (_bs, _max_num_images): (usize, usize),
    ) -> Result<PreprocessedImages> {
        assert!(videos.is_empty());

        let mut patch_masks = Vec::new();
        let mut pixel_values = Vec::new();

        // Image splitting
        if config.do_image_splitting.is_some_and(|x| x) {
            let mut new_images = Vec::new();
            for image in images {
                let (w, h) = image.dimensions();
                let mid_w = w / 2;
                let mid_h = h / 2;
                new_images.push(image.crop_imm(0, 0, mid_w, mid_h));
                new_images.push(image.crop_imm(mid_w, 0, w, mid_h));
                new_images.push(image.crop_imm(0, mid_h, mid_w, h));
                new_images.push(image.crop_imm(mid_w, mid_h, w, h));
                new_images.push(image);
            }
            images = new_images;
        }

        for image in images.iter_mut() {
            // Resize
            if config.do_resize.is_some_and(|x| x) {
                let size = config.size.as_ref().unwrap();
                let (h, w) = if size.contains_key("shortest_edge")
                    && size.contains_key("longest_edge")
                {
                    mistralrs_vision::get_resize_image_size(
                        (image.dimensions().1 as usize, image.dimensions().0 as usize),
                        (
                            size["shortest_edge"] as usize,
                            size["longest_edge"] as usize,
                        ),
                    )
                } else if size.contains_key("height") && size.contains_key("width") {
                    (size["height"] as usize, size["width"] as usize)
                } else {
                    candle_core::bail!("Size must be a map of `shortest_edge` and `longest_edge` or `height` and `width`.");
                };

                *image = image.resize_exact(w as u32, h as u32, config.resampling.to_filter()?);
            }
        }

        if let Some(max_edge) = self.max_edge {
            images = mistralrs_vision::pad_to_max_edge(&images, max_edge);
        }

        let mut max_h = 0;
        let mut max_w = 0;
        for image in &images {
            let (w, h) = image.dimensions();
            if w > max_w {
                max_w = w;
            }
            if h > max_h {
                max_h = h;
            }
        }

        for image in images.iter_mut() {
            // Convert to rgb
            if config.do_convert_rgb.is_some_and(|x| x) {
                *image = DynamicImage::ImageRgb8(image.to_rgb8());
            }

            let transforms = Transforms {
                input: &ToTensorNoNorm,
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

            let mut image = image.apply(transforms, device)?;
            // Pad images, calculating attention mask.
            if config.do_pad.is_some_and(|x| x) {
                let (_c, h, w) = image.dims3()?;
                let padded = mistralrs_vision::pad(&image, max_h as usize, max_w as usize)?;
                let mask = mistralrs_vision::make_pixel_mask(&padded, h, w)?;
                patch_masks.push(mask.unsqueeze(0)?);
                image = padded;
            }

            // Get pixel values
            pixel_values.push(image.unsqueeze(0)?)
        }

        Ok(PreprocessedImages {
            pixel_values: Tensor::cat(&pixel_values, 0)?,
            pixel_attention_mask: Some(Tensor::cat(&patch_masks, 0)?),
            image_sizes: None,
            num_img_tokens: None,
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
    use std::collections::HashMap;

    use super::*;
    use crate::vision_models::multimodal_layout::MultimodalEncoderOutputs;

    #[test]
    fn split_images_group_placeholder_spans_by_original_item() {
        let features =
            idefics2_mm_features(&[7, 7, 1, 7, 7, 2, 7, 7, 3, 7, 7, 4, 7, 7], 7, &[11], 5, 2)
                .unwrap();

        assert_eq!(features.len(), 1);
        assert_eq!(features[0].item_range, 0..1);
        assert_eq!((features[0].offset, features[0].length), (0, 14));
    }

    #[test]
    fn packed_layout_handles_unequal_media_and_text_rows() {
        let tokens = [4, 7, 7, 5];
        let features = idefics2_mm_features(&tokens, 7, &[19], 1, 2).unwrap();
        let layout = PackedMultimodalLayout::new(&[
            RequestMultimodalLayout {
                sequence_id: 1,
                query: 0..4,
                items: idefics2_layout_items(&tokens, &features, 7, 1, 2).unwrap(),
            },
            RequestMultimodalLayout {
                sequence_id: 2,
                query: 0..2,
                items: vec![],
            },
        ])
        .unwrap();
        let text = Tensor::zeros((1, 6, 2), candle_core::DType::F32, &Device::Cpu).unwrap();
        let image = Tensor::from_vec(vec![1f32, 2., 3., 4.], (2, 2), &Device::Cpu).unwrap();
        let outputs: MultimodalEncoderOutputs = HashMap::from([(
            MultimodalEncoderKey {
                kind: MultimodalKind::Image,
                hash: 19,
            },
            vec![image],
        )]);

        let result = layout
            .splice_embeddings(&text, &outputs)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(result, vec![0., 0., 1., 2., 3., 4., 0., 0., 0., 0., 0., 0.]);
    }

    #[test]
    fn local_image_selection_handles_cached_prefix_and_chunk_views() {
        assert_eq!(
            image_item_selection(false, None, None, 1, 2, 3).unwrap(),
            Some((0..2, 1..3))
        );
        assert_eq!(
            image_item_selection(true, Some(2..3), Some(0..1), 0, 1, 3).unwrap(),
            Some((0..1, 2..3))
        );
        assert!(image_item_selection(true, Some(2..3), Some(1..2), 0, 1, 3).is_err());
    }

    #[test]
    fn malformed_placeholder_cardinality_is_rejected() {
        let cardinality = idefics2_mm_features(&[7, 7, 1, 7, 7], 7, &[11], 1, 2).unwrap_err();
        let length = idefics2_mm_features(&[7, 1], 7, &[11], 1, 2).unwrap_err();
        let internal = idefics2_mm_features(&[7, 7], 7, &[11], 0, 2).unwrap_err();

        assert!(cardinality.is::<InputsProcessorValidationError>());
        assert!(length.is::<InputsProcessorValidationError>());
        assert!(!internal.is::<InputsProcessorValidationError>());
    }

    #[test]
    fn image_spans_cannot_be_partially_packed() {
        let tokens = [4, 7, 7, 5];
        let features = idefics2_mm_features(&tokens, 7, &[19], 1, 2).unwrap();
        let items = idefics2_layout_items(&tokens, &features, 7, 1, 2).unwrap();

        assert!(validate_idefics2_image_spans(&(0..2), &items).is_err());
        assert!(validate_idefics2_image_spans(&(0..4), &items).is_ok());
        assert!(validate_idefics2_image_spans(&(3..4), &items).is_ok());
    }
}
