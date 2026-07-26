#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{any::Any, ops::Range, sync::Arc};

use candle_core::{Device, IndexOp, Result, Tensor};
use image::{imageops::FilterType, DynamicImage, GenericImageView};
use mistralrs_vision::{ApplyTransforms, Normalize, ToTensor, Transforms};
use tokenizers::Tokenizer;

use crate::{
    device_map::DeviceMapper,
    paged_attention::block_hash::{MultiModalFeature, MultimodalAttentionPolicy, MultimodalKind},
    pipeline::{
        text_models_inputs_processor::{
            self, get_completion_input, get_prompt_input, PagedAttentionMeta,
        },
        InputProcessorOutput, InputsProcessor, InputsProcessorType, MessagesAction, Processor,
    },
    sequence::Sequence,
    vision_models::{
        multimodal_layout::{
            MultimodalEmbeddingMap, MultimodalEncoderKey, MultimodalItemLayout,
            PackedMultimodalLayout, RequestMultimodalLayout,
        },
        ModelInputs,
    },
};

use crate::vision_models::{
    image_processor::{ImagePreProcessor, PreprocessedImages},
    preprocessor_config::PreProcessorConfig,
    processor_config::ProcessorConfig,
};

use super::{MiniCpmOLegacyMap, MiniCpmOSpecificArgs, MiniCpmOVisualInput};

const DEFAULT_MAX_SLICE_NUMS: usize = 9;
const DEFAULT_SCALE_RESOLUTION: usize = 448;
const DEFAULT_PATCH_SIZE: usize = 14;
const DEFAULT_IMAGE_FEATURE_SIZE: usize = 64;
const DEFAULT_IM_START_TOKEN: &str = "<image>";
const DEFAULT_IM_END_TOKEN: &str = "</image>";
const DEFAULT_IM_ID_START: &str = "<image_id>";
const DEFAULT_IM_ID_END: &str = "</image_id>";
const DEFAULT_SLICE_START_TOKEN: &str = "<slice>";
const DEFAULT_SLICE_END_TOKEN: &str = "</slice>";
const DEFAULT_UNK_TOKEN: &str = "<unk>";
const DEFAULT_USE_IMAGE_ID: bool = false;
const DEFAULT_SLICE_MODE: bool = true;
const RAW_IMAGE_TAG: &str = "(<image>./</image>)";

#[derive(Clone, Copy)]
struct MiniCpmOTokenIds {
    image_start: u32,
    image_end: u32,
    slice_start: u32,
    slice_end: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct MiniCpmOPromptItem {
    placeholder: Range<usize>,
    embedding_spans: Vec<Range<usize>>,
}

#[derive(Clone, Debug)]
struct SelectedPromptItem {
    source_index: usize,
    item: MiniCpmOPromptItem,
}

pub struct MiniCpmOImageProcessor {
    config: PreProcessorConfig,
}

pub struct MiniCpmOProcessor {
    preprocessor_config: PreProcessorConfig,
}

impl MiniCpmOProcessor {
    pub fn new(
        _config: ProcessorConfig,
        preprocessor_config: PreProcessorConfig,
        _max_edge: Option<u32>,
    ) -> Self {
        Self {
            preprocessor_config,
        }
    }
}

impl Processor for MiniCpmOProcessor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(MiniCpmOImageProcessor {
            config: self.preprocessor_config.clone(),
        })
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &[
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_SLICE_START_TOKEN,
            DEFAULT_SLICE_END_TOKEN,
            DEFAULT_UNK_TOKEN,
        ]
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::FlattenOnlyText
    }
}

fn parse_prompt_items(
    tokens: &[u32],
    token_ids: MiniCpmOTokenIds,
) -> anyhow::Result<Vec<MiniCpmOPromptItem>> {
    enum OpenSpan {
        Image(usize),
        Slice(usize),
    }

    let mut items = Vec::new();
    let mut current = None;
    let mut open = None;
    for (position, &token) in tokens.iter().enumerate() {
        match open {
            Some(OpenSpan::Image(start)) => {
                if token == token_ids.image_end {
                    if position == start + 1 {
                        anyhow::bail!("MiniCPMO image placeholder is empty");
                    }
                    current = Some(MiniCpmOPromptItem {
                        placeholder: start..position + 1,
                        embedding_spans: std::iter::once(start + 1..position).collect(),
                    });
                    open = None;
                } else if token == token_ids.image_start
                    || token == token_ids.slice_start
                    || token == token_ids.slice_end
                {
                    anyhow::bail!("MiniCPMO image placeholder delimiters are malformed");
                }
            }
            Some(OpenSpan::Slice(start)) => {
                if token == token_ids.slice_end {
                    if position == start + 1 {
                        anyhow::bail!("MiniCPMO slice placeholder is empty");
                    }
                    let item = current.as_mut().ok_or_else(|| {
                        anyhow::Error::msg("MiniCPMO slice appears before an image placeholder")
                    })?;
                    item.embedding_spans.push(start + 1..position);
                    item.placeholder.end = position + 1;
                    open = None;
                } else if token == token_ids.image_start
                    || token == token_ids.image_end
                    || token == token_ids.slice_start
                {
                    anyhow::bail!("MiniCPMO slice placeholder delimiters are malformed");
                }
            }
            None if token == token_ids.image_start => {
                if let Some(item) = current.take() {
                    items.push(item);
                }
                open = Some(OpenSpan::Image(position));
            }
            None if token == token_ids.slice_start => {
                if current.is_none() {
                    anyhow::bail!("MiniCPMO slice appears before an image placeholder");
                }
                open = Some(OpenSpan::Slice(position));
            }
            None if token == token_ids.image_end || token == token_ids.slice_end => {
                anyhow::bail!("MiniCPMO placeholder has an unmatched closing delimiter");
            }
            None => {}
        }
    }
    if open.is_some() {
        anyhow::bail!("MiniCPMO placeholder has an unmatched opening delimiter");
    }
    if let Some(item) = current {
        items.push(item);
    }
    Ok(items)
}

fn prompt_features(
    items: &[MiniCpmOPromptItem],
    hashes: &[u64],
) -> anyhow::Result<Vec<MultiModalFeature>> {
    if items.len() != hashes.len() {
        anyhow::bail!(
            "MiniCPMO has {} image placeholders but {} image inputs",
            items.len(),
            hashes.len()
        );
    }
    Ok(items
        .iter()
        .zip(hashes)
        .enumerate()
        .map(|(item_index, (item, &hash))| MultiModalFeature {
            kind: MultimodalKind::Image,
            item_range: item_index..item_index + 1,
            hashes: vec![hash],
            offset: item.placeholder.start,
            length: item.placeholder.len(),
            attention_policy: MultimodalAttentionPolicy::Causal,
            splittable: false,
        })
        .collect())
}

fn select_prompt_items(
    items: &[MiniCpmOPromptItem],
    query: Range<usize>,
    token_count: usize,
) -> anyhow::Result<Vec<SelectedPromptItem>> {
    if query.start > query.end || query.end > token_count {
        anyhow::bail!("MiniCPMO active prompt range is outside the token sequence");
    }
    let mut selected = Vec::new();
    for (source_index, item) in items.iter().enumerate() {
        let overlaps = item.placeholder.start < query.end && query.start < item.placeholder.end;
        if !overlaps {
            continue;
        }
        if item.placeholder.start < query.start || item.placeholder.end > query.end {
            anyhow::bail!("MiniCPMO image placeholder must be scheduled as a complete span");
        }
        selected.push(SelectedPromptItem {
            source_index,
            item: MiniCpmOPromptItem {
                placeholder: item.placeholder.start - query.start
                    ..item.placeholder.end - query.start,
                embedding_spans: item
                    .embedding_spans
                    .iter()
                    .map(|span| span.start - query.start..span.end - query.start)
                    .collect(),
            },
        });
    }
    Ok(selected)
}

impl InputsProcessor for MiniCpmOImageProcessor {
    fn get_type(&self) -> InputsProcessorType {
        InputsProcessorType::Vision
    }

    fn prepare_for_paged_prompt_planning(
        &self,
        tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut Sequence],
        _device: &Device,
        _other_config: Option<Arc<dyn Any>>,
        mut paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> anyhow::Result<()> {
        let tokenizer = tokenizer.ok_or_else(|| {
            anyhow::Error::msg("MiniCpmOImageProcessor requires a specified tokenizer.")
        })?;
        for seq in input_seqs {
            self.prepare_prompt(&tokenizer, seq, paged_attn_metadata.as_deref_mut())?;
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
                "MiniCpmOImageProcessor requires a specified tokenizer.",
            ));
        };

        let config = other_config.expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");

        let preserve_images = input_seqs
            .iter()
            .map(|seq| !seq.multimodal.has_changed_prompt)
            .collect::<Vec<_>>();
        if is_prompt {
            for seq in input_seqs.iter_mut() {
                self.prepare_prompt(&tokenizer, seq, paged_attn_metadata.as_mut())?;
            }
        }

        let token_ids = self.token_ids(&tokenizer)?;
        let expected_span_len = config
            .image_feature_size
            .unwrap_or(DEFAULT_IMAGE_FEATURE_SIZE);
        let mut visual_inputs = Vec::new();
        let mut legacy_maps = (0..input_seqs.len())
            .map(|_| Vec::new())
            .collect::<Vec<_>>();
        let mut requests = Vec::with_capacity(input_seqs.len());

        if is_prompt {
            for (seq_index, seq) in input_seqs.iter_mut().enumerate() {
                let tokens = seq.get_toks().to_vec();
                let items = parse_prompt_items(&tokens, token_ids)?;
                let query = seq
                    .active_prompt_local_query_range()
                    .unwrap_or_else(|| seq.prefix_cache_len().min(tokens.len())..tokens.len());
                let selected = select_prompt_items(&items, query.clone(), tokens.len())?;
                let chunked_view = seq.is_chunked_prefill_view();

                let mut selected_media = Vec::with_capacity(selected.len());
                if !selected.is_empty() {
                    let hashes = seq.image_hashes().unwrap_or_default().to_vec();
                    let images = if preserve_images[seq_index] {
                        seq.clone_images()
                    } else {
                        seq.take_images()
                    }
                    .ok_or_else(|| {
                        anyhow::Error::msg("MiniCPMO image placeholders are missing image inputs")
                    })?;

                    if chunked_view {
                        if selected.len() != images.len() || selected.len() != hashes.len() {
                            anyhow::bail!(
                                "MiniCPMO active prompt has {} placeholders, {} images, and {} hashes",
                                selected.len(),
                                images.len(),
                                hashes.len()
                            );
                        }
                        for ((selected, image), hash) in
                            selected.into_iter().zip(images).zip(hashes)
                        {
                            selected_media.push((selected, image, hash));
                        }
                    } else {
                        for selected in selected {
                            let image =
                                images.get(selected.source_index).cloned().ok_or_else(|| {
                                    anyhow::Error::msg(
                                        "MiniCPMO image inputs do not cover the active prompt",
                                    )
                                })?;
                            let hash = *hashes.get(selected.source_index).ok_or_else(|| {
                                anyhow::Error::msg(
                                    "MiniCPMO image hashes do not cover the active prompt",
                                )
                            })?;
                            selected_media.push((selected, image, hash));
                        }
                    }
                }

                let mut layout_items = Vec::with_capacity(selected_media.len());
                for (item_index, (selected, image, hash)) in selected_media.into_iter().enumerate()
                {
                    if selected
                        .item
                        .embedding_spans
                        .iter()
                        .any(|span| span.len() != expected_span_len)
                    {
                        anyhow::bail!(
                            "MiniCPMO image placeholder span does not match the configured feature size"
                        );
                    }
                    let PreprocessedImages {
                        pixel_values_list,
                        tgt_sizes,
                        ..
                    } = self.preprocess(
                        vec![image],
                        vec![],
                        config,
                        device,
                        (usize::MAX, usize::MAX),
                    )?;
                    let pixel_values = pixel_values_list.ok_or_else(|| {
                        anyhow::Error::msg("MiniCPMO preprocessing omitted pixel values")
                    })?;
                    let tgt_sizes = tgt_sizes.ok_or_else(|| {
                        anyhow::Error::msg("MiniCPMO preprocessing omitted target sizes")
                    })?;
                    if pixel_values.len() != selected.item.embedding_spans.len()
                        || tgt_sizes.dim(0)? != selected.item.embedding_spans.len()
                    {
                        anyhow::bail!(
                            "MiniCPMO produced {} image slices for {} placeholder spans",
                            pixel_values.len(),
                            selected.item.embedding_spans.len()
                        );
                    }

                    let key = MultimodalEncoderKey {
                        kind: MultimodalKind::Image,
                        hash,
                    };
                    visual_inputs.push(MiniCpmOVisualInput {
                        key,
                        pixel_values,
                        tgt_sizes,
                    });
                    let mut embedding_maps =
                        Vec::with_capacity(selected.item.embedding_spans.len());
                    for (source_output, destination) in
                        selected.item.embedding_spans.iter().cloned().enumerate()
                    {
                        legacy_maps[seq_index].push(MiniCpmOLegacyMap {
                            key,
                            source_output,
                            destination: destination.clone(),
                        });
                        embedding_maps.push(MultimodalEmbeddingMap::contiguous(
                            destination,
                            0,
                            source_output,
                        )?);
                    }
                    layout_items.push(MultimodalItemLayout::new(
                        key,
                        item_index,
                        selected.item.placeholder,
                        MultimodalAttentionPolicy::Causal,
                        embedding_maps,
                    )?);
                }
                requests.push(RequestMultimodalLayout {
                    sequence_id: *seq.id(),
                    query: 0..query.len(),
                    items: layout_items,
                });
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

        if !is_prompt {
            legacy_maps = (0..input.dim(0)?).map(|_| Vec::new()).collect();
        }
        let packed_layout = if is_prompt && flash_meta.packed {
            let query_lens = paged_attn_meta
                .as_ref()
                .and_then(|metadata| metadata.query_lens.as_deref())
                .ok_or_else(|| {
                    anyhow::Error::msg("packed MiniCPMO prefill requires logical query lengths")
                })?;
            if requests.len() != query_lens.len() {
                anyhow::bail!("MiniCPMO packed request metadata length mismatch");
            }
            for (request, &query_len) in requests.iter().zip(query_lens) {
                if request.query.len() != query_len {
                    anyhow::bail!(
                        "MiniCPMO packed query has {} tokens but input metadata has {query_len}",
                        request.query.len()
                    );
                }
            }
            let layout = PackedMultimodalLayout::new(&requests)?;
            if layout.token_count() != input.dim(1)? {
                anyhow::bail!(
                    "MiniCPMO packed layout has {} tokens but input has {}",
                    layout.token_count(),
                    input.dim(1)?
                );
            }
            Some(layout)
        } else {
            None
        };
        let args = MiniCpmOSpecificArgs {
            visual_inputs,
            legacy_maps,
            packed_layout,
        };

        let inputs: Box<dyn Any> = Box::new(ModelInputs {
            input_ids: input,
            seqlen_offsets: positions,
            context_lens,
            position_ids,
            pixel_values: None,
            model_specific_args: Box::new(args),
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

impl MiniCpmOImageProcessor {
    fn token_ids(&self, tokenizer: &Tokenizer) -> anyhow::Result<MiniCpmOTokenIds> {
        let token_id =
            |configured: &Option<String>, default: &str, name: &str| -> anyhow::Result<u32> {
                let token = configured.as_deref().unwrap_or(default);
                tokenizer.token_to_id(token).ok_or_else(|| {
                    anyhow::Error::msg(format!(
                        "MiniCPMO tokenizer is missing the {name} token {token:?}"
                    ))
                })
            };
        Ok(MiniCpmOTokenIds {
            image_start: token_id(
                &self.config.im_start_token,
                DEFAULT_IM_START_TOKEN,
                "image start",
            )?,
            image_end: token_id(&self.config.im_end_token, DEFAULT_IM_END_TOKEN, "image end")?,
            slice_start: token_id(
                &self.config.slice_start_token,
                DEFAULT_SLICE_START_TOKEN,
                "slice start",
            )?,
            slice_end: token_id(
                &self.config.slice_end_token,
                DEFAULT_SLICE_END_TOKEN,
                "slice end",
            )?,
        })
    }

    fn prepare_prompt(
        &self,
        tokenizer: &Tokenizer,
        seq: &mut Sequence,
        paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> anyhow::Result<()> {
        if seq.multimodal.has_changed_prompt || !seq.has_images() {
            return Ok(());
        }
        let images = seq
            .images()
            .ok_or_else(|| anyhow::Error::msg("MiniCPMO prompt is missing image inputs"))?;
        let prompt = tokenizer
            .decode(seq.get_toks(), false)
            .map_err(|err| anyhow::Error::msg(err.to_string()))?;
        let fragments = prompt.split(RAW_IMAGE_TAG).collect::<Vec<_>>();
        let raw_placeholder_count = fragments.len() - 1;
        let prompt = if raw_placeholder_count == 0 {
            prompt
        } else {
            if raw_placeholder_count != images.len() {
                anyhow::bail!(
                    "MiniCPMO has {raw_placeholder_count} raw image placeholders but {} image inputs",
                    images.len()
                );
            }
            let mut expanded = String::with_capacity(prompt.len());
            for (image_index, fragment) in fragments[..raw_placeholder_count].iter().enumerate() {
                expanded.push_str(fragment);
                expanded.push_str(
                    &self
                        .get_slice_image_placeholder(images[image_index].dimensions(), image_index),
                );
            }
            expanded.push_str(fragments[raw_placeholder_count]);
            expanded
        };
        let input_ids = tokenizer
            .encode_fast(prompt.as_str(), false)
            .map_err(|err| anyhow::Error::msg(err.to_string()))?
            .get_ids()
            .to_vec();
        let items = parse_prompt_items(&input_ids, self.token_ids(tokenizer)?)?;
        let hashes = seq.image_hashes().unwrap_or_default().to_vec();
        let features = prompt_features(&items, &hashes)?;
        let expected_span_len = self
            .config
            .image_feature_size
            .unwrap_or(DEFAULT_IMAGE_FEATURE_SIZE);
        if items
            .iter()
            .flat_map(|item| &item.embedding_spans)
            .any(|span| span.len() != expected_span_len)
        {
            anyhow::bail!(
                "MiniCPMO image placeholder span does not match the configured feature size"
            );
        }

        let has_prefill_toks = seq.has_prefill_toks();
        seq.set_initial_prompt(prompt);
        seq.set_mm_features(features);
        seq.set_toks_and_reallocate(input_ids.clone(), paged_attn_metadata);
        if has_prefill_toks {
            seq.set_prefill_toks(input_ids);
        }
        seq.multimodal.has_changed_prompt = true;
        Ok(())
    }

    fn get_sliced_grid(
        &self,
        (w, h): (usize, usize),
        max_slice_nums: usize,
        scale_resolution: usize,
        never_split: bool,
    ) -> Option<(usize, usize)> {
        let log_ratio = (w as f32 / h as f32).ln();
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
            w = (h as f32 * r) as usize;
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

        let grid_w = refine_w / grid_x;
        let grid_h = refine_h / grid_y;

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
        if !self.config.slice_mode.unwrap_or(DEFAULT_SLICE_MODE) {
            return vec![image.clone()];
        }

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

    fn get_image_id_placeholder(&self, image_idx: usize) -> String {
        format!(
            "{}{image_idx}{}",
            self.config
                .im_id_start
                .clone()
                .unwrap_or(DEFAULT_IM_ID_START.to_string()),
            self.config
                .im_id_end
                .clone()
                .unwrap_or(DEFAULT_IM_ID_END.to_string())
        )
    }

    fn get_grid_placeholder(&self, grid: Option<(usize, usize)>) -> String {
        if let Some(grid) = grid {
            let slice_image_placeholder = format!(
                "{}{}{}",
                self.config
                    .slice_start_token
                    .clone()
                    .unwrap_or(DEFAULT_SLICE_START_TOKEN.to_string()),
                self.config
                    .unk_token
                    .clone()
                    .unwrap_or(DEFAULT_UNK_TOKEN.to_string())
                    .repeat(
                        self.config
                            .image_feature_size
                            .unwrap_or(DEFAULT_IMAGE_FEATURE_SIZE)
                    ),
                self.config
                    .slice_end_token
                    .clone()
                    .unwrap_or(DEFAULT_SLICE_END_TOKEN.to_string())
            );

            let (cols, rows) = grid;
            let mut slices = Vec::new();
            for _ in 0..rows {
                let mut lines = Vec::new();
                for _ in 0..cols {
                    lines.push(slice_image_placeholder.clone());
                }
                slices.push(lines.join(""));
            }

            slices.join("\n")
        } else {
            "".to_string()
        }
    }

    fn get_slice_image_placeholder(&self, image_size: (u32, u32), image_idx: usize) -> String {
        let max_slice_nums = self.config.max_slice_nums.unwrap_or(DEFAULT_MAX_SLICE_NUMS);
        let use_image_id = self.config.use_image_id.unwrap_or(DEFAULT_USE_IMAGE_ID);
        let slice_mode = self.config.slice_mode.unwrap_or(DEFAULT_SLICE_MODE);

        let grid = self.get_sliced_grid(
            (image_size.0 as usize, image_size.1 as usize),
            max_slice_nums,
            self.config
                .scale_resolution
                .unwrap_or(DEFAULT_SCALE_RESOLUTION),
            false,
        );

        let image_placeholder = format!(
            "{}{}{}",
            self.config
                .im_start_token
                .clone()
                .unwrap_or(DEFAULT_IM_START_TOKEN.to_string()),
            self.config
                .unk_token
                .clone()
                .unwrap_or(DEFAULT_UNK_TOKEN.to_string())
                .repeat(
                    self.config
                        .image_feature_size
                        .unwrap_or(DEFAULT_IMAGE_FEATURE_SIZE)
                ),
            self.config
                .im_end_token
                .clone()
                .unwrap_or(DEFAULT_IM_END_TOKEN.to_string())
        );

        let final_placeholder = if use_image_id {
            format!(
                "{}{image_placeholder}",
                self.get_image_id_placeholder(image_idx)
            )
        } else {
            image_placeholder
        };

        if slice_mode {
            format!("{final_placeholder}{}", self.get_grid_placeholder(grid))
        } else {
            final_placeholder
        }
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
            let max_slice_nums = config.max_slice_nums.unwrap_or(DEFAULT_MAX_SLICE_NUMS);
            let scale_resolution = config.scale_resolution.unwrap_or(DEFAULT_SCALE_RESOLUTION);
            let patch_size = config.patch_size.unwrap_or(DEFAULT_PATCH_SIZE);

            let image_patches =
                self.get_sliced_images(&image, max_slice_nums, scale_resolution, patch_size);

            for slice_image in image_patches {
                let (w, h) = slice_image.dimensions();
                let to_tensor_rescale = Transforms {
                    input: &ToTensor,
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
        // Dummy pixel values
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
            num_crops: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOKEN_IDS: MiniCpmOTokenIds = MiniCpmOTokenIds {
        image_start: 1,
        image_end: 2,
        slice_start: 3,
        slice_end: 4,
    };

    #[test]
    fn prompt_parser_groups_slices_with_their_raw_image() {
        let items =
            parse_prompt_items(&[9, 1, 7, 7, 2, 8, 3, 7, 7, 4, 9, 1, 7, 7, 2], TOKEN_IDS).unwrap();
        assert_eq!(
            items,
            vec![
                MiniCpmOPromptItem {
                    placeholder: 1..10,
                    embedding_spans: vec![2..4, 7..9],
                },
                MiniCpmOPromptItem {
                    placeholder: 11..15,
                    embedding_spans: std::iter::once(12..14).collect(),
                },
            ]
        );
    }

    #[test]
    fn prompt_parser_rejects_malformed_delimiters() {
        assert!(parse_prompt_items(&[3, 7, 4], TOKEN_IDS).is_err());
        assert!(parse_prompt_items(&[1, 7, 3, 2], TOKEN_IDS).is_err());
        assert!(parse_prompt_items(&[1, 7], TOKEN_IDS).is_err());
        assert!(parse_prompt_items(&[1, 2, 4], TOKEN_IDS).is_err());
    }

    #[test]
    fn prompt_features_cover_the_complete_raw_item() {
        let items = parse_prompt_items(&[1, 7, 2, 3, 7, 4, 9, 1, 7, 2], TOKEN_IDS).unwrap();
        let features = prompt_features(&items, &[11, 12]).unwrap();
        assert_eq!(features.len(), 2);
        assert_eq!(features[0].offset, 0);
        assert_eq!(features[0].length, 6);
        assert_eq!(features[0].item_range, 0..1);
        assert_eq!(features[0].hashes, vec![11]);
        assert_eq!(features[1].offset, 7);
        assert_eq!(features[1].length, 3);
        assert_eq!(features[1].item_range, 1..2);
        assert!(prompt_features(&items, &[11]).is_err());
    }

    #[test]
    fn active_query_requires_and_shifts_a_complete_item() {
        let items = vec![MiniCpmOPromptItem {
            placeholder: 4..11,
            embedding_spans: vec![5..7, 8..10],
        }];
        let selected = select_prompt_items(&items, 4..12, 12).unwrap();
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].item.placeholder, 0..7);
        assert_eq!(selected[0].item.embedding_spans, vec![1..3, 4..6]);
        assert!(select_prompt_items(&items, 5..12, 12).is_err());
        assert!(select_prompt_items(&items, 0..10, 12).is_err());
    }

    #[test]
    fn packed_layout_keeps_text_only_rows_and_slice_outputs() {
        let key = MultimodalEncoderKey {
            kind: MultimodalKind::Image,
            hash: 17,
        };
        let layout = PackedMultimodalLayout::new(&[
            RequestMultimodalLayout {
                sequence_id: 1,
                query: 0..6,
                items: vec![MultimodalItemLayout::new(
                    key,
                    0,
                    1..5,
                    MultimodalAttentionPolicy::Causal,
                    vec![
                        MultimodalEmbeddingMap::contiguous(2..3, 0, 0).unwrap(),
                        MultimodalEmbeddingMap::contiguous(4..5, 0, 1).unwrap(),
                    ],
                )
                .unwrap()],
            },
            RequestMultimodalLayout {
                sequence_id: 2,
                query: 0..3,
                items: Vec::new(),
            },
        ])
        .unwrap();
        assert_eq!(layout.token_count(), 9);
        assert_eq!(layout.destination_positions(0), vec![2, 4]);
    }

    #[test]
    fn image_slicing_matches_reference_geometry() {
        let processor = MiniCpmOImageProcessor {
            config: PreProcessorConfig::default(),
        };
        assert_eq!(
            processor.get_sliced_grid((896, 448), 9, 448, false),
            Some((2, 1))
        );
        assert_eq!(
            processor.get_sliced_grid((448, 896), 9, 448, false),
            Some((1, 2))
        );
        assert_eq!(
            processor.find_best_resize((1600, 900), 448, 14, false),
            (602, 336)
        );
        assert_eq!(
            processor.get_refine_size((1600, 900), (2, 1), 448, 14, true),
            (840, 476)
        );
    }
}
