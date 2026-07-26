//! Inputs processor: turns a templated prompt + one image into the engine's
//! `ModelInputs` for PaddleOCR-VL. Single image, no video, no deepstack, batch-1 vision path.
//!
//! The chat template emits ONE `<|IMAGE_PLACEHOLDER|>` between
//! `<|IMAGE_START|>`/`<|IMAGE_END|>`; this processor expands that single placeholder into
//! `t*h*w / merge^2` copies (161 for the ocr fixture: 1*14*46/4) so the token stream matches the
//! reference input_ids (`...101305 [IMG x161] 101306...`). pixel_values + grid come from
//! `preprocess::preprocess_decoded`.

use std::{any::Any, sync::Arc};

use anyhow::Result;
use candle_core::{Device, Tensor};
use either::Either;
use image::DynamicImage;
use indexmap::IndexMap;
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::{
    device_map::DeviceMapper,
    paged_attention::block_hash::MultimodalKind,
    pipeline::{
        processing::default_process,
        text_models_inputs_processor::{
            self, get_completion_input, get_prompt_input, PagedAttentionMeta,
        },
        InputProcessorOutput, InputsProcessor, InputsProcessorType, MessagesAction, Processor,
    },
    request::ReasoningEffort,
    sequence::{build_mm_features_from_ranges, find_placeholder_delimited_ranges, Sequence},
    vision_models::{
        image_processor::{ImagePreProcessor, PreprocessedImages},
        preprocessor_config::PreProcessorConfig,
        ModelInputs,
    },
    MessageContent, Tool,
};

use super::preprocess::{preprocess_decoded, MERGE};
use super::PaddleOcrVlVisionSpecificArgs;

/// The `Processor` (message-level). Registers the special tokens the tokenizer must treat atomically
/// and hands out the `InputsProcessor`.
pub struct PaddleOcrVlProcessor;

impl PaddleOcrVlProcessor {
    pub const IMAGE_START: &'static str = "<|IMAGE_START|>";
    pub const IMAGE_PLACEHOLDER: &'static str = "<|IMAGE_PLACEHOLDER|>";
    pub const IMAGE_END: &'static str = "<|IMAGE_END|>";
    /// Temp marker used while expanding so the `while contains(PLACEHOLDER)` loop can't re-match the
    /// copies it just inserted; swapped back to the real placeholder before re-encoding.
    const EXPAND_MARKER: &'static str = "<|IMAGE_EXPAND_TMP|>";
}

impl Processor for PaddleOcrVlProcessor {
    // The checkpoint's template reads content as a list of typed parts (it has to, to find the image
    // part), but a plain text message arrives as a bare string. `Keep` hands that straight to jinja,
    // which iterates it per character, so `content["type"]` never matches and the text is dropped:
    // the model then decodes an empty user turn into byte-fallback garbage. Wrap it first.
    fn process(
        &self,
        pipeline: &dyn crate::pipeline::Pipeline,
        messages: Vec<IndexMap<String, MessageContent>>,
        add_generation_prompt: bool,
        add_special_tokens: bool,
        enable_thinking: Option<bool>,
        reasoning_effort: Option<ReasoningEffort>,
        tools: Vec<Tool>,
    ) -> anyhow::Result<(Vec<u32>, String)> {
        let messages = messages
            .into_iter()
            .map(|message| {
                message
                    .into_iter()
                    .map(|(key, value)| match (key.as_str(), value) {
                        ("content", Either::Left(text)) => (
                            key,
                            Either::Right(vec![IndexMap::from([
                                ("type".to_string(), Value::String("text".to_string())),
                                ("text".to_string(), Value::String(text)),
                            ])]),
                        ),
                        (_, value) => (key, value),
                    })
                    .collect()
            })
            .collect();
        default_process(
            pipeline,
            messages,
            add_generation_prompt,
            add_special_tokens,
            enable_thinking,
            reasoning_effort,
            self.template_action(),
            tools,
        )
    }
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(PaddleOcrVlImageProcessor)
    }
    fn get_special_tokens(&self) -> &[&'static str] {
        &[Self::IMAGE_START, Self::IMAGE_PLACEHOLDER, Self::IMAGE_END]
    }
    fn template_action(&self) -> MessagesAction {
        MessagesAction::Keep
    }
}

struct PaddleOcrVlImageProcessor;

fn replace_first_occurrence(text: &str, to_replace: &str, replacement: &str) -> String {
    if let Some(pos) = text.find(to_replace) {
        let mut result = text.to_string();
        result.replace_range(pos..pos + to_replace.len(), replacement);
        result
    } else {
        text.to_string()
    }
}

/// Expand every `<|IMAGE_PLACEHOLDER|>` in `text` into `product(grid)/merge^2` copies, taking the
/// i-th grid for the i-th placeholder (single image => one grid, one placeholder). Extracted so the
/// count arithmetic is unit-testable in isolation.
fn expand_placeholders(text: &str, grids: &[(usize, usize, usize)], merge: usize) -> String {
    let merge_length = merge * merge;
    let mut out = text.to_string();
    let mut index = 0;
    while out.contains(PaddleOcrVlProcessor::IMAGE_PLACEHOLDER) {
        let (t, h, w) = grids[index];
        let n = t * h * w / merge_length;
        out = replace_first_occurrence(
            &out,
            PaddleOcrVlProcessor::IMAGE_PLACEHOLDER,
            &PaddleOcrVlProcessor::EXPAND_MARKER.repeat(n),
        );
        index += 1;
    }
    out.replace(
        PaddleOcrVlProcessor::EXPAND_MARKER,
        PaddleOcrVlProcessor::IMAGE_PLACEHOLDER,
    )
}

/// Tag the `<|IMAGE_START|>..<|IMAGE_END|>` span with the image content hash for the paged prefix
/// cache. Every expanded placeholder is the same token id, so without this two requests that share a
/// prompt and a grid shape hash to identical blocks and the second one reuses the first one's image
/// KV. It also keeps a cache hit off the middle of the span, which would desync the connector rows
/// `Merger::forward` scatters (it counts image slots from the start of `input_ids`).
fn register_image_span(seq: &mut Sequence, ids: &[u32], tokenizer: &Tokenizer) {
    if !seq.mm_features().is_empty() {
        return;
    }
    let (Some(hashes), Some(pad_id), Some(start_id), Some(end_id)) = (
        seq.image_hashes().map(<[u64]>::to_vec),
        tokenizer.token_to_id(PaddleOcrVlProcessor::IMAGE_PLACEHOLDER),
        tokenizer.token_to_id(PaddleOcrVlProcessor::IMAGE_START),
        tokenizer.token_to_id(PaddleOcrVlProcessor::IMAGE_END),
    ) else {
        return;
    };
    let ranges = find_placeholder_delimited_ranges(ids, pad_id, start_id, end_id);
    let features = build_mm_features_from_ranges(&ranges, &hashes, MultimodalKind::Image);
    if !features.is_empty() {
        seq.set_mm_features(features);
    }
}

/// Every `(t, h, w)` row of a `[n_images, 3]` grid tensor, in message order.
fn grid_rows(grid: &Tensor) -> Vec<(usize, usize, usize)> {
    grid.to_vec2::<u32>()
        .unwrap()
        .into_iter()
        .map(|g| (g[0] as usize, g[1] as usize, g[2] as usize))
        .collect()
}

impl InputsProcessor for PaddleOcrVlImageProcessor {
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
        sliding_window: Option<usize>,
        other_config: Option<Arc<dyn Any>>,
        mut paged_attn_metadata: Option<PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
    ) -> Result<InputProcessorOutput> {
        if is_xlora {
            anyhow::bail!("Cannot make inputs for X-LoRA vision model.");
        }
        if no_kv_cache {
            anyhow::bail!("Vision model must have kv cache.");
        }
        let Some(tokenizer) = tokenizer else {
            anyhow::bail!("PaddleOcrVlImageProcessor requires a specified tokenizer.");
        };
        let config = other_config.expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");

        // Per row, independently: `grids[row]` is that row's image grid, and `vision_rows` is the
        // subset whose patches are in this pass's `pixel_values`. `Sequence::has_images` is
        // window-scoped once mm_features are set, so a decode step or a prompt chunk past the span
        // keeps the grid but skips the tower. Deciding per row (not `all`) keeps a batch correct
        // when rows sit at different prefill chunks, or when a text-only request shares the batch.
        //
        // A grid is only attached once the row's token window actually holds its image tokens:
        // get_rope_index emits a position block for every grid it is handed, and the first prompt
        // chunk stops before the span, so a grid there would emit positions for absent tokens.
        let image_pad_id = tokenizer.token_to_id(PaddleOcrVlProcessor::IMAGE_PLACEHOLDER);
        let mut grids: Vec<Vec<(usize, usize, usize)>> = Vec::with_capacity(input_seqs.len());
        let mut hashes: Vec<Vec<u64>> = Vec::with_capacity(input_seqs.len());
        let mut pixel_values_accum = Vec::new();
        let mut vision_rows: Vec<usize> = Vec::new();

        for (row, seq) in input_seqs.iter_mut().enumerate() {
            let window_has_image_toks = image_pad_id.is_some_and(|id| seq.get_toks().contains(&id));
            if !seq.has_images() {
                grids.push(
                    seq.multimodal
                        .cached_img_thw
                        .as_ref()
                        .filter(|_| window_has_image_toks)
                        .map(grid_rows)
                        .unwrap_or_default(),
                );
                hashes.push(Vec::new());
                continue;
            }
            let (pixel_values, row_grids) = match &seq.multimodal.cached_pixel_values {
                Some(cached) => (
                    cached.clone(),
                    grid_rows(seq.multimodal.cached_img_thw.as_ref().unwrap()),
                ),
                None => {
                    let PreprocessedImages {
                        pixel_values,
                        image_grid_thw,
                        ..
                    } = self.preprocess(
                        seq.clone_images().expect("Need images by this point."),
                        vec![],
                        config,
                        device,
                        (usize::MAX, usize::MAX),
                    )?;
                    seq.multimodal.cached_pixel_values = Some(pixel_values.clone());
                    seq.multimodal.cached_img_thw = image_grid_thw.clone();
                    (pixel_values, grid_rows(image_grid_thw.as_ref().unwrap()))
                }
            };

            if !seq.multimodal.has_changed_prompt {
                let detok = tokenizer
                    .decode(seq.get_toks(), false)
                    .expect("Detokenization failed!");
                let detok = expand_placeholders(&detok, &row_grids, MERGE);
                let ids = tokenizer
                    .encode_fast(detok.clone(), false)
                    .expect("Tokenization failed!")
                    .get_ids()
                    .to_vec();
                seq.set_initial_prompt(detok);
                // Before set_toks_and_reallocate: the block hashes it triggers must see the span.
                register_image_span(seq, &ids, &tokenizer);
                seq.set_toks_and_reallocate(ids, paged_attn_metadata.as_mut());
                seq.multimodal.has_changed_prompt = true;
            }

            grids.push(row_grids);
            hashes.push(seq.image_hashes().map(<[u64]>::to_vec).unwrap_or_default());
            if is_prompt {
                // Keep pixel_values as [N_patches, 3, 14, 14] (the shape the parity-verified tower
                // expects); rows concatenate on dim 0 and the model splits them back by grid.
                pixel_values_accum.push(pixel_values);
                vision_rows.push(row);
            }
        }

        let pixel_values =
            (!pixel_values_accum.is_empty()).then(|| Tensor::cat(&pixel_values_accum, 0).unwrap());
        // All-empty means no row carries an image at all: let the model take the text-embed path.
        let image_grid_thw = if grids.iter().all(Vec::is_empty) {
            Vec::new()
        } else {
            grids
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
            )
            .unwrap()
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
            )
            .unwrap()
        };

        // The model recomputes mrope positions from the full token history, so input_ids_full is each
        // row's whole sequence (prompt + generated), not just this pass's window.
        let max_len = input_seqs
            .iter()
            .map(|seq| seq.get_toks().len())
            .max()
            .unwrap_or(0);
        let mut rows = Vec::with_capacity(input_seqs.len());
        for seq in input_seqs.iter() {
            let mut ids = seq.get_toks().to_vec();
            ids.resize(max_len, 0);
            rows.push(Tensor::new(ids, device).unwrap());
        }
        let input_ids_full = Tensor::stack(&rows, 0).unwrap();

        let inputs: Box<dyn Any> = Box::new(ModelInputs {
            input_ids: input,
            seqlen_offsets: positions,
            context_lens,
            position_ids,
            pixel_values,
            model_specific_args: Box::new(PaddleOcrVlVisionSpecificArgs {
                input_ids_full,
                image_grid_thw,
                image_hashes: hashes,
                vision_rows,
            }),
            paged_attn_meta,
            flash_meta,
            recurrent_batch_kind: if is_prompt {
                crate::pipeline::RecurrentBatchKind::Prefill
            } else {
                crate::pipeline::RecurrentBatchKind::Decode
            },
        });
        Ok(InputProcessorOutput {
            inputs,
            seq_indices,
        })
    }
}

impl ImagePreProcessor for PaddleOcrVlImageProcessor {
    const DEFAULT_MEAN: [f64; 3] = [0.5, 0.5, 0.5];
    const DEFAULT_STD: [f64; 3] = [0.5, 0.5, 0.5];

    fn preprocess(
        &self,
        images: Vec<DynamicImage>,
        _videos: Vec<Vec<DynamicImage>>,
        _config: &PreProcessorConfig,
        device: &Device,
        (_, _): (usize, usize),
    ) -> candle_core::Result<PreprocessedImages> {
        if images.is_empty() {
            candle_core::bail!("PaddleOCR-VL needs at least one image.");
        }
        // Patches of every image concatenated on dim 0, with one grid row each, in message order.
        // The forward splits them back by each grid's t*h*w.
        let mut patches = Vec::with_capacity(images.len());
        let mut grid = Vec::with_capacity(images.len() * 3);
        for img in &images {
            let (px, (t, h, w)) = preprocess_decoded(img, device)?;
            patches.push(px);
            grid.extend([t as u32, h as u32, w as u32]);
        }
        let grid = Tensor::from_vec(grid, (images.len(), 3), device)?;
        Ok(PreprocessedImages {
            pixel_values: Tensor::cat(&patches, 0)?,
            pixel_attention_mask: None,
            image_sizes: None,
            num_img_tokens: None,
            aspect_ratio_ids: None,
            aspect_ratio_mask: None,
            num_tiles: None,
            image_grid_thw: Some(grid),
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
    use super::*;
    use crate::paged_attention::block_hash::compute_block_hashes;
    use crate::sequence::clamp_prefix_cache_len_for_mm_features;

    // Reference ocr fixture ids; only their distinctness matters here.
    const IMAGE_START_ID: u32 = 101305;
    const IMAGE_PAD_ID: u32 = 101304;
    const IMAGE_END_ID: u32 = 101306;
    const BLOCK: usize = 16;

    fn expanded_ids(n_image_toks: usize) -> Vec<u32> {
        let mut ids = vec![7u32; 20];
        ids.push(IMAGE_START_ID);
        ids.extend(std::iter::repeat_n(IMAGE_PAD_ID, n_image_toks));
        ids.push(IMAGE_END_ID);
        ids.extend([8u32, 9]);
        ids
    }

    // Every expanded placeholder is the same token id, so two OCR requests that share a prompt and a
    // grid shape have byte-identical token streams. Only the span's content hash can tell their paged
    // blocks apart; without it the second request reuses the first one's image KV.
    #[test]
    fn image_span_separates_prefix_cache_blocks() {
        let ids = expanded_ids(161);
        let ranges =
            find_placeholder_delimited_ranges(&ids, IMAGE_PAD_ID, IMAGE_START_ID, IMAGE_END_ID);
        assert_eq!(
            ranges,
            vec![(20, 163)],
            "span must cover START..END inclusive"
        );

        let feats =
            |hash: u64| build_mm_features_from_ranges(&ranges, &[hash], MultimodalKind::Image);
        let a = compute_block_hashes(&ids, BLOCK, &feats(0xAAAA_AAAA), &[]);
        let b = compute_block_hashes(&ids, BLOCK, &feats(0xBBBB_BBBB), &[]);
        assert!(!a.is_empty(), "prompt must span at least one full block");
        assert_ne!(a, b, "different images hashed to the same blocks");
        // Guard: the collision is real without the span, so the assert above is not vacuous.
        assert_eq!(
            compute_block_hashes(&ids, BLOCK, &[], &[]),
            compute_block_hashes(&ids, BLOCK, &[], &[])
        );
    }

    // A hit inside the span would leave `input_ids` with fewer image slots than the connector emits
    // rows, and `Merger::forward` counts slots from the start of `input_ids`.
    #[test]
    fn prefix_cache_hit_cannot_land_inside_image_span() {
        let ids = expanded_ids(161);
        let ranges =
            find_placeholder_delimited_ranges(&ids, IMAGE_PAD_ID, IMAGE_START_ID, IMAGE_END_ID);
        let features =
            build_mm_features_from_ranges(&ranges, &[0xAAAA_AAAA], MultimodalKind::Image);
        for hit in [21usize, 100, 182] {
            let clamped = clamp_prefix_cache_len_for_mm_features(hit, BLOCK, &features);
            assert!(
                clamped <= 20,
                "hit {hit} clamped to {clamped}, inside the span"
            );
        }
        // Past the span the whole image is cached and `input_ids` has no image slots left: legal.
        assert_eq!(
            clamp_prefix_cache_len_for_mm_features(183, BLOCK, &features),
            183
        );
    }

    // Two images in one message used to be silently reduced to the first, then panic in
    // `expand_placeholders` on the second placeholder's missing grid. Each image must get its own
    // grid row, and the patches must concatenate in message order.
    #[test]
    fn every_image_gets_its_own_grid_row() {
        let call = |images: Vec<DynamicImage>| {
            PaddleOcrVlImageProcessor.preprocess(
                images,
                vec![],
                &PreProcessorConfig::default(),
                &Device::Cpu,
                (usize::MAX, usize::MAX),
            )
        };
        let out = call(vec![
            DynamicImage::new_rgb8(64, 64),
            DynamicImage::new_rgb8(128, 64),
        ])
        .expect("two images must be accepted");
        let grid = out.image_grid_thw.expect("grid");
        assert_eq!(grid.dims(), &[2, 3], "one grid row per image");
        let rows = grid_rows(&grid);
        assert_ne!(
            rows[0], rows[1],
            "differently sized images need different grids"
        );
        let patches: usize = rows.iter().map(|&(t, h, w)| t * h * w).sum();
        assert_eq!(
            out.pixel_values.dim(0).unwrap(),
            patches,
            "patches must be both images concatenated"
        );
        assert!(call(vec![]).is_err(), "zero images must not panic");
    }

    #[test]
    fn expand_placeholder_count_matches_grid() {
        // ocr fixture grid (t=1, h=14, w=46) => 1*14*46 / 2^2 = 161 image tokens.
        let text = format!(
            "User: {}{}{}OCR:",
            PaddleOcrVlProcessor::IMAGE_START,
            PaddleOcrVlProcessor::IMAGE_PLACEHOLDER,
            PaddleOcrVlProcessor::IMAGE_END,
        );
        let expanded = expand_placeholders(&text, &[(1, 14, 46)], MERGE);
        let count = expanded
            .matches(PaddleOcrVlProcessor::IMAGE_PLACEHOLDER)
            .count();
        assert_eq!(count, 161);
        // No temp marker leaks and the surrounding text is intact.
        assert!(!expanded.contains(PaddleOcrVlProcessor::EXPAND_MARKER));
        assert!(expanded.contains(PaddleOcrVlProcessor::IMAGE_START));
        assert!(expanded.contains("OCR:"));
    }
}
