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
use image::DynamicImage;
use tokenizers::Tokenizer;

use crate::{
    device_map::DeviceMapper,
    pipeline::{
        text_models_inputs_processor::{
            self, get_completion_input, get_prompt_input, PagedAttentionMeta,
        },
        InputProcessorOutput, InputsProcessor, InputsProcessorType, MessagesAction, Processor,
    },
    sequence::Sequence,
    vision_models::{
        image_processor::{ImagePreProcessor, PreprocessedImages},
        preprocessor_config::PreProcessorConfig,
        ModelInputs,
    },
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

fn grid_tuple(grid: &Tensor) -> (usize, usize, usize) {
    let g = grid.to_vec2::<u32>().unwrap();
    (g[0][0] as usize, g[0][1] as usize, g[0][2] as usize)
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

        let has_images = input_seqs.iter().all(|seq| seq.has_images());

        // (padded input_ids_full, pixel_values [N,3,14,14], grid (t,h,w)). grid stays Some on decode
        // too (via the cache) so the model's mrope `delta` recomputes identically each step.
        let (new_input, pixel_values, image_grid_thw) = if has_images {
            let mut pixel_values_accum = Vec::new();
            let mut grid_accum: Vec<(usize, usize, usize)> = Vec::new();

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
                let (pixel_values, grid) = if let Some(cached) = &seq.multimodal.cached_pixel_values
                {
                    (
                        cached.clone(),
                        grid_tuple(seq.multimodal.cached_img_thw.as_ref().unwrap()),
                    )
                } else {
                    let PreprocessedImages {
                        pixel_values,
                        image_grid_thw,
                        ..
                    } = self
                        .preprocess(
                            seq.clone_images().expect("Need images by this point."),
                            vec![],
                            config,
                            device,
                            (usize::MAX, usize::MAX),
                        )
                        .expect("Preprocessing failed");
                    seq.multimodal.cached_pixel_values = Some(pixel_values.clone());
                    seq.multimodal.cached_img_thw = image_grid_thw.clone();
                    (pixel_values, grid_tuple(image_grid_thw.as_ref().unwrap()))
                };
                // Single-image batch-1: keep pixel_values as [N_patches, 3, 14, 14] (the shape the
                // parity-verified vision tower expects); do NOT prepend a batch dim.
                pixel_values_accum.push(pixel_values);
                grid_accum.push(grid);
            }

            if is_prompt {
                for ((text, seq), &grid) in detok_seqs
                    .iter_mut()
                    .zip(input_seqs.iter_mut())
                    .zip(grid_accum.iter())
                {
                    if seq.multimodal.has_changed_prompt {
                        continue;
                    }
                    *text = expand_placeholders(text, &[grid], MERGE);
                }
            }

            let mut all_ids = Vec::new();
            for (detok, seq) in detok_seqs.into_iter().zip(input_seqs.iter_mut()) {
                let ids = tokenizer
                    .encode_fast(detok.clone(), false)
                    .expect("Tokenization failed!")
                    .get_ids()
                    .to_vec();
                if !seq.multimodal.has_changed_prompt {
                    seq.set_initial_prompt(detok.clone());
                    seq.set_toks_and_reallocate(ids.clone(), paged_attn_metadata.as_mut());
                    seq.multimodal.has_changed_prompt = true;
                }
                all_ids.push(ids);
            }

            let max_len = all_ids.iter().map(|ids| ids.len()).max().unwrap();
            let mut rows = Vec::new();
            for ids in all_ids {
                let pad = max_len - ids.len();
                rows.push(Tensor::new([ids, vec![0; pad]].concat(), device).unwrap());
            }
            (
                Some(Tensor::stack(&rows, 0).unwrap()),
                Some(Tensor::cat(&pixel_values_accum, 0).unwrap()),
                grid_accum.first().copied(),
            )
        } else {
            (None, None, None)
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

        // On decode the model recomputes mrope positions from the full token history, so
        // input_ids_full must be the whole sequence (prompt + generated), not just the new token.
        let input_ids_full = match (new_input, is_prompt) {
            (Some(new_input), _) => new_input,
            (None, _) => {
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
                Tensor::stack(&rows, 0).unwrap()
            }
        };

        let pixel_values = if is_prompt { pixel_values } else { None };

        let inputs: Box<dyn Any> = Box::new(ModelInputs {
            input_ids: input,
            seqlen_offsets: positions,
            context_lens,
            position_ids,
            pixel_values,
            model_specific_args: Box::new(PaddleOcrVlVisionSpecificArgs {
                input_ids_full,
                image_grid_thw,
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
        mut images: Vec<DynamicImage>,
        _videos: Vec<Vec<DynamicImage>>,
        _config: &PreProcessorConfig,
        device: &Device,
        (_, _): (usize, usize),
    ) -> candle_core::Result<PreprocessedImages> {
        // Single image (batch-1 vision path); the model's forward assumes one grid.
        let img = images.remove(0);
        let (pixel_values, (t, h, w)) = preprocess_decoded(&img, device)?;
        let grid = Tensor::from_vec(vec![t as u32, h as u32, w as u32], (1, 3), device)?;
        Ok(PreprocessedImages {
            pixel_values,
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
