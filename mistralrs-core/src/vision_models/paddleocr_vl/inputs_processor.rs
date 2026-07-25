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

        // Per row, independently: `grids[row]` is that row's image grid whenever its prompt has one,
        // because mrope is recomputed from the whole prompt on every pass. `vision_rows` is the
        // subset whose image tokens actually land in THIS pass, in the order their patches are
        // concatenated into `pixel_values`. `Sequence::has_images` is window-scoped once mm_features
        // are set, so decode steps and prompt chunks outside the span keep the grid but skip the
        // tower. Deciding per row (not `all`) keeps a batch correct when rows are at different
        // prefill chunks, or when a text-only request shares the batch with an OCR one.
        let mut grids: Vec<Option<(usize, usize, usize)>> = Vec::with_capacity(input_seqs.len());
        let mut pixel_values_accum = Vec::new();
        let mut vision_rows: Vec<usize> = Vec::new();

        for (row, seq) in input_seqs.iter_mut().enumerate() {
            if !seq.has_images() {
                grids.push(seq.multimodal.cached_img_thw.as_ref().map(grid_tuple));
                continue;
            }
            let (pixel_values, grid) = match &seq.multimodal.cached_pixel_values {
                Some(cached) => (
                    cached.clone(),
                    grid_tuple(seq.multimodal.cached_img_thw.as_ref().unwrap()),
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
                    (pixel_values, grid_tuple(image_grid_thw.as_ref().unwrap()))
                }
            };

            if !seq.multimodal.has_changed_prompt {
                let detok = tokenizer
                    .decode(seq.get_toks(), false)
                    .expect("Detokenization failed!");
                let detok = expand_placeholders(&detok, &[grid], MERGE);
                let ids = tokenizer
                    .encode_fast(detok.clone(), false)
                    .expect("Tokenization failed!")
                    .get_ids()
                    .to_vec();
                seq.set_initial_prompt(detok);
                seq.set_toks_and_reallocate(ids, paged_attn_metadata.as_mut());
                seq.multimodal.has_changed_prompt = true;
            }

            grids.push(Some(grid));
            if is_prompt {
                // Keep pixel_values as [N_patches, 3, 14, 14] (the shape the parity-verified tower
                // expects); rows concatenate on dim 0 and the model splits them back by grid.
                pixel_values_accum.push(pixel_values);
                vision_rows.push(row);
            }
        }

        let pixel_values =
            (!pixel_values_accum.is_empty()).then(|| Tensor::cat(&pixel_values_accum, 0).unwrap());
        // All-None means no row carries an image at all: let the model take the text-embed path.
        let image_grid_thw = if grids.iter().all(Option::is_none) {
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
        // One image per request: the forward maps one grid to one batch row, and the placeholder
        // expansion has exactly one grid to spend per row. Reject rather than drop the extras.
        let [img] = &images[..] else {
            candle_core::bail!(
                "PaddleOCR-VL takes one image per request, got {}. Send one region crop per request.",
                images.len()
            );
        };
        let (pixel_values, (t, h, w)) = preprocess_decoded(img, device)?;
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
    // Two images in one message used to be silently reduced to the first, then panic in
    // `expand_placeholders` on the second placeholder's missing grid. It must be a clean error.
    #[test]
    fn multiple_images_are_rejected_not_dropped() {
        let one = || DynamicImage::new_rgb8(64, 64);
        let call = |images: Vec<DynamicImage>| {
            PaddleOcrVlImageProcessor.preprocess(
                images,
                vec![],
                &PreProcessorConfig::default(),
                &Device::Cpu,
                (usize::MAX, usize::MAX),
            )
        };
        let err = call(vec![one(), one()])
            .err()
            .expect("two images must be rejected")
            .to_string();
        assert!(err.contains("one image per request"), "{err}");
        assert!(call(vec![]).is_err(), "zero images must not panic either");
        assert!(call(vec![one()]).is_ok(), "one image must still work");
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
