#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
use std::any::Any;
use std::sync::Arc;

use candle_core::Result;
use candle_core::{DType, Device, Tensor};
use image::GenericImageView;
use image::Rgb;
use itertools::Itertools;
use regex_automata::meta::Regex;
use tokenizers::Tokenizer;

use super::llava15::LLaVAVisionSpecificArgs;
use super::utils::{expand2square, LLaVAImageProcessor};
use crate::device_map::DeviceMapper;
use crate::pipeline::text_models_inputs_processor::{
    get_completion_input, get_prompt_input, PagedAttentionMeta,
};
use crate::pipeline::{
    text_models_inputs_processor, InputProcessorOutput, InputsProcessor, InputsProcessorType,
    MessagesAction, Processor,
};
use crate::sequence::{build_mm_features_from_ranges, Sequence};
use crate::vision_models::image_processor::{self, ImagePreProcessor, PreprocessedImages};
use crate::vision_models::llava::config::Config as LLaVAConfig;
use crate::vision_models::preprocessor_config::{PreProcessorConfig, ToFilter};
use crate::vision_models::{preprocessor_config, ModelInputs};

pub struct LLaVAProcessor {
    inputs_processor: Arc<LLaVAInputProcessor>,
}

impl Processor for LLaVAProcessor {
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

impl LLaVAProcessor {
    pub fn new(config: &str) -> Self {
        let model_config =
            serde_json::from_str::<LLaVAConfig>(config).expect("Failed to parse model config.");
        let image_tag_splitter = Regex::new(r"<image>").expect("Failed to compile split regex.");
        let inputs_processor = Arc::new(LLaVAInputProcessor {
            image_tag_splitter,
            model_config: model_config.clone(),
        });
        Self { inputs_processor }
    }
}

pub struct LLaVAInputProcessor {
    image_tag_splitter: Regex,
    model_config: LLaVAConfig,
}

impl LLaVAInputProcessor {
    pub fn get_num_image_tokens(cfg: &LLaVAConfig) -> usize {
        let patch_size = cfg.vision_config.patch_size;
        let patch_per_side = cfg.vision_config.image_size / patch_size;
        patch_per_side * patch_per_side
    }
}

// Copy from phi3_inputs_processor. different is (1) calculate of num_image_token (2) process_anyres_image (3)image_ids_pad
impl InputsProcessor for LLaVAInputProcessor {
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
                "LLaVAInputProcessor requires a specified tokenizer.",
            ));
        };

        let config = other_config
            .clone()
            .expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");

        let has_images = input_seqs.iter().all(|seq| seq.has_images());

        let (pixel_values, num_img_tokens) = if has_images {
            let mut pixel_values_accum = Vec::new();
            let mut num_img_tokens_accum = Vec::new();
            for seq in input_seqs.iter_mut() {
                let imgs = seq
                    .take_images()
                    .expect("Need to have images by this point.");
                let PreprocessedImages {
                    pixel_values,
                    pixel_attention_mask: _,
                    image_sizes: _,
                    num_img_tokens,
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
                } = self
                    .preprocess(
                        imgs.clone(),
                        vec![],
                        config,
                        device,
                        (usize::MAX, usize::MAX),
                    )
                    .expect("Preprocessor failed");
                pixel_values_accum.push(pixel_values);
                num_img_tokens_accum.push(num_img_tokens.unwrap());
            }
            (
                Some(Tensor::cat(&pixel_values_accum, 0).unwrap()),
                Some(num_img_tokens_accum),
            )
        } else {
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
                    } = *inputs
                        .downcast::<text_models_inputs_processor::ModelInputs>()
                        .expect("Downcast failed.");

                    let inputs: Box<dyn Any> = Box::new(ModelInputs {
                        input_ids,
                        seqlen_offsets,
                        context_lens,
                        position_ids,
                        pixel_values: None,
                        model_specific_args: Box::new(LLaVAVisionSpecificArgs {
                            image_hashes: vec![],
                        }),
                        paged_attn_meta,
                        flash_meta,
                    });
                    InputProcessorOutput {
                        inputs,
                        seq_indices,
                    }
                });
        };

        let mut toks = Vec::new();
        let detokenized = tokenizer
            .decode_batch(
                &input_seqs
                    .iter()
                    .map(|seq| seq.get_toks())
                    .collect::<Vec<_>>(),
                false,
            )
            .expect("Decoding failed");

        for (detokenized, (seq, num_img_tokens)) in detokenized.into_iter().zip(
            input_seqs
                .iter_mut()
                .zip(num_img_tokens.unwrap().into_iter()),
        ) {
            let splits = self
                .image_tag_splitter
                .split(&detokenized)
                .map(|span| &detokenized[span.range()])
                .collect::<Vec<_>>();
            let prompt_chunks = splits
                .iter()
                .map(|s| {
                    // we don't use encode_batch here, because encode_batch will pad 0 to the end of the shor sequences, which will cause the image_ids_pad to be wrong.
                    tokenizer
                        .encode_fast(*s, false)
                        .unwrap()
                        .get_ids()
                        .to_vec()
                        .iter()
                        .map(|x| *x as i64)
                        .collect()
                })
                .collect::<Vec<Vec<_>>>();
            let mut image_ids_pad = Vec::new();
            for (i, num_img_token) in num_img_tokens.iter().enumerate() {
                let mut image_id_pad = vec![0; *num_img_token];
                image_id_pad[0] = -(i as i64 + 1);
                image_ids_pad.push(image_id_pad);
            }

            // Compute image placeholder ranges from interleave structure
            let mut img_ranges = Vec::new();
            {
                let mut offset = 0;
                for (chunk, pad) in prompt_chunks.iter().zip(image_ids_pad.iter()) {
                    offset += chunk.len();
                    img_ranges.push((offset, pad.len()));
                    offset += pad.len();
                }
            }

            let mut input_ids: Vec<i64> = Vec::new();
            for item in prompt_chunks
                .iter()
                .map(|x| x.to_vec())
                .interleave(image_ids_pad)
            {
                input_ids.extend(item);
            }
            let new_ids = input_ids
                .iter()
                .map(|x| if *x < 0 { 0u32 } else { *x as u32 })
                .collect::<Vec<_>>();
            if !seq.multimodal.has_changed_prompt {
                let new_prompt = tokenizer.decode(&new_ids, false).unwrap();
                seq.set_initial_prompt(new_prompt);

                // Build mm_features for position-aware prefix cache hashing
                if seq.mm_features().is_empty() {
                    if let Some(hashes) = seq.image_hashes().map(|h| h.to_vec()) {
                        seq.set_mm_features(build_mm_features_from_ranges(
                            &img_ranges,
                            &hashes,
                            "img",
                        ));
                    }
                }

                // NOTE(EricLBuehler): Casting to u32 is fine, we don't care about the other toks
                seq.set_toks_and_reallocate(new_ids, paged_attn_metadata.as_mut());
                seq.multimodal.has_changed_prompt = true;
            }

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
            )
        };

        metadata.map(|metadata| {
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
            } = metadata;
            let image_hashes: Vec<u64> = input_seqs
                .iter()
                .flat_map(|seq| {
                    seq.image_hashes()
                        .map(|h| {
                            let cached = seq.count_prefix_cached_mm_items();
                            if cached < h.len() {
                                h[cached..].to_vec()
                            } else {
                                vec![]
                            }
                        })
                        .unwrap_or_default()
                })
                .collect();
            let inputs: Box<dyn Any> = Box::new(ModelInputs {
                input_ids: input,
                seqlen_offsets: positions,
                context_lens,
                position_ids,
                pixel_values: pixel_values.clone(),
                model_specific_args: Box::new(LLaVAVisionSpecificArgs { image_hashes }),
                paged_attn_meta,
                flash_meta,
            });
            InputProcessorOutput {
                inputs,
                seq_indices,
            }
        })
    }
}

impl ImagePreProcessor for LLaVAInputProcessor {
    #[allow(clippy::excessive_precision)]
    const DEFAULT_MEAN: [f64; 3] = [0.48145466, 0.4578275, 0.40821073];
    #[allow(clippy::excessive_precision)]
    const DEFAULT_STD: [f64; 3] = [0.26862954, 0.26130258, 0.27577711];
    fn preprocess(
        &self,
        images: Vec<image::DynamicImage>,
        videos: Vec<Vec<image::DynamicImage>>,
        config: &preprocessor_config::PreProcessorConfig,
        device: &candle_core::Device,
        (_, _): (usize, usize),
    ) -> candle_core::Result<image_processor::PreprocessedImages> {
        if images.len() > 1 {
            candle_core::bail!("Can only process one image per batch"); // This is no different from phi3_input_processor
        };
        assert!(videos.is_empty());
        let resized_size = *config.size.as_ref().unwrap().get("shortest_edge").unwrap() as usize;

        let original_size = images[0].dimensions();
        let filter = config.resampling.to_filter()?;
        let image_mean = config
            .image_mean
            .unwrap_or(Self::DEFAULT_MEAN)
            .map(|x| x as f32);
        let mean_color = image_mean
            .iter()
            .map(|x| ((*x) * 255.0) as u8)
            .collect::<Vec<u8>>();
        let mean_color = Rgb::from([mean_color[0], mean_color[1], mean_color[2]]);
        let image = expand2square(&images[0], mean_color);
        let image_std = config
            .image_std
            .unwrap_or(Self::DEFAULT_STD)
            .map(|x| x as f32);
        let pixel_values = [image]
            .iter()
            .map(|x| {
                LLaVAImageProcessor::process_one_image(
                    x,
                    config,
                    resized_size as u32,
                    filter,
                    DType::BF16,
                    device,
                    &image_mean,
                    &image_std,
                )
            })
            .collect::<Result<Vec<Tensor>>>()?;
        let pixel_values = Tensor::stack(&pixel_values, 0)?;

        Ok(image_processor::PreprocessedImages {
            pixel_values,
            pixel_attention_mask: None,
            image_sizes: Some((original_size.0 as usize, original_size.1 as usize)),
            num_img_tokens: Some(vec![LLaVAInputProcessor::get_num_image_tokens(
                &self.model_config,
            )]),
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
