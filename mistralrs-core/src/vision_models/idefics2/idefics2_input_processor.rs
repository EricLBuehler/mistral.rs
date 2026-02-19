#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{any::Any, sync::Arc};

use candle_core::{Device, Result, Tensor};
use image::{DynamicImage, GenericImageView};
use indexmap::IndexMap;
use mistralrs_vision::{ApplyTransforms, Normalize, Rescale, ToTensorNoNorm, Transforms};
use tokenizers::Tokenizer;

use crate::{
    device_map::DeviceMapper,
    pipeline::{
        apply_chat_template,
        text_models_inputs_processor::{
            self, get_completion_input, get_prompt_input, PagedAttentionMeta,
        },
        InputProcessorOutput, InputsProcessor, InputsProcessorType, MessagesAction, Processor,
    },
    request::ReasoningEffort,
    sequence::Sequence,
    vision_models::ModelInputs,
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
        })
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &["<fake_token_around_image>", "<image>", "<end_of_utterance>"]
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::Keep
    }
}

impl InputsProcessor for Idefics2ImageProcessor {
    fn get_type(&self) -> InputsProcessorType {
        InputsProcessorType::Vision
    }
    fn process_inputs(
        &self,
        _: Option<Arc<Tokenizer>>,
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
            )
            .unwrap()
        };
        let config = other_config.expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");

        let has_images = input_seqs.iter().all(|seq| seq.has_images());

        let (pixel_values, pixel_attention_mask) = if has_images {
            let mut pixel_values_accum = Vec::new();
            let mut pixel_attention_mask_accum = Vec::new();
            for seq in input_seqs.iter_mut() {
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
                pixel_attention_mask_accum
                    .push(pixel_attention_mask.unwrap().unsqueeze(0).unwrap());
            }
            (
                Some(Tensor::cat(&pixel_values_accum, 0).unwrap()),
                Some(Tensor::cat(&pixel_attention_mask_accum, 0).unwrap()),
            )
        } else {
            (None, None)
        };

        let image_hashes: Vec<u64> = if is_prompt {
            input_seqs
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
                .collect()
        } else {
            vec![]
        };

        let inputs: Box<dyn Any> = Box::new(ModelInputs {
            input_ids: input,
            seqlen_offsets: positions,
            context_lens,
            position_ids,
            pixel_values,
            model_specific_args: Box::new(super::Idefics2SpecificArgs {
                pixel_attention_mask,
                image_hashes,
            }),
            paged_attn_meta,
            flash_meta,
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
