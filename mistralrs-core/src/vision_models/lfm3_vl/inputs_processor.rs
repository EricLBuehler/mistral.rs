use std::sync::Arc;

use crate::{
    pipeline::{InputsProcessor, MessagesAction, Processor},
    vision_models::{
        grid_processor::{GridProcessor, GridProcessorOptions},
        preprocessor_config::PreProcessorConfig,
    },
};

use super::config::Config;

const IMAGE_MIN_PIXELS: usize = 65_536;
const IMAGE_MAX_PIXELS: usize = 16_777_216;
const VIDEO_MIN_PIXELS: usize = 4_096;
const VIDEO_MAX_PIXELS: usize = 25_165_824;
const IMAGE_MEAN: [f64; 3] = [0.5; 3];
const IMAGE_STD: [f64; 3] = [0.5; 3];
const BICUBIC: usize = 3;
const VIDEO_FPS_DECIMAL_PLACES: u32 = 6;

pub(crate) struct Lfm3VlProcessor(GridProcessor);

impl Lfm3VlProcessor {
    pub(crate) fn new(
        config: &Config,
        preprocessor: &PreProcessorConfig,
        max_edge: Option<u32>,
    ) -> Self {
        let mut preprocessor = preprocessor.clone();
        let mut video = preprocessor
            .video
            .take()
            .map(|config| *config)
            .unwrap_or_default();
        Self::defaults(
            &mut preprocessor,
            config,
            IMAGE_MIN_PIXELS,
            IMAGE_MAX_PIXELS,
        );
        Self::defaults(&mut video, config, VIDEO_MIN_PIXELS, VIDEO_MAX_PIXELS);
        preprocessor.video = Some(Box::new(video));
        Self(GridProcessor::with_options(
            max_edge,
            GridProcessorOptions {
                image_token: "<|image|>",
                video_token: "<|video|>",
                mrope: false,
                allow_small_images: true,
                round_to_even: true,
                timestamp_fps_precision: Some(VIDEO_FPS_DECIMAL_PLACES),
                config: Some(preprocessor),
            },
        ))
    }

    fn defaults(
        config: &mut PreProcessorConfig,
        model: &Config,
        min_pixels: usize,
        max_pixels: usize,
    ) {
        config
            .patch_size
            .get_or_insert(model.vision_config.patch_size);
        config
            .merge_size
            .get_or_insert(model.vision_config.spatial_merge_size);
        config
            .temporal_patch_size
            .get_or_insert(model.vision_config.temporal_patch_size);
        config.image_mean.get_or_insert(IMAGE_MEAN);
        config.image_std.get_or_insert(IMAGE_STD);
        config.resampling.get_or_insert(BICUBIC);
        if config.size.is_none() {
            config.min_pixels.get_or_insert(min_pixels);
            config.max_pixels.get_or_insert(max_pixels);
        }
    }
}

impl Processor for Lfm3VlProcessor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        self.0.inputs_processor()
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        self.0.get_special_tokens()
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::Keep
    }
}
