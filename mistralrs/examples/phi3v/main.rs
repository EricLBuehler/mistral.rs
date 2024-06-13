use either::Either;
use image::{ColorType, DynamicImage};
use indexmap::IndexMap;
use std::sync::Arc;
use tokio::sync::mpsc::channel;

use mistralrs::{
    Constraint, Device, DeviceMapMetadata, MistralRs, MistralRsBuilder, ModelDType, NormalRequest,
    Request, RequestMessage, Response, SamplingParams, SchedulerMethod, TokenSource,
    VisionLoaderBuilder, VisionLoaderType, VisionSpecificConfig,
};

fn setup() -> anyhow::Result<Arc<MistralRs>> {
    // Select a Mistral model
    let loader = VisionLoaderBuilder::new(
        VisionSpecificConfig {
            use_flash_attn: false,
            repeat_last_n: 64,
        },
        None,
        None,
        Some("microsoft/Phi-3-vision-128k-instruct".to_string()),
    )
    .build(VisionLoaderType::Phi3V);
    // Load, into a Pipeline
    let pipeline = loader.load_model_from_hf(
        None,
        TokenSource::CacheToken,
        &ModelDType::Auto,
        &Device::cuda_if_available(0)?,
        false,
        DeviceMapMetadata::dummy(),
        None,
    )?;
    // Create the MistralRs, which is a runner
    Ok(MistralRsBuilder::new(pipeline, SchedulerMethod::Fixed(5.try_into().unwrap())).build())
}

fn main() -> anyhow::Result<()> {
    let mistralrs = setup()?;

    let (tx, mut rx) = channel(10_000);
    let request = Request::Normal(NormalRequest {
        messages: RequestMessage::VisionChat {
            images: vec![DynamicImage::new(1280, 720, ColorType::Rgb8)],
            messages: vec![IndexMap::from([
                ("role".to_string(), Either::Left("user".to_string())),
                (
                    "content".to_string(),
                    Either::Left("<|image_1|>\nWhat is shown in this image?".to_string()),
                ),
            ])],
        },
        sampling_params: SamplingParams::default(),
        response: tx,
        return_logprobs: false,
        is_streaming: false,
        id: 0,
        constraint: Constraint::None,
        suffix: None,
        adapters: None,
    });
    mistralrs.get_sender()?.blocking_send(request)?;

    let response = rx.blocking_recv().unwrap();
    match response {
        Response::Done(c) => println!("Text: {}", c.choices[0].message.content),
        _ => unreachable!(),
    }
    Ok(())
}
