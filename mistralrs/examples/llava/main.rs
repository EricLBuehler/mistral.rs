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
        Some("chat_templates/llava.json".to_string()),
        None,
        Some("/root/autodl-tmp/cache/huggingface/hub/models--llava-hf--llava-v1.6-vicuna-7b-hf/snapshots/0524afe4453163103dcefe78eb0a58b3f6424eac".to_string()),
    )
    .build(VisionLoaderType::LLaVANext);
    // Load, into a Pipeline

    let pipeline = loader.load_model_from_hf(
        None,
        TokenSource::CacheToken,
        &ModelDType::F16, // how can we load from config?
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
                    Either::Left("<|image_1|>what is this image show?".to_string()),
                ),
            ])],
        },
        /*
        messages: RequestMessage::Completion {
            text: "Hello! My name is ".to_string(),
            echo_prompt: false,
            best_of: 1,
        },
        */
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
        Response::InternalError(e) => println!("Internal error: {:?}", e),
        Response::ValidationError(e) => println!("Validation error: {:?}", e),
        Response::ModelError(s, r) => println!("Model error: {:?} {:?}", s, r),
        Response::Chunk(_) => println!("Chunk"),
        Response::CompletionModelError(_, _) => println!("Completion model error"),
        Response::CompletionDone(c) => println!("Text: {}", c.choices[0].text),
    }
    Ok(())
}
