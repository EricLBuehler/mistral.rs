use std::sync::{mpsc::channel, Arc};

use candle_core::Device;
use mistralrs::{
    Constraint, DeviceMapMetadata, GGUFLoaderBuilder, GGUFSpecificConfig, MistralRs,
    MistralRsBuilder, Request, RequestMessage, Response, SamplingParams, SchedulerMethod,
    TokenSource,
};

fn setup() -> anyhow::Result<Arc<MistralRs>> {
    // Select a Mistral model
    let loader = GGUFLoaderBuilder::new(
        GGUFSpecificConfig { repeat_last_n: 64 },
        None,
        None,
        Some("mistralai/Mistral-7B-Instruct-v0.1".to_string()),
        "TheBloke/Mistral-7B-Instruct-v0.1-GGUF".to_string(),
        "mistral-7b-instruct-v0.1.Q4_K_M.gguf".to_string(),
    )
    .build();
    // Load, into a Pipeline
    let pipeline = loader.load_model(
        None,
        TokenSource::CacheToken,
        None,
        &Device::cuda_if_available(0)?,
        false,
        DeviceMapMetadata::dummy(),
    )?;
    // Create the MistralRs, which is a runner
    Ok(MistralRsBuilder::new(pipeline, SchedulerMethod::Fixed(5.try_into().unwrap())).build())
}

fn main() -> anyhow::Result<()> {
    let mistralrs = setup()?;

    let (tx, rx) = channel();
    let request = Request {
        messages: RequestMessage::Completion {
            text: "Hello! My name is ".to_string(),
            echo_prompt: false,
            best_of: 1,
        },
        sampling_params: SamplingParams::default(),
        response: tx,
        return_logprobs: false,
        is_streaming: false,
        id: 0,
        constraint: Constraint::None,
        suffix: None,
    };
    mistralrs.get_sender().send(request)?;

    let response = rx.recv().unwrap();
    match response {
        Response::CompletionDone(c) => println!("Text: {}", c.choices[0].text),
        _ => unreachable!(),
    }
    Ok(())
}
