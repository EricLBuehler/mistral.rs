use candle_core::Device;
use futures::executor::block_on;
use mistralrs::{
    Constraint, DeviceMapMetadata, MistralRs, MistralRsBuilder, NormalLoaderBuilder,
    NormalLoaderType, NormalSpecificConfig, Request, RequestMessage, Response, SamplingParams,
    SchedulerMethod, TokenSource,
};
use std::sync::Arc;
use tokio::sync::mpsc::channel;

fn setup() -> anyhow::Result<Arc<MistralRs>> {
    // Select a Mistral model
    let loader = NormalLoaderBuilder::new(
        NormalSpecificConfig {
            use_flash_attn: false,
            repeat_last_n: 64,
        },
        None,
        None,
        Some("mistralai/Mistral-7B-Instruct-v0.1".to_string()),
    )
    .build(NormalLoaderType::Mistral);
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

    let (tx, mut rx) = channel(10_000);
    let request = Request {
        messages: RequestMessage::Completion {
            text: "I like to code in the following language: ".to_string(),
            echo_prompt: false,
            best_of: 1,
        },
        sampling_params: SamplingParams::default(),
        response: tx,
        return_logprobs: false,
        is_streaming: false,
        id: 0,
        constraint: Constraint::Regex("(- [^\n]*\n)+(- [^\n]*)(\n\n)?".to_string()), // Bullet list regex
        suffix: None,
    };
    block_on(mistralrs.get_sender().send(request))?;

    let response = block_on(rx.recv()).unwrap();
    match response {
        Response::CompletionDone(c) => println!("Text: {}", c.choices[0].text),
        _ => unreachable!(),
    }
    Ok(())
}
