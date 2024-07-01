use std::sync::Arc;
use tokio::sync::mpsc::channel;

use mistralrs::{
    Constraint, Device, DeviceMapMetadata, GGUFLoaderBuilder, GGUFSpecificConfig, MistralRs,
    MistralRsBuilder, ModelDType, NormalRequest, Request, RequestMessage, Response, Result,
    SamplingParams, SchedulerMethod, TokenSource,
};

/// Gets the best device, cpu, cuda if compiled with CUDA
pub(crate) fn best_device() -> Result<Device> {
    #[cfg(not(feature = "metal"))]
    {
        Device::cuda_if_available(0)
    }
    #[cfg(feature = "metal")]
    {
        Device::new_metal(0)
    }
}

fn setup() -> anyhow::Result<Arc<MistralRs>> {
    // Select a Mistral model
    // This uses a model, tokenizer, and chat template, from HF hub.
    let loader = GGUFLoaderBuilder::new(
        GGUFSpecificConfig { repeat_last_n: 64 },
        None,
        Some("mistralai/Mistral-7B-Instruct-v0.1".to_string()),
        "TheBloke/Mistral-7B-Instruct-v0.1-GGUF".to_string(),
        "mistral-7b-instruct-v0.1.Q4_K_M.gguf".to_string(),
    )
    .build();
    // Load, into a Pipeline
    let pipeline = loader.load_model_from_hf(
        None,
        TokenSource::CacheToken,
        &ModelDType::default(),
        &best_device()?,
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
        adapters: None,
    });
    mistralrs.get_sender()?.blocking_send(request)?;

    let response = rx.blocking_recv().unwrap();
    match response {
        Response::CompletionDone(c) => println!("Text: {}", c.choices[0].text),
        _ => unreachable!(),
    }
    Ok(())
}
