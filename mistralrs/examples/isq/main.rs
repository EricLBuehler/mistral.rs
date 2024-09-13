use either::Either;
use indexmap::IndexMap;
use std::sync::Arc;
use tokio::sync::mpsc::channel;

use mistralrs::{
    Constraint, DefaultSchedulerMethod, Device, DeviceMapMetadata, IsqType, MistralRs,
    MistralRsBuilder, ModelDType, NormalLoaderBuilder, NormalRequest, NormalSpecificConfig,
    Request, RequestMessage, ResponseOk, Result, SamplingParams, SchedulerConfig, TokenSource,
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
    let loader = NormalLoaderBuilder::new(
        NormalSpecificConfig {
            use_flash_attn: false,
            prompt_batchsize: None,
            topology: None,
            organization: Default::default(),
            write_uqff: None,
            from_uqff: None,
        },
        None,
        None,
        Some("microsoft/Phi-3.5-mini-instruct".to_string()),
    )
    .build(None)?;
    // Load, into a Pipeline
    let pipeline = loader.load_model_from_hf(
        None,
        TokenSource::CacheToken,
        &ModelDType::Auto,
        &best_device()?,
        false,
        DeviceMapMetadata::dummy(),
        Some(IsqType::Q4K), // In-situ quantize the model into q4k
        None,               // No PagedAttention.
    )?;
    // Create the MistralRs, which is a runner
    Ok(MistralRsBuilder::new(
        pipeline,
        SchedulerConfig::DefaultScheduler {
            method: DefaultSchedulerMethod::Fixed(5.try_into().unwrap()),
        },
    )
    .build())
}

fn main() -> anyhow::Result<()> {
    let mistralrs = setup()?;

    let (tx, mut rx) = channel(10_000);
    let request = Request::Normal(NormalRequest {
        messages: RequestMessage::Chat(vec![IndexMap::from([
            ("role".to_string(), Either::Left("user".to_string())),
            ("content".to_string(), Either::Left("Hello!".to_string())),
        ])]),
        sampling_params: SamplingParams::default(),
        response: tx,
        return_logprobs: false,
        is_streaming: false,
        id: 0,
        constraint: Constraint::None,
        suffix: None,
        adapters: None,
        tools: None,
        tool_choice: None,
        logits_processors: None,
    });
    mistralrs.get_sender()?.blocking_send(request)?;

    let response = rx.blocking_recv().unwrap().as_result().unwrap();
    match response {
        ResponseOk::Done(c) => println!(
            "Text: {}, Prompt T/s: {}, Completion T/s: {}",
            c.choices[0].message.content.as_ref().unwrap(),
            c.usage.avg_prompt_tok_per_sec,
            c.usage.avg_compl_tok_per_sec
        ),
        _ => unreachable!(),
    }
    Ok(())
}
