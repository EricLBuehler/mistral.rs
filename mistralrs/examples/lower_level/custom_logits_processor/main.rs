use either::Either;
use indexmap::IndexMap;
use rand::Rng;
use std::sync::Arc;
use tokio::sync::mpsc::channel;

use mistralrs::{
    Constraint, CustomLogitsProcessor, DefaultSchedulerMethod, Device, DeviceMapMetadata,
    MistralRs, MistralRsBuilder, ModelDType, NormalLoaderBuilder, NormalRequest,
    NormalSpecificConfig, Request, RequestMessage, ResponseOk, Result, SamplingParams,
    SchedulerConfig, Tensor, TokenSource,
};

struct ThresholdLogitsProcessor {
    threshold: f64,
}

impl CustomLogitsProcessor for ThresholdLogitsProcessor {
    fn apply(&self, logits: &Tensor, _context: &[u32]) -> Result<Tensor> {
        // Mask is 1 for true, 0 for false.
        let mask = logits.ge(self.threshold)?;
        logits.broadcast_mul(&mask.to_dtype(logits.dtype())?)
    }
}

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
        Some("mistralai/Mistral-7B-Instruct-v0.1".to_string()),
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
        None,
        None, // No PagedAttention.
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

    let mut rng = rand::thread_rng();
    let random_value: f64 = rng.gen_range(0.0..=1.0);
    let threshold: f64 = rng.gen_range(0.0..=0.5);

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
        logits_processors: Some(vec![
            Arc::new(move |logits: &Tensor, _context: &[u32]| logits * random_value),
            Arc::new(ThresholdLogitsProcessor { threshold }),
        ]),
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
