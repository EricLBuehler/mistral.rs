use either::Either;
use indexmap::IndexMap;
use std::sync::Arc;
use tokio::sync::mpsc::channel;

use mistralrs::{
    initialize_logging, ChatCompletionResponse, Constraint, Device, DeviceMapMetadata,
    GGUFLoaderBuilder, GGUFSpecificConfig, MemoryGpuConfig, MistralRs, MistralRsBuilder,
    ModelDType, NormalRequest, PagedAttentionConfig, Request, RequestMessage, ResponseOk,
    SamplingParams, SchedulerConfig, TokenSource, Usage,
};

async fn setup() -> anyhow::Result<Arc<MistralRs>> {
    // Select a Mistral model
    // This uses a model, tokenizer, and chat template, from HF hub.
    let loader = GGUFLoaderBuilder::new(
        None,
        Some("mistralai/Mistral-7B-Instruct-v0.1".to_string()),
        "TheBloke/Mistral-7B-Instruct-v0.1-GGUF".to_string(),
        vec!["mistral-7b-instruct-v0.1.Q4_K_M.gguf".to_string()],
        GGUFSpecificConfig {
            prompt_chunksize: None,
            topology: None,
        },
    )
    .build();
    // Load, into a Pipeline
    let pipeline = loader.load_model_from_hf(
        None,
        TokenSource::CacheToken,
        &ModelDType::default(),
        &Device::cuda_if_available(0)?,
        false,
        DeviceMapMetadata::dummy(),
        None,
        Some(PagedAttentionConfig::new(
            Some(32),
            500,
            MemoryGpuConfig::Utilization(0.9),
        )?), // No PagedAttention.
    )?;
    let config = pipeline
        .lock()
        .await
        .get_metadata()
        .cache_config
        .as_ref()
        .unwrap()
        .clone();
    // Create the MistralRs, which is a runner
    Ok(MistralRsBuilder::new(
        pipeline,
        SchedulerConfig::PagedAttentionMeta {
            max_num_seqs: 500,
            config,
        },
    )
    .with_throughput_logging()
    .build())
}

async fn bench_mistralrs(n_requests: usize) -> anyhow::Result<()> {
    initialize_logging();
    let mistralrs = setup().await?;

    let mut handles = Vec::new();
    for _ in 0..n_requests {
        let (tx, rx) = channel(10_000);
        let request = Request::Normal(NormalRequest {
            messages: RequestMessage::Chat(vec![IndexMap::from([
                ("role".to_string(), Either::Left("user".to_string())),
                (
                    "content".to_string(),
                    Either::Left("What is graphene".to_string()),
                ),
            ])]),
            sampling_params: SamplingParams::default(),
            response: tx,
            return_logprobs: false,
            is_streaming: false,
            id: 0,
            constraint: Constraint::None,
            suffix: None,
            tools: None,
            tool_choice: None,
            logits_processors: None,
        });
        mistralrs.get_sender()?.send(request).await?;
        handles.push(rx);
    }

    let responses = futures::future::join_all(handles.iter_mut().map(|x| x.recv())).await;

    let mut max_prompt = f32::MIN;
    let mut max_completion = f32::MIN;

    for response in responses {
        let ResponseOk::Done(ChatCompletionResponse {
            usage:
                Usage {
                    avg_compl_tok_per_sec,
                    avg_prompt_tok_per_sec,
                    ..
                },
            ..
        }) = response.unwrap().as_result().unwrap()
        else {
            unreachable!()
        };
        dbg!(avg_compl_tok_per_sec, avg_prompt_tok_per_sec);
        if avg_compl_tok_per_sec > max_prompt {
            max_prompt = avg_prompt_tok_per_sec;
        }
        if avg_compl_tok_per_sec > max_completion {
            max_completion = avg_compl_tok_per_sec;
        }
    }
    println!("Individual sequence stats: {max_prompt} max PP T/s, {max_completion} max TG T/s");

    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    bench_mistralrs(10).await
}
