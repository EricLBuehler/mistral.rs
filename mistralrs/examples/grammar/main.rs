use std::sync::{mpsc::channel, Arc};

use candle_core::Device;
use mistralrs::{
    Constraint, LoaderBuilder, ModelSelected, Request, RequestMessage, Response, SamplingParams,
    SchedulerMethod, TokenSource,
};
use mistralrs_core::{MistralRs, MistralRsBuilder};

fn setup() -> anyhow::Result<Arc<MistralRs>> {
    // Select a Mistral model
    let selected = ModelSelected::Mistral {
        model_id: "mistralai/Mistral-7B-Instruct-v0.1".to_string(),
        tokenizer_json: None,
        repeat_last_n: 64,
    };
    // Create a loader
    let loader = LoaderBuilder::new(selected)
        .with_chat_template(None)
        .with_no_kv_cache(false)
        .with_use_flash_attn(false)
        .build()?;
    // Load, into a Pipeline
    let pipeline = loader.load_model(
        None,
        TokenSource::CacheToken,
        None,
        &Device::cuda_if_available(0)?,
    )?;
    // Create the MistralRs, which is a runner
    Ok(MistralRsBuilder::new(pipeline, SchedulerMethod::Fixed(5.try_into().unwrap())).build())
}

fn main() -> anyhow::Result<()> {
    let mistralrs = setup()?;

    let (tx, rx) = channel();
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
    mistralrs.get_sender().send(request)?;

    let response = rx.recv().unwrap();
    match response {
        Response::CompletionDone(c) => println!("Text: {}", c.choices[0].text),
        _ => unreachable!(),
    }
    Ok(())
}
