use std::sync::{mpsc::channel, Arc};

use either::Either;
use indexmap::IndexMap;
use mistralrs_core::{Constraint, MistralRs, Request, RequestType, Response, SamplingParams};
use tracing::{error, info};

pub fn prompt_mode(
    mistralrs: Arc<MistralRs>,
    prompt: String,
    prompt_concurrency: usize,
    prompt_max_tokens: usize,
) {
    let sender = mistralrs.get_sender();
    let mut messages = Vec::new();

    let sampling_params = SamplingParams {
        temperature: Some(0.1),
        top_k: Some(32),
        top_p: Some(0.1),
        top_n_logprobs: 0,
        frequency_penalty: Some(0.1),
        presence_penalty: Some(0.1),
        max_len: Some(prompt_max_tokens),
        stop_toks: None,
        logits_bias: None,
        n_choices: 1,
    };
    info!("Running the prompt `{prompt}`, concurrency of {prompt_concurrency} and with sampling params: {sampling_params:?}");

    let mut user_message = IndexMap::new();
    user_message.insert("role".to_string(), "user".to_string());
    user_message.insert("content".to_string(), prompt);
    messages.push(user_message);

    let (tx, rx) = channel();
    let req = Request {
        id: mistralrs.next_request_id(),
        messages: Either::Left(messages.clone()),
        sampling_params: sampling_params.clone(),
        response: tx,
        return_logprobs: false,
        is_streaming: false,
        constraint: Constraint::None,
        request_type: RequestType::Chat,
        suffix: None,
        best_of: None,
    };
    for _ in 0..prompt_concurrency {
        sender.send(req.clone()).unwrap();
    }

    let mut completion_tokens = 0;
    let mut prompt_tokens = 0;

    let mut total_time_sec = 0.0;
    let mut total_prompt_time_sec = 0.0;

    for _ in 0..prompt_concurrency {
        let resp = rx.recv();
        if let Ok(resp) = resp {
            match resp {
                Response::InternalError(e) => {
                    error!("Got an internal error: {e:?}");
                }
                Response::ModelError(e, resp) => {
                    error!("Got a model error: {e:?}, response: {resp:?}");
                }
                Response::ValidationError(e) => {
                    error!("Got a validation error: {e:?}");
                }
                Response::Done(res) => {
                    println!("{}", res.choices[0].message.content);
                    println!("=======================");

                    completion_tokens += res.usage.completion_tokens;
                    prompt_tokens += res.usage.prompt_tokens;
                    total_time_sec += res.usage.total_time_sec;
                    total_prompt_time_sec += res.usage.total_prompt_time_sec;
                }
                Response::Chunk(_) => unreachable!(),
                Response::CompletionDone(_) => unreachable!(),
                Response::CompletionModelError(_, _) => unreachable!(),
            }
        }
    }

    let total_completion_time_sec = total_time_sec - total_prompt_time_sec;

    let avg_compl_tok_per_sec =
        completion_tokens as f32 / (total_completion_time_sec / prompt_concurrency as f32);
    let avg_prompt_tok_per_sec =
        prompt_tokens as f32 / (total_prompt_time_sec / prompt_concurrency as f32);

    println!("Prompt T/s = {}", avg_prompt_tok_per_sec);
    println!("Completion T/s = {}", avg_compl_tok_per_sec);
}
