use indexmap::IndexMap;
use mistralrs_core::{
    Constraint, MistralRs, NormalRequest, Request, RequestMessage, Response, SamplingParams,
};
use std::{
    io::{self, Write},
    sync::Arc,
};
use tokio::sync::mpsc::channel;
use tracing::{error, info};

pub async fn interactive_mode(mistralrs: Arc<MistralRs>) {
    let sender = mistralrs.get_sender();
    let mut messages = Vec::new();

    let sampling_params = SamplingParams {
        temperature: Some(0.1),
        top_k: Some(32),
        top_p: Some(0.1),
        top_n_logprobs: 0,
        frequency_penalty: Some(0.1),
        presence_penalty: Some(0.1),
        max_len: Some(4096),
        stop_toks: None,
        logits_bias: None,
        n_choices: 1,
    };
    info!("Starting interactive loop with sampling params: {sampling_params:?}");
    'outer: loop {
        let mut prompt = String::new();
        print!("> ");
        io::stdout().flush().unwrap();
        io::stdin()
            .read_line(&mut prompt)
            .expect("Failed to get input");
        if prompt.is_empty() {
            return;
        }
        let mut user_message = IndexMap::new();
        user_message.insert("role".to_string(), "user".to_string());
        user_message.insert("content".to_string(), prompt);
        messages.push(user_message);

        let (tx, mut rx) = channel(10_000);
        let req = Request::Normal(NormalRequest {
            id: mistralrs.next_request_id(),
            messages: RequestMessage::Chat(messages.clone()),
            sampling_params: sampling_params.clone(),
            response: tx,
            return_logprobs: false,
            is_streaming: true,
            constraint: Constraint::None,
            suffix: None,
        });
        sender.send(req).await.unwrap();

        let mut assistant_output = String::new();

        while let Some(resp) = rx.recv().await {
            match resp {
                Response::Chunk(chunk) => {
                    let choice = &chunk.choices[0];
                    assistant_output.push_str(&choice.delta.content);
                    print!("{}", choice.delta.content);
                    io::stdout().flush().unwrap();
                    if choice.finish_reason.is_some() {
                        if matches!(choice.finish_reason.as_ref().unwrap().as_str(), "length") {
                            print!("...");
                        }
                        break;
                    }
                }
                Response::InternalError(e) => {
                    error!("Got an internal error: {e:?}");
                    break 'outer;
                }
                Response::ModelError(e, resp) => {
                    error!("Got a model error: {e:?}, response: {resp:?}");
                    break 'outer;
                }
                Response::ValidationError(e) => {
                    error!("Got a validation error: {e:?}");
                    break 'outer;
                }
                Response::Done(_) => unreachable!(),
                Response::CompletionDone(_) => unreachable!(),
                Response::CompletionModelError(_, _) => unreachable!(),
            }
        }
        let mut assistant_message = IndexMap::new();
        assistant_message.insert("role".to_string(), "assistant".to_string());
        assistant_message.insert("content".to_string(), assistant_output);
        messages.push(assistant_message);
        println!();
    }
}
