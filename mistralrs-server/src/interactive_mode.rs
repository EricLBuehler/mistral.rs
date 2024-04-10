use std::{
    io,
    sync::{mpsc::channel, Arc},
};

use either::Either;
use indexmap::IndexMap;
use mistralrs_core::{MistralRs, Request, Response, SamplingParams};

pub fn interactive_mode(mistralrs: Arc<MistralRs>) {
    let sender = mistralrs.get_sender();
    let mut messages = Vec::new();
    loop {
        let mut prompt = String::new();
        print!("> ");
        io::stdin()
            .read_line(&mut prompt)
            .expect("Failed to get input");
        let mut user_message = IndexMap::new();
        user_message.insert("role".to_string(), "user".to_string());
        user_message.insert("content".to_string(), prompt);
        messages.push(user_message);

        let (tx, rx) = channel();
        let req = Request {
            id: mistralrs.next_request_id(),
            messages: Either::Left(messages.clone()),
            sampling_params: SamplingParams {
                temperature: Some(0.1),
                top_k: Some(32),
                top_p: Some(0.1),
                top_n_logprobs: 0,
                frequency_penalty: Some(0.1),
                presence_penalty: Some(0.1),
                max_len: Some(256),
                stop_toks: None,
                logits_bias: None,
                n_choices: 1,
            },
            response: tx,
            return_logprobs: false,
            is_streaming: true,
        };
        sender.send(req).unwrap();

        let mut assistant_output = String::new();
        loop {
            if let Ok(Response::Chunk(chunk)) = rx.try_recv() {
                let choice = &chunk.choices[0];
                print!("{}", choice.delta.content);
                assistant_output.push_str(&choice.delta.content);
                if choice.stopreason.is_some() {
                    if matches!(choice.stopreason.as_ref().unwrap().as_str(), "length") {
                        print!("...");
                    }
                    break;
                }
            }
        }
        let mut assistant_message = IndexMap::new();
        assistant_message.insert("role".to_string(), "assistant".to_string());
        assistant_message.insert("content".to_string(), assistant_output);
        messages.push(assistant_message);
        println!();
    }
}
