use either::Either;
use indexmap::IndexMap;
use mistralrs_core::{
    Constraint, DiffusionGenerationParams, DrySamplingParams, ImageGenerationResponseFormat,
    MessageContent, MistralRs, ModelCategory, NormalRequest, Request, RequestMessage, Response,
    ResponseOk, SamplingParams, TERMINATE_ALL_NEXT_STEP,
};
use once_cell::sync::Lazy;
use std::{
    io::{self, Write},
    sync::{atomic::Ordering, Arc, Mutex},
    time::Instant,
};
use tokio::sync::mpsc::channel;
use tracing::{error, info};

use crate::util;

fn exit_handler() {
    std::process::exit(0);
}

fn terminate_handler() {
    TERMINATE_ALL_NEXT_STEP.store(true, Ordering::SeqCst);
}

static CTRLC_HANDLER: Lazy<Mutex<&'static (dyn Fn() + Sync)>> =
    Lazy::new(|| Mutex::new(&exit_handler));

pub async fn interactive_mode(mistralrs: Arc<MistralRs>, throughput: bool) {
    match mistralrs.get_model_category() {
        ModelCategory::Text => text_interactive_mode(mistralrs, throughput).await,
        ModelCategory::Vision { .. } => vision_interactive_mode(mistralrs, throughput).await,
        ModelCategory::Diffusion => diffusion_interactive_mode(mistralrs).await,
    }
}

const TEXT_INTERACTIVE_HELP: &str = r#"
Welcome to interactive mode! Because this model is a text model, you can enter prompts and chat with the model.

Commands:
- `\help`: Display this message.
- `\exit`: Quit interactive mode.
- `\system <system message here>`:
    Add a system message to the chat without running the model.
    Ex: `\system Always respond as a pirate.`
"#;

const VISION_INTERACTIVE_HELP: &str = r#"
Welcome to interactive mode! Because this model is a vision model, you can enter prompts and chat with the model.

To specify a message with an image, use the `\image` command detailed below.

Commands:
- `\help`: Display this message.
- `\exit`: Quit interactive mode.
- `\system <system message here>`:
    Add a system message to the chat without running the model.
    Ex: `\system Always respond as a pirate.`
- `\image <image URL or local path here> <message here>`: 
    Add a message paired with an image. You are responsible for prefixing the message with anything the model
    requires.
    Ex: `\image path/to/image.jpg Describe what is in this image.`
"#;

const DIFFUSION_INTERACTIVE_HELP: &str = r#"
Welcome to interactive mode! Because this model is a diffusion model, you can enter prompts and the model will generate an image.

Commands:
- `\help`: Display this message.
- `\exit`: Quit interactive mode.
"#;

const HELP_CMD: &str = "\\help";
const EXIT_CMD: &str = "\\exit";
const SYSTEM_CMD: &str = "\\system";
const IMAGE_CMD: &str = "\\image";

async fn text_interactive_mode(mistralrs: Arc<MistralRs>, throughput: bool) {
    let sender = mistralrs.get_sender().unwrap();
    let mut messages: Vec<IndexMap<String, MessageContent>> = Vec::new();

    let sampling_params = SamplingParams {
        temperature: Some(0.1),
        top_k: Some(32),
        top_p: Some(0.1),
        min_p: Some(0.05),
        top_n_logprobs: 0,
        frequency_penalty: Some(0.1),
        presence_penalty: Some(0.1),
        max_len: Some(4096),
        stop_toks: None,
        logits_bias: None,
        n_choices: 1,
        dry_params: Some(DrySamplingParams::default()),
    };

    info!("Starting interactive loop with sampling params: {sampling_params:?}");
    println!(
        "{}{TEXT_INTERACTIVE_HELP}{}",
        "=".repeat(20),
        "=".repeat(20)
    );

    // Set the handler to process exit
    *CTRLC_HANDLER.lock().unwrap() = &exit_handler;

    ctrlc::set_handler(move || CTRLC_HANDLER.lock().unwrap()())
        .expect("Failed to set CTRL-C handler for interactive mode");

    'outer: loop {
        // Set the handler to process exit
        *CTRLC_HANDLER.lock().unwrap() = &exit_handler;

        let mut prompt = String::new();
        print!("> ");
        io::stdout().flush().unwrap();
        io::stdin()
            .read_line(&mut prompt)
            .expect("Failed to get input");

        match prompt.as_str().trim() {
            "" => continue,
            HELP_CMD => {
                println!(
                    "{}{TEXT_INTERACTIVE_HELP}{}",
                    "=".repeat(20),
                    "=".repeat(20)
                );
                continue;
            }
            EXIT_CMD => {
                break;
            }
            prompt if prompt.trim().starts_with(SYSTEM_CMD) => {
                let parsed = match &prompt.split(SYSTEM_CMD).collect::<Vec<_>>()[..] {
                    &["", a] => a.trim(),
                    _ => {
                        println!("Error: Setting the system command should be done with this format: `{SYSTEM_CMD} This is a system message.`");
                        continue;
                    }
                };
                info!("Set system message to `{parsed}`.");
                let mut user_message: IndexMap<String, MessageContent> = IndexMap::new();
                user_message.insert("role".to_string(), Either::Left("system".to_string()));
                user_message.insert("content".to_string(), Either::Left(parsed.to_string()));
                messages.push(user_message);
                continue;
            }
            message => {
                let mut user_message: IndexMap<String, MessageContent> = IndexMap::new();
                user_message.insert("role".to_string(), Either::Left("user".to_string()));
                user_message.insert("content".to_string(), Either::Left(message.to_string()));
                messages.push(user_message);
            }
        }

        // Set the handler to terminate all seqs, so allowing cancelling running
        *CTRLC_HANDLER.lock().unwrap() = &terminate_handler;

        let request_messages = RequestMessage::Chat(messages.clone());

        let (tx, mut rx) = channel(10_000);
        let req = Request::Normal(NormalRequest {
            id: mistralrs.next_request_id(),
            messages: request_messages,
            sampling_params: sampling_params.clone(),
            response: tx,
            return_logprobs: false,
            is_streaming: true,
            constraint: Constraint::None,
            suffix: None,
            adapters: None,
            tool_choice: None,
            tools: None,
            logits_processors: None,
        });
        sender.send(req).await.unwrap();

        let mut assistant_output = String::new();

        let start = Instant::now();
        let mut toks = 0;
        while let Some(resp) = rx.recv().await {
            match resp {
                Response::Chunk(chunk) => {
                    let choice = &chunk.choices[0];
                    assistant_output.push_str(&choice.delta.content);
                    print!("{}", choice.delta.content);
                    toks += 3usize; // NOTE: we send toks every 3.
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
                Response::CompletionChunk(_) => unreachable!(),
                Response::ImageGeneration(_) => unreachable!(),
            }
        }
        if throughput {
            let time = Instant::now().duration_since(start).as_secs_f64();
            println!();
            info!("Average T/s: {}", toks as f64 / time);
        }
        let mut assistant_message: IndexMap<String, Either<String, Vec<IndexMap<String, String>>>> =
            IndexMap::new();
        assistant_message.insert("role".to_string(), Either::Left("assistant".to_string()));
        assistant_message.insert("content".to_string(), Either::Left(assistant_output));
        messages.push(assistant_message);
        println!();
    }
}

async fn vision_interactive_mode(mistralrs: Arc<MistralRs>, throughput: bool) {
    let sender = mistralrs.get_sender().unwrap();
    let mut messages: Vec<IndexMap<String, MessageContent>> = Vec::new();
    let mut images = Vec::new();

    let sampling_params = SamplingParams {
        temperature: Some(0.1),
        top_k: Some(32),
        top_p: Some(0.1),
        min_p: Some(0.05),
        top_n_logprobs: 0,
        frequency_penalty: Some(0.1),
        presence_penalty: Some(0.1),
        max_len: Some(4096),
        stop_toks: None,
        logits_bias: None,
        n_choices: 1,
        dry_params: Some(DrySamplingParams::default()),
    };

    info!("Starting interactive loop with sampling params: {sampling_params:?}");
    println!(
        "{}{VISION_INTERACTIVE_HELP}{}",
        "=".repeat(20),
        "=".repeat(20)
    );

    // Set the handler to process exit
    *CTRLC_HANDLER.lock().unwrap() = &exit_handler;

    ctrlc::set_handler(move || CTRLC_HANDLER.lock().unwrap()())
        .expect("Failed to set CTRL-C handler for interactive mode");

    'outer: loop {
        // Set the handler to process exit
        *CTRLC_HANDLER.lock().unwrap() = &exit_handler;

        let mut prompt = String::new();
        print!("> ");
        io::stdout().flush().unwrap();
        io::stdin()
            .read_line(&mut prompt)
            .expect("Failed to get input");

        match prompt.as_str().trim() {
            "" => continue,
            HELP_CMD => {
                println!(
                    "{}{VISION_INTERACTIVE_HELP}{}",
                    "=".repeat(20),
                    "=".repeat(20)
                );
                continue;
            }
            EXIT_CMD => {
                break;
            }
            prompt if prompt.trim().starts_with(SYSTEM_CMD) => {
                let parsed = match &prompt.split(SYSTEM_CMD).collect::<Vec<_>>()[..] {
                    &["", a] => a.trim(),
                    _ => {
                        println!("Error: Setting the system command should be done with this format: `{SYSTEM_CMD} This is a system message.`");
                        continue;
                    }
                };
                info!("Set system message to `{parsed}`.");
                let mut user_message: IndexMap<String, MessageContent> = IndexMap::new();
                user_message.insert("role".to_string(), Either::Left("system".to_string()));
                user_message.insert("content".to_string(), Either::Left(parsed.to_string()));
                messages.push(user_message);
                continue;
            }
            prompt if prompt.trim().starts_with(IMAGE_CMD) => {
                let mut parts = prompt.trim().strip_prefix(IMAGE_CMD).unwrap().split(' ');
                // No space??
                if !parts.next().unwrap().is_empty() {
                    println!("Error: Adding an image message should be done with this format: `{IMAGE_CMD} path/to/image.jpg Describe what is in this image.`");
                }
                let url = match parts.next() {
                    Some(p) => p.trim(),
                    None => {
                        println!("Error: Adding an image message should be done with this format: `{IMAGE_CMD} path/to/image.jpg Describe what is in this image.`");
                        continue;
                    }
                };
                let message = parts.collect::<Vec<_>>().join(" ");

                let image = util::parse_image_url(url)
                    .await
                    .expect("Failed to read image from URL/path");
                images.push(image);

                let mut user_message: IndexMap<String, MessageContent> = IndexMap::new();
                user_message.insert("role".to_string(), Either::Left("user".to_string()));
                user_message.insert("content".to_string(), Either::Left(message));
                messages.push(user_message);
            }
            message => {
                let mut user_message: IndexMap<String, MessageContent> = IndexMap::new();
                user_message.insert("role".to_string(), Either::Left("user".to_string()));
                user_message.insert("content".to_string(), Either::Left(message.to_string()));
                messages.push(user_message);
            }
        };

        // Set the handler to terminate all seqs, so allowing cancelling running
        *CTRLC_HANDLER.lock().unwrap() = &terminate_handler;

        let request_messages = RequestMessage::VisionChat {
            images: images.clone(),
            messages: messages.clone(),
        };

        let (tx, mut rx) = channel(10_000);
        let req = Request::Normal(NormalRequest {
            id: mistralrs.next_request_id(),
            messages: request_messages,
            sampling_params: sampling_params.clone(),
            response: tx,
            return_logprobs: false,
            is_streaming: true,
            constraint: Constraint::None,
            suffix: None,
            adapters: None,
            tool_choice: None,
            tools: None,
            logits_processors: None,
        });
        sender.send(req).await.unwrap();

        let mut assistant_output = String::new();

        let start = Instant::now();
        let mut toks = 0;
        while let Some(resp) = rx.recv().await {
            match resp {
                Response::Chunk(chunk) => {
                    let choice = &chunk.choices[0];
                    assistant_output.push_str(&choice.delta.content);
                    print!("{}", choice.delta.content);
                    toks += 3usize; // NOTE: we send toks every 3.
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
                Response::CompletionChunk(_) => unreachable!(),
                Response::ImageGeneration(_) => unreachable!(),
            }
        }
        if throughput {
            let time = Instant::now().duration_since(start).as_secs_f64();
            println!();
            info!("Average T/s: {}", toks as f64 / time);
        }
        let mut assistant_message: IndexMap<String, Either<String, Vec<IndexMap<String, String>>>> =
            IndexMap::new();
        assistant_message.insert("role".to_string(), Either::Left("assistant".to_string()));
        assistant_message.insert("content".to_string(), Either::Left(assistant_output));
        messages.push(assistant_message);
        println!();
    }
}

async fn diffusion_interactive_mode(mistralrs: Arc<MistralRs>) {
    let sender = mistralrs.get_sender().unwrap();

    let diffusion_params = DiffusionGenerationParams::default();

    info!("Starting interactive loop with generation params: {diffusion_params:?}");
    println!(
        "{}{TEXT_INTERACTIVE_HELP}{}",
        "=".repeat(20),
        "=".repeat(20)
    );

    // Set the handler to process exit
    *CTRLC_HANDLER.lock().unwrap() = &exit_handler;

    ctrlc::set_handler(move || CTRLC_HANDLER.lock().unwrap()())
        .expect("Failed to set CTRL-C handler for interactive mode");

    loop {
        // Set the handler to process exit
        *CTRLC_HANDLER.lock().unwrap() = &exit_handler;

        let mut prompt = String::new();
        print!("> ");
        io::stdout().flush().unwrap();
        io::stdin()
            .read_line(&mut prompt)
            .expect("Failed to get input");

        let prompt = match prompt.as_str().trim() {
            "" => continue,
            HELP_CMD => {
                println!(
                    "{}{DIFFUSION_INTERACTIVE_HELP}{}",
                    "=".repeat(20),
                    "=".repeat(20)
                );
                continue;
            }
            EXIT_CMD => {
                break;
            }
            prompt => prompt.to_string(),
        };

        // Set the handler to terminate all seqs, so allowing cancelling running
        *CTRLC_HANDLER.lock().unwrap() = &terminate_handler;

        let (tx, mut rx) = channel(10_000);
        let req = Request::Normal(NormalRequest {
            id: 0,
            messages: RequestMessage::ImageGeneration {
                prompt: prompt.to_string(),
                format: ImageGenerationResponseFormat::Url,
                generation_params: diffusion_params.clone(),
            },
            sampling_params: SamplingParams::deterministic(),
            response: tx,
            return_logprobs: false,
            is_streaming: false,
            suffix: None,
            constraint: Constraint::None,
            adapters: None,
            tool_choice: None,
            tools: None,
            logits_processors: None,
        });
        sender.send(req).await.unwrap();

        let ResponseOk::ImageGeneration(response) = rx.recv().await.unwrap().as_result().unwrap()
        else {
            panic!("Got unexpected response type.")
        };

        println!(
            "Image generated can be found at: image is at {}",
            response.data[0].url.as_ref().unwrap()
        );

        println!();
    }
}
