use either::Either;
use indexmap::IndexMap;
use mistralrs_core::{
    ChunkChoice, Constraint, Delta, DiffusionGenerationParams, DrySamplingParams,
    ImageGenerationResponseFormat, MessageContent, MistralRs, ModelCategory, NormalRequest,
    Request, RequestMessage, Response, ResponseOk, SamplingParams, WebSearchOptions,
    TERMINATE_ALL_NEXT_STEP,
};
use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::Value;
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

pub async fn interactive_mode(mistralrs: Arc<MistralRs>, do_search: bool) {
    match mistralrs.get_model_category() {
        ModelCategory::Text => text_interactive_mode(mistralrs, do_search).await,
        ModelCategory::Vision { .. } => vision_interactive_mode(mistralrs, do_search).await,
        ModelCategory::Diffusion => diffusion_interactive_mode(mistralrs, do_search).await,
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
    Add a message paired with an image. The image will be fed to the model as if it were the first item in this prompt.
    You do not need to modify your prompt for specific models.
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

async fn text_interactive_mode(mistralrs: Arc<MistralRs>, do_search: bool) {
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
            tool_choice: None,
            tools: None,
            logits_processors: None,
            return_raw_logits: false,
            web_search_options: do_search.then(WebSearchOptions::default),
        });
        sender.send(req).await.unwrap();

        let mut assistant_output = String::new();

        let mut last_usage = None;
        while let Some(resp) = rx.recv().await {
            match resp {
                Response::Chunk(chunk) => {
                    last_usage = chunk.usage.clone();
                    if let ChunkChoice {
                        delta:
                            Delta {
                                content: Some(content),
                                ..
                            },
                        finish_reason,
                        ..
                    } = &chunk.choices[0]
                    {
                        assistant_output.push_str(content);
                        print!("{}", content);
                        io::stdout().flush().unwrap();
                        if finish_reason.is_some() {
                            if matches!(finish_reason.as_ref().unwrap().as_str(), "length") {
                                print!("...");
                            }
                            break;
                        }
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
                Response::Raw { .. } => unreachable!(),
            }
        }

        if let Some(last_usage) = last_usage {
            println!();
            println!();
            println!("Stats:");
            println!(
                "Prompt: {} tokens, {:.2} T/s",
                last_usage.prompt_tokens, last_usage.avg_prompt_tok_per_sec
            );
            println!(
                "Decode: {} tokens, {:.2} T/s",
                last_usage.completion_tokens, last_usage.avg_compl_tok_per_sec
            );
        }
        let mut assistant_message: IndexMap<String, Either<String, Vec<IndexMap<String, Value>>>> =
            IndexMap::new();
        assistant_message.insert("role".to_string(), Either::Left("assistant".to_string()));
        assistant_message.insert("content".to_string(), Either::Left(assistant_output));
        messages.push(assistant_message);
        println!();
    }
}

fn parse_image_path_and_message(input: &str) -> Option<(String, String)> {
    // Regex to capture the image path and the following message
    let re = Regex::new(r#"\\image\s+"([^"]+)"\s*(.*)|\\image\s+(\S+)\s*(.*)"#).unwrap();

    if let Some(captures) = re.captures(input) {
        // Capture either the quoted or unquoted path and the message
        if let Some(path) = captures.get(1) {
            if let Some(message) = captures.get(2) {
                return Some((
                    path.as_str().trim().to_string(),
                    message.as_str().trim().to_string(),
                ));
            }
        } else if let Some(path) = captures.get(3) {
            if let Some(message) = captures.get(4) {
                return Some((
                    path.as_str().trim().to_string(),
                    message.as_str().trim().to_string(),
                ));
            }
        }
    }

    None
}

async fn vision_interactive_mode(mistralrs: Arc<MistralRs>, do_search: bool) {
    let sender = mistralrs.get_sender().unwrap();
    let mut messages: Vec<IndexMap<String, MessageContent>> = Vec::new();
    let mut images = Vec::new();

    let prefixer = match &mistralrs.config().category {
        ModelCategory::Text | ModelCategory::Diffusion => {
            panic!("`add_image_message` expects a vision model.")
        }
        ModelCategory::Vision {
            has_conv2d: _,
            prefixer,
        } => prefixer,
    };

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
                let Some((url, message)) = parse_image_path_and_message(prompt.trim()) else {
                    println!("Error: Adding an image message should be done with this format: `{IMAGE_CMD} path/to/image.jpg Describe what is in this image.`");
                    continue;
                };
                let message = prefixer.prefix_image(images.len(), &message);

                let image = util::parse_image_url(&url)
                    .await
                    .expect("Failed to read image from URL/path");
                images.push(image);

                let mut user_message: IndexMap<String, MessageContent> = IndexMap::new();
                user_message.insert("role".to_string(), Either::Left("user".to_string()));
                user_message.insert(
                    "content".to_string(),
                    Either::Right(vec![
                        IndexMap::from([("type".to_string(), Value::String("image".to_string()))]),
                        IndexMap::from([
                            ("type".to_string(), Value::String("text".to_string())),
                            ("text".to_string(), Value::String(message)),
                        ]),
                    ]),
                );
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
            tool_choice: None,
            tools: None,
            logits_processors: None,
            return_raw_logits: false,
            web_search_options: do_search.then(WebSearchOptions::default),
        });
        sender.send(req).await.unwrap();

        let mut assistant_output = String::new();

        let mut last_usage = None;
        while let Some(resp) = rx.recv().await {
            match resp {
                Response::Chunk(chunk) => {
                    last_usage = chunk.usage.clone();
                    if let ChunkChoice {
                        delta:
                            Delta {
                                content: Some(content),
                                ..
                            },
                        finish_reason,
                        ..
                    } = &chunk.choices[0]
                    {
                        assistant_output.push_str(content);
                        print!("{}", content);
                        io::stdout().flush().unwrap();
                        if finish_reason.is_some() {
                            if matches!(finish_reason.as_ref().unwrap().as_str(), "length") {
                                print!("...");
                            }
                            break;
                        }
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
                Response::Raw { .. } => unreachable!(),
            }
        }

        if let Some(last_usage) = last_usage {
            println!();
            println!();
            println!("Stats:");
            println!(
                "Prompt: {} tokens, {:.2} T/s",
                last_usage.prompt_tokens, last_usage.avg_prompt_tok_per_sec
            );
            println!(
                "Decode: {} tokens, {:.2} T/s",
                last_usage.completion_tokens, last_usage.avg_compl_tok_per_sec
            );
        }
        let mut assistant_message: IndexMap<String, Either<String, Vec<IndexMap<String, Value>>>> =
            IndexMap::new();
        assistant_message.insert("role".to_string(), Either::Left("assistant".to_string()));
        assistant_message.insert("content".to_string(), Either::Left(assistant_output));
        messages.push(assistant_message);
        println!();
    }
}

async fn diffusion_interactive_mode(mistralrs: Arc<MistralRs>, do_search: bool) {
    let sender = mistralrs.get_sender().unwrap();

    let diffusion_params = DiffusionGenerationParams::default();

    info!("Starting interactive loop with generation params: {diffusion_params:?}");
    println!(
        "{}{DIFFUSION_INTERACTIVE_HELP}{}",
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
            tool_choice: None,
            tools: None,
            logits_processors: None,
            return_raw_logits: false,
            web_search_options: do_search.then(WebSearchOptions::default),
        });

        let start = Instant::now();
        sender.send(req).await.unwrap();

        let ResponseOk::ImageGeneration(response) = rx.recv().await.unwrap().as_result().unwrap()
        else {
            panic!("Got unexpected response type.")
        };
        let end = Instant::now();

        let duration = end.duration_since(start).as_secs_f32();
        let pixels_per_s = (diffusion_params.height * diffusion_params.width) as f32 / duration;

        println!(
            "Image generated can be found at: image is at `{}`. Took {duration:.2}s ({pixels_per_s:.2} pixels/s).",
            response.data[0].url.as_ref().unwrap(),
        );

        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::parse_image_path_and_message;

    #[test]
    fn test_parse_image_with_unquoted_path_and_message() {
        let input = r#"\image image.jpg What is this"#;
        let result = parse_image_path_and_message(input);
        assert_eq!(
            result,
            Some(("image.jpg".to_string(), "What is this".to_string()))
        );
    }

    #[test]
    fn test_parse_image_with_quoted_path_and_message() {
        let input = r#"\image "image name.jpg" What is this?"#;
        let result = parse_image_path_and_message(input);
        assert_eq!(
            result,
            Some(("image name.jpg".to_string(), "What is this?".to_string()))
        );
    }

    #[test]
    fn test_parse_image_with_only_unquoted_path() {
        let input = r#"\image image.jpg"#;
        let result = parse_image_path_and_message(input);
        assert_eq!(result, Some(("image.jpg".to_string(), "".to_string())));
    }

    #[test]
    fn test_parse_image_with_only_quoted_path() {
        let input = r#"\image "image name.jpg""#;
        let result = parse_image_path_and_message(input);
        assert_eq!(result, Some(("image name.jpg".to_string(), "".to_string())));
    }

    #[test]
    fn test_parse_image_with_extra_spaces() {
        let input = r#"\image    "image with spaces.jpg"    This is a test message with spaces  "#;
        let result = parse_image_path_and_message(input);
        assert_eq!(
            result,
            Some((
                "image with spaces.jpg".to_string(),
                "This is a test message with spaces".to_string()
            ))
        );
    }

    #[test]
    fn test_parse_image_with_no_message() {
        let input = r#"\image "image.jpg""#;
        let result = parse_image_path_and_message(input);
        assert_eq!(result, Some(("image.jpg".to_string(), "".to_string())));
    }

    #[test]
    fn test_parse_image_missing_path() {
        let input = r#"\image"#;
        let result = parse_image_path_and_message(input);
        assert_eq!(result, None);
    }

    #[test]
    fn test_parse_image_invalid_command() {
        let input = r#"\img "image.jpg" This should fail"#;
        let result = parse_image_path_and_message(input);
        assert_eq!(result, None);
    }

    #[test]
    fn test_parse_image_with_non_image_text() {
        let input = r#"Some random text without command"#;
        let result = parse_image_path_and_message(input);
        assert_eq!(result, None);
    }

    #[test]
    fn test_parse_image_with_path_and_message_special_chars() {
        let input = r#"\image "path with special chars @#$%^&*().jpg" This is a message with special chars !@#$%^&*()"#;
        let result = parse_image_path_and_message(input);
        assert_eq!(
            result,
            Some((
                "path with special chars @#$%^&*().jpg".to_string(),
                "This is a message with special chars !@#$%^&*()".to_string()
            ))
        );
    }
}
