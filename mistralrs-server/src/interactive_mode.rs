use directories::ProjectDirs;
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
use rustyline::{error::ReadlineError, history::History, DefaultEditor, Editor, Helper};
use serde_json::Value;
use std::{
    fs,
    io::{self, Write},
    path::PathBuf,
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

fn history_file_path() -> PathBuf {
    // Replace these with your own org/app identifiers.
    let proj_dirs = ProjectDirs::from("com", "", "mistral.rs")
        .expect("Could not determine project directories");
    let config_dir = proj_dirs.config_dir();

    // Ensure the directory exists:
    fs::create_dir_all(config_dir).expect("Failed to create config directory");

    // e.g. ~/.config/MyApp/history.txt
    config_dir.join("history.txt")
}

fn read_line<H: Helper, I: History>(editor: &mut Editor<H, I>) -> String {
    let r = editor.readline("> ");
    match r {
        Err(ReadlineError::Interrupted) => {
            editor.save_history(&history_file_path()).unwrap();
            // Ctrl+C
            std::process::exit(0);
        }

        Err(ReadlineError::Eof) => {
            editor.save_history(&history_file_path()).unwrap();
            // CTRL-D
            std::process::exit(0);
        }

        Err(e) => {
            editor.save_history(&history_file_path()).unwrap();
            eprintln!("Error reading input: {:?}", e);
            std::process::exit(1);
        }
        Ok(prompt) => {
            editor.add_history_entry(prompt.clone()).unwrap();
            prompt
        }
    }
}

static CTRLC_HANDLER: Lazy<Mutex<&'static (dyn Fn() + Sync)>> =
    Lazy::new(|| Mutex::new(&exit_handler));

pub async fn interactive_mode(
    mistralrs: Arc<MistralRs>,
    do_search: bool,
    enable_thinking: Option<bool>,
) {
    match mistralrs.get_model_category() {
        ModelCategory::Text => text_interactive_mode(mistralrs, do_search, enable_thinking).await,
        ModelCategory::Vision { .. } => {
            vision_interactive_mode(mistralrs, do_search, enable_thinking).await
        }
        ModelCategory::Audio => audio_interactive_mode(mistralrs, do_search, enable_thinking).await,
        ModelCategory::Diffusion => diffusion_interactive_mode(mistralrs, do_search).await,
    }
}

const COMMAND_COMMANDS: &str = r#"
Commands:
- `\help`: Display this message.
- `\exit`: Quit interactive mode.
- `\system <system message here>`:
    Add a system message to the chat without running the model.
    Ex: `\system Always respond as a pirate.`
- `\clear`: Clear the chat history.
"#;

const TEXT_INTERACTIVE_HELP: &str = r#"
Welcome to interactive mode! Because this model is a text model, you can enter prompts and chat with the model.
"#;

const VISION_INTERACTIVE_HELP: &str = r#"
Welcome to interactive mode! Because this model is a vision model, you can enter prompts and chat with the model.

To specify a message with one or more images, simply include the image URL or path:

- `Please describe this image: path/to/image1.jpg path/to/image2.png`
- `What is in this image: <url here>`
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
const CLEAR_CMD: &str = "\\clear";

fn interactive_sample_parameters() -> SamplingParams {
    SamplingParams {
        temperature: Some(0.1),
        top_k: Some(32),
        top_p: Some(0.1),
        min_p: Some(0.05),
        top_n_logprobs: 0,
        frequency_penalty: Some(0.1),
        presence_penalty: Some(0.1),
        max_len: None,
        stop_toks: None,
        logits_bias: None,
        n_choices: 1,
        dry_params: Some(DrySamplingParams::default()),
    }
}

async fn text_interactive_mode(
    mistralrs: Arc<MistralRs>,
    do_search: bool,
    enable_thinking: Option<bool>,
) {
    let sender = mistralrs.get_sender().unwrap();
    let mut messages: Vec<IndexMap<String, MessageContent>> = Vec::new();

    let sampling_params = interactive_sample_parameters();

    info!("Starting interactive loop with sampling params: {sampling_params:?}");
    println!(
        "{}{TEXT_INTERACTIVE_HELP}{COMMAND_COMMANDS}{}",
        "=".repeat(20),
        "=".repeat(20)
    );

    // Set the handler to process exit
    *CTRLC_HANDLER.lock().unwrap() = &exit_handler;

    ctrlc::set_handler(move || CTRLC_HANDLER.lock().unwrap()())
        .expect("Failed to set CTRL-C handler for interactive mode");

    let mut rl = DefaultEditor::new().expect("Failed to open input");
    let _ = rl.load_history(&history_file_path());
    'outer: loop {
        // Set the handler to process exit
        *CTRLC_HANDLER.lock().unwrap() = &exit_handler;

        let prompt = read_line(&mut rl);

        match prompt.as_str().trim() {
            "" => continue,
            HELP_CMD => {
                println!(
                    "{}{TEXT_INTERACTIVE_HELP}{COMMAND_COMMANDS}{}",
                    "=".repeat(20),
                    "=".repeat(20)
                );
                continue;
            }
            EXIT_CMD => {
                break;
            }
            CLEAR_CMD => {
                messages.clear();
                info!("Cleared chat history.");
                continue;
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

        let request_messages = RequestMessage::Chat {
            messages: messages.clone(),
            enable_thinking,
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

    rl.save_history(&history_file_path()).unwrap();
}

fn parse_files_and_message(input: &str, regex: &Regex) -> (Vec<String>, String) {
    // Collect all URLs
    let urls: Vec<String> = regex
        .captures_iter(input)
        .filter_map(|cap| cap.get(1).map(|m| m.as_str().to_string()))
        .collect();
    // Remove the URLs from the input to get the message text
    let text = regex.replace_all(input, "").trim().to_string();
    (urls, text)
}

async fn vision_interactive_mode(
    mistralrs: Arc<MistralRs>,
    do_search: bool,
    enable_thinking: Option<bool>,
) {
    // Capture HTTP/HTTPS URLs and local file paths ending with common image extensions
    let image_regex =
        Regex::new(r#"((?:https?://|file://)?\S+\.(?:png|jpe?g|bmp|gif|webp))"#).unwrap();

    let sender = mistralrs.get_sender().unwrap();
    let mut messages: Vec<IndexMap<String, MessageContent>> = Vec::new();
    let mut images = Vec::new();

    let prefixer = match &mistralrs.config().category {
        ModelCategory::Vision {
            has_conv2d: _,
            prefixer,
        } => prefixer,
        _ => {
            panic!("`add_image_message` expects a vision model.")
        }
    };

    let sampling_params = interactive_sample_parameters();

    info!("Starting interactive loop with sampling params: {sampling_params:?}");
    println!(
        "{}{VISION_INTERACTIVE_HELP}{COMMAND_COMMANDS}{}",
        "=".repeat(20),
        "=".repeat(20)
    );

    // Set the handler to process exit
    *CTRLC_HANDLER.lock().unwrap() = &exit_handler;

    ctrlc::set_handler(move || CTRLC_HANDLER.lock().unwrap()())
        .expect("Failed to set CTRL-C handler for interactive mode");

    let mut rl = DefaultEditor::new().expect("Failed to open input");
    let _ = rl.load_history(&history_file_path());
    'outer: loop {
        // Set the handler to process exit
        *CTRLC_HANDLER.lock().unwrap() = &exit_handler;

        let prompt = read_line(&mut rl);

        match prompt.as_str().trim() {
            "" => continue,
            HELP_CMD => {
                println!(
                    "{}{VISION_INTERACTIVE_HELP}{COMMAND_COMMANDS}{}",
                    "=".repeat(20),
                    "=".repeat(20)
                );
                continue;
            }
            EXIT_CMD => {
                break;
            }
            CLEAR_CMD => {
                messages.clear();
                images.clear();
                info!("Cleared chat history.");
                continue;
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
            // Extract any image URLs and the remaining text
            _ => {
                let (urls, text) = parse_files_and_message(prompt.trim(), &image_regex);
                if !urls.is_empty() {
                    let mut image_indexes = Vec::new();
                    // Load all images first
                    for url in &urls {
                        match util::parse_image_url(url).await {
                            Ok(image) => {
                                image_indexes.push(images.len());
                                images.push(image);
                            }
                            Err(e) => {
                                error!("Failed to read image from URL/path {}: {}", url, e);
                                continue 'outer;
                            }
                        }
                    }
                    // Build a single user message with multiple images and then the text
                    let mut content_vec: Vec<IndexMap<String, Value>> = Vec::new();
                    // Add one image part per URL
                    for _ in &urls {
                        content_vec.push(IndexMap::from([(
                            "type".to_string(),
                            Value::String("image".to_string()),
                        )]));
                    }
                    // Add the text part once
                    let text = prefixer.prefix_image(image_indexes, &text);
                    content_vec.push(IndexMap::from([
                        ("type".to_string(), Value::String("text".to_string())),
                        ("text".to_string(), Value::String(text.clone())),
                    ]));
                    let mut user_message: IndexMap<String, MessageContent> = IndexMap::new();
                    user_message.insert("role".to_string(), Either::Left("user".to_string()));
                    user_message.insert("content".to_string(), Either::Right(content_vec));
                    messages.push(user_message);
                } else {
                    // Default: handle as text-only prompt
                    let mut user_message: IndexMap<String, MessageContent> = IndexMap::new();
                    user_message.insert("role".to_string(), Either::Left("user".to_string()));
                    user_message.insert(
                        "content".to_string(),
                        Either::Left(prompt.trim().to_string()),
                    );
                    messages.push(user_message);
                }
            }
        };

        // Set the handler to terminate all seqs, so allowing cancelling running
        *CTRLC_HANDLER.lock().unwrap() = &terminate_handler;

        let request_messages = RequestMessage::VisionChat {
            images: images.clone(),
            messages: messages.clone(),
            enable_thinking,
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

    rl.save_history(&history_file_path()).unwrap();
}

async fn audio_interactive_mode(
    _mistralrs: Arc<MistralRs>,
    _do_search: bool,
    _enable_thinking: Option<bool>,
) {
    unimplemented!("Using audio models interactively isn't supported yet")
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

    let mut rl = DefaultEditor::new().expect("Failed to open input");
    let _ = rl.load_history(&history_file_path());
    loop {
        // Set the handler to process exit
        *CTRLC_HANDLER.lock().unwrap() = &exit_handler;

        let prompt = read_line(&mut rl);

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

    rl.save_history(&history_file_path()).unwrap();
}
