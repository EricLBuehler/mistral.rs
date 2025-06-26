use directories::ProjectDirs;
use either::Either;
use indexmap::IndexMap;
use mistralrs_core::{
    speech_utils, ChunkChoice, Constraint, Delta, DiffusionGenerationParams, DrySamplingParams,
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

use mistralrs_server_core::util;

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
            eprintln!("Error reading input: {e:?}");
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
    match mistralrs.get_model_category(None) {
        Ok(ModelCategory::Text) => {
            text_interactive_mode(mistralrs, do_search, enable_thinking).await
        }
        Ok(ModelCategory::Vision { .. }) => {
            vision_interactive_mode(mistralrs, do_search, enable_thinking).await
        }
        Ok(ModelCategory::Diffusion) => diffusion_interactive_mode(mistralrs, do_search).await,
        Ok(ModelCategory::Audio) => {
            audio_interactive_mode(mistralrs, do_search, enable_thinking).await
        }
        Ok(ModelCategory::Speech) => speech_interactive_mode(mistralrs, do_search).await,
        Err(e) => eprintln!("Error getting model category: {e}"),
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
- `\temperature <float>`: Set sampling temperature (0.0 to 2.0).
- `\topk <int>`: Set top-k sampling value (>0).
- `\topp <float>`: Set top-p sampling value in (0.0 to 1.0).
"#;

const TEXT_INTERACTIVE_HELP: &str = r#"
Welcome to interactive mode! Because this model is a text model, you can enter prompts and chat with the model.
"#;

const VISION_INTERACTIVE_HELP: &str = r#"
Welcome to interactive mode! Because this model is a vision model, you can enter prompts and chat with the model.

To specify a message with one or more images or audios, simply include the image/audio URL or path:

- `Describe these images: path/to/image1.jpg path/to/image2.png`
- `Describe the image and transcribe the audio: path/to/image1.jpg path/to/sound.mp3`
"#;

const DIFFUSION_INTERACTIVE_HELP: &str = r#"
Welcome to interactive mode! Because this model is a diffusion model, you can enter prompts and the model will generate an image.

Commands:
- `\help`: Display this message.
- `\exit`: Quit interactive mode.
"#;

const SPEECH_INTERACTIVE_HELP: &str = r#"
Welcome to interactive mode! Because this model is a speech generation model, you can enter prompts and the model will generate audio.

Commands:
- `\help`: Display this message.
- `\exit`: Quit interactive mode.
"#;

const HELP_CMD: &str = "\\help";
const EXIT_CMD: &str = "\\exit";
const SYSTEM_CMD: &str = "\\system";
const CLEAR_CMD: &str = "\\clear";
const TEMPERATURE_CMD: &str = "\\temperature";
const TOPK_CMD: &str = "\\topk";
const TOPP_CMD: &str = "\\topp";

/// Regex string used to extract image URLs from prompts.
const IMAGE_REGEX: &str = r#"((?:https?://|file://)?\S+?\.(?:png|jpe?g|bmp|gif|webp)(?:\?\S+?)?)"#;
const AUDIO_REGEX: &str = r#"((?:https?://|file://)?\S+?\.(?:wav|mp3|flac|ogg)(?:\?\S+?)?)"#;

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

/// Handles sampling commands (\temperature, \topk, \topp) and updates the sampling_params accordingly.
/// Returns true if the prompt was a handled sampling command, otherwise false.
fn handle_sampling_command(prompt: &str, sampling_params: &mut SamplingParams) -> bool {
    let trimmed = prompt.trim();
    if trimmed.starts_with(TEMPERATURE_CMD) {
        let parts: Vec<&str> = trimmed.splitn(2, ' ').collect();
        if let [_, value] = parts.as_slice() {
            match value.trim().parse::<f64>() {
                Ok(v) if v > 0.0 && v <= 2.0 => {
                    sampling_params.temperature = Some(v);
                    info!("Set temperature to {v}");
                }
                Ok(_) => {
                    println!("Error: temperature must be in (0.0, 2.0]");
                }
                Err(_) => println!("Error: format is `{TEMPERATURE_CMD} <float>`"),
            }
        } else {
            println!("Error: format is `{TEMPERATURE_CMD} <float>`");
        }
        return true;
    }
    if trimmed.starts_with(TOPK_CMD) {
        let parts: Vec<&str> = trimmed.splitn(2, ' ').collect();
        if let [_, value] = parts.as_slice() {
            match value.trim().parse::<usize>() {
                Ok(v) if v > 0 => {
                    sampling_params.top_k = Some(v);
                    info!("Set top-k to {v}");
                }
                Ok(_) => {
                    println!("Error: top-k must be a positive integer");
                }
                Err(_) => println!("Error: format is `{TOPK_CMD} <int>`"),
            }
        } else {
            println!("Error: format is `{TOPK_CMD} <int>`");
        }
        return true;
    }
    if trimmed.starts_with(TOPP_CMD) {
        let parts: Vec<&str> = trimmed.splitn(2, ' ').collect();
        if let [_, value] = parts.as_slice() {
            match value.trim().parse::<f64>() {
                Ok(v) if v > 0.0 && v <= 1.0 => {
                    sampling_params.top_p = Some(v);
                    info!("Set top-p to {v}");
                }
                Ok(_) => {
                    println!("Error: top-p must be in (0.0, 1.0]");
                }
                Err(_) => println!("Error: format is `{TOPP_CMD} <float>`"),
            }
        } else {
            println!("Error: format is `{TOPP_CMD} <float>`");
        }
        return true;
    }
    false
}

async fn text_interactive_mode(
    mistralrs: Arc<MistralRs>,
    do_search: bool,
    enable_thinking: Option<bool>,
) {
    let sender = mistralrs.get_sender(None).unwrap();
    let mut messages: Vec<IndexMap<String, MessageContent>> = Vec::new();

    let mut sampling_params = interactive_sample_parameters();

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

        let prompt_trimmed = prompt.as_str().trim();
        if prompt_trimmed.is_empty() {
            continue;
        }
        if handle_sampling_command(prompt_trimmed, &mut sampling_params) {
            continue;
        }
        match prompt_trimmed {
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
            _ if prompt_trimmed.starts_with(SYSTEM_CMD) => {
                let parsed = match &prompt_trimmed.split(SYSTEM_CMD).collect::<Vec<_>>()[..] {
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
        let req = Request::Normal(Box::new(NormalRequest {
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
            model_id: None,
        }));
        sender.send(req).await.unwrap();
        let start_ttft = Instant::now();
        let mut first_token_duration: Option<std::time::Duration> = None;

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
                        if first_token_duration.is_none() {
                            let ttft = Instant::now().duration_since(start_ttft);
                            first_token_duration = Some(ttft);
                        }
                        assistant_output.push_str(content);
                        print!("{content}");
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
                Response::Speech { .. } => unreachable!(),
                Response::Raw { .. } => unreachable!(),
            }
        }

        if let Some(last_usage) = last_usage {
            println!();
            println!();
            println!("Stats:");
            if let Some(ttft) = first_token_duration {
                println!("Time to first token: {:.2?}s", ttft.as_secs_f32());
            }
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
        .filter_map(|cap| {
            cap.get(1).map(|m| {
                m.as_str()
                    .trim_end_matches(|c: char| {
                        matches!(
                            c,
                            '.' | ',' | ';' | ':' | '!' | '?' | ')' | ']' | '}' | '"' | '\''
                        )
                    })
                    .to_string()
            })
        })
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
    let image_regex = Regex::new(IMAGE_REGEX).unwrap();
    let audio_regex = Regex::new(AUDIO_REGEX).unwrap();

    let sender = mistralrs.get_sender(None).unwrap();
    let mut messages: Vec<IndexMap<String, MessageContent>> = Vec::new();
    let mut images = Vec::new();
    let mut audios = Vec::new();

    let config = mistralrs.config(None).unwrap();
    let prefixer = match &config.category {
        ModelCategory::Vision { prefixer } => prefixer,
        ModelCategory::Text
        | ModelCategory::Diffusion
        | ModelCategory::Speech
        | ModelCategory::Audio => {
            panic!("`add_image_message` expects a vision model.")
        }
    };

    let mut sampling_params = interactive_sample_parameters();

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

        let prompt_trimmed = prompt.as_str().trim();
        if prompt_trimmed.is_empty() {
            continue;
        }
        if handle_sampling_command(prompt_trimmed, &mut sampling_params) {
            continue;
        }
        match prompt_trimmed {
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
            _ if prompt_trimmed.starts_with(SYSTEM_CMD) => {
                let parsed = match &prompt_trimmed.split(SYSTEM_CMD).collect::<Vec<_>>()[..] {
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
            _ => {
                let (urls_image, text_without_images) =
                    parse_files_and_message(prompt_trimmed, &image_regex);
                let (urls_audio, text) =
                    parse_files_and_message(&text_without_images, &audio_regex);
                if !urls_image.is_empty() || !urls_audio.is_empty() {
                    // Load images
                    let mut image_indexes = Vec::new();
                    for url in &urls_image {
                        match util::parse_image_url(url).await {
                            Ok(image) => {
                                info!("Added image at `{url}`");
                                image_indexes.push(images.len());
                                images.push(image);
                            }
                            Err(e) => {
                                error!("Failed to read image from URL/path {}: {}", url, e);
                                continue 'outer;
                            }
                        }
                    }
                    // Load audios
                    let mut audio_indexes = Vec::new();
                    for url in &urls_audio {
                        match util::parse_audio_url(url).await {
                            Ok(audio) => {
                                info!("Added audio at `{url}`");
                                audio_indexes.push(audios.len());
                                audios.push(audio);
                            }
                            Err(e) => {
                                error!("Failed to read audio from URL/path {}: {}", url, e);
                                continue 'outer;
                            }
                        }
                    }
                    // Build mixed content parts
                    let mut content_vec: Vec<IndexMap<String, Value>> = Vec::new();
                    for _ in &urls_image {
                        content_vec.push(IndexMap::from([(
                            "type".to_string(),
                            Value::String("image".to_string()),
                        )]));
                    }
                    for _ in &urls_audio {
                        content_vec.push(IndexMap::from([(
                            "type".to_string(),
                            Value::String("audio".to_string()),
                        )]));
                    }
                    // Prefix the text with any media context
                    let mut prefixed_text = text.clone();
                    if !image_indexes.is_empty() {
                        prefixed_text =
                            prefixer.prefix_image(image_indexes.clone(), &prefixed_text);
                    }
                    if !audio_indexes.is_empty() {
                        prefixed_text =
                            prefixer.prefix_audio(audio_indexes.clone(), &prefixed_text);
                    }
                    // Add the final text part
                    content_vec.push(IndexMap::from([
                        ("type".to_string(), Value::String("text".to_string())),
                        ("text".to_string(), Value::String(prefixed_text)),
                    ]));
                    // Push the combined user message
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
                        Either::Left(prompt_trimmed.to_string()),
                    );
                    messages.push(user_message);
                }
            }
        }

        // Set the handler to terminate all seqs, so allowing cancelling running
        *CTRLC_HANDLER.lock().unwrap() = &terminate_handler;

        let request_messages = RequestMessage::VisionChat {
            images: images.clone(),
            audios: audios.clone(),
            messages: messages.clone(),
            enable_thinking,
        };

        let (tx, mut rx) = channel(10_000);
        let req = Request::Normal(Box::new(NormalRequest {
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
            model_id: None,
        }));
        sender.send(req).await.unwrap();
        let start_ttft = Instant::now();
        let mut first_token_duration: Option<std::time::Duration> = None;

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
                        if first_token_duration.is_none() {
                            let ttft = Instant::now().duration_since(start_ttft);
                            first_token_duration = Some(ttft);
                        }
                        assistant_output.push_str(content);
                        print!("{content}");
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
                Response::Speech { .. } => unreachable!(),
                Response::Raw { .. } => unreachable!(),
            }
        }

        if let Some(last_usage) = last_usage {
            println!();
            println!();
            println!("Stats:");
            if let Some(ttft) = first_token_duration {
                println!("Time to first token: {:.2?}s", ttft.as_secs_f32());
            }
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
    unimplemented!("Using audio models isn't supported yet")
}

async fn diffusion_interactive_mode(mistralrs: Arc<MistralRs>, do_search: bool) {
    let sender = mistralrs.get_sender(None).unwrap();

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
        let req = Request::Normal(Box::new(NormalRequest {
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
            model_id: None,
        }));

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

async fn speech_interactive_mode(mistralrs: Arc<MistralRs>, do_search: bool) {
    let sender = mistralrs.get_sender(None).unwrap();

    info!("Starting interactive loop for speech");
    println!(
        "{}{SPEECH_INTERACTIVE_HELP}{}",
        "=".repeat(20),
        "=".repeat(20)
    );

    // Set the handler to process exit
    *CTRLC_HANDLER.lock().unwrap() = &exit_handler;

    ctrlc::set_handler(move || CTRLC_HANDLER.lock().unwrap()())
        .expect("Failed to set CTRL-C handler for interactive mode");

    let mut rl = DefaultEditor::new().expect("Failed to open input");
    let _ = rl.load_history(&history_file_path());

    let mut n = 0;
    loop {
        // Set the handler to process exit
        *CTRLC_HANDLER.lock().unwrap() = &exit_handler;

        let prompt = read_line(&mut rl);

        let prompt = match prompt.as_str().trim() {
            "" => continue,
            HELP_CMD => {
                println!(
                    "{}{SPEECH_INTERACTIVE_HELP}{}",
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
        let req = Request::Normal(Box::new(NormalRequest {
            id: 0,
            messages: RequestMessage::SpeechGeneration {
                prompt: prompt.to_string(),
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
            model_id: None,
        }));

        let start = Instant::now();
        sender.send(req).await.unwrap();

        let ResponseOk::Speech {
            pcm,
            rate,
            channels,
        } = rx.recv().await.unwrap().as_result().unwrap()
        else {
            panic!("Got unexpected response type.")
        };
        let end = Instant::now();

        let out_file = format!("speech-{n}.wav");
        let mut output = std::fs::File::create(&out_file).unwrap();
        speech_utils::write_pcm_as_wav(&mut output, &pcm, rate as u32, channels as u16).unwrap();

        let duration = end.duration_since(start).as_secs_f32();
        println!("Speech generated can be found at `{out_file}`. Took {duration:.2}s.");

        n += 1;

        println!();
    }

    rl.save_history(&history_file_path()).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_files_and_message_trims_trailing_punctuation() {
        let regex = Regex::new(IMAGE_REGEX).unwrap();
        let input = "Look at this https://example.com/test.png.";
        let (urls, text) = parse_files_and_message(input, &regex);
        assert_eq!(urls, vec!["https://example.com/test.png"]);
        assert_eq!(text, "Look at this .");
    }
}
