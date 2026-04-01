//! Interactive mode implementation
//!
//! Ported from mistralrs-server/src/interactive_mode.rs

use directories::ProjectDirs;
use either::Either;
use indexmap::IndexMap;
use mistralrs_core::{
    speech_utils, Constraint, DiffusionGenerationParams, DrySamplingParams,
    ImageGenerationResponseFormat, MessageContent, MistralRs, ModelCategory, NormalRequest,
    Request, RequestMessage, Response, ResponseOk, SamplingParams, Usage, WebSearchOptions,
    TERMINATE_ALL_NEXT_STEP,
};
use regex::Regex;
use rustyline::{error::ReadlineError, history::History, DefaultEditor, Editor, Helper};
use serde_json::Value;
use std::{
    fs,
    io::{self, Write},
    path::PathBuf,
    sync::{atomic::Ordering, Arc, LazyLock, Mutex},
    time::Instant,
};
use tokio::sync::mpsc::{channel, Receiver};
use tracing::{error, info};

use mistralrs_server_core::util;
use mistralrs_server_core::video::parse_video_url;

fn exit_handler() {
    std::process::exit(0);
}

fn terminate_handler() {
    TERMINATE_ALL_NEXT_STEP.store(true, Ordering::SeqCst);
}

fn history_file_path() -> PathBuf {
    let proj_dirs = ProjectDirs::from("com", "", "mistral.rs")
        .expect("Could not determine project directories");
    let config_dir = proj_dirs.config_dir();

    // Ensure the directory exists:
    fs::create_dir_all(config_dir).expect("Failed to create config directory");

    // e.g. ~/.config/mistral.rs/history.txt
    config_dir.join("history.txt")
}

fn format_sampling_params(params: &SamplingParams) -> String {
    fn fmt_opt<T: std::fmt::Display>(v: &Option<T>) -> String {
        match v {
            Some(v) => v.to_string(),
            None => "off".to_string(),
        }
    }
    let mut parts = vec![
        format!("temp={}", fmt_opt(&params.temperature)),
        format!("top_k={}", fmt_opt(&params.top_k)),
        format!("top_p={}", fmt_opt(&params.top_p)),
        format!("min_p={}", fmt_opt(&params.min_p)),
    ];
    if params.frequency_penalty.is_some() {
        parts.push(format!("freq_pen={}", fmt_opt(&params.frequency_penalty)));
    }
    if params.presence_penalty.is_some() {
        parts.push(format!("pres_pen={}", fmt_opt(&params.presence_penalty)));
    }
    if params.repetition_penalty.is_some() {
        parts.push(format!("rep_pen={}", fmt_opt(&params.repetition_penalty)));
    }
    parts.join(", ")
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

static CTRLC_HANDLER: LazyLock<Mutex<&'static (dyn Fn() + Sync)>> =
    LazyLock::new(|| Mutex::new(&exit_handler));

pub async fn interactive_mode(
    mistralrs: Arc<MistralRs>,
    do_search: bool,
    enable_thinking: Option<bool>,
) {
    match mistralrs.get_model_category(None) {
        Ok(ModelCategory::Text) => {
            text_interactive_mode(mistralrs, do_search, enable_thinking).await
        }
        Ok(ModelCategory::Multimodal { .. }) => {
            multimodal_interactive_mode(mistralrs, do_search, enable_thinking).await
        }
        Ok(ModelCategory::Diffusion) => diffusion_interactive_mode(mistralrs, do_search).await,
        Ok(ModelCategory::Audio) => {
            audio_interactive_mode(mistralrs, do_search, enable_thinking).await
        }
        Ok(ModelCategory::Speech) => speech_interactive_mode(mistralrs, do_search).await,
        Ok(ModelCategory::Embedding) => error!(
            "Embedding models do not support interactive mode. Use the server or Python/Rust APIs."
        ),
        Err(e) => eprintln!("Error getting model category: {e}"),
    }
}

const COMMAND_COMMANDS: &str = r#"
Commands:
- `/help`: Display this message.
- `/exit`: Quit interactive mode.
- `/system <system message here>`:
    Add a system message to the chat without running the model.
    Ex: `/system Always respond as a pirate.`
- `/clear`: Clear the chat history.
- `/temperature <float>`: Set sampling temperature (0.0 to 2.0).
- `/topk <int>`: Set top-k sampling value (>0).
- `/topp <float>`: Set top-p sampling value in (0.0 to 1.0).
"#;

const TEXT_INTERACTIVE_HELP: &str = r#"
Welcome to interactive mode! Because this model is a text model, you can enter prompts and chat with the model.
"#;

const VISION_INTERACTIVE_HELP: &str = r#"
Welcome to interactive mode! Because this model is a multimodal model, you can enter prompts and chat with the model.

To specify a message with one or more images, audios, or videos, simply include the image/audio/video URL or path:

- `Describe these images: path/to/image1.jpg path/to/image2.png`
- `Describe the image and transcribe the audio: path/to/image1.jpg path/to/sound.mp3`
- `Describe this video: path/to/video.mp4`
"#;

const DIFFUSION_INTERACTIVE_HELP: &str = r#"
Welcome to interactive mode! Because this model is a diffusion model, you can enter prompts and the model will generate an image.

Commands:
- `/help`: Display this message.
- `/exit`: Quit interactive mode.
"#;

const SPEECH_INTERACTIVE_HELP: &str = r#"
Welcome to interactive mode! Because this model is a speech generation model, you can enter prompts and the model will generate audio.

Commands:
- `/help`: Display this message.
- `/exit`: Quit interactive mode.
"#;

const HELP_CMD: &str = "/help";
const EXIT_CMD: &str = "/exit";
const SYSTEM_CMD: &str = "/system";
const CLEAR_CMD: &str = "/clear";
const TEMPERATURE_CMD: &str = "/temperature";
const TOPK_CMD: &str = "/topk";
const TOPP_CMD: &str = "/topp";

/// Regex string used to extract image URLs from prompts.
const IMAGE_REGEX: &str = r#"((?:https?://|file://)?\S+?\.(?:png|jpe?g|bmp|gif|webp)(?:\?\S+?)?)"#;
const AUDIO_REGEX: &str = r#"((?:https?://|file://)?\S+?\.(?:wav|mp3|flac|ogg)(?:\?\S+?)?)"#;
const VIDEO_REGEX: &str = r#"((?:https?://|file://)?\S+?\.(?:mp4|avi|mov|mkv|webm|gif|m4v)(?:\?\S+?)?)"#;

fn interactive_fallback_sample_parameters() -> SamplingParams {
    SamplingParams {
        temperature: Some(0.1),
        top_k: Some(32),
        top_p: Some(0.1),
        min_p: Some(0.05),
        top_n_logprobs: 0,
        frequency_penalty: None,
        presence_penalty: None,
        repetition_penalty: None,
        max_len: None,
        stop_toks: None,
        logits_bias: None,
        n_choices: 1,
        dry_params: Some(DrySamplingParams::default()),
    }
}

fn interactive_sample_parameters(mistralrs: &Arc<MistralRs>) -> SamplingParams {
    match mistralrs
        .config(None)
        .ok()
        .and_then(|cfg| cfg.generation_defaults)
    {
        Some(defaults) => {
            let mut params = SamplingParams {
                dry_params: Some(DrySamplingParams::default()),
                ..SamplingParams::neutral()
            };
            params.apply_model_defaults(&defaults);
            params
        }
        None => interactive_fallback_sample_parameters(),
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
                Ok(v) if (0.0..=2.0).contains(&v) => {
                    sampling_params.temperature = if v == 0.0 { None } else { Some(v) };
                    info!("Set temperature to {v}");
                }
                Ok(_) => {
                    println!("Error: temperature must be in [0.0, 2.0]");
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

    let mut sampling_params = interactive_sample_parameters(&mistralrs);

    info!("Starting interactive loop with sampling params: {sampling_params:?}");
    println!(
        "{}{TEXT_INTERACTIVE_HELP}{COMMAND_COMMANDS}\nSampling: {}\n{}",
        "=".repeat(20),
        format_sampling_params(&sampling_params),
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
            reasoning_effort: None,
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
            truncate_sequence: false,
        }));
        sender.send(req).await.unwrap();
        let start_ttft = Instant::now();
        let (assistant_output, first_token_duration, last_usage) =
            match stream_assistant_response(&mut rx, start_ttft).await {
                Ok(response) => response,
                Err(e) => {
                    error!("{e}");
                    break 'outer;
                }
            };

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
            if let Ok(logger) = mistralrs.get_logger(None) {
                let (prefix_hits, prefix_total) = logger.prefix_cache_stats();
                if prefix_total > 0 {
                    println!(
                        "Prefix cache: {} hits / {} turns",
                        prefix_hits, prefix_total
                    );
                }
            }
            println!("Sampling: {}", format_sampling_params(&sampling_params));
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

async fn stream_assistant_response(
    rx: &mut Receiver<Response>,
    start_ttft: Instant,
) -> Result<(String, Option<std::time::Duration>, Option<Usage>), String> {
    let mut assistant_output = String::new();
    let mut first_token_duration = None;
    let mut last_usage = None;

    const GRAY: &str = "\x1b[90m";
    const RESET: &str = "\x1b[0m";
    let mut was_reasoning = false;

    while let Some(resp) = rx.recv().await {
        match resp {
            Response::Chunk(chunk) => {
                last_usage = chunk.usage.clone();
                let choice = &chunk.choices[0];

                let has_any_content =
                    choice.delta.content.is_some() || choice.delta.reasoning_content.is_some();
                if has_any_content && first_token_duration.is_none() {
                    first_token_duration = Some(Instant::now().duration_since(start_ttft));
                }

                if let Some(ref reasoning) = choice.delta.reasoning_content {
                    print!("{GRAY}{reasoning}{RESET}");
                    io::stdout().flush().unwrap();
                    was_reasoning = true;
                }

                if let Some(ref content) = choice.delta.content {
                    if was_reasoning {
                        println!();
                        was_reasoning = false;
                    }
                    assistant_output.push_str(content);
                    print!("{content}");
                    io::stdout().flush().unwrap();
                }

                if let Some(ref finish_reason) = choice.finish_reason {
                    if was_reasoning {
                        println!();
                    }
                    if matches!(finish_reason.as_str(), "length") {
                        print!("...");
                    }
                    break;
                }
            }
            Response::InternalError(e) => return Err(format!("Got an internal error: {e:?}")),
            Response::ModelError(e, resp) => {
                return Err(format!("Got a model error: {e:?}, response: {resp:?}"));
            }
            Response::ValidationError(e) => return Err(format!("Got a validation error: {e:?}")),
            Response::Done(_) => unreachable!(),
            Response::CompletionDone(_) => unreachable!(),
            Response::CompletionModelError(_, _) => unreachable!(),
            Response::CompletionChunk(_) => unreachable!(),
            Response::ImageGeneration(_) => unreachable!(),
            Response::Speech { .. } => unreachable!(),
            Response::Raw { .. } => unreachable!(),
            Response::Embeddings { .. } => unreachable!(),
        }
    }

    Ok((assistant_output, first_token_duration, last_usage))
}

async fn multimodal_interactive_mode(
    mistralrs: Arc<MistralRs>,
    do_search: bool,
    enable_thinking: Option<bool>,
) {
    // Capture HTTP/HTTPS URLs and local file paths ending with common image extensions
    let image_regex = Regex::new(IMAGE_REGEX).unwrap();
    let audio_regex = Regex::new(AUDIO_REGEX).unwrap();
    let video_regex = Regex::new(VIDEO_REGEX).unwrap();

    let sender = mistralrs.get_sender(None).unwrap();
    let mut messages: Vec<IndexMap<String, MessageContent>> = Vec::new();
    let mut images = Vec::new();
    let mut audios = Vec::new();
    let mut videos = Vec::new();

    let config = mistralrs.config(None).unwrap();
    let prefixer = match &config.category {
        ModelCategory::Multimodal { prefixer } => prefixer,
        _ => {
            panic!("`add_image_message` expects a multimodal model.")
        }
    };

    let mut sampling_params = interactive_sample_parameters(&mistralrs);
    let mut prev_encoder_hits: usize = 0;
    let mut prev_encoder_misses: usize = 0;

    info!("Starting interactive loop with sampling params: {sampling_params:?}");
    println!(
        "{}{VISION_INTERACTIVE_HELP}{COMMAND_COMMANDS}\nSampling: {}\n{}",
        "=".repeat(20),
        format_sampling_params(&sampling_params),
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
                audios.clear();
                videos.clear();
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
                let (urls_audio, text_without_audios) =
                    parse_files_and_message(&text_without_images, &audio_regex);
                let (urls_video, text) =
                    parse_files_and_message(&text_without_audios, &video_regex);
                if !urls_image.is_empty() || !urls_audio.is_empty() || !urls_video.is_empty() {
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
                    // Load audios and retain earlier turns so multimodal history can be
                    // replayed with stable audio indices and matching payloads.
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
                    // Load videos
                    let mut video_indexes = Vec::new();
                    for url in &urls_video {
                        match parse_video_url(url, None).await {
                            Ok(video) => {
                                info!("Added video at `{url}`");
                                video_indexes.push(videos.len());
                                videos.push(video);
                            }
                            Err(e) => {
                                error!("Failed to read video from URL/path {}: {}", url, e);
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
                    for _ in &urls_video {
                        content_vec.push(IndexMap::from([(
                            "type".to_string(),
                            Value::String("video".to_string()),
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
                    if !video_indexes.is_empty() {
                        prefixed_text =
                            prefixer.prefix_video(video_indexes.clone(), &prefixed_text);
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

        let request_messages = RequestMessage::MultimodalChat {
            images: images.clone(),
            audios: audios.clone(),
            videos: videos.clone(),
            messages: messages.clone(),
            enable_thinking,
            reasoning_effort: None,
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
            truncate_sequence: false,
        }));
        sender.send(req).await.unwrap();
        let start_ttft = Instant::now();
        let (assistant_output, first_token_duration, last_usage) =
            match stream_assistant_response(&mut rx, start_ttft).await {
                Ok(response) => response,
                Err(e) => {
                    error!("{e}");
                    break 'outer;
                }
            };

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
            if let Ok(logger) = mistralrs.get_logger(None) {
                let (prefix_hits, prefix_total) = logger.prefix_cache_stats();
                if prefix_total > 0 {
                    println!(
                        "Prefix cache: {} hits / {} turns",
                        prefix_hits, prefix_total
                    );
                }
                if let Some((hits, misses)) = logger.encoder_cache_stats() {
                    let turn_hits = hits - prev_encoder_hits;
                    let turn_lookups = (hits + misses) - (prev_encoder_hits + prev_encoder_misses);
                    if turn_lookups > 0 {
                        println!("Encoder cache: {}/{} hits", turn_hits, turn_lookups);
                    }
                    prev_encoder_hits = hits;
                    prev_encoder_misses = misses;
                }
            }
            println!("Sampling: {}", format_sampling_params(&sampling_params));
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
                save_file: None,
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
            truncate_sequence: false,
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
            truncate_sequence: false,
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
