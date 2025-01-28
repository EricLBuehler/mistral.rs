use anyhow::Result;
use mistralrs::{
    DrySamplingParams, Model, ModelCategory, ResponseOk, SamplingParams, TextMessageRole, TextMessages, TERMINATE_ALL_NEXT_STEP
};
use once_cell::sync::Lazy;
use std::{
    io::{self, Write},
    sync::{atomic::Ordering, Mutex},
    time::Instant,
};
use tracing::info;

use crate::util;

fn exit_handler() {
    std::process::exit(0);
}

fn terminate_handler() {
    TERMINATE_ALL_NEXT_STEP.store(true, Ordering::SeqCst);
}

static CTRLC_HANDLER: Lazy<Mutex<&'static (dyn Fn() + Sync)>> =
    Lazy::new(|| Mutex::new(&exit_handler));

pub async fn launch_interactive_mode(mistralrs: Model, throughput: bool) -> Result<()> {
    match mistralrs.get_model_category() {
        ModelCategory::Text => text_interactive_mode(mistralrs, throughput).await,
        ModelCategory::Vision { .. } => vision_interactive_mode(mistralrs, throughput).await,
        ModelCategory::Diffusion => {
            anyhow::bail!("Diffusion interactive mode is unsupported for now!")
        }
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

const HELP_CMD: &str = "\\help";
const EXIT_CMD: &str = "\\exit";
const SYSTEM_CMD: &str = "\\system";
const IMAGE_CMD: &str = "\\image";

async fn text_interactive_mode(model: Model, throughput: bool) -> Result<()> {
    let mut messages = TextMessages::new();

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
                break 'outer;
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
                messages = messages.add_message(TextMessageRole::Assistant, parsed.to_string());
                continue;
            }
            message => {
                messages = messages.add_message(TextMessageRole::User, message.to_string());
            }
        }

        // Set the handler to terminate all seqs, so allowing cancelling running
        *CTRLC_HANDLER.lock().unwrap() = &terminate_handler;

        let mut assistant_output = String::new();
        let start = Instant::now();
        let mut toks = 0;

        let mut stream = model.stream_chat_request(messages.clone()).await?;

        while let Some(chunk) = stream.next().await {
            let ResponseOk::Chunk(chunk) = chunk.as_result()? else {
                anyhow::bail!("Expected chunk response");
            };
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
        if throughput {
            let time = Instant::now().duration_since(start).as_secs_f64();
            println!();
            info!("Average T/s: {}", toks as f64 / time);
        }
        messages = messages.add_message(TextMessageRole::Assistant, assistant_output);
        println!();
    }

    Ok(())
}

async fn vision_interactive_mode(model: Model, throughput: bool) -> Result<()> {
    let mut messages = TextMessages::new();
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

    loop {
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
                messages = messages.add_message(TextMessageRole::Assistant, parsed.to_string());
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

                messages = messages.add_message(TextMessageRole::User, message.to_string());
            }
            message => {
                messages = messages.add_message(TextMessageRole::User, message.to_string());
            }
        };

        // Set the handler to terminate all seqs, so allowing cancelling running
        *CTRLC_HANDLER.lock().unwrap() = &terminate_handler;

        let mut assistant_output = String::new();
        let start = Instant::now();
        let mut toks = 0;

        let mut stream = model.stream_chat_request(messages.clone()).await?;

        while let Some(chunk) = stream.next().await {
            let ResponseOk::Chunk(chunk) = chunk.as_result()? else {
                anyhow::bail!("Expected chunk response");
            };
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
        if throughput {
            let time = Instant::now().duration_since(start).as_secs_f64();
            println!();
            info!("Average T/s: {}", toks as f64 / time);
        }
        messages = messages.add_message(TextMessageRole::Assistant, assistant_output);
        println!();
    }

    Ok(())
}
