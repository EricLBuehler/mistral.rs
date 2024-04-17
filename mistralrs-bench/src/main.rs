use candle_core::Device;
use clap::Parser;
use cli_table::{format::Justify, print_stdout, Cell, CellStruct, Style, Table};
use mistralrs_core::{
    Constraint, Loader, MistralLoader, MistralRs, MistralRsConfig, MistralSpecificConfig,
    ModelKind, Request, RequestMessage, Response, SamplingParams, SchedulerMethod, TokenSource,
    Usage,
};
use std::sync::Arc;
use std::{fmt::Display, sync::mpsc::channel};
use tracing::{info, warn};

enum TestName {
    Prompt(usize),
    Gen(usize),
}

impl Display for TestName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            TestName::Prompt(n) => format!("pp {}", n),
            TestName::Gen(n) => format!("tg {}", n),
        };
        write!(f, "{}", name)
    }
}

struct BenchResult {
    usages: Vec<Usage>,
    batch_size: usize,
    test_name: TestName,
}

fn run_bench(
    mistralrs: Arc<MistralRs>,
    prompt: RequestMessage,
    n_gen: usize,
    batch_size: usize,
    repetitions: usize,
    test_name: TestName,
) -> anyhow::Result<BenchResult> {
    let sampling_params = SamplingParams {
        temperature: Some(0.1),
        top_k: Some(32),
        top_p: Some(0.1),
        top_n_logprobs: 0,
        frequency_penalty: Some(0.1),
        presence_penalty: Some(0.1),
        max_len: Some(n_gen),
        stop_toks: None,
        logits_bias: None,
        n_choices: 1,
    };
    let sender = mistralrs.get_sender();
    let (tx, rx) = channel();

    let req = Request {
        id: mistralrs.next_request_id(),
        messages: prompt,
        sampling_params: sampling_params.clone(),
        response: tx,
        return_logprobs: false,
        is_streaming: false,
        constraint: Constraint::None,
        suffix: None,
    };

    let mut usages = Vec::new();

    for _ in 0..repetitions {
        for _ in 0..batch_size {
            sender.send(req.clone()).unwrap();
        }
        for _ in 0..batch_size {
            match rx.recv() {
                Ok(r) => match r {
                    Response::InternalError(e) => {
                        unreachable!("Got an internal error: {e:?}");
                    }
                    Response::ModelError(e, resp) => {
                        unreachable!("Got a model error: {e:?}, response: {resp:?}");
                    }
                    Response::ValidationError(e) => {
                        unreachable!("Got a validation error: {e:?}");
                    }
                    Response::Done(res) => {
                        usages.push(res.usage);
                    }
                    Response::Chunk(_) => unreachable!(),
                    Response::CompletionModelError(_, _) => unreachable!(),
                    Response::CompletionDone(res) => {
                        usages.push(res.usage);
                    }
                },
                Err(e) => unreachable!("Expected a Done response, got: {:?}", e),
            }
        }
    }

    Ok(BenchResult {
        usages,
        batch_size,
        test_name,
    })
}

fn get_tok_s(result: &BenchResult) -> f32 {
    match result.test_name {
        TestName::Prompt(_) => {
            // let tokens = result.usages.iter().map(|u| u.prompt_tokens).sum::<usize>();
            // let time = result
            //     .usages
            //     .iter()
            //     .map(|u| u.total_prompt_time_sec)
            //     .sum::<f32>();

            // tokens as f32 / time
            let sum_of_avg = result
                .usages
                .iter()
                .map(|u| u.avg_prompt_tok_per_sec)
                .sum::<f32>();
            sum_of_avg / result.usages.len() as f32
        }
        TestName::Gen(_) => {
            // let tokens = result
            //     .usages
            //     .iter()
            //     .map(|u| u.completion_tokens)
            //     .sum::<usize>();
            // let time = result
            //     .usages
            //     .iter()
            //     .map(|u| u.total_completion_time_sec)
            //     .sum::<f32>();

            // tokens as f32 / time

            let sum_of_avg = result
                .usages
                .iter()
                .map(|u| u.avg_compl_tok_per_sec)
                .sum::<f32>();

            sum_of_avg / result.usages.len() as f32
        }
    }
}

fn print_usage(model: &str, device: &Device, results: Vec<BenchResult>) {
    let backend = match device {
        Device::Cpu => "CPU",
        Device::Cuda(_) => "CUDA",
        Device::Metal(_) => "Metal",
    };
    let results: Vec<Vec<CellStruct>> = results
        .into_iter()
        .map(|r| {
            vec![
                model.cell(),
                backend.cell(),
                r.batch_size.cell().justify(Justify::Right),
                r.test_name.to_string().cell(),
                get_tok_s(&r).cell().justify(Justify::Right),
            ]
        })
        .collect();

    let table = results
        .table()
        .title(vec![
            "model".cell().bold(true),
            // "size".cell().bold(true),
            // "params".cell().bold(true),
            "backend".cell().bold(true),
            // "ngl".cell().bold(true),
            "n_batch".cell().bold(true),
            "test".cell().bold(true),
            "t/s".cell().bold(true),
        ])
        .bold(true);
    print_stdout(table).expect("print table");
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long, short = 'p', default_value_t = 512)]
    n_prompt: usize,

    #[arg(long, short = 'n', default_value_t = 128)]
    n_gen: usize,

    #[arg(long, short, default_value_t = 512)]
    batch_size: usize,

    #[arg(long, short, default_value_t = 5)]
    repetitions: usize,

    /// Number of prefix caches to hold on the device. Other caches are evicted to the CPU based on a LRU strategy.
    #[arg(long, default_value_t = 16)]
    prefix_cache_n: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let tok_model_id = "mistralai/Mistral-7B-Instruct-v0.1";
    let tokenizer_json = None;
    let quantized_model_id = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF";
    let quantized_filename = "mistral-7b-instruct-v0.1.Q4_K_M.gguf";
    let repeat_last_n = 64;

    let no_kv_cache = false;
    let chat_template = None;

    #[cfg(not(feature = "flash-attn"))]
    let use_flash_attn = false;
    #[cfg(feature = "flash-attn")]
    let use_flash_attn = true;

    let loader = MistralLoader::new(
        tok_model_id.to_string(),
        MistralSpecificConfig {
            use_flash_attn,
            repeat_last_n,
        },
        Some(quantized_model_id.to_string()),
        Some(quantized_filename.to_string()),
        None,
        ModelKind::QuantizedGGUF,
        None,
        no_kv_cache,
        chat_template,
        tokenizer_json,
        None,
    );

    #[cfg(feature = "metal")]
    let device = Device::new_metal(0)?;
    #[cfg(not(feature = "metal"))]
    let device = Device::cuda_if_available(0)?;

    tracing_subscriber::fmt().init();
    let token_source = TokenSource::CacheToken;
    info!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );
    info!("Sampling method: penalties -> temperature -> topk -> topp -> multinomial");
    info!("Loading model `{}` on {device:?}...", loader.get_id());
    if use_flash_attn {
        info!("Using flash attention.");
    }
    if use_flash_attn
        && matches!(
            loader.get_kind(),
            ModelKind::QuantizedGGML
                | ModelKind::QuantizedGGUF
                | ModelKind::XLoraGGML
                | ModelKind::XLoraGGUF
        )
    {
        warn!("Using flash attention with a quantized model has no effect!")
    }
    info!("Model kind is: {}", loader.get_kind().as_ref());
    let pipeline = loader.load_model(None, token_source, None, &device)?;
    info!("Model loaded.");

    let config = MistralRsConfig::new(
        pipeline,
        SchedulerMethod::Fixed(args.batch_size.try_into().unwrap()),
    )
    .with_no_prefix_cache(true)
    .with_prefix_cache_n(args.prefix_cache_n)
    .with_disable_eos_stop(true);

    let mistralrs = MistralRs::new(config);

    let mut results = vec![];

    if args.n_gen > 0 {
        let r = run_bench(
            mistralrs.clone(),
            RequestMessage::Completion {
                text: "Rust".to_string(),
                echo_prompt: false,
                best_of: 1,
            },
            args.n_gen - 1,
            args.batch_size,
            args.repetitions,
            TestName::Gen(args.n_gen),
        )?;
        results.push(r);
    }

    if args.n_prompt > 0 {
        let tks = (1000..1000 + args.n_prompt as u32).collect();
        let r = run_bench(
            mistralrs,
            RequestMessage::CompletionTokens(tks),
            1,
            args.batch_size,
            args.repetitions,
            TestName::Prompt(args.n_prompt),
        )?;

        results.push(r);
    }

    print_usage(quantized_model_id, &device, results);

    Ok(())
}
