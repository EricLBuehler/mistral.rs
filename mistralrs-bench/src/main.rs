use candle_core::Device;
use clap::Parser;
use cli_table::{format::Justify, print_stdout, Cell, CellStruct, Style, Table};
use mistralrs_core::{
    Constraint, DeviceMapMetadata, Loader, LoaderBuilder, MistralRs, MistralRsBuilder, ModelKind,
    ModelSelected, Request, RequestMessage, Response, SamplingParams, SchedulerMethod, TokenSource,
    Usage,
};
use std::fmt::Display;
use std::sync::Arc;
use tokio::sync::mpsc::channel;
use tracing::info;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::EnvFilter;

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
    concurrency: usize,
    test_name: TestName,
}

struct UncertainTokSec {
    mean: f32,
    std_dev: f32,
}

impl Display for UncertainTokSec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.3}±{:.3}", self.mean, self.std_dev)
    }
}

fn run_bench(
    mistralrs: Arc<MistralRs>,
    prompt: RequestMessage,
    n_gen: usize,
    concurrency: usize,
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
    let (tx, mut rx) = channel(10_000);

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
        for _ in 0..concurrency {
            sender
                .blocking_send(req.clone())
                .expect("Expected receiver.");
        }
        for _ in 0..concurrency {
            match rx.blocking_recv() {
                Some(r) => match r {
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
                None => unreachable!("Expected a Done response, got None",),
            }
        }
    }

    Ok(BenchResult {
        usages,
        concurrency,
        test_name,
    })
}

fn get_tok_s(result: &BenchResult) -> UncertainTokSec {
    let ts_measurements = match result.test_name {
        TestName::Prompt(_) => result
            .usages
            .iter()
            .map(|u| u.avg_prompt_tok_per_sec)
            .collect::<Vec<_>>(),
        TestName::Gen(_) => result
            .usages
            .iter()
            .map(|u| u.avg_compl_tok_per_sec)
            .collect::<Vec<_>>(),
    };
    // Calculate uncertainty
    let mean = ts_measurements.iter().sum::<f32>() / ts_measurements.len() as f32;
    let variance = ts_measurements
        .iter()
        .map(|e| (mean - e).powf(2.))
        .sum::<f32>()
        / ts_measurements.len() as f32;
    let std_dev = variance.sqrt();
    UncertainTokSec { mean, std_dev }
}

fn get_ms_tok(result: &BenchResult) -> UncertainTokSec {
    let ms_tok_measurements = match result.test_name {
        TestName::Prompt(_) => result
            .usages
            .iter()
            .map(|u| 1000. / u.avg_prompt_tok_per_sec)
            .collect::<Vec<_>>(),
        TestName::Gen(_) => result
            .usages
            .iter()
            .map(|u| 1000. / u.avg_compl_tok_per_sec)
            .collect::<Vec<_>>(),
    };
    // Calculate uncertainty
    let mean = ms_tok_measurements.iter().sum::<f32>() / ms_tok_measurements.len() as f32;
    let variance = ms_tok_measurements
        .iter()
        .map(|e| (mean - e).powf(2.))
        .sum::<f32>()
        / ms_tok_measurements.len() as f32;
    let std_dev = variance.sqrt();
    UncertainTokSec { mean, std_dev }
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
                r.test_name.to_string().cell(),
                get_tok_s(&r).cell().justify(Justify::Right),
                get_ms_tok(&r).cell().justify(Justify::Right),
                r.concurrency.cell().justify(Justify::Right),
                (get_tok_s(&r).mean * r.concurrency as f32)
                    .cell()
                    .justify(Justify::Right),
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
            "test".cell().bold(true),
            "t/s".cell().bold(true),
            "ms/t".cell().bold(true),
            "concurrency".cell().bold(true),
            "throughput/s".cell().bold(true),
        ])
        .bold(true);
    print_stdout(table).expect("print table");
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    /// Model
    #[clap(subcommand)]
    model: ModelSelected,

    /// Number of prompt tokens to run.
    #[arg(long, short = 'p', default_value_t = 512)]
    n_prompt: usize,

    /// Number of generations tokens to run.
    #[arg(long, short = 'g', default_value_t = 128)]
    n_gen: usize,

    /// Number of concurrent requests to run. Default is 1
    #[clap(short, long, value_parser, value_delimiter = ',')]
    concurrency: Option<Vec<usize>>,

    /// Number of times to repeat each test.
    #[arg(long, short, default_value_t = 5)]
    repetitions: usize,

    /// Number of device layers to load and run on the device. All others will be on the CPU.
    #[arg(short, long)]
    num_device_layers: Option<usize>,
}

fn main() -> anyhow::Result<()> {
    let mut args = Args::parse();
    args.concurrency = Some(args.concurrency.unwrap_or(vec![1]));

    #[cfg(not(feature = "flash-attn"))]
    let use_flash_attn = false;
    #[cfg(feature = "flash-attn")]
    let use_flash_attn = true;

    let loader: Box<dyn Loader> = LoaderBuilder::new(args.model)
        .with_use_flash_attn(use_flash_attn)
        .build()?;
    let model_name = loader.get_id();

    #[cfg(feature = "metal")]
    let device = Device::new_metal(0)?;
    #[cfg(not(feature = "metal"))]
    let device = Device::cuda_if_available(0)?;

    let filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();
    tracing_subscriber::fmt().with_env_filter(filter).init();

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
        info!("⚠️ WARNING: Using flash attention with a quantized model has no effect!")
    }
    info!("Model kind is: {}", loader.get_kind().to_string());
    let pipeline = loader.load_model(
        None,
        token_source,
        None,
        &device,
        false,
        args.num_device_layers
            .map(DeviceMapMetadata::from_num_device_layers)
            .unwrap_or(DeviceMapMetadata::dummy()),
        None,
    )?;
    info!("Model loaded.");

    let mistralrs = MistralRsBuilder::new(
        pipeline,
        SchedulerMethod::Fixed(
            (*args.concurrency.as_ref().unwrap().iter().max().unwrap())
                .try_into()
                .unwrap(),
        ),
    )
    .with_no_prefix_cache(true)
    .with_disable_eos_stop(true)
    .build();

    for concurrency in args.concurrency.as_ref().unwrap() {
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
                *concurrency,
                args.repetitions,
                TestName::Gen(args.n_gen),
            )?;
            results.push(r);
        }

        if args.n_prompt > 0 {
            let tks = (1000..1000 + args.n_prompt as u32).collect();
            let r = run_bench(
                mistralrs.clone(),
                RequestMessage::CompletionTokens(tks),
                1,
                *concurrency,
                args.repetitions,
                TestName::Prompt(args.n_prompt),
            )?;

            results.push(r);
        }

        print_usage(&model_name, &device, results);
    }

    Ok(())
}
