use clap::Parser;
use either::Either;
use image::DynamicImage;
use indexmap::IndexMap;
use std::sync::Arc;
use tokio::sync::mpsc::{channel, Sender};
use tokio::task;

use mistralrs::{
    Constraint, DefaultSchedulerMethod, Device, DeviceMapMetadata, MistralRs, MistralRsBuilder,
    ModelDType, NormalRequest, Request, RequestMessage, ResponseOk, Result, SamplingParams,
    SchedulerConfig, TokenSource, VisionLoaderBuilder, VisionLoaderType, VisionSpecificConfig,
};
use mistralrs_core::{initialize_logging, DeviceLayerMapMetadata, DeviceMapSetting, Response};

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "Qwen/Qwen2.5-VL-7B-Instruct")]
    model_id: String,
}

/// Gets the best device, cpu, cuda if compiled with CUDA
pub(crate) fn best_device() -> Result<Device> {
    #[cfg(not(feature = "metal"))]
    {
        Device::cuda_if_available(0)
    }
    #[cfg(feature = "metal")]
    {
        Device::new_metal(0)
    }
}

fn setup() -> anyhow::Result<Arc<MistralRs>> {
    let args = Args::parse();

    //map all layers to gpu
    let device_mapping = DeviceMapSetting::Map(DeviceMapMetadata::from_num_device_layers(vec![
        DeviceLayerMapMetadata {
            ordinal: 0,
            layers: 999,
        },
    ]));

    // Select a model
    let loader = VisionLoaderBuilder::new(
        VisionSpecificConfig {
            use_flash_attn: false,
            prompt_chunksize: None,
            topology: None,
            write_uqff: None,
            from_uqff: None,
            max_edge: None,
            imatrix: None,
            calibration_file: None,
        },
        None,
        None,
        Some(args.model_id),
    )
    .build(VisionLoaderType::Qwen2_5VL);
    // Load, into a Pipeline
    let pipeline = loader.load_model_from_hf(
        None,
        TokenSource::CacheToken,
        &ModelDType::Auto,
        &best_device()?,
        false,
        device_mapping,
        None,
        None, // No PagedAttention.
    )?;
    // Create the MistralRs, which is a runner
    Ok(MistralRsBuilder::new(
        pipeline,
        SchedulerConfig::DefaultScheduler {
            method: DefaultSchedulerMethod::Fixed(5.try_into().unwrap()),
        },
    )
    .build())
}

fn gen_request(id: usize, image: DynamicImage, tx: Sender<Response>) -> Request {
    Request::Normal(NormalRequest {
        messages: RequestMessage::VisionChat {
            images: vec![image],
            messages: vec![IndexMap::from([
                ("role".to_string(), Either::Left("user".to_string())),
                (
                    "content".to_string(),
                    Either::Left(
                        "<|vision_start|><|image_pad|><|vision_end|>What is depicted here?"
                            .to_string(),
                    ),
                ),
            ])],
        },
        sampling_params: SamplingParams::deterministic(),
        response: tx,
        return_logprobs: false,
        is_streaming: false,
        id,
        constraint: Constraint::None,
        suffix: None,
        tools: None,
        tool_choice: None,
        logits_processors: None,
        return_raw_logits: false,
    })
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    initialize_logging();
    let mistralrs = setup()?;

    let bytes = task::spawn_blocking(|| {
        match reqwest::blocking::get(
            "https://cdn.britannica.com/45/5645-050-B9EC0205/head-treasure-flower-disk-flowers-inflorescence-ray.jpg",
        ) {
            Ok(http_resp) => Ok(http_resp.bytes()?.to_vec()),
            Err(e) => anyhow::bail!(e),
        }
    }).await??;

    let image = image::load_from_memory(&bytes)?;

    let (tx, mut rx) = channel(10_000);

    let _ = task::spawn_blocking(move || {
        for _ in 0..3 {
            let id = mistralrs.next_request_id();
            let request = gen_request(id, image.clone(), tx.clone());
            let _ = mistralrs
                .get_sender()
                .expect("get sender error")
                .blocking_send(request);
            let response = rx.blocking_recv().unwrap().as_result().unwrap();
            match response {
                ResponseOk::Done(c) => println!(
                    "Text: {}, Prompt T/s: {}, Completion T/s: {}",
                    c.choices[0].message.content.as_ref().unwrap(),
                    c.usage.avg_prompt_tok_per_sec,
                    c.usage.avg_compl_tok_per_sec
                ),
                _ => unreachable!(),
            }
        }
    })
    .await?;

    Ok(())
}
