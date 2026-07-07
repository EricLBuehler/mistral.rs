//! PaddleOCR-VL recognition example.
//!
//! Reads a layout manifest (`manifest.json` + crop PNGs), builds ONE CPU/f32 mistral.rs engine on
//! a local checkpoint (auto-detects `PaddleOCRVLForConditionalGeneration` -> `PaddleOcrVlLoader`),
//! runs each crop through the VLM with its manifest-resolved task `prompt` (greedy), and writes
//! `results.json` = `[{read_order, class, text}]` next to the manifest.
//!
//! Run: `PADDLEOCR_VL_WEIGHTS=<checkpoint_dir> \
//!   cargo run --release --example paddleocr_vl_recognize -p mistralrs -- <manifest_dir>`

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use mistralrs::{
    ModelDType, MultimodalMessages, MultimodalModelBuilder, RequestBuilder, TextMessageRole,
};
use serde::{Deserialize, Serialize};

const WEIGHTS_ENV: &str = "PADDLEOCR_VL_WEIGHTS";
/// Cap per-crop generation so a pathological region can't run away; real crops stop on EOS well
/// under this. Fixed cap; raise if a legitimately long table/formula ever truncates.
const MAX_NEW_TOKENS: usize = 2048;

/// One layout task, as emitted by `layout::assemble::manifest_json` (H1). The `prompt` is already
/// resolved by class in the layout stage, so recognition is a dumb crop+prompt -> text mapper.
#[derive(Debug, Deserialize)]
struct Task {
    read_order: i64,
    class: String,
    prompt: String,
    crop: String,
}

/// One recognized block, the contract H3 (`assemble::assemble_markdown`) reads back.
#[derive(Debug, Serialize)]
struct Recognized {
    read_order: i64,
    class: String,
    text: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let weights_dir = std::env::var(WEIGHTS_ENV)
        .with_context(|| format!("set {WEIGHTS_ENV} to the local PaddleOCR-VL checkpoint dir"))?;
    let manifest_dir = match std::env::args().nth(1) {
        Some(d) => PathBuf::from(d),
        None => bail!("usage: paddleocr_vl_recognize <manifest_dir>  (dir holding manifest.json)"),
    };
    let manifest_path = manifest_dir.join("manifest.json");

    let manifest = std::fs::read_to_string(&manifest_path)
        .with_context(|| format!("reading manifest {}", manifest_path.display()))?;
    let tasks: Vec<Task> =
        serde_json::from_str(&manifest).context("parsing manifest.json into tasks")?;
    println!(
        "loaded {} task(s) from {}",
        tasks.len(),
        manifest_path.display()
    );

    let model = MultimodalModelBuilder::new(&weights_dir)
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .with_logging()
        .build()
        .await?;

    let mut results = Vec::with_capacity(tasks.len());
    for task in &tasks {
        let crop_path = manifest_dir.join(&task.crop);
        let image = image::open(&crop_path)
            .with_context(|| format!("opening crop {}", crop_path.display()))?;

        // A spotting 2x-upscale override would live here, but no layout class maps
        // to the `Spotting:` prompt (spotting is a whole-page task, not a per-region class), so it
        // never triggers from this pipeline. Add it only if a spotting task ever reaches here.
        let req = RequestBuilder::from(MultimodalMessages::new().add_image_message(
            TextMessageRole::User,
            &task.prompt,
            vec![image],
        ))
        .set_sampler_max_len(MAX_NEW_TOKENS);

        let resp = model.send_chat_request(req).await?;
        let text = resp.choices[0].message.content.clone().unwrap_or_default();

        println!(
            "[{}] {} ({}) -> {text:?}",
            task.read_order, task.crop, task.class
        );
        results.push(Recognized {
            read_order: task.read_order,
            class: task.class.clone(),
            text,
        });
    }

    // Keep reading order stable regardless of manifest ordering.
    results.sort_by_key(|r| r.read_order);

    let results_path = manifest_dir.join("results.json");
    let json = serde_json::to_string_pretty(&results)?;
    std::fs::write(&results_path, format!("{json}\n"))
        .with_context(|| format!("writing {}", results_path.display()))?;
    println!(
        "wrote {} result(s) -> {}",
        results.len(),
        results_path.display()
    );

    Ok(())
}
