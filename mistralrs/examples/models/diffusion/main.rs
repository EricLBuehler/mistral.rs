//! Image generation using a diffusion model.
//!
//! Run with: `cargo run --release --example diffusion -p mistralrs`

use std::time::Instant;

use anyhow::Result;
use mistralrs::{
    DiffusionGenerationParams, DiffusionLoaderType, DiffusionModelBuilder,
    ImageGenerationResponseFormat,
};

#[tokio::main]
async fn main() -> Result<()> {
    let model = DiffusionModelBuilder::new(
        "black-forest-labs/FLUX.1-schnell",
        DiffusionLoaderType::FluxOffloaded,
    )
    .with_logging()
    .build()
    .await?;

    let start = Instant::now();

    let response = model
        .generate_image(
            "A vibrant sunset in the mountains, 4k, high quality.".to_string(),
            ImageGenerationResponseFormat::Url,
            DiffusionGenerationParams::default(),
            None,
        )
        .await?;

    let finished = Instant::now();

    println!(
        "Done! Took {} s. Image saved at: {}",
        finished.duration_since(start).as_secs_f32(),
        response.data[0].url.as_ref().unwrap()
    );

    Ok(())
}
