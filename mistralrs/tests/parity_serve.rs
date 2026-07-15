//! PaddleOCR-VL end-to-end greedy serve-parity smoke test (env-gated, opt-in).
//!
//! CI coverage of the engine paths (quant layers, Sdpa, paged branch, batched decode) lives in the
//! weight-free unit tests under `vision_models::paddleocr_vl`. This is the optional local check that
//! the full mistral.rs serve path reproduces the transformers golden on real weights. The 1.8 GB
//! checkpoint is gitignored, so it skips gracefully when `PADDLEOCR_VL_WEIGHTS` is unset.

use std::path::Path;

use mistralrs::{
    ModelDType, MultimodalMessages, MultimodalModelBuilder, RequestBuilder, TextMessageRole,
};

const WEIGHTS_ENV: &str = "PADDLEOCR_VL_WEIGHTS";
const FIXTURES_ENV: &str = "PADDLEOCR_VL_FIXTURES";
const EOS: u32 = 2;

// One representative fixture: the transformers-5.13 greedy golden for the "OCR:" task on ocr.png.
const IMAGE: &str = "ocr.png";
const PROMPT: &str = "OCR:";
const GOLDEN: &[u32] = &[16276, 93919, 4, 5, 6, 2];

fn strip_eos(ids: &[u32]) -> &[u32] {
    match ids.last() {
        Some(&EOS) => &ids[..ids.len() - 1],
        _ => ids,
    }
}

#[tokio::test]
async fn serve_greedy_parity_smoke() -> anyhow::Result<()> {
    let Some(weights_dir) = std::env::var(WEIGHTS_ENV)
        .ok()
        .filter(|d| Path::new(d).exists())
    else {
        eprintln!(
            "SKIP serve parity: set {WEIGHTS_ENV} to the local checkpoint dir (1.8 GB, gitignored)"
        );
        return Ok(());
    };
    let fix = std::env::var(FIXTURES_ENV).unwrap_or_else(|_| "tests/fixtures".to_string());

    let model = MultimodalModelBuilder::new(&weights_dir)
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?;

    let image = image::open(format!("{fix}/{IMAGE}"))?;
    let req = RequestBuilder::from(MultimodalMessages::new().add_image_message(
        TextMessageRole::User,
        PROMPT,
        vec![image],
    ))
    .return_logprobs(true)
    .set_sampler_topn_logprobs(1)
    .set_sampler_max_len(GOLDEN.len());

    let resp = model.send_chat_request(req).await?;
    let choice = &resp.choices[0];

    let ids: Vec<u32> = choice
        .logprobs
        .as_ref()
        .and_then(|lp| lp.content.as_ref())
        .map(|toks| toks.iter().map(|t| t.top_logprobs[0].token).collect())
        .unwrap_or_default();

    assert_eq!(
        strip_eos(&ids),
        strip_eos(GOLDEN),
        "greedy token ids differ from transformers golden"
    );
    Ok(())
}
