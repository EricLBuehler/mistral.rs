//! PaddleOCR-VL PagedAttention serve test (env-gated, opt-in, needs the `cuda` feature).
//!
//! The weight-free unit tests cover the block-hash and clamp arithmetic; this drives the real paged
//! path, where the failure mode is silent: pages of a scanned PDF share a size, so their prompts are
//! byte-identical token streams and the prefix cache will hand page 2 page 1's image KV unless the
//! image span is registered. Skips when `PADDLEOCR_VL_WEIGHTS` is unset.

#![cfg(feature = "cuda")]

use std::path::Path;

use mistralrs::{
    MultimodalMessages, MultimodalModelBuilder, PagedAttentionMetaBuilder, RequestBuilder,
    TextMessageRole,
};

const WEIGHTS_ENV: &str = "PADDLEOCR_VL_WEIGHTS";
const PAGES_ENV: &str = "PADDLEOCR_VL_PAGES";
const PROMPT: &str = "OCR:";
const MAX_LEN: usize = 48;

fn skip_reason() -> Option<(String, String)> {
    let weights = std::env::var(WEIGHTS_ENV)
        .ok()
        .filter(|d| Path::new(d).exists())?;
    let pages = std::env::var(PAGES_ENV).unwrap_or_else(|_| "../tests/pdf_pages".to_string());
    Path::new(&pages).exists().then_some((weights, pages))
}

#[tokio::test]
async fn paged_prefix_cache_does_not_leak_between_images() -> anyhow::Result<()> {
    let Some((weights_dir, pages_dir)) = skip_reason() else {
        eprintln!("SKIP paged serve: set {WEIGHTS_ENV} (and optionally {PAGES_ENV})");
        return Ok(());
    };

    let model = MultimodalModelBuilder::new(&weights_dir)
        .with_dtype(mistralrs::ModelDType::BF16)
        .with_paged_attn(PagedAttentionMetaBuilder::default().build()?)
        // The builder defaults the prefix cacher off; the server/CLI path runs with it on, and it is
        // the only configuration where the block-hash collision can bite. Test what ships.
        .with_prefix_cache_n(Some(16))
        .build()
        .await?;

    let ocr = |name: &str| {
        let path = format!("{pages_dir}/{name}");
        async {
            let image = image::open(path)?;
            let req = RequestBuilder::from(MultimodalMessages::new().add_image_message(
                TextMessageRole::User,
                PROMPT,
                vec![image],
            ))
            .set_sampler_max_len(MAX_LEN);
            let resp = model.send_chat_request(req).await?;
            anyhow::Ok(resp.choices[0].message.content.clone().unwrap_or_default())
        }
    };

    // Same size, same prompt -> identical token streams. Only the registered image span keeps their
    // paged blocks apart; without it page_01 comes back as page_00's text.
    let first = ocr("page_00.png").await?;
    let second = ocr("page_01.png").await?;
    assert!(!first.trim().is_empty(), "page_00 produced no output");
    assert_ne!(
        first, second,
        "different pages returned identical text: the prefix cache served page_00's image KV"
    );

    // Re-running the first page must reuse its own blocks, not the ones page_01 just wrote.
    let first_again = ocr("page_00.png").await?;
    assert_eq!(
        first, first_again,
        "re-running page_00 changed after page_01 ran: prefix cache reuse is corrupt"
    );
    Ok(())
}

/// A text-only request sharing the batch with an OCR one must not disturb the OCR row: they take
/// different branches of the per-row vision path.
#[tokio::test]
async fn paged_mixed_text_and_image_batch() -> anyhow::Result<()> {
    let Some((weights_dir, pages_dir)) = skip_reason() else {
        eprintln!("SKIP paged mixed batch: set {WEIGHTS_ENV}");
        return Ok(());
    };

    let model = MultimodalModelBuilder::new(&weights_dir)
        .with_dtype(mistralrs::ModelDType::BF16)
        .with_paged_attn(PagedAttentionMetaBuilder::default().build()?)
        // The builder defaults the prefix cacher off; the server/CLI path runs with it on, and it is
        // the only configuration where the block-hash collision can bite. Test what ships.
        .with_prefix_cache_n(Some(16))
        .build()
        .await?;

    let image = image::open(format!("{pages_dir}/page_00.png"))?;
    let image_req = RequestBuilder::from(MultimodalMessages::new().add_image_message(
        TextMessageRole::User,
        PROMPT,
        vec![image.clone()],
    ))
    .set_sampler_max_len(MAX_LEN);
    let alone = model.send_chat_request(image_req.clone()).await?.choices[0]
        .message
        .content
        .clone()
        .unwrap_or_default();

    let text_req = RequestBuilder::new()
        .add_message(TextMessageRole::User, "Say hello.")
        .set_sampler_max_len(8);
    let (image_res, _text_res) = tokio::join!(
        model.send_chat_request(image_req),
        model.send_chat_request(text_req)
    );
    let batched = image_res?.choices[0]
        .message
        .content
        .clone()
        .unwrap_or_default();

    assert_eq!(
        alone, batched,
        "OCR output changed when a text-only request shared the batch"
    );
    Ok(())
}
