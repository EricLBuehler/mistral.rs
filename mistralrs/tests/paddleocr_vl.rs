//! PaddleOCR-VL end-to-end tests against a real checkpoint. Opt-in: every test skips unless
//! `PADDLEOCR_VL_WEIGHTS` points at a local checkpoint dir (the weights are ~1.9 GB and gitignored).
//!
//! The weight-free unit tests under `vision_models::paddleocr_vl` cover the arithmetic; these cover
//! the paths whose failure modes are silent on real weights, so they are worth the download:
//! greedy parity against the transformers golden, and the paged/prefix-cache behaviour where a
//! wrong answer looks like a plausible answer.

use std::path::Path;

use mistralrs::{
    Model, ModelDType, MultimodalMessages, MultimodalModelBuilder, RequestBuilder, TextMessageRole,
};

const WEIGHTS_ENV: &str = "PADDLEOCR_VL_WEIGHTS";
const FIXTURES_ENV: &str = "PADDLEOCR_VL_FIXTURES";
const PAGES_ENV: &str = "PADDLEOCR_VL_PAGES";
const PROMPT: &str = "OCR:";
const EOS: u32 = 2;

/// transformers-5.13 greedy golden for `OCR:` on `ocr.png`.
const GOLDEN: &[u32] = &[16276, 93919, 4, 5, 6, 2];
/// transformers-5.13 greedy golden for the text-only prompt below, with no image in the request.
/// The template renders text through a typed content part, so a bare string that never reaches the
/// prompt shows up here as byte-fallback garbage rather than these ids.
const TEXT_ONLY_PROMPT: &str = "Reply with the single word: ok";
const TEXT_ONLY_GOLDEN: &[u32] = &[715, 275, 318, 290, 93919, 5, 3, 364, 315, 6644, 93937, 2];

fn weights() -> Option<String> {
    std::env::var(WEIGHTS_ENV)
        .ok()
        .filter(|d| Path::new(d).exists())
}

fn fixtures() -> String {
    std::env::var(FIXTURES_ENV).unwrap_or_else(|_| "tests/fixtures".to_string())
}

fn pages() -> String {
    std::env::var(PAGES_ENV).unwrap_or_else(|_| "../tests/pdf_pages".to_string())
}

fn strip_eos(ids: &[u32]) -> &[u32] {
    match ids.last() {
        Some(&EOS) => &ids[..ids.len() - 1],
        _ => ids,
    }
}

fn greedy_ids(resp: &mistralrs::ChatCompletionResponse) -> Vec<u32> {
    resp.choices[0]
        .logprobs
        .as_ref()
        .and_then(|lp| lp.content.as_ref())
        .map(|toks| toks.iter().map(|t| t.top_logprobs[0].token).collect())
        .unwrap_or_default()
}

fn text(resp: &mistralrs::ChatCompletionResponse) -> String {
    resp.choices[0].message.content.clone().unwrap_or_default()
}

fn ocr_request(image: image::DynamicImage, max_len: usize) -> RequestBuilder {
    RequestBuilder::from(MultimodalMessages::new().add_image_message(
        TextMessageRole::User,
        PROMPT,
        vec![image],
    ))
    .set_sampler_max_len(max_len)
}

/// Paged attention needs CUDA/Metal, so the paged tests below are additionally feature-gated.
async fn build(paged: bool) -> anyhow::Result<Model> {
    let dir = weights().expect("checked by caller");
    let mut builder = MultimodalModelBuilder::new(&dir).with_dtype(if paged {
        ModelDType::BF16
    } else {
        ModelDType::F32
    });
    if !paged {
        builder = builder.with_force_cpu();
    }
    #[cfg(any(feature = "cuda", feature = "metal"))]
    if paged {
        builder = builder
            .with_paged_attn(mistralrs::PagedAttentionMetaBuilder::default().build()?)
            // The builder defaults the prefix cacher off; the server turns it on, and it is the
            // only configuration where a block-hash collision can serve one image's KV for another.
            .with_prefix_cache_n(Some(16));
    }
    builder.build().await
}

macro_rules! skip_unless_weights {
    ($what:literal) => {
        if weights().is_none() {
            eprintln!(
                "SKIP {}: set {} to a local checkpoint dir",
                $what, WEIGHTS_ENV
            );
            return Ok(());
        }
    };
}

/// Greedy decode must reproduce the transformers golden token for token, on CPU f32.
#[tokio::test]
async fn greedy_matches_transformers_golden() -> anyhow::Result<()> {
    skip_unless_weights!("greedy parity");
    let model = build(false).await?;
    let image = image::open(format!("{}/ocr.png", fixtures()))?;
    let resp = model
        .send_chat_request(
            ocr_request(image, GOLDEN.len())
                .return_logprobs(true)
                .set_sampler_topn_logprobs(1),
        )
        .await?;
    assert_eq!(
        strip_eos(&greedy_ids(&resp)),
        strip_eos(GOLDEN),
        "greedy token ids differ from the transformers golden"
    );
    Ok(())
}

/// A text-only request carries its content as a bare string. The checkpoint template reads content
/// as typed parts, so without normalising it the text never reaches the prompt and the model decodes
/// an empty user turn into byte-fallback garbage instead of these ids.
#[tokio::test]
async fn text_only_matches_transformers_golden() -> anyhow::Result<()> {
    skip_unless_weights!("text-only parity");
    let model = build(false).await?;
    let resp = model
        .send_chat_request(
            RequestBuilder::new()
                .add_message(TextMessageRole::User, TEXT_ONLY_PROMPT)
                .return_logprobs(true)
                .set_sampler_topn_logprobs(1)
                .set_sampler_max_len(TEXT_ONLY_GOLDEN.len()),
        )
        .await?;
    assert_eq!(
        strip_eos(&greedy_ids(&resp)),
        strip_eos(TEXT_ONLY_GOLDEN),
        "text-only token ids differ from the transformers golden"
    );
    Ok(())
}

/// One image per request. Two used to be silently reduced to the first and then panic on the second
/// placeholder's missing grid, taking the engine worker down with it.
#[tokio::test]
async fn extra_images_are_rejected_and_the_engine_survives() -> anyhow::Result<()> {
    skip_unless_weights!("multi-image rejection");
    let model = build(false).await?;
    let image = image::open(format!("{}/ocr.png", fixtures()))?;
    let two = RequestBuilder::from(MultimodalMessages::new().add_image_message(
        TextMessageRole::User,
        PROMPT,
        vec![image.clone(), image.clone()],
    ))
    .set_sampler_max_len(8);
    let err = model
        .send_chat_request(two)
        .await
        .err()
        .expect("two images must be rejected")
        .to_string();
    assert!(err.contains("one image per request"), "{err}");
    assert!(
        !text(&model.send_chat_request(ocr_request(image, 8)).await?).is_empty(),
        "engine did not survive the rejected request"
    );
    Ok(())
}

#[cfg(any(feature = "cuda", feature = "metal"))]
mod paged {
    use super::*;

    fn page(name: &str) -> anyhow::Result<image::DynamicImage> {
        Ok(image::open(format!("{}/{name}", pages()))?)
    }

    macro_rules! skip_unless_pages {
        ($what:literal) => {
            skip_unless_weights!($what);
            if !Path::new(&pages()).exists() {
                eprintln!("SKIP {}: set {} to a dir of page PNGs", $what, PAGES_ENV);
                return Ok(());
            }
        };
    }

    /// Pages of a scanned PDF share a size, so their prompts are byte-identical token streams: every
    /// expanded placeholder is the same id. Only the registered image span keeps their paged blocks
    /// apart, and without it the second page comes back with the first page's text.
    #[tokio::test]
    async fn prefix_cache_does_not_serve_one_image_for_another() -> anyhow::Result<()> {
        skip_unless_pages!("paged prefix cache");
        let model = build(true).await?;
        let run = async |name: &str| -> anyhow::Result<String> {
            Ok(text(
                &model
                    .send_chat_request(ocr_request(page(name)?, 48))
                    .await?,
            ))
        };
        let first = run("page_00.png").await?;
        let second = run("page_01.png").await?;
        assert!(!first.trim().is_empty(), "page_00 produced no output");
        assert_ne!(
            first, second,
            "different pages returned identical text: the prefix cache served page_00's image KV"
        );
        assert_eq!(
            first,
            run("page_00.png").await?,
            "re-running page_00 changed after page_01: prefix cache reuse is corrupt"
        );
        Ok(())
    }

    /// A text-only request in flight next to an image one used to rotate forever in the completion
    /// scheduler, pinning a core with no request ever completing.
    #[tokio::test]
    async fn mixed_text_and_image_batch_makes_progress() -> anyhow::Result<()> {
        skip_unless_pages!("paged mixed batch");
        let model = build(true).await?;
        let image = ocr_request(page("page_00.png")?, 32);
        let alone = text(&model.send_chat_request(image.clone()).await?);
        let (batched, _) = tokio::join!(
            model.send_chat_request(image),
            model.send_chat_request(
                RequestBuilder::new()
                    .add_message(TextMessageRole::User, TEXT_ONLY_PROMPT)
                    .set_sampler_max_len(8)
            )
        );
        assert_eq!(
            alone,
            text(&batched?),
            "OCR output changed when a text-only request shared the batch"
        );
        Ok(())
    }
}
