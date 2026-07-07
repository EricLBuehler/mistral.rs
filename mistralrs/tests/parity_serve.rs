//! PaddleOCR-VL-1.5 end-to-end greedy serve-parity gate.
//!
//! Loads the local checkpoint on CPU/f32 through the real mistral.rs engine (auto-detects
//! `PaddleOCRVLForConditionalGeneration` -> `PaddleOcrVlLoader`) once, then for each task fixture
//! feeds the image + task prompt, greedy-decodes (capped at the golden length), and asserts the
//! emitted token ids match the transformers-5.13 golden.
//!
//! The 1.8 GB weights are gitignored, so this skips gracefully when `ref/weights/` is absent; a
//! clean `cargo test` must not hard-fail. The golden token ids are inlined, so the assertion
//! needs no gitignored golden artifacts.

use std::path::Path;

use mistralrs::{
    ModelDType, MultimodalMessages, MultimodalModelBuilder, RequestBuilder, TextMessageRole,
};

// Local checkpoint + fixture dirs, supplied via env so no absolute path is compiled in. Absent =>
// the test skips (weights are a 1.8 GB gitignored artifact, never present on CI).
const WEIGHTS_ENV: &str = "PADDLEOCR_VL_WEIGHTS";
const FIXTURES_ENV: &str = "PADDLEOCR_VL_FIXTURES";
const EOS: u32 = 2;

struct Fixture {
    name: &'static str,
    image: &'static str,
    prompt: &'static str,
    /// golden `greedy_new_tokens`, including the trailing EOS when generation stopped on it.
    golden: &'static [u32],
}

const FIXTURES: &[Fixture] = &[
    Fixture {
        name: "ocr",
        image: "ocr.png",
        prompt: "OCR:",
        golden: &[16276, 93919, 4, 5, 6, 2],
    },
    Fixture {
        name: "table",
        image: "table.png",
        prompt: "Table Recognition:",
        golden: &[
            101309, 93957, 101309, 93978, 101313, 101309, 4, 101309, 5, 101313, 2,
        ],
    },
    Fixture {
        name: "formula",
        image: "formula.png",
        prompt: "Formula Recognition:",
        golden: &[2812, 93962, 93950, 93933, 284, 1305, 5, 1498, 93989, 2],
    },
    Fixture {
        name: "spotting",
        image: "spotting.png",
        prompt: "Spotting:",
        golden: &[
            1818, 93919, 10, 100344, 100630, 100493, 100630, 100493, 100797, 100344, 100797, 2,
        ],
    },
    Fixture {
        name: "seal",
        image: "seal.png",
        prompt: "Seal Recognition:",
        golden: &[858, 763, 2],
    },
    // chart hit the golden dump's 16-token cap without EOS; strip_eos leaves all 16 to compare.
    Fixture {
        name: "chart",
        image: "chart.png",
        prompt: "Chart Recognition:",
        golden: &[
            2221, 697, 8914, 23, 4, 697, 93919, 4, 3, 3, 23, 5, 697, 93919, 4, 3,
        ],
    },
];

fn strip_eos(ids: &[u32]) -> &[u32] {
    match ids.last() {
        Some(&EOS) => &ids[..ids.len() - 1],
        _ => ids,
    }
}

#[tokio::test]
async fn serve_greedy_parity_all_fixtures() -> anyhow::Result<()> {
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

    let mut failed = Vec::new();
    for fx in FIXTURES {
        let image = image::open(format!("{fix}/{}", fx.image))?;
        let req = RequestBuilder::from(MultimodalMessages::new().add_image_message(
            TextMessageRole::User,
            fx.prompt,
            vec![image],
        ))
        .return_logprobs(true)
        .set_sampler_topn_logprobs(1)
        .set_sampler_max_len(fx.golden.len());

        let resp = model.send_chat_request(req).await?;
        let choice = &resp.choices[0];

        let ids: Vec<u32> = choice
            .logprobs
            .as_ref()
            .and_then(|lp| lp.content.as_ref())
            .map(|toks| toks.iter().map(|t| t.top_logprobs[0].token).collect())
            .unwrap_or_default();

        println!("[{}] token ids: {ids:?}", fx.name);
        if strip_eos(&ids) != strip_eos(fx.golden) {
            println!("[{}] MISMATCH: golden {:?}", fx.name, fx.golden);
            failed.push(fx.name);
        }
    }

    assert!(
        failed.is_empty(),
        "greedy token ids differ from golden for: {failed:?}"
    );
    Ok(())
}
