//! This crate provides an asynchronous, multithreaded API to `mistral.rs`.
//!
//! ## Example
//! ```no_run
//! use std::sync::Arc;
//! use tokio::sync::mpsc::channel;
//!
//! use mistralrs::{
//!     Constraint, DeviceMapMetadata, MistralRs, MistralRsBuilder,
//!     NormalLoaderType, NormalRequest, Request, RequestMessage, Response,
//!     SamplingParams, SchedulerMethod, TokenSource,
//! };
//!
//! fn setup() -> anyhow::Result<Arc<MistralRs>> {
//!     // See the examples for how to load your model.
//!     todo!()
//! }
//!
//! fn main() -> anyhow::Result<()> {
//!     let mistralrs = setup()?;
//!
//!     let (tx, mut rx) = channel(10_000);
//!     let request = Request::Normal(NormalRequest {
//!         messages: RequestMessage::Completion {
//!             text: "Hello! My name is ".to_string(),
//!             echo_prompt: false,
//!             best_of: 1,
//!         },
//!         sampling_params: SamplingParams::default(),
//!         response: tx,
//!         return_logprobs: false,
//!         is_streaming: false,
//!         id: 0,
//!         constraint: Constraint::None,
//!         suffix: None,
//!         adapters: None,
//!     });
//!     mistralrs.get_sender()?.blocking_send(request)?;
//!
//!     let response = rx.blocking_recv().unwrap();
//!     match response {
//!         Response::CompletionDone(c) => println!("Text: {}", c.choices[0].text),
//!         _ => unreachable!(),
//!     }
//!     Ok(())
//! }
//! ```

pub use candle_core::{quantized::GgmlDType, DType, Device, Result};
pub use mistralrs_core::*;

#[macro_export]
macro_rules! load_normal_model {
    (id = $model_id:expr, kind = $kind:ident, device = $device:expr, use_flash_attn = $use_flash:expr) => {
        load_normal_model!(
            id = $model_id,
            kind = $kind,
            dtype = Auto,
            device = $device,
            cache_tok = mistralrs::TokenSource::CacheToken,
            mapper = mistralrs::DeviceMapMetadata::dummy(),
            isq = None,
            use_flash_attn = $use_flash
        )
    };

    (id = $model_id:expr, kind = $kind:ident, device = $device:expr, use_flash_attn = $use_flash:expr, isq = $isq:expr) => {
        load_normal_model!(
            id = $model_id,
            kind = $kind,
            dtype = Auto,
            device = $device,
            cache_tok = mistralrs::TokenSource::CacheToken,
            mapper = mistralrs::DeviceMapMetadata::dummy(),
            isq = Some($isq),
            use_flash_attn = $use_flash
        )
    };

    (id = $model_id:expr, kind = $kind:ident, dtype = $ty:ident, device = $device:expr, cache_tok = $cache_tok:expr, mapper = $mapper:expr, isq = $isq:expr, use_flash_attn = $use_flash:expr) => {{
        let loader = mistralrs::NormalLoaderBuilder::new(
            mistralrs::NormalSpecificConfig {
                use_flash_attn: $use_flash,
                repeat_last_n: 64,
            },
            None,
            None,
            Some($model_id),
        )
        .build(mistralrs::NormalLoaderType::$kind);
        // Load, into a Pipeline
        let pipeline = loader.load_model_from_hf(
            None,
            $cache_tok,
            &mistralrs::ModelDType::$ty,
            &$device,
            false,
            $mapper,
            $isq,
        )?;
        mistralrs::MistralRsBuilder::new(
            pipeline,
            mistralrs::SchedulerMethod::Fixed(32.try_into().unwrap()),
        )
        .build()
    }};
}
