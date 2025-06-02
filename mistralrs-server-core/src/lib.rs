//! > **mistral.rs server core**
//!
//! ## About
//!
//! This crate powers mistral.rs server. It exposes the underlying functionality
//! allowing others to implement and extend the server implementation.
//!
//! ### Features
//! 1. Incorporate mistral.rs server into another axum.rs project.
//! 2. Hook into the mistral.rs server lifecycle.
//!
//! ### Example
//! ```no_run
//! use std::sync::Arc;
//!
//! use axum::{
//!     Json, Router,
//!     extract::State,
//!     routing::{get, post},
//! };
//! use utoipa::OpenApi;
//! use utoipa_swagger_ui::SwaggerUi;
//!
//! use mistralrs::{AutoDeviceMapParams, ChatCompletionChunkResponse, ModelDType, ModelSelected};
//! use mistralrs_server_core::{
//!     chat_completion::{
//!         ChatCompletionResponder, OnChunkCallback, OnDoneCallback, create_chat_streamer,
//!         create_response_channel, handle_chat_completion_error, parse_request,
//!         process_non_streaming_chat_response, send_request,
//!     },
//!     mistralrs_for_server_builder::MistralRsForServerBuilder,
//!     mistralrs_server_router_builder::MistralRsServerRouterBuilder,
//!     openai::ChatCompletionRequest,
//!     openapi_doc::get_openapi_doc,
//!     types::SharedMistralState,
//! };
//!
//! #[derive(OpenApi)]
//! #[openapi(
//!     paths(root, custom_chat),
//!     tags(
//!         (name = "hello", description = "Hello world endpoints")
//!     ),
//!     info(
//!         title = "Hello World API",
//!         version = "1.0.0",
//!         description = "A simple API that responds with a greeting"
//!     )
//! )]
//! struct ApiDoc;
//!
//! #[derive(Clone)]
//! pub struct AppState {
//!     pub mistral_state: SharedMistralState,
//!     pub db_create: fn(),
//! }
//!
//! #[tokio::main]
//! async fn main() {
//!     let plain_model_id = String::from("meta-llama/Llama-3.2-1B-Instruct");
//!     let tokenizer_json = None;
//!     let arch = None;
//!     let organization = None;
//!     let write_uqff = None;
//!     let from_uqff = None;
//!     let imatrix = None;
//!     let calibration_file = None;
//!     let hf_cache_path = None;
//!
//!     let dtype = ModelDType::Auto;
//!     let topology = None;
//!     let max_seq_len = AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN;
//!     let max_batch_size = AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE;
//!
//!     let model = ModelSelected::Plain {
//!         model_id: plain_model_id,
//!         tokenizer_json,
//!         arch,
//!         dtype,
//!         topology,
//!         organization,
//!         write_uqff,
//!         from_uqff,
//!         imatrix,
//!         calibration_file,
//!         max_seq_len,
//!         max_batch_size,
//!         hf_cache_path,
//!     };
//!
//!     let shared_mistralrs = MistralRsForServerBuilder::new()
//!         .with_model(model)
//!         .with_in_situ_quant("8".to_string())
//!         .with_paged_attn(true)
//!         .build()
//!         .await
//!         .unwrap();
//!
//!     let mistral_base_path = "/api/mistral";
//!
//!     let mistral_routes = MistralRsServerRouterBuilder::new()
//!         .with_mistralrs(shared_mistralrs.clone())
//!         .with_include_swagger_routes(false)
//!         .with_base_path(mistral_base_path)
//!         .build()
//!         .await
//!         .unwrap();
//!
//!     let mistral_doc = get_openapi_doc(Some(mistral_base_path));
//!     let mut api_docs = ApiDoc::openapi();
//!     api_docs.merge(mistral_doc);
//!
//!     let app_state = Arc::new(AppState {
//!         mistral_state: shared_mistralrs,
//!         db_create: mock_db_call,
//!     });
//!
//!     let app = Router::new()
//!         .route("/", get(root))
//!         .route("/chat", post(custom_chat))
//!         .with_state(app_state.clone())
//!         .nest(mistral_base_path, mistral_routes)
//!         .merge(SwaggerUi::new("/api-docs").url("/api-docs/openapi.json", api_docs));
//!
//!     let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
//!     axum::serve(listener, app).await.unwrap();
//!
//!     println!("Listening on 0.0.0.0:3000");
//! }
//!
//! #[utoipa::path(
//!     get,
//!     path = "/",
//!     tag = "hello",
//!     responses(
//!         (status = 200, description = "Successful response with greeting message", body = String)
//!     )
//! )]
//! async fn root() -> &'static str {
//!     "Hello, World!"
//! }
//!
//! #[utoipa::path(
//!   post,
//!   tag = "Custom",
//!   path = "/chat",
//!   request_body = ChatCompletionRequest,
//!   responses((status = 200, description = "Chat completions"))
//! )]
//! pub async fn custom_chat(
//!     State(state): State<Arc<AppState>>,
//!     Json(oai_request): Json<ChatCompletionRequest>,
//! ) -> ChatCompletionResponder {
//!     let mistral_state = state.mistral_state.clone();
//!     let (tx, mut rx) = create_response_channel();
//!
//!     let (request, is_streaming) = match parse_request(oai_request, mistral_state.clone(), tx).await
//!     {
//!         Ok(x) => x,
//!         Err(e) => return handle_chat_completion_error(mistral_state, e.into()),
//!     };
//!
//!     dbg!(request.clone());
//!
//!     if let Err(e) = send_request(&mistral_state, request).await {
//!         return handle_chat_completion_error(mistral_state, e.into());
//!     }
//!
//!     if is_streaming {
//!         let db_fn = state.db_create;
//!
//!         let on_chunk: OnChunkCallback = Box::new(move |mut chunk: ChatCompletionChunkResponse| {
//!             dbg!(&chunk);
//!
//!             if let Some(original_content) = &chunk.choices[0].delta.content {
//!                 chunk.choices[0].delta.content = Some(format!("CHANGED! {}", original_content));
//!             }
//!
//!             chunk.clone()
//!         });
//!
//!         let on_done: OnDoneCallback = Box::new(move |chunks: &[ChatCompletionChunkResponse]| {
//!             dbg!(chunks);
//!             (db_fn)();
//!         });
//!
//!         let streamer =
//!             create_chat_streamer(rx, mistral_state.clone(), Some(on_chunk), Some(on_done));
//!
//!         ChatCompletionResponder::Sse(streamer)
//!     } else {
//!         let response = process_non_streaming_chat_response(&mut rx, mistral_state.clone()).await;
//!
//!         match &response {
//!             ChatCompletionResponder::Json(json_response) => {
//!                 dbg!(json_response);
//!             }
//!             _ => {
//!                 //
//!             }
//!         }
//!
//!         response
//!     }
//! }
//!
//! pub fn mock_db_call() {
//!     println!("Saving to DB");
//! }
//! ```

pub mod chat_completion;
mod completions;
mod handlers;
mod image_generation;
pub mod mistralrs_for_server_builder;
pub mod mistralrs_server_router_builder;
pub mod openai;
pub mod openapi_doc;
mod speech_generation;
pub mod types;
pub mod util;
