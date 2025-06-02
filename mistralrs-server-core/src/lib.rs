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
//! ```ignore
//! #[derive(OpenApi)]
//! #[openapi(
//!     paths(root, controllers::custom_chat),
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
//!         mistral_state: shared_mistralrs
//!     });
//!
//!     let app = Router::new()
//!         .route("/", get(root))
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
