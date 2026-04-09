//! mistralrs-serve — HTTP server and web UI for mistral.rs
//!
//! This crate contains the serve command logic, web UI, and all related
//! infrastructure. It is extracted from `mistralrs-cli` to isolate
//! UI/server compilation from the CLI binary.

pub mod config;
mod serve;
pub mod ui;

pub use config::{DeviceConfig, GlobalConfig, PagedAttnConfig, RuntimeConfig, ServerConfig};
pub use serve::run_server;
