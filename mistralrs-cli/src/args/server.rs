//! Server configuration options

use clap::{Args, ValueEnum};
use mistralrs_server_core::metrics::{AccessLogFormat, ObservabilityConfig};
use serde::Deserialize;

/// HTTP server configuration
#[derive(Args, Clone, Deserialize)]
pub struct ServerOptions {
    /// HTTP server port
    #[arg(short = 'p', long, default_value_t = 1234)]
    #[serde(default = "default_port")]
    pub port: u16,

    /// Bind address
    #[arg(long, default_value = "0.0.0.0")]
    #[serde(default = "default_host")]
    pub host: String,

    /// Disable the built-in web UI (served at /ui by default).
    #[arg(long)]
    #[serde(default)]
    pub no_ui: bool,

    /// Also expose the loaded model as an MCP server on this port (JSON-RPC 2.0 at POST /mcp).
    #[arg(long)]
    #[serde(default)]
    pub mcp_port: Option<u16>,

    /// Default maximum tool-call rounds for the agentic loop.
    /// Per-request values from the HTTP API override this. Safety cap: 256 if unset.
    #[arg(long)]
    #[serde(default)]
    pub max_tool_rounds: Option<usize>,

    /// URL to POST tool calls to for server-side execution.
    /// For security, this is only configurable server-side (not per-request via HTTP API).
    #[arg(long)]
    #[serde(default)]
    pub tool_dispatch_url: Option<String>,

    /// Disable per-request HTTP access logs.
    #[arg(long)]
    #[serde(default)]
    pub disable_access_log: bool,

    /// Format for HTTP access logs.
    #[arg(long, value_enum, default_value = "text")]
    #[serde(default)]
    pub access_log_format: AccessLogFormatArg,

    /// Include health, metrics, docs, and UI requests in HTTP access logs.
    #[arg(long)]
    #[serde(default)]
    pub access_log_health: bool,

    /// Disable the x-request-id response header.
    #[arg(long)]
    #[serde(default)]
    pub disable_request_id_header: bool,

    /// Disable Prometheus HTTP metrics and the metrics recorder.
    #[arg(long)]
    #[serde(default)]
    pub disable_metrics: bool,
}

impl Default for ServerOptions {
    fn default() -> Self {
        Self {
            port: 1234,
            host: "0.0.0.0".to_string(),
            no_ui: false,
            mcp_port: None,
            max_tool_rounds: None,
            tool_dispatch_url: None,
            disable_access_log: false,
            access_log_format: AccessLogFormatArg::Text,
            access_log_health: false,
            disable_request_id_header: false,
            disable_metrics: false,
        }
    }
}

impl ServerOptions {
    pub fn observability_config(&self) -> ObservabilityConfig {
        ObservabilityConfig {
            access_log: !self.disable_access_log,
            access_log_health: self.access_log_health,
            access_log_format: self.access_log_format.into(),
            request_id_header: !self.disable_request_id_header,
            metrics: !self.disable_metrics,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AccessLogFormatArg {
    #[default]
    Text,
    Json,
}

impl From<AccessLogFormatArg> for AccessLogFormat {
    fn from(format: AccessLogFormatArg) -> Self {
        match format {
            AccessLogFormatArg::Text => AccessLogFormat::Text,
            AccessLogFormatArg::Json => AccessLogFormat::Json,
        }
    }
}

fn default_port() -> u16 {
    1234
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}
