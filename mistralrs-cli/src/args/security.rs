//! Opt-in HTTP security options

use clap::Args;
use serde::Deserialize;

/// Application-layer security for the HTTP server. Everything is off unless `--security` is set.
#[derive(Args, Clone, Default, Deserialize)]
pub struct SecurityCliOptions {
    /// Enable the security layer (needs a build with the `security` feature).
    #[arg(long)]
    #[serde(default)]
    pub security: bool,

    /// IPs or CIDR ranges allowed to reach the control routes.
    #[arg(long, value_delimiter = ',')]
    #[serde(default)]
    pub security_control_allow_ips: Vec<String>,

    /// API keys for the control routes, from TOML or `MISTRALRS_SECURITY_API_KEYS`; never a flag, so not in `ps`.
    #[arg(skip)]
    #[serde(default)]
    pub security_api_keys: Vec<String>,

    /// Header carrying the API key. Default: x-api-key. For `authorization` the `Bearer ` prefix is accepted.
    #[arg(long)]
    #[serde(default)]
    pub security_api_key_header: Option<String>,

    /// Apply the IP allowlist and API key to every route, not only the control routes.
    #[arg(long)]
    #[serde(default)]
    pub security_protect_all_routes: bool,

    /// Proxy IPs or CIDR ranges whose X-Forwarded-For is believed.
    #[arg(long, value_delimiter = ',')]
    #[serde(default)]
    pub security_trusted_proxies: Vec<String>,

    /// Requests per client per minute across all routes. Default: 1200.
    #[arg(long)]
    #[serde(default)]
    pub security_rate_limit: Option<u32>,

    /// Requests per client per minute on each GPU route (inference, images, speech, embeddings). Default: 120.
    #[arg(long)]
    #[serde(default)]
    pub security_gpu_rate_limit: Option<u32>,

    /// Largest accepted file or skill upload, in MB. The 50 MB body limit still applies.
    #[arg(long)]
    #[serde(default)]
    pub security_max_upload_mb: Option<u64>,

    /// Country codes to block (needs an ipinfo token).
    #[arg(long, value_delimiter = ',')]
    #[serde(default)]
    pub security_blocked_countries: Vec<String>,

    /// Country codes to allow exclusively (needs an ipinfo token).
    #[arg(long, value_delimiter = ',')]
    #[serde(default)]
    pub security_allowed_countries: Vec<String>,

    /// ipinfo.io token for country rules, from TOML or `MISTRALRS_SECURITY_IPINFO_TOKEN`; never a flag.
    #[arg(skip)]
    #[serde(default)]
    pub security_ipinfo_token: Option<String>,

    /// Cloud providers to block (AWS, GCP, Azure).
    #[arg(long, value_delimiter = ',')]
    #[serde(default)]
    pub security_block_cloud_providers: Vec<String>,

    /// Answer 503 to everyone outside the emergency whitelist.
    #[arg(long)]
    #[serde(default)]
    pub security_emergency_mode: bool,

    /// IPs or CIDR ranges still served in emergency mode.
    #[arg(long, value_delimiter = ',')]
    #[serde(default)]
    pub security_emergency_whitelist: Vec<String>,

    /// Log what would be blocked without blocking; the control allowlist and API key still apply.
    #[arg(long)]
    #[serde(default)]
    pub security_passive_mode: bool,
}
