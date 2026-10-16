//! Wires the opt-in security layer into the serve and from-config commands

use anyhow::Result;
use axum::Router;
use mistralrs_server_core::mistralrs_server_router_builder::MistralRsServerRouterBuilder;

use crate::args::SecurityCliOptions;

#[cfg(feature = "security")]
use mistralrs_server_core::{
    mistralrs_server_router_builder::DEFAULT_MAX_BODY_LIMIT,
    security::{SecurityGuard, SecurityOptions},
};

#[cfg(feature = "security")]
const API_KEYS_ENV: &str = "MISTRALRS_SECURITY_API_KEYS";
#[cfg(feature = "security")]
const IPINFO_TOKEN_ENV: &str = "MISTRALRS_SECURITY_IPINFO_TOKEN";

pub(crate) struct ServerSecurity {
    #[cfg(feature = "security")]
    guard: Option<SecurityGuard>,
}

impl ServerSecurity {
    #[cfg(feature = "security")]
    pub(crate) fn new(options: &SecurityCliOptions) -> Result<Self> {
        let guard = options
            .security
            .then(|| SecurityGuard::new(&security_options(options)))
            .transpose()?;
        Ok(Self { guard })
    }

    #[cfg(not(feature = "security"))]
    pub(crate) fn new(options: &SecurityCliOptions) -> Result<Self> {
        let secrets_set =
            !options.security_api_keys.is_empty() || options.security_ipinfo_token.is_some();
        if options.security || secrets_set {
            anyhow::bail!("security settings need a build with the `security` feature");
        }
        Ok(Self {})
    }

    pub(crate) fn attach(
        &self,
        builder: MistralRsServerRouterBuilder,
    ) -> MistralRsServerRouterBuilder {
        #[cfg(feature = "security")]
        if let Some(guard) = &self.guard {
            return builder.with_security(guard.clone());
        }
        builder
    }

    pub(crate) fn protect_ui(&self, router: Router) -> Router {
        #[cfg(feature = "security")]
        if let Some(guard) = &self.guard {
            return guard.apply(router, DEFAULT_MAX_BODY_LIMIT);
        }
        router
    }
}

#[cfg(feature = "security")]
fn security_options(options: &SecurityCliOptions) -> SecurityOptions {
    let defaults = SecurityOptions::default();
    let mut api_keys = options.security_api_keys.clone();
    api_keys.extend(
        std::env::var(API_KEYS_ENV)
            .iter()
            .flat_map(|keys| keys.split(','))
            .filter(|key| !key.is_empty())
            .map(str::to_string),
    );
    SecurityOptions {
        control_allow_ips: options.security_control_allow_ips.clone(),
        api_keys,
        api_key_header: options
            .security_api_key_header
            .clone()
            .unwrap_or(defaults.api_key_header),
        protect_all_routes: options.security_protect_all_routes,
        trusted_proxies: options.security_trusted_proxies.clone(),
        rate_limit_per_minute: options
            .security_rate_limit
            .unwrap_or(defaults.rate_limit_per_minute),
        gpu_rate_limit_per_minute: options
            .security_gpu_rate_limit
            .unwrap_or(defaults.gpu_rate_limit_per_minute),
        max_upload_mb: options.security_max_upload_mb,
        blocked_countries: options.security_blocked_countries.clone(),
        allowed_countries: options.security_allowed_countries.clone(),
        ipinfo_token: options
            .security_ipinfo_token
            .clone()
            .or_else(|| std::env::var(IPINFO_TOKEN_ENV).ok()),
        block_cloud_providers: options.security_block_cloud_providers.clone(),
        emergency_mode: options.security_emergency_mode,
        emergency_whitelist: options.security_emergency_whitelist.clone(),
        passive_mode: options.security_passive_mode,
    }
}
