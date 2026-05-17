//! Thin re-export for backwards compatibility with the linux module layout.
//! The actual scrub logic now lives in `crate::env_scrub` so macOS shares it.

pub(crate) use crate::env_scrub::apply;
