use crate::{EffectiveProtection, Sandbox, SandboxError, SandboxPolicy};

pub struct NullSandbox;

impl Sandbox for NullSandbox {
    fn harden(
        &self,
        _cmd: &mut tokio::process::Command,
        _policy: &SandboxPolicy,
    ) -> Result<(), SandboxError> {
        Ok(())
    }

    fn name(&self) -> &'static str {
        "null"
    }

    fn effective(&self, _policy: &SandboxPolicy) -> EffectiveProtection {
        EffectiveProtection::default()
    }
}
