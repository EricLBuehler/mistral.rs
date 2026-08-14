#[derive(Clone, Debug)]
pub enum SpeculativeAttachKind {
    Mtp { assistant: String, n_predict: usize },
    DFlash { assistant: String, n_predict: usize },
}

#[derive(Clone, Debug)]
pub struct SpeculativeAttachInfo {
    pub kind: SpeculativeAttachKind,
}

impl SpeculativeAttachInfo {
    pub fn mtp(assistant: String, n_predict: usize) -> Self {
        Self {
            kind: SpeculativeAttachKind::Mtp {
                assistant,
                n_predict,
            },
        }
    }

    pub fn dflash(assistant: String, n_predict: usize) -> Self {
        Self {
            kind: SpeculativeAttachKind::DFlash {
                assistant,
                n_predict,
            },
        }
    }

    pub fn disables_prefix_cache(&self) -> bool {
        matches!(self.kind, SpeculativeAttachKind::DFlash { .. })
    }
}

pub fn log_attach(info: &SpeculativeAttachInfo) {
    match &info.kind {
        SpeculativeAttachKind::Mtp {
            assistant,
            n_predict,
        } => tracing::info!(
            "Speculative decoding enabled: MTP assistant `{assistant}` with n_predict={n_predict}"
        ),
        SpeculativeAttachKind::DFlash {
            assistant,
            n_predict,
        } => tracing::info!(
            "Speculative decoding enabled: DFlash assistant `{assistant}` with n_predict={n_predict}"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dflash_disables_prefix_cache_before_engine_start() {
        assert!(SpeculativeAttachInfo::dflash("assistant".to_string(), 15).disables_prefix_cache());
        assert!(!SpeculativeAttachInfo::mtp("assistant".to_string(), 6).disables_prefix_cache());
    }
}
