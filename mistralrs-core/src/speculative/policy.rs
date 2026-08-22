#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpeculativeBatchPlan {
    pub proposal_len: usize,
    pub needs_target_hiddens: bool,
}

impl SpeculativeBatchPlan {
    pub const fn new(proposal_len: usize) -> Self {
        Self {
            proposal_len,
            needs_target_hiddens: true,
        }
    }

    pub const fn without_target_hiddens(mut self) -> Self {
        self.needs_target_hiddens = false;
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpeculativeGraphPlan {
    pub proposal_len: usize,
    pub max_batch_size: Option<usize>,
}

impl SpeculativeGraphPlan {
    pub const fn new(proposal_len: usize, max_batch_size: Option<usize>) -> Self {
        Self {
            proposal_len,
            max_batch_size,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpeculativeBatchObservation {
    pub batch_size: usize,
    pub proposal_len: usize,
    pub sequences: usize,
    pub proposed_drafts: usize,
    pub accepted_drafts: usize,
}

#[cfg(test)]
mod tests {
    use super::SpeculativeBatchPlan;

    #[test]
    fn target_hiddens_are_required_by_default() {
        assert!(SpeculativeBatchPlan::new(7).needs_target_hiddens);
        assert!(
            !SpeculativeBatchPlan::new(7)
                .without_target_hiddens()
                .needs_target_hiddens
        );
    }
}
