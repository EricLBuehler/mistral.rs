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

    /// Whether a verify graph of width `q_len` may be replayed or captured at the padded batch `bucket`.
    pub fn allows(plans: &[Self], q_len: usize, bucket: usize) -> bool {
        plans
            .iter()
            .filter(|plan| 1 + plan.proposal_len == q_len)
            .all(|plan| plan.max_batch_size.is_none_or(|max| bucket <= max))
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
    #[test]
    fn graph_plans_cap_their_own_width_only() {
        let plans = [
            super::SpeculativeGraphPlan::new(3, Some(4)),
            super::SpeculativeGraphPlan::new(1, None),
        ];
        assert!(super::SpeculativeGraphPlan::allows(&plans, 4, 4));
        assert!(!super::SpeculativeGraphPlan::allows(&plans, 4, 8));
        assert!(super::SpeculativeGraphPlan::allows(&plans, 2, 32));
        assert!(super::SpeculativeGraphPlan::allows(&plans, 6, 32));
    }

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
