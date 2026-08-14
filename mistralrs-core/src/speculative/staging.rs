use crate::sequence::Sequence;

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum StagedBatchState {
    None,
    Homogeneous(usize),
    Mixed,
}

pub(crate) fn staged_batch_state(seqs: &[&mut Sequence]) -> StagedBatchState {
    staged_batch_state_from_widths(seqs.iter().map(|seq| seq.active_staged_speculative_len()))
}

fn staged_batch_state_from_widths(widths: impl IntoIterator<Item = usize>) -> StagedBatchState {
    let mut width = None;
    let mut saw_empty = false;
    for len in widths {
        if len == 0 {
            if width.is_some() {
                return StagedBatchState::Mixed;
            }
            saw_empty = true;
            continue;
        }
        if saw_empty {
            return StagedBatchState::Mixed;
        }
        match width {
            Some(existing) if existing != len => return StagedBatchState::Mixed,
            Some(_) => {}
            None => width = Some(len),
        }
    }
    width.map_or(StagedBatchState::None, StagedBatchState::Homogeneous)
}

pub(crate) fn staged_batch_width(seqs: &[&mut Sequence]) -> Option<usize> {
    match staged_batch_state(seqs) {
        StagedBatchState::Homogeneous(width) => Some(width),
        StagedBatchState::None | StagedBatchState::Mixed => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_staged_widths_disable_batched_verification_input() {
        assert_eq!(
            staged_batch_state_from_widths([15, 0]),
            StagedBatchState::Mixed
        );
        assert_eq!(
            staged_batch_state_from_widths([15, 7]),
            StagedBatchState::Mixed
        );
        assert_eq!(
            staged_batch_state_from_widths([15, 15]),
            StagedBatchState::Homogeneous(15)
        );
    }
}
