use crate::sequence::Sequence;

#[derive(Debug)]
pub(crate) enum StagedBatchState {
    None,
    Homogeneous(usize),
    Mixed,
}

pub(crate) fn staged_batch_state(seqs: &[&mut Sequence]) -> StagedBatchState {
    let mut width = None;
    let mut saw_empty = false;
    for seq in seqs {
        let len = seq.active_staged_speculative_len();
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
