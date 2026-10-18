const MIN_SHARED_PREFIX_TOKENS: usize = 32;

pub(super) struct SharedPrefixBatch {
    pub tokens: Vec<u32>,
    pub positions: Vec<u32>,
    pub pooled_indices: Vec<u32>,
    prefix_len: usize,
    branch_starts: Vec<usize>,
}

impl SharedPrefixBatch {
    pub fn new(sequences: &[&[u32]]) -> Option<Self> {
        if sequences.len() < 2 {
            return None;
        }
        let first = sequences[0];
        let prefix_len = sequences[1..]
            .iter()
            .map(|sequence| {
                first
                    .iter()
                    .zip(*sequence)
                    .take_while(|(a, b)| a == b)
                    .count()
            })
            .min()?;
        if prefix_len < MIN_SHARED_PREFIX_TOKENS {
            return None;
        }
        let mut tokens = first[..prefix_len].to_vec();
        let mut positions: Vec<_> = (0..prefix_len as u32).collect();
        let mut branch_starts = vec![0; prefix_len];
        let mut pooled_indices = Vec::with_capacity(sequences.len());
        for sequence in sequences {
            let start = tokens.len();
            tokens.extend_from_slice(&sequence[prefix_len..]);
            positions.extend(prefix_len as u32..sequence.len() as u32);
            branch_starts.resize(tokens.len(), start);
            pooled_indices.push(if sequence.len() == prefix_len {
                (prefix_len - 1) as u32
            } else {
                (tokens.len() - 1) as u32
            });
        }
        Some(Self {
            tokens,
            positions,
            pooled_indices,
            prefix_len,
            branch_starts,
        })
    }

    pub fn mask(&self) -> Vec<f32> {
        (0..self.tokens.len())
            .flat_map(|query| {
                (0..self.tokens.len()).map(move |key| {
                    if key <= query && (key < self.prefix_len || key >= self.branch_starts[query]) {
                        0.
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn branches_share_only_prefix_and_preserve_positions() {
        let prefix: Vec<_> = (0..MIN_SHARED_PREFIX_TOKENS as u32).collect();
        let a = [prefix.as_slice(), &[101, 102]].concat();
        let b = [prefix.as_slice(), &[201, 202, 203]].concat();
        let batch = SharedPrefixBatch::new(&[&a, &prefix, &b]).unwrap();
        let p = prefix.len();
        assert_eq!(batch.tokens.len(), p + 5);
        assert_eq!(
            &batch.positions[p..],
            &[p as u32, p as u32 + 1, p as u32, p as u32 + 1, p as u32 + 2]
        );
        assert_eq!(
            batch.pooled_indices,
            vec![(p + 1) as u32, (p - 1) as u32, (p + 4) as u32]
        );
        let mask = batch.mask();
        for query in 0..batch.tokens.len() {
            for key in 0..batch.tokens.len() {
                let same_branch = (query < p + 2) == (key < p + 2);
                let allowed = key <= query && (key < p || same_branch);
                assert_eq!(mask[query * batch.tokens.len() + key].is_finite(), allowed);
            }
        }
        assert!(SharedPrefixBatch::new(&[&a]).is_none());
        assert!(SharedPrefixBatch::new(&[&a, &[999]]).is_none());
    }
}
