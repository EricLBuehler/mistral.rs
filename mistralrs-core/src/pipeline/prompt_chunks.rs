use crate::paged_attention::block_hash::{MultiModalFeature, MultimodalAttentionPolicy};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PromptChunkPlan {
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) attention_policy: MultimodalAttentionPolicy,
}

pub(crate) fn build_prompt_chunk_plan(
    total_len: usize,
    prefix_len: usize,
    chunk_size: usize,
    features: &[MultiModalFeature],
) -> Vec<PromptChunkPlan> {
    let mut pos = prefix_len.min(total_len);
    let mut chunks = Vec::new();
    let mut features = features
        .iter()
        .filter(|feature| feature.offset < total_len && feature.end() > pos)
        .collect::<Vec<_>>();
    features.sort_by_key(|feature| feature.offset);

    while pos < total_len {
        let active_features = features
            .iter()
            .filter(|feature| feature.offset <= pos && feature.end() > pos)
            .collect::<Vec<_>>();
        if !active_features.is_empty() {
            let attention_policy = if active_features
                .iter()
                .any(|feature| feature.attention_policy == MultimodalAttentionPolicy::NonCausal)
            {
                MultimodalAttentionPolicy::NonCausal
            } else {
                MultimodalAttentionPolicy::Causal
            };
            let next_feature_start = features
                .iter()
                .filter(|feature| feature.offset > pos)
                .map(|feature| feature.offset)
                .min()
                .unwrap_or(total_len);
            let next_feature_end = active_features
                .iter()
                .map(|feature| feature.end())
                .min()
                .unwrap_or(total_len);
            let end = next_feature_start.min(next_feature_end).min(total_len);
            chunks.push(PromptChunkPlan {
                start: pos,
                end,
                attention_policy,
            });
            pos = end;
            continue;
        }

        let next_feature_start = features
            .iter()
            .filter(|feature| feature.offset > pos)
            .map(|feature| feature.offset)
            .min()
            .unwrap_or(total_len);
        let end = (pos + chunk_size).min(next_feature_start).min(total_len);
        chunks.push(PromptChunkPlan {
            start: pos,
            end,
            attention_policy: MultimodalAttentionPolicy::Causal,
        });
        pos = end;
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paged_attention::block_hash::MultimodalKind;

    fn feature(
        offset: usize,
        length: usize,
        attention_policy: MultimodalAttentionPolicy,
    ) -> MultiModalFeature {
        MultiModalFeature {
            kind: MultimodalKind::Image,
            item_range: 0..1,
            hashes: vec![1],
            offset,
            length,
            attention_policy,
            splittable: false,
        }
    }

    fn policies(chunks: Vec<PromptChunkPlan>) -> Vec<(usize, usize, MultimodalAttentionPolicy)> {
        chunks
            .into_iter()
            .map(|chunk| (chunk.start, chunk.end, chunk.attention_policy))
            .collect()
    }

    #[test]
    fn keeps_media_spans_policy_homogeneous() {
        let chunks = build_prompt_chunk_plan(
            25,
            0,
            8,
            &[feature(10, 6, MultimodalAttentionPolicy::NonCausal)],
        );

        assert_eq!(
            chunks,
            vec![
                PromptChunkPlan {
                    start: 0,
                    end: 8,
                    attention_policy: MultimodalAttentionPolicy::Causal,
                },
                PromptChunkPlan {
                    start: 8,
                    end: 10,
                    attention_policy: MultimodalAttentionPolicy::Causal,
                },
                PromptChunkPlan {
                    start: 10,
                    end: 16,
                    attention_policy: MultimodalAttentionPolicy::NonCausal,
                },
                PromptChunkPlan {
                    start: 16,
                    end: 24,
                    attention_policy: MultimodalAttentionPolicy::Causal,
                },
                PromptChunkPlan {
                    start: 24,
                    end: 25,
                    attention_policy: MultimodalAttentionPolicy::Causal,
                },
            ]
        );
    }

    #[test]
    fn splits_active_media_on_overlapping_boundaries() {
        let chunks = build_prompt_chunk_plan(
            20,
            0,
            8,
            &[
                feature(4, 8, MultimodalAttentionPolicy::NonCausal),
                feature(8, 4, MultimodalAttentionPolicy::NonCausal),
            ],
        );

        assert_eq!(
            policies(chunks),
            vec![
                (0, 4, MultimodalAttentionPolicy::Causal),
                (4, 8, MultimodalAttentionPolicy::NonCausal),
                (8, 12, MultimodalAttentionPolicy::NonCausal),
                (12, 20, MultimodalAttentionPolicy::Causal),
            ]
        );
    }

    #[test]
    fn mixed_overlapping_policies_use_non_causal() {
        let chunks = build_prompt_chunk_plan(
            14,
            0,
            8,
            &[
                feature(2, 8, MultimodalAttentionPolicy::Causal),
                feature(4, 4, MultimodalAttentionPolicy::NonCausal),
            ],
        );

        assert_eq!(
            policies(chunks),
            vec![
                (0, 2, MultimodalAttentionPolicy::Causal),
                (2, 4, MultimodalAttentionPolicy::Causal),
                (4, 8, MultimodalAttentionPolicy::NonCausal),
                (8, 10, MultimodalAttentionPolicy::Causal),
                (10, 14, MultimodalAttentionPolicy::Causal),
            ]
        );
    }
}
