use crate::paged_attention::block_hash::{MultiModalFeature, MultimodalAttentionPolicy};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PromptChunkPlan {
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) attention_policy: MultimodalAttentionPolicy,
}

pub(crate) fn next_prompt_chunk_group(
    plan_indices: &[usize],
    chunk_plans: &[Vec<PromptChunkPlan>],
    require_uniform_query_len: bool,
) -> Option<(Vec<usize>, MultimodalAttentionPolicy, bool)> {
    let (attention_policy, is_final, query_len) =
        plan_indices
            .iter()
            .zip(chunk_plans)
            .find_map(|(&plan_idx, plan)| {
                plan.get(plan_idx).map(|chunk| {
                    (
                        chunk.attention_policy,
                        plan_idx + 1 == plan.len(),
                        chunk.end - chunk.start,
                    )
                })
            })?;
    let active_indices = plan_indices
        .iter()
        .zip(chunk_plans)
        .enumerate()
        .filter_map(|(idx, (&plan_idx, plan))| {
            plan.get(plan_idx)
                .filter(|chunk| {
                    chunk.attention_policy == attention_policy
                        && (plan_idx + 1 == plan.len()) == is_final
                        && (!require_uniform_query_len || chunk.end - chunk.start == query_len)
                })
                .map(|_| idx)
        })
        .collect();
    Some((active_indices, attention_policy, is_final))
}

fn noncausal_component_end(
    pos: usize,
    total_len: usize,
    features: &[&MultiModalFeature],
) -> Option<usize> {
    let mut end = features
        .iter()
        .filter(|feature| {
            feature.attention_policy == MultimodalAttentionPolicy::NonCausal
                && feature.offset <= pos
                && feature.end() > pos
        })
        .map(|feature| feature.end())
        .max()?
        .min(total_len);

    loop {
        let extended_end = features
            .iter()
            .filter(|feature| {
                feature.attention_policy == MultimodalAttentionPolicy::NonCausal
                    && feature.offset < end
                    && feature.end() > pos
            })
            .map(|feature| feature.end())
            .max()
            .unwrap_or(end)
            .min(total_len);
        if extended_end == end {
            return Some(end);
        }
        end = extended_end;
    }
}

fn causal_unsplittable_component_end(
    pos: usize,
    total_len: usize,
    features: &[&MultiModalFeature],
) -> Option<usize> {
    let mut end = features
        .iter()
        .filter(|feature| {
            feature.attention_policy == MultimodalAttentionPolicy::Causal
                && !feature.splittable
                && feature.offset <= pos
                && feature.end() > pos
        })
        .map(|feature| feature.end())
        .max()?
        .min(total_len);

    loop {
        let extended_end = features
            .iter()
            .filter(|feature| {
                feature.attention_policy == MultimodalAttentionPolicy::Causal
                    && !feature.splittable
                    && feature.offset < end
                    && feature.end() > pos
            })
            .map(|feature| feature.end())
            .max()
            .unwrap_or(end)
            .min(total_len);
        if extended_end == end {
            return Some(end);
        }
        end = extended_end;
    }
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
        if let Some(end) = noncausal_component_end(pos, total_len, &features) {
            chunks.push(PromptChunkPlan {
                start: pos,
                end,
                attention_policy: MultimodalAttentionPolicy::NonCausal,
            });
            pos = end;
            continue;
        }

        if let Some(end) = causal_unsplittable_component_end(pos, total_len, &features) {
            let end = features
                .iter()
                .filter(|feature| {
                    feature.attention_policy == MultimodalAttentionPolicy::NonCausal
                        && feature.offset > pos
                        && feature.offset < end
                })
                .map(|feature| feature.offset)
                .min()
                .unwrap_or(end);
            chunks.push(PromptChunkPlan {
                start: pos,
                end,
                attention_policy: MultimodalAttentionPolicy::Causal,
            });
            pos = end;
            continue;
        }

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
            let end = next_feature_start
                .min(next_feature_end)
                .min(pos.saturating_add(chunk_size))
                .min(total_len);
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
    fn chunk_groups_do_not_mix_final_and_nonfinal_sequences() {
        let plans = vec![
            vec![PromptChunkPlan {
                start: 0,
                end: 4,
                attention_policy: MultimodalAttentionPolicy::Causal,
            }],
            vec![
                PromptChunkPlan {
                    start: 0,
                    end: 2,
                    attention_policy: MultimodalAttentionPolicy::Causal,
                },
                PromptChunkPlan {
                    start: 2,
                    end: 4,
                    attention_policy: MultimodalAttentionPolicy::Causal,
                },
            ],
        ];

        assert_eq!(
            next_prompt_chunk_group(&[0, 0], &plans, false),
            Some((vec![0], MultimodalAttentionPolicy::Causal, true))
        );
        assert_eq!(
            next_prompt_chunk_group(&[1, 0], &plans, false),
            Some((vec![1], MultimodalAttentionPolicy::Causal, false))
        );
    }

    #[test]
    fn uniform_chunk_group_separates_unequal_final_tails() {
        let plans = vec![
            vec![
                PromptChunkPlan {
                    start: 0,
                    end: 4,
                    attention_policy: MultimodalAttentionPolicy::Causal,
                },
                PromptChunkPlan {
                    start: 4,
                    end: 5,
                    attention_policy: MultimodalAttentionPolicy::Causal,
                },
            ],
            vec![
                PromptChunkPlan {
                    start: 0,
                    end: 4,
                    attention_policy: MultimodalAttentionPolicy::Causal,
                },
                PromptChunkPlan {
                    start: 4,
                    end: 7,
                    attention_policy: MultimodalAttentionPolicy::Causal,
                },
            ],
        ];

        assert_eq!(
            next_prompt_chunk_group(&[1, 1], &plans, false),
            Some((vec![0, 1], MultimodalAttentionPolicy::Causal, true))
        );
        assert_eq!(
            next_prompt_chunk_group(&[1, 1], &plans, true),
            Some((vec![0], MultimodalAttentionPolicy::Causal, true))
        );
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
    fn only_splittable_causal_media_respects_chunk_size() {
        let unsplittable = feature(2, 10, MultimodalAttentionPolicy::Causal);
        let mut splittable = unsplittable.clone();
        splittable.splittable = true;

        assert_eq!(
            policies(build_prompt_chunk_plan(16, 0, 4, &[unsplittable])),
            vec![
                (0, 2, MultimodalAttentionPolicy::Causal),
                (2, 12, MultimodalAttentionPolicy::Causal),
                (12, 16, MultimodalAttentionPolicy::Causal),
            ]
        );
        assert_eq!(
            policies(build_prompt_chunk_plan(16, 0, 4, &[splittable])),
            vec![
                (0, 2, MultimodalAttentionPolicy::Causal),
                (2, 6, MultimodalAttentionPolicy::Causal),
                (6, 10, MultimodalAttentionPolicy::Causal),
                (10, 12, MultimodalAttentionPolicy::Causal),
                (12, 16, MultimodalAttentionPolicy::Causal),
            ]
        );
    }

    #[test]
    fn overlapping_unsplittable_causal_media_keeps_its_atomic_boundary() {
        let unsplittable = feature(2, 10, MultimodalAttentionPolicy::Causal);
        let mut splittable = feature(2, 4, MultimodalAttentionPolicy::Causal);
        splittable.splittable = true;

        assert_eq!(
            policies(build_prompt_chunk_plan(
                16,
                0,
                4,
                &[unsplittable, splittable],
            )),
            vec![
                (0, 2, MultimodalAttentionPolicy::Causal),
                (2, 12, MultimodalAttentionPolicy::Causal),
                (12, 16, MultimodalAttentionPolicy::Causal),
            ]
        );
    }

    #[test]
    fn overlapping_noncausal_media_is_one_atomic_chunk() {
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
                (4, 12, MultimodalAttentionPolicy::NonCausal),
                (12, 20, MultimodalAttentionPolicy::Causal),
            ]
        );
    }

    #[test]
    fn transitively_overlapping_noncausal_media_is_one_atomic_chunk() {
        let chunks = build_prompt_chunk_plan(
            24,
            0,
            8,
            &[
                feature(4, 6, MultimodalAttentionPolicy::NonCausal),
                feature(8, 6, MultimodalAttentionPolicy::NonCausal),
                feature(13, 5, MultimodalAttentionPolicy::NonCausal),
            ],
        );

        assert_eq!(
            policies(chunks),
            vec![
                (0, 4, MultimodalAttentionPolicy::Causal),
                (4, 18, MultimodalAttentionPolicy::NonCausal),
                (18, 24, MultimodalAttentionPolicy::Causal),
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
