use std::{
    collections::{HashMap, HashSet},
    ops::Range,
};

use candle_core::{DType, Device, Result, Tensor};

use crate::paged_attention::block_hash::{MultimodalAttentionPolicy, MultimodalKind};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct MultimodalEncoderKey {
    pub kind: MultimodalKind,
    pub hash: u64,
}

pub(crate) type MultimodalEncoderOutputs = HashMap<MultimodalEncoderKey, Vec<Tensor>>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MultimodalEmbeddingMap {
    destination_positions: Vec<usize>,
    source_positions: Vec<usize>,
    source_output: usize,
    target_output: usize,
}

impl MultimodalEmbeddingMap {
    pub(crate) fn new(
        destination_positions: Vec<usize>,
        source_positions: Vec<usize>,
        source_output: usize,
    ) -> Result<Self> {
        Self::new_for_output(destination_positions, source_positions, source_output, 0)
    }

    pub(crate) fn new_for_output(
        destination_positions: Vec<usize>,
        source_positions: Vec<usize>,
        source_output: usize,
        target_output: usize,
    ) -> Result<Self> {
        if destination_positions.len() != source_positions.len() {
            candle_core::bail!(
                "multimodal embedding map has {} destinations but {} sources",
                destination_positions.len(),
                source_positions.len()
            );
        }
        let mut unique_destinations = HashSet::with_capacity(destination_positions.len());
        if destination_positions
            .iter()
            .any(|position| !unique_destinations.insert(*position))
        {
            candle_core::bail!("multimodal embedding map contains duplicate destinations");
        }
        Ok(Self {
            destination_positions,
            source_positions,
            source_output,
            target_output,
        })
    }

    pub(crate) fn contiguous(
        destination: Range<usize>,
        source_start: usize,
        source_output: usize,
    ) -> Result<Self> {
        let source_end = source_start
            .checked_add(destination.len())
            .ok_or_else(|| candle_core::Error::Msg("multimodal source range overflow".into()))?;
        Self::new(
            destination.collect(),
            (source_start..source_end).collect(),
            source_output,
        )
    }

    fn entries(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.destination_positions
            .iter()
            .copied()
            .zip(self.source_positions.iter().copied())
    }
}

#[derive(Clone, Debug)]
pub(crate) struct MultimodalItemLayout {
    pub key: MultimodalEncoderKey,
    pub item_index: usize,
    pub placeholder: Range<usize>,
    pub attention_policy: MultimodalAttentionPolicy,
    pub embedding_maps: Vec<MultimodalEmbeddingMap>,
}

impl MultimodalItemLayout {
    pub(crate) fn new(
        key: MultimodalEncoderKey,
        item_index: usize,
        placeholder: Range<usize>,
        attention_policy: MultimodalAttentionPolicy,
        embedding_maps: Vec<MultimodalEmbeddingMap>,
    ) -> Result<Self> {
        if placeholder.start > placeholder.end {
            candle_core::bail!("multimodal placeholder range is reversed");
        }
        let mut destinations = HashSet::new();
        for map in &embedding_maps {
            for destination in &map.destination_positions {
                if !placeholder.contains(destination) {
                    candle_core::bail!(
                        "multimodal embedding destination {destination} is outside placeholder {:?}",
                        placeholder
                    );
                }
                if !destinations.insert((map.target_output, *destination)) {
                    candle_core::bail!(
                        "multimodal item contains duplicate embedding destination {destination} for output {}",
                        map.target_output
                    );
                }
            }
        }
        Ok(Self {
            key,
            item_index,
            placeholder,
            attention_policy,
            embedding_maps,
        })
    }
}

#[derive(Clone, Debug)]
pub(crate) struct RequestMultimodalLayout {
    pub sequence_id: usize,
    pub query: Range<usize>,
    pub items: Vec<MultimodalItemLayout>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PackedEmbeddingCopy {
    key: MultimodalEncoderKey,
    source_output: usize,
    target_output: usize,
    source_positions: Vec<usize>,
    destination_positions: Vec<usize>,
}

#[derive(Clone, Debug)]
pub(crate) struct PackedMultimodalLayout {
    token_count: usize,
    copies: Vec<PackedEmbeddingCopy>,
}

impl PackedMultimodalLayout {
    pub(crate) fn new(requests: &[RequestMultimodalLayout]) -> Result<Self> {
        let mut packed_offset = 0usize;
        let mut copies = Vec::new();
        let mut packed_destinations = HashSet::new();
        let mut sequence_ids = HashSet::with_capacity(requests.len());

        for request in requests {
            if !sequence_ids.insert(request.sequence_id) {
                candle_core::bail!(
                    "packed multimodal layout contains duplicate sequence {}",
                    request.sequence_id
                );
            }
            if request.query.start > request.query.end {
                candle_core::bail!("multimodal query range is reversed");
            }
            for item in &request.items {
                let overlaps_query = item.placeholder.start < request.query.end
                    && request.query.start < item.placeholder.end;
                if overlaps_query
                    && item.attention_policy == MultimodalAttentionPolicy::NonCausal
                    && (request.query.start > item.placeholder.start
                        || request.query.end < item.placeholder.end)
                {
                    candle_core::bail!(
                        "noncausal multimodal item {} in sequence {} must be scheduled as a complete span",
                        item.item_index,
                        request.sequence_id
                    );
                }

                for map in &item.embedding_maps {
                    let mut source_positions = Vec::new();
                    let mut destination_positions = Vec::new();
                    for (destination, source) in map.entries() {
                        if request.query.contains(&destination) {
                            let destination = packed_offset + destination - request.query.start;
                            if !packed_destinations.insert((map.target_output, destination)) {
                                candle_core::bail!(
                                    "duplicate packed multimodal destination {destination} for output {}",
                                    map.target_output
                                );
                            }
                            source_positions.push(source);
                            destination_positions.push(destination);
                        }
                    }
                    if !source_positions.is_empty() {
                        copies.push(PackedEmbeddingCopy {
                            key: item.key,
                            source_output: map.source_output,
                            target_output: map.target_output,
                            source_positions,
                            destination_positions,
                        });
                    }
                }
            }
            packed_offset = packed_offset
                .checked_add(request.query.len())
                .ok_or_else(|| {
                    candle_core::Error::Msg("packed multimodal token count overflow".into())
                })?;
        }

        Ok(Self {
            token_count: packed_offset,
            copies,
        })
    }

    pub(crate) fn token_count(&self) -> usize {
        self.token_count
    }

    pub(crate) fn splice_embeddings(
        &self,
        text_embeddings: &Tensor,
        encoder_outputs: &MultimodalEncoderOutputs,
    ) -> Result<Tensor> {
        self.splice_output_embeddings(0, text_embeddings, encoder_outputs)
    }

    pub(crate) fn destination_positions(&self, target_output: usize) -> Vec<usize> {
        let mut destinations = self
            .copies
            .iter()
            .filter(|copy| copy.target_output == target_output)
            .flat_map(|copy| copy.destination_positions.iter().copied())
            .collect::<Vec<_>>();
        destinations.sort_unstable();
        destinations
    }

    pub(crate) fn gather_output_embeddings(
        &self,
        target_output: usize,
        reference: &Tensor,
        encoder_outputs: &MultimodalEncoderOutputs,
    ) -> Result<Tensor> {
        let hidden_size = reference.dim(candle_core::D::Minus1)?;
        let mut sources = Vec::with_capacity(self.copies.len());
        let mut destinations = Vec::new();
        for copy in self
            .copies
            .iter()
            .filter(|copy| copy.target_output == target_output)
        {
            let outputs = encoder_outputs.get(&copy.key).ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "missing {:?} encoder output with hash {}",
                    copy.key.kind, copy.key.hash
                ))
            })?;
            let output = outputs.get(copy.source_output).ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "missing encoder output {} for {:?} hash {}",
                    copy.source_output, copy.key.kind, copy.key.hash
                ))
            })?;
            if output.dim(candle_core::D::Minus1)? != hidden_size {
                candle_core::bail!(
                    "encoder output hidden size {} does not match text hidden size {hidden_size}",
                    output.dim(candle_core::D::Minus1)?
                );
            }
            let output = output
                .to_device(reference.device())?
                .to_dtype(reference.dtype())?
                .reshape(((), hidden_size))?;
            let output_rows = output.dim(0)?;
            if copy
                .source_positions
                .iter()
                .any(|position| *position >= output_rows)
            {
                candle_core::bail!(
                    "multimodal embedding source is outside encoder output with {} rows",
                    output_rows
                );
            }
            let source_indices = positions_tensor(&copy.source_positions, reference.device())?;
            sources.push(output.index_select(&source_indices, 0)?);
            destinations.extend_from_slice(&copy.destination_positions);
        }
        if sources.is_empty() {
            return Tensor::zeros((0, hidden_size), reference.dtype(), reference.device());
        }

        let source = Tensor::cat(&sources, 0)?;
        let mut permutation = (0..destinations.len()).collect::<Vec<_>>();
        permutation.sort_unstable_by_key(|index| destinations[*index]);
        if permutation.iter().copied().eq(0..permutation.len()) {
            Ok(source)
        } else {
            let permutation = positions_tensor(&permutation, reference.device())?;
            source.index_select(&permutation, 0)
        }
    }

    pub(crate) fn splice_output_embeddings(
        &self,
        target_output: usize,
        text_embeddings: &Tensor,
        encoder_outputs: &MultimodalEncoderOutputs,
    ) -> Result<Tensor> {
        let original_shape = text_embeddings.shape().clone();
        let hidden_size = text_embeddings.dim(candle_core::D::Minus1)?;
        let row_count = text_embeddings.elem_count() / hidden_size;
        if row_count != self.token_count {
            candle_core::bail!(
                "packed text embeddings have {row_count} tokens but layout has {}",
                self.token_count
            );
        }
        if !self
            .copies
            .iter()
            .any(|copy| copy.target_output == target_output)
        {
            return Ok(text_embeddings.clone());
        }

        let mut flat = text_embeddings.reshape((row_count, hidden_size))?;
        let source =
            self.gather_output_embeddings(target_output, text_embeddings, encoder_outputs)?;
        let destinations = self.destination_positions(target_output);
        let destination_indices = positions_tensor(&destinations, text_embeddings.device())?;
        let current = flat.index_select(&destination_indices, 0)?;
        let destination_indices = destination_indices.unsqueeze(1)?.repeat((1, hidden_size))?;
        flat = flat.scatter_add(&destination_indices, &(source - current)?, 0)?;
        flat.reshape(original_shape)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct MropePositionSource {
    pub position_ids: Tensor,
    pub delta: i64,
}

pub(crate) fn gather_packed_mrope_positions(
    sources: &[MropePositionSource],
    query_ranges: &[Range<usize>],
    device: &Device,
) -> Result<Tensor> {
    if sources.len() != query_ranges.len() {
        candle_core::bail!(
            "MRoPE source count {} does not match query count {}",
            sources.len(),
            query_ranges.len()
        );
    }
    if sources.is_empty() {
        candle_core::bail!("cannot gather an empty MRoPE batch");
    }

    let mut planes = None;
    let mut slices = Vec::with_capacity(sources.len());
    for (source, query) in sources.iter().zip(query_ranges) {
        if query.start > query.end {
            candle_core::bail!("MRoPE query range is reversed");
        }
        let position_ids = normalize_mrope_positions(&source.position_ids)?;
        let source_planes = position_ids.dim(0)?;
        if planes
            .replace(source_planes)
            .is_some_and(|value| value != source_planes)
        {
            candle_core::bail!("MRoPE sources have different plane counts");
        }
        let stored_len = position_ids.dim(1)?;
        let slice = if query.end <= stored_len {
            position_ids.narrow(1, query.start, query.len())?
        } else if query.start >= stored_len {
            let start = i64::try_from(query.start).map_err(candle_core::Error::wrap)?;
            let end = i64::try_from(query.end).map_err(candle_core::Error::wrap)?;
            Tensor::arange(start, end, device)?
                .broadcast_add(&Tensor::new(source.delta, device)?)?
                .reshape((1, query.len()))?
                .repeat((source_planes, 1))?
        } else {
            candle_core::bail!(
                "MRoPE query {:?} crosses the stored position boundary {stored_len}",
                query
            );
        };
        slices.push(slice.to_device(device)?.to_dtype(DType::I64)?);
    }
    Tensor::cat(&slices, 1)?.unsqueeze(1)
}

fn normalize_mrope_positions(position_ids: &Tensor) -> Result<Tensor> {
    match position_ids.dims() {
        [_, _] => Ok(position_ids.clone()),
        [_, 1, _] => position_ids.squeeze(1),
        shape => candle_core::bail!(
            "MRoPE positions must have shape [planes, length] or [planes, 1, length], got {shape:?}"
        ),
    }
}

fn positions_tensor(positions: &[usize], device: &Device) -> Result<Tensor> {
    let positions = positions
        .iter()
        .map(|position| u32::try_from(*position).map_err(candle_core::Error::wrap))
        .collect::<Result<Vec<_>>>()?;
    let len = positions.len();
    Tensor::from_vec(positions, len, device)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(
        hash: u64,
        item_index: usize,
        placeholder: Range<usize>,
        attention_policy: MultimodalAttentionPolicy,
        embedding_maps: Vec<MultimodalEmbeddingMap>,
    ) -> MultimodalItemLayout {
        MultimodalItemLayout::new(
            MultimodalEncoderKey {
                kind: MultimodalKind::Image,
                hash,
            },
            item_index,
            placeholder,
            attention_policy,
            embedding_maps,
        )
        .unwrap()
    }

    #[test]
    fn masked_placeholder_maps_only_embedding_positions() {
        let map = MultimodalEmbeddingMap::new(vec![3, 4, 6], vec![4, 5, 6], 0).unwrap();
        let layout = PackedMultimodalLayout::new(&[RequestMultimodalLayout {
            sequence_id: 0,
            query: 4..7,
            items: vec![item(
                11,
                0,
                2..7,
                MultimodalAttentionPolicy::Causal,
                vec![map],
            )],
        }])
        .unwrap();

        assert_eq!(layout.token_count(), 3);
        assert_eq!(
            layout.copies,
            vec![PackedEmbeddingCopy {
                key: MultimodalEncoderKey {
                    kind: MultimodalKind::Image,
                    hash: 11,
                },
                source_output: 0,
                target_output: 0,
                source_positions: vec![5, 6],
                destination_positions: vec![0, 2],
            }]
        );
    }

    #[test]
    fn ragged_requests_translate_to_flat_destinations() {
        let layout = PackedMultimodalLayout::new(&[
            RequestMultimodalLayout {
                sequence_id: 10,
                query: 10..13,
                items: vec![item(
                    1,
                    0,
                    11..13,
                    MultimodalAttentionPolicy::Causal,
                    vec![MultimodalEmbeddingMap::contiguous(11..13, 0, 0).unwrap()],
                )],
            },
            RequestMultimodalLayout {
                sequence_id: 11,
                query: 4..8,
                items: vec![item(
                    2,
                    0,
                    5..7,
                    MultimodalAttentionPolicy::Causal,
                    vec![MultimodalEmbeddingMap::contiguous(5..7, 2, 0).unwrap()],
                )],
            },
        ])
        .unwrap();

        assert_eq!(layout.token_count(), 7);
        assert_eq!(layout.copies[0].destination_positions, vec![1, 2]);
        assert_eq!(layout.copies[1].destination_positions, vec![4, 5]);
        assert_eq!(layout.copies[1].source_positions, vec![2, 3]);
    }

    #[test]
    fn splice_replaces_only_mapped_rows() {
        let layout = PackedMultimodalLayout::new(&[RequestMultimodalLayout {
            sequence_id: 0,
            query: 0..4,
            items: vec![item(
                7,
                0,
                1..4,
                MultimodalAttentionPolicy::Causal,
                vec![MultimodalEmbeddingMap::new(vec![1, 3], vec![1, 2], 0).unwrap()],
            )],
        }])
        .unwrap();
        let text = Tensor::from_vec(
            vec![0f32, 1., 2., 3., 4., 5., 6., 7.],
            (1, 4, 2),
            &Device::Cpu,
        )
        .unwrap();
        let encoder =
            Tensor::from_vec(vec![10f32, 11., 20., 21., 30., 31.], (3, 2), &Device::Cpu).unwrap();
        let outputs = HashMap::from([(
            MultimodalEncoderKey {
                kind: MultimodalKind::Image,
                hash: 7,
            },
            vec![encoder],
        )]);

        let result = layout
            .splice_embeddings(&text, &outputs)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(result, vec![0., 1., 20., 21., 4., 5., 30., 31.]);
    }

    #[test]
    fn splice_selects_independent_main_and_auxiliary_outputs() {
        let layout = PackedMultimodalLayout::new(&[RequestMultimodalLayout {
            sequence_id: 0,
            query: 0..2,
            items: vec![item(
                9,
                0,
                0..2,
                MultimodalAttentionPolicy::Causal,
                vec![
                    MultimodalEmbeddingMap::new(vec![0, 1], vec![1, 0], 0).unwrap(),
                    MultimodalEmbeddingMap::new_for_output(vec![0, 1], vec![2, 0], 1, 1).unwrap(),
                ],
            )],
        }])
        .unwrap();
        let text = Tensor::zeros((2, 2), DType::F32, &Device::Cpu).unwrap();
        let outputs = HashMap::from([(
            MultimodalEncoderKey {
                kind: MultimodalKind::Image,
                hash: 9,
            },
            vec![
                Tensor::from_vec(vec![1f32, 2., 3., 4.], (2, 2), &Device::Cpu).unwrap(),
                Tensor::from_vec(vec![10f32, 11., 20., 21., 30., 31.], (3, 2), &Device::Cpu)
                    .unwrap(),
            ],
        )]);

        assert_eq!(
            layout
                .splice_embeddings(&text, &outputs)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![3., 4., 1., 2.]
        );
        assert_eq!(
            layout
                .splice_output_embeddings(1, &text, &outputs)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![30., 31., 10., 11.]
        );
    }

    #[test]
    fn gather_orders_rows_by_packed_destination() {
        let layout = PackedMultimodalLayout::new(&[RequestMultimodalLayout {
            sequence_id: 0,
            query: 0..3,
            items: vec![item(
                12,
                0,
                0..3,
                MultimodalAttentionPolicy::Causal,
                vec![MultimodalEmbeddingMap::new(vec![2, 0], vec![0, 2], 0).unwrap()],
            )],
        }])
        .unwrap();
        let reference = Tensor::zeros((1, 3, 2), DType::F32, &Device::Cpu).unwrap();
        let outputs = HashMap::from([(
            MultimodalEncoderKey {
                kind: MultimodalKind::Image,
                hash: 12,
            },
            vec![
                Tensor::from_vec(vec![10f32, 11., 20., 21., 30., 31.], (3, 2), &Device::Cpu)
                    .unwrap(),
            ],
        )]);

        assert_eq!(layout.destination_positions(0), vec![0, 2]);
        assert_eq!(
            layout
                .gather_output_embeddings(0, &reference, &outputs)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![30., 31., 10., 11.]
        );
    }

    #[test]
    fn noncausal_item_cannot_be_split() {
        let error = PackedMultimodalLayout::new(&[RequestMultimodalLayout {
            sequence_id: 0,
            query: 3..5,
            items: vec![item(
                1,
                0,
                2..6,
                MultimodalAttentionPolicy::NonCausal,
                vec![MultimodalEmbeddingMap::contiguous(2..6, 0, 0).unwrap()],
            )],
        }])
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("must be scheduled as a complete span"));
    }

    #[test]
    fn gathers_prompt_and_decode_mrope_positions() {
        let first = Tensor::from_vec(
            vec![0i64, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23],
            (3, 4),
            &Device::Cpu,
        )
        .unwrap();
        let second =
            Tensor::from_vec(vec![5i64, 6, 15, 16, 25, 26], (3, 1, 2), &Device::Cpu).unwrap();
        let result = gather_packed_mrope_positions(
            &[
                MropePositionSource {
                    position_ids: first,
                    delta: 0,
                },
                MropePositionSource {
                    position_ids: second,
                    delta: -2,
                },
            ],
            &[1..3, 4..5],
            &Device::Cpu,
        )
        .unwrap();

        assert_eq!(result.dims(), &[3, 1, 3]);
        assert_eq!(
            result.flatten_all().unwrap().to_vec1::<i64>().unwrap(),
            vec![1, 2, 2, 11, 12, 2, 21, 22, 2]
        );
    }

    #[test]
    fn mrope_rejects_query_crossing_stored_boundary() {
        let query = 3..5;
        let error = gather_packed_mrope_positions(
            &[MropePositionSource {
                position_ids: Tensor::zeros((3, 4), DType::I64, &Device::Cpu).unwrap(),
                delta: 0,
            }],
            std::slice::from_ref(&query),
            &Device::Cpu,
        )
        .unwrap_err();
        assert!(error.to_string().contains("crosses the stored"));
    }
}
