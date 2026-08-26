use std::{collections::HashMap, ops::Range};

use candle_core::{Result, Tensor};

use crate::pipeline::RecurrentBatchKind;

use super::{GatedDeltaNet, GdnLayerCache};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct UniformPackedShape {
    batch_size: usize,
    seq_len: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PackedGdnRow {
    token_range: Range<usize>,
    state_index: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PackedGdnGroup {
    seq_len: usize,
    rows: Vec<PackedGdnRow>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PackedGdnPlan {
    groups: Vec<PackedGdnGroup>,
    token_count: usize,
}

fn uniform_packed_shape(query_lens: &[usize], has_active_lora: bool) -> Option<UniformPackedShape> {
    let &seq_len = query_lens.first()?;
    if has_active_lora || seq_len == 0 || query_lens.iter().any(|&len| len != seq_len) {
        return None;
    }
    Some(UniformPackedShape {
        batch_size: query_lens.len(),
        seq_len,
    })
}

fn reshape_packed_input(x: &Tensor, shape: UniformPackedShape) -> Result<Tensor> {
    let (physical_batch, physical_tokens, hidden_size) = x.dims3()?;
    let expected_tokens = shape
        .batch_size
        .checked_mul(shape.seq_len)
        .ok_or_else(|| candle_core::Error::msg("packed GDN token count overflow"))?;
    if physical_batch != 1 || physical_tokens != expected_tokens {
        candle_core::bail!(
            "packed GDN cannot reshape [{physical_batch}, {physical_tokens}, {hidden_size}] into [{}, {}, {hidden_size}]",
            shape.batch_size,
            shape.seq_len
        );
    }
    x.reshape((shape.batch_size, shape.seq_len, hidden_size))
}

fn restore_packed_output(output: Tensor, physical_tokens: usize) -> Result<Tensor> {
    let (batch_size, seq_len, hidden_size) = output.dims3()?;
    let output_tokens = batch_size
        .checked_mul(seq_len)
        .ok_or_else(|| candle_core::Error::msg("packed GDN output token count overflow"))?;
    if output_tokens != physical_tokens {
        candle_core::bail!(
            "packed GDN returned {output_tokens} tokens for {physical_tokens} packed inputs"
        );
    }
    output.reshape((1, physical_tokens, hidden_size))
}

fn packed_gdn_plan(query_lens: &[usize]) -> Result<PackedGdnPlan> {
    let mut groups = Vec::<PackedGdnGroup>::new();
    let mut group_by_len = HashMap::<usize, usize>::new();
    let mut token_count = 0usize;
    for (state_index, &seq_len) in query_lens.iter().enumerate() {
        if seq_len == 0 {
            candle_core::bail!("packed GDN query lengths cannot contain zero");
        }
        let token_end = token_count
            .checked_add(seq_len)
            .ok_or_else(|| candle_core::Error::msg("packed GDN token count overflow"))?;
        let row = PackedGdnRow {
            token_range: token_count..token_end,
            state_index,
        };
        if let Some(&group_index) = group_by_len.get(&seq_len) {
            groups[group_index].rows.push(row);
        } else {
            group_by_len.insert(seq_len, groups.len());
            groups.push(PackedGdnGroup {
                seq_len,
                rows: vec![row],
            });
        }
        token_count = token_end;
    }
    Ok(PackedGdnPlan {
        groups,
        token_count,
    })
}

fn gather_tensor_rows(source: &Tensor, rows: &[PackedGdnRow]) -> Result<Tensor> {
    if let [row] = rows {
        return source.narrow(0, row.state_index, 1);
    }
    let rows = rows
        .iter()
        .map(|row| source.narrow(0, row.state_index, 1))
        .collect::<Result<Vec<_>>>()?;
    Tensor::cat(&rows, 0)
}

fn gather_token_rows(source: &Tensor, rows: &[PackedGdnRow]) -> Result<Tensor> {
    if let [row] = rows {
        return source.narrow(1, row.token_range.start, row.token_range.len());
    }
    let rows = rows
        .iter()
        .map(|row| source.narrow(1, row.token_range.start, row.token_range.len()))
        .collect::<Result<Vec<_>>>()?;
    Tensor::cat(&rows, 0)
}

fn restore_logical_rows(
    mut rows: Vec<(usize, Tensor)>,
    logical_batch: usize,
    concat_dim: usize,
    kind: &str,
) -> Result<Tensor> {
    rows.sort_unstable_by_key(|(state_index, _)| *state_index);
    if rows.len() != logical_batch
        || rows
            .iter()
            .enumerate()
            .any(|(expected, (state_index, _))| expected != *state_index)
    {
        candle_core::bail!("packed GDN returned an invalid {kind} row mapping");
    }
    let rows = rows
        .into_iter()
        .map(|(_, tensor)| tensor)
        .collect::<Vec<_>>();
    Tensor::cat(&rows, concat_dim)
}

fn try_forward_uniform_packed_gdn(
    gdn: &GatedDeltaNet,
    x: &Tensor,
    cache: &mut GdnLayerCache,
    query_lens: &[usize],
) -> Result<Option<Tensor>> {
    let Some(shape) = uniform_packed_shape(query_lens, gdn.is_dynamic_lora_active()) else {
        return Ok(None);
    };
    let physical_tokens = x.dim(1)?;
    let x = reshape_packed_input(x, shape)?;
    let output = gdn.forward(&x, cache, RecurrentBatchKind::Prefill)?;
    restore_packed_output(output, physical_tokens).map(Some)
}

pub(crate) fn try_forward_grouped_packed_gdn(
    gdn: &GatedDeltaNet,
    x: &Tensor,
    cache: &mut GdnLayerCache,
    query_lens: &[usize],
) -> Result<Option<Tensor>> {
    if let Some(output) = try_forward_uniform_packed_gdn(gdn, x, cache, query_lens)? {
        return Ok(Some(output));
    }
    if gdn.is_dynamic_lora_active() || query_lens.is_empty() {
        return Ok(None);
    }
    if cache.slots.is_some()
        || cache.conv_state.dim(0)? != query_lens.len()
        || cache.recurrent_state.dim(0)? != query_lens.len()
    {
        candle_core::bail!("packed GDN requires gathered logical state rows");
    }

    let plan = packed_gdn_plan(query_lens)?;
    let (physical_batch, physical_tokens, _) = x.dims3()?;
    if physical_batch != 1 || physical_tokens != plan.token_count {
        candle_core::bail!(
            "packed GDN input has shape {:?}, expected one batch row and {} tokens",
            x.dims(),
            plan.token_count
        );
    }
    if plan.groups.iter().all(|group| group.rows.len() == 1) {
        return Ok(None);
    }

    let logical_batch = query_lens.len();
    let mut outputs = Vec::with_capacity(logical_batch);
    let mut next_conv_states = Vec::with_capacity(logical_batch);
    let mut next_recurrent_states = Vec::with_capacity(logical_batch);
    for group in plan.groups {
        let group_x = gather_token_rows(x, &group.rows)?;
        let mut group_cache = GdnLayerCache::gathered(
            gather_tensor_rows(&cache.conv_state, &group.rows)?,
            gather_tensor_rows(&cache.recurrent_state, &group.rows)?,
            cache.state_layout,
        );
        let output = gdn.forward(&group_x, &mut group_cache, RecurrentBatchKind::Prefill)?;
        let output_shape = output.dims3()?;
        if output_shape.0 != group.rows.len() || output_shape.1 != group.seq_len {
            candle_core::bail!("packed GDN grouped forward returned an incompatible shape");
        }
        for (group_index, row) in group.rows.into_iter().enumerate() {
            outputs.push((row.state_index, output.narrow(0, group_index, 1)?));
            next_conv_states.push((
                row.state_index,
                group_cache.conv_state.narrow(0, group_index, 1)?,
            ));
            next_recurrent_states.push((
                row.state_index,
                group_cache.recurrent_state.narrow(0, group_index, 1)?,
            ));
        }
    }

    cache.conv_state = restore_logical_rows(next_conv_states, logical_batch, 0, "state")?;
    cache.recurrent_state = restore_logical_rows(next_recurrent_states, logical_batch, 0, "state")?;
    restore_logical_rows(outputs, logical_batch, 1, "output").map(Some)
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::{
        gather_tensor_rows, gather_token_rows, packed_gdn_plan, reshape_packed_input,
        restore_logical_rows, restore_packed_output, uniform_packed_shape, UniformPackedShape,
    };

    #[test]
    fn uniform_packed_shape_selects_only_safe_batches() {
        assert_eq!(
            uniform_packed_shape(&[4, 4, 4], false),
            Some(UniformPackedShape {
                batch_size: 3,
                seq_len: 4,
            })
        );
        assert_eq!(uniform_packed_shape(&[4, 3, 4], false), None);
        assert_eq!(uniform_packed_shape(&[4, 4, 4], true), None);
        assert_eq!(uniform_packed_shape(&[], false), None);
        assert_eq!(uniform_packed_shape(&[0, 0], false), None);
    }

    #[test]
    fn uniform_packed_reshape_preserves_logical_row_order() -> candle_core::Result<()> {
        let x = Tensor::from_vec((0..24).collect::<Vec<u32>>(), (1, 6, 4), &Device::Cpu)?;
        let batched = reshape_packed_input(
            &x,
            UniformPackedShape {
                batch_size: 3,
                seq_len: 2,
            },
        )?;
        assert_eq!(batched.dims(), &[3, 2, 4]);
        assert_eq!(
            batched.get(0)?.flatten_all()?.to_vec1::<u32>()?,
            (0..8).collect::<Vec<_>>()
        );
        assert_eq!(
            batched.get(1)?.flatten_all()?.to_vec1::<u32>()?,
            (8..16).collect::<Vec<_>>()
        );
        assert_eq!(
            batched.get(2)?.flatten_all()?.to_vec1::<u32>()?,
            (16..24).collect::<Vec<_>>()
        );

        let restored = restore_packed_output(batched, 6)?;
        assert_eq!(restored.dims(), &[1, 6, 4]);
        assert_eq!(
            restored.flatten_all()?.to_vec1::<u32>()?,
            (0..24).collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn uniform_packed_reshape_rejects_inconsistent_token_counts() -> candle_core::Result<()> {
        let x = Tensor::zeros((1, 5, 4), candle_core::DType::F32, &Device::Cpu)?;
        assert!(reshape_packed_input(
            &x,
            UniformPackedShape {
                batch_size: 3,
                seq_len: 2,
            },
        )
        .is_err());
        Ok(())
    }

    #[test]
    fn ragged_packed_plan_groups_equal_lengths_stably() -> candle_core::Result<()> {
        let plan = packed_gdn_plan(&[2, 5, 2, 1, 5])?;
        assert_eq!(plan.token_count, 15);
        assert_eq!(
            plan.groups
                .iter()
                .map(|group| (group.seq_len, group.rows.len()))
                .collect::<Vec<_>>(),
            vec![(2, 2), (5, 2), (1, 1)]
        );
        assert_eq!(plan.groups[0].rows[0].token_range, 0..2);
        assert_eq!(plan.groups[0].rows[1].token_range, 7..9);
        assert!(packed_gdn_plan(&[2, 0, 2]).is_err());
        assert!(packed_gdn_plan(&[usize::MAX, 1]).is_err());
        Ok(())
    }

    #[test]
    fn ragged_packed_gathers_and_restores_logical_order() -> candle_core::Result<()> {
        let plan = packed_gdn_plan(&[2, 3, 2])?;
        let tokens = Tensor::from_vec((0..7).collect::<Vec<u32>>(), (1, 7, 1), &Device::Cpu)?;
        let states = Tensor::from_vec(vec![10u32, 20, 30], (3, 1), &Device::Cpu)?;
        let repeated_group = &plan.groups[0].rows;
        assert_eq!(
            gather_token_rows(&tokens, repeated_group)?
                .flatten_all()?
                .to_vec1::<u32>()?,
            vec![0, 1, 5, 6]
        );
        assert_eq!(
            gather_tensor_rows(&states, repeated_group)?
                .flatten_all()?
                .to_vec1::<u32>()?,
            vec![10, 30]
        );

        let restored = restore_logical_rows(
            vec![
                (2, Tensor::new(&[5u32, 6], &Device::Cpu)?),
                (0, Tensor::new(&[0u32, 1], &Device::Cpu)?),
                (1, Tensor::new(&[2u32, 3, 4], &Device::Cpu)?),
            ],
            3,
            0,
            "test",
        )?;
        assert_eq!(
            restored.flatten_all()?.to_vec1::<u32>()?,
            (0..7).collect::<Vec<_>>()
        );
        Ok(())
    }
}
