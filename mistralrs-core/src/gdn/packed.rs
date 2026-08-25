use candle_core::{Result, Tensor};

use crate::pipeline::RecurrentBatchKind;

use super::{GatedDeltaNet, GdnLayerCache};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct UniformPackedShape {
    batch_size: usize,
    seq_len: usize,
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

pub(crate) fn try_forward_uniform_packed_gdn(
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

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::{
        reshape_packed_input, restore_packed_output, uniform_packed_shape, UniformPackedShape,
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
}
