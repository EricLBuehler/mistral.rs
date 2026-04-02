use candle_core::{Result, Tensor};

use super::NormalCache;

#[derive(Debug, Clone)]
pub struct RotatingCache {
    pub all_data: Option<Tensor>,
    pub dim: usize,
    // The total size of the sequence seen so far.
    pub current_seq_len: usize,
    // max_seq_len is the number of retained tokens in the sliding window.
    pub max_seq_len: usize,
    pub capacity_seq_len: usize,
}

impl RotatingCache {
    pub fn new(dim: usize, max_seq_len: usize, capacity_seq_len: usize) -> Self {
        Self {
            all_data: None,
            dim,
            current_seq_len: 0,
            max_seq_len,
            capacity_seq_len: capacity_seq_len.min(max_seq_len),
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn current_seq_len(&self) -> usize {
        self.current_seq_len
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn all_data(&self) -> Option<&Tensor> {
        self.all_data.as_ref()
    }

    pub fn current_data(&self) -> Result<Option<Tensor>> {
        let data = match self.all_data.as_ref() {
            None => None,
            Some(d) => {
                if self.current_seq_len >= self.max_seq_len {
                    Some(d.clone())
                } else {
                    Some(d.narrow(self.dim, 0, self.current_seq_len)?)
                }
            }
        };
        Ok(data)
    }

    pub fn reset(&mut self) {
        self.current_seq_len = 0;
        self.all_data = None;
    }

    pub fn try_set_len(&self, len: usize) -> candle_core::Result<()> {
        // Once the retained window has dropped old tokens, rollback would require
        // data that is no longer present.
        if self.current_seq_len > self.max_seq_len && len < self.current_seq_len {
            candle_core::bail!(
                "Sliding KV cache cannot roll back after truncation \
                 (current_seq_len {} > max_seq_len {}, requested len {})",
                self.current_seq_len,
                self.max_seq_len,
                len,
            );
        }
        if self.current_seq_len.saturating_sub(len) > self.max_seq_len {
            candle_core::bail!(
                "Sliding KV cache tried to reset to len {len} while current is {} and max retained is {}",
                self.current_seq_len,
                self.max_seq_len
            );
        }
        Ok(())
    }

    pub fn set_len(&mut self, len: usize) -> candle_core::Result<()> {
        self.try_set_len(len)?;
        self.current_seq_len = len;
        Ok(())
    }

    pub fn append(&mut self, src: &Tensor) -> Result<Tensor> {
        let seq_len = src.dim(self.dim)?;
        if self.all_data.is_none() {
            let mut shape = src.dims().to_vec();
            shape[self.dim] = self.capacity_seq_len;
            self.all_data = Some(Tensor::zeros(shape, src.dtype(), src.device())?);
        }

        let retained_len = self.current_seq_len.min(self.max_seq_len);
        if self.current_seq_len + seq_len > self.capacity_seq_len {
            let diff = self.current_seq_len + seq_len - self.capacity_seq_len;
            let n_blocks_needed = diff.div_ceil(NormalCache::CACHE_GROW_SIZE);
            self.capacity_seq_len += n_blocks_needed * NormalCache::CACHE_GROW_SIZE;
            self.capacity_seq_len = self.capacity_seq_len.min(self.max_seq_len);
            let mut shape = src.dims().to_vec();
            shape[self.dim] = self.capacity_seq_len;
            let ad = Tensor::zeros(shape, src.dtype(), src.device())?;
            if retained_len > 0 {
                let retained = self
                    .all_data
                    .as_ref()
                    .unwrap()
                    .narrow(self.dim, 0, retained_len)?
                    .contiguous()?;
                ad.slice_set(&retained, self.dim, 0)?;
            }
            self.all_data = Some(ad);
        }

        let ad = self.all_data.as_mut().unwrap();

        // During prefill (seq_len > 1), if total tokens exceed the sliding window,
        // we need the full K/V (retained + new) for correct attention: different
        // query positions attend to different windows. Read retained BEFORE the
        // buffer is overwritten below.
        let prefill_full_kv = if seq_len > 1 && (retained_len + seq_len) > self.max_seq_len {
            Some(if retained_len > 0 {
                let retained = ad.narrow(self.dim, 0, retained_len)?.contiguous()?;
                Tensor::cat(&[&retained, &src.contiguous()?], self.dim)?
            } else {
                src.clone()
            })
        } else {
            None
        };

        self.current_seq_len += seq_len;

        if seq_len >= self.max_seq_len {
            let to_copy = src
                .narrow(self.dim, seq_len - self.max_seq_len, self.max_seq_len)?
                .contiguous()?;
            ad.slice_set(&to_copy, self.dim, 0)?;
            return if let Some(full_kv) = prefill_full_kv {
                Ok(full_kv)
            } else {
                Ok(ad.clone())
            };
        }

        let keep_from_old = retained_len.min(self.max_seq_len - seq_len);
        if keep_from_old > 0 {
            let keep_start = retained_len - keep_from_old;
            let kept = ad
                .narrow(self.dim, keep_start, keep_from_old)?
                .copy()?
                .contiguous()?;
            ad.slice_set(&kept, self.dim, 0)?;
        }

        ad.slice_set(&src.contiguous()?, self.dim, keep_from_old)?;

        if let Some(full_kv) = prefill_full_kv {
            Ok(full_kv)
        } else if self.current_seq_len >= self.max_seq_len {
            Ok(ad.clone())
        } else {
            Ok(ad.narrow(self.dim, 0, self.current_seq_len)?)
        }
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::RotatingCache;

    fn make_src(values: &[f32]) -> candle_core::Result<Tensor> {
        Tensor::new(values.to_vec(), &Device::Cpu)?.reshape((1, 1, values.len(), 1))
    }

    #[test]
    fn retains_last_window_in_order() -> candle_core::Result<()> {
        let mut cache = RotatingCache::new(2, 4, 4);

        let first = cache.append(&make_src(&[0., 1., 2.])?)?;
        assert_eq!(first.flatten_all()?.to_vec1::<f32>()?, vec![0., 1., 2.]);
        assert_eq!(cache.current_seq_len(), 3);

        let second = cache.append(&make_src(&[3., 4., 5.])?)?;
        // During multi-token append (prefill), full K/V is returned so all
        // query positions can attend to their correct sliding windows.
        assert_eq!(
            second.flatten_all()?.to_vec1::<f32>()?,
            vec![0., 1., 2., 3., 4., 5.]
        );
        assert_eq!(cache.current_seq_len(), 6);

        let current = cache.current_data()?.unwrap();
        assert_eq!(
            current.flatten_all()?.to_vec1::<f32>()?,
            vec![2., 3., 4., 5.]
        );

        Ok(())
    }

    #[test]
    fn rejects_rollback_after_truncation() -> candle_core::Result<()> {
        let mut cache = RotatingCache::new(2, 4, 4);
        let _ = cache.append(&make_src(&[0., 1., 2., 3., 4.])?)?;

        assert!(cache.try_set_len(4).is_err());
        assert!(cache.set_len(4).is_err());

        Ok(())
    }

    #[test]
    fn returns_full_kv_on_large_prefill() -> candle_core::Result<()> {
        // Sliding window = 4, but prefill has 7 tokens
        let mut cache = RotatingCache::new(2, 4, 4);

        // Prefill with more tokens than the window
        let result = cache.append(&make_src(&[0., 1., 2., 3., 4., 5., 6.])?)?;
        // Should return ALL 7 tokens for correct attention during prefill
        assert_eq!(
            result.flatten_all()?.to_vec1::<f32>()?,
            vec![0., 1., 2., 3., 4., 5., 6.]
        );
        assert_eq!(cache.current_seq_len(), 7);

        // Internal buffer should only retain the last 4
        let current = cache.current_data()?.unwrap();
        assert_eq!(
            current.flatten_all()?.to_vec1::<f32>()?,
            vec![3., 4., 5., 6.]
        );

        // Subsequent decode (single token) should work normally
        let decode = cache.append(&make_src(&[7.])?)?;
        assert_eq!(
            decode.flatten_all()?.to_vec1::<f32>()?,
            vec![4., 5., 6., 7.]
        );

        Ok(())
    }

    #[test]
    fn returns_full_kv_on_prefill_with_retained() -> candle_core::Result<()> {
        // Sliding window = 4, initial small append, then large prefill
        let mut cache = RotatingCache::new(2, 4, 4);

        // First: small append (fits in window)
        let first = cache.append(&make_src(&[0., 1., 2.])?)?;
        assert_eq!(first.flatten_all()?.to_vec1::<f32>()?, vec![0., 1., 2.]);

        // Second: prefill that overflows window (retained=3 + new=3 = 6 > 4)
        let second = cache.append(&make_src(&[3., 4., 5.])?)?;
        // Should return retained + new = all 6 tokens
        assert_eq!(
            second.flatten_all()?.to_vec1::<f32>()?,
            vec![0., 1., 2., 3., 4., 5.]
        );

        // Internal buffer should only retain the last 4
        let current = cache.current_data()?.unwrap();
        assert_eq!(
            current.flatten_all()?.to_vec1::<f32>()?,
            vec![2., 3., 4., 5.]
        );

        Ok(())
    }
}
