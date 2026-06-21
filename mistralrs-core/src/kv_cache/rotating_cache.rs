use candle_core::{Result, Tensor};

use super::NormalCache;

#[derive(Debug, Clone)]
pub struct RotatingCacheSnapshot {
    pub dim: usize,
    pub current_seq_len: usize,
    pub max_seq_len: usize,
    pub capacity_seq_len: usize,
    pub retained: Option<Tensor>,
}

#[derive(Debug, Clone)]
pub struct RotatingCache {
    pub all_data: Option<Tensor>,
    pub dim: usize,
    // The total size of the sequence seen so far.
    pub current_seq_len: usize,
    // max_seq_len is the number of retained tokens in the sliding window.
    pub max_seq_len: usize,
    pub capacity_seq_len: usize,
    // The full K/V tensor returned by the last `append()` call.
    // During prefill this may be larger than the internal buffer (retained + new),
    // which is what shared KV layers need for correct attention.
    pub last_append_result: Option<Tensor>,
}

impl RotatingCache {
    pub fn new(dim: usize, max_seq_len: usize, capacity_seq_len: usize) -> Self {
        Self {
            all_data: None,
            dim,
            current_seq_len: 0,
            max_seq_len,
            capacity_seq_len: capacity_seq_len.min(max_seq_len),
            last_append_result: None,
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

    pub fn last_append_result(&self) -> Option<&Tensor> {
        self.last_append_result.as_ref()
    }

    pub fn snapshot(&self) -> Result<RotatingCacheSnapshot> {
        Ok(RotatingCacheSnapshot {
            dim: self.dim,
            current_seq_len: self.current_seq_len,
            max_seq_len: self.max_seq_len,
            capacity_seq_len: self.capacity_seq_len,
            retained: self.current_data()?,
        })
    }

    pub fn can_append_from_snapshot(
        &self,
        snapshot: &RotatingCacheSnapshot,
        append_len: usize,
    ) -> bool {
        snapshot.current_seq_len == self.current_seq_len
            && snapshot.max_seq_len == self.max_seq_len
            && append_len <= self.max_seq_len
    }

    pub fn accepted_append_from_batched_append(
        &self,
        snapshot: &RotatingCacheSnapshot,
        keep_len: usize,
        row_idx: usize,
        batch_len: usize,
    ) -> Result<Option<Tensor>> {
        let accepted_len = keep_len
            .checked_sub(snapshot.current_seq_len)
            .ok_or_else(|| {
                candle_core::Error::Msg("rotating cache rollback keep_len underflow".into())
            })?;
        if accepted_len == 0 {
            return Ok(None);
        }
        let appended = self.last_append_result.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("missing rotating cache append result".into())
        })?;
        let dim0 = appended.dim(0)?;
        if batch_len == 0 || dim0 % batch_len != 0 {
            candle_core::bail!(
                "rotating cache batch shape mismatch: dim0={dim0}, batch_len={batch_len}"
            );
        }
        let per_row = dim0 / batch_len;
        let retained_len = snapshot.current_seq_len.min(snapshot.max_seq_len);
        appended
            .narrow(0, row_idx * per_row, per_row)?
            .narrow(snapshot.dim, retained_len, accepted_len)?
            .contiguous()
            .map(Some)
    }

    pub fn restore_from_snapshot(
        snapshot: &RotatingCacheSnapshot,
        accepted_append: Option<Tensor>,
        keep_len: usize,
    ) -> Result<Self> {
        let accepted_len = keep_len
            .checked_sub(snapshot.current_seq_len)
            .ok_or_else(|| {
                candle_core::Error::Msg("rotating cache rollback keep_len underflow".into())
            })?;
        if let Some(accepted_append) = accepted_append.as_ref() {
            if accepted_append.dim(snapshot.dim)? != accepted_len {
                candle_core::bail!(
                    "rotating cache rollback accepted append length mismatch: got {}, expected {accepted_len}",
                    accepted_append.dim(snapshot.dim)?
                );
            }
        } else if accepted_len != 0 {
            candle_core::bail!(
                "rotating cache rollback missing accepted append for accepted_len={accepted_len}"
            );
        }

        let retained = match (snapshot.retained.as_ref(), accepted_append.as_ref()) {
            (Some(retained), Some(accepted)) => Tensor::cat(&[retained, accepted], snapshot.dim)?,
            (Some(retained), None) => retained.clone(),
            (None, Some(accepted)) => accepted.clone(),
            (None, None) => {
                return Ok(Self {
                    all_data: None,
                    dim: snapshot.dim,
                    current_seq_len: keep_len,
                    max_seq_len: snapshot.max_seq_len,
                    capacity_seq_len: snapshot.capacity_seq_len.min(snapshot.max_seq_len),
                    last_append_result: None,
                });
            }
        };

        let retained_len = retained.dim(snapshot.dim)?;
        let keep = retained_len.min(snapshot.max_seq_len);
        let retained = retained
            .narrow(snapshot.dim, retained_len - keep, keep)?
            .contiguous()?;
        let capacity_seq_len = snapshot
            .capacity_seq_len
            .max(keep)
            .min(snapshot.max_seq_len)
            .max(keep);
        let mut shape = retained.dims().to_vec();
        shape[snapshot.dim] = capacity_seq_len;
        let all_data = Tensor::zeros(shape, retained.dtype(), retained.device())?;
        if keep > 0 {
            all_data.slice_set(&retained, snapshot.dim, 0)?;
        }

        Ok(Self {
            all_data: Some(all_data),
            dim: snapshot.dim,
            current_seq_len: keep_len,
            max_seq_len: snapshot.max_seq_len,
            capacity_seq_len,
            last_append_result: None,
        })
    }

    pub fn reset(&mut self) {
        self.current_seq_len = 0;
        self.all_data = None;
        self.last_append_result = None;
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
        self.last_append_result = None;
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

        let result = if seq_len >= self.max_seq_len {
            let to_copy = src
                .narrow(self.dim, seq_len - self.max_seq_len, self.max_seq_len)?
                .contiguous()?;
            ad.slice_set(&to_copy, self.dim, 0)?;
            if let Some(full_kv) = prefill_full_kv {
                full_kv
            } else {
                ad.clone()
            }
        } else {
            let keep_from_old = retained_len.min(self.max_seq_len - seq_len);
            // Only rotate when keep_start > 0 (the cache has reached the sliding window and we actually need to shift entries left).
            if keep_from_old > 0 {
                let keep_start = retained_len - keep_from_old;
                if keep_start > 0 {
                    let kept = ad
                        .narrow(self.dim, keep_start, keep_from_old)?
                        .copy()?
                        .contiguous()?;
                    ad.slice_set(&kept, self.dim, 0)?;
                }
            }

            ad.slice_set(&src.contiguous()?, self.dim, keep_from_old)?;

            if let Some(full_kv) = prefill_full_kv {
                full_kv
            } else if self.current_seq_len >= self.max_seq_len {
                ad.clone()
            } else {
                ad.narrow(self.dim, 0, self.current_seq_len)?
            }
        };

        self.last_append_result = Some(result.clone());
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::RotatingCache;

    fn make_src(values: &[f32]) -> candle_core::Result<Tensor> {
        Tensor::new(values.to_vec(), &Device::Cpu)?.reshape((1, 1, values.len(), 1))
    }

    fn make_batched_src(rows: &[&[f32]]) -> candle_core::Result<Tensor> {
        let len = rows.first().map(|row| row.len()).unwrap_or(0);
        assert!(rows.iter().all(|row| row.len() == len));
        let values = rows
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect::<Vec<_>>();
        Tensor::new(values, &Device::Cpu)?.reshape((rows.len(), 1, len, 1))
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
    fn restores_from_snapshot_after_sliding_window_advance() -> candle_core::Result<()> {
        let mut cache = RotatingCache::new(2, 4, 4);
        let _ = cache.append(&make_src(&[0., 1., 2., 3., 4.])?)?;
        let snapshot = cache.snapshot()?;

        let _ = cache.append(&make_src(&[5., 6., 7.])?)?;
        let accepted = cache.accepted_append_from_batched_append(&snapshot, 7, 0, 1)?;
        assert_eq!(
            accepted.as_ref().unwrap().flatten_all()?.to_vec1::<f32>()?,
            vec![5., 6.]
        );

        let restored = RotatingCache::restore_from_snapshot(&snapshot, accepted, 7)?;
        assert_eq!(restored.current_seq_len(), 7);
        assert_eq!(
            restored
                .current_data()?
                .unwrap()
                .flatten_all()?
                .to_vec1::<f32>()?,
            vec![3., 4., 5., 6.]
        );

        Ok(())
    }

    #[test]
    fn extracts_accepted_append_from_batched_append_row() -> candle_core::Result<()> {
        let mut single = RotatingCache::new(2, 4, 4);
        let _ = single.append(&make_src(&[0., 1., 2.])?)?;
        let snapshot = single.snapshot()?;

        let mut batched = RotatingCache::new(2, 4, 4);
        let _ = batched.append(&make_batched_src(&[&[0., 1., 2.], &[10., 11., 12.]])?)?;
        let _ = batched.append(&make_batched_src(&[&[3., 4., 5.], &[13., 14., 15.]])?)?;

        let accepted = batched.accepted_append_from_batched_append(&snapshot, 5, 1, 2)?;
        assert_eq!(
            accepted.unwrap().flatten_all()?.to_vec1::<f32>()?,
            vec![13., 14.]
        );

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
