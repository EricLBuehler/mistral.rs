use std::sync::Arc;

use candle_core::{Result, Tensor};

use super::codec::{KvCacheCodec, KvCacheCodecRef};
use super::NormalCache;

#[derive(Debug, Clone)]
pub struct SingleCache {
    // all_data is an option on a Tensor, this makes it possible to only create the actual tensor
    // on the first call where the batch size is easily known.
    // Also this makes it safe to clone a KvCache that has been reset (as in it will not share
    // its internal state with the cloned instance).
    pub all_data: Option<Tensor>,
    pub dim: usize,
    pub current_seq_len: usize,
    pub capacity_seq_len: usize,
    pub max_seq_len: usize,
    // Optional encode/decode hook. `None` is the bit-exact default; `Some`
    // installs a quantization codec (e.g. fp8, TurboQuant). See
    // `super::codec::KvCacheCodec` for the shape/dtype contract.
    pub codec: KvCacheCodecRef,
}

impl SingleCache {
    pub fn new(dim: usize, max_seq_len: usize, capacity_seq_len: usize) -> Self {
        Self {
            all_data: None,
            dim,
            current_seq_len: 0,
            max_seq_len,
            capacity_seq_len,
            codec: None,
        }
    }

    /// Install a compression codec. Call before the first `append()` — the
    /// codec only applies to subsequent writes, so switching mid-stream would
    /// leave a mix of encoded / unencoded data in the buffer.
    pub fn set_codec(&mut self, codec: Arc<dyn KvCacheCodec>) {
        self.codec = Some(codec);
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
                let view = d.narrow(self.dim, 0, self.current_seq_len)?;
                Some(match &self.codec {
                    Some(codec) => codec.decode(&view)?,
                    None => view,
                })
            }
        };
        Ok(data)
    }

    pub fn reset(&mut self) {
        self.current_seq_len = 0;
        self.all_data = None;
    }

    pub fn try_set_len(&self, len: usize) -> candle_core::Result<()> {
        if len > self.capacity_seq_len {
            candle_core::bail!(
                "kv-cache: requested length ({}) exceeds current capacity ({})",
                len,
                self.capacity_seq_len
            );
        }
        Ok(())
    }

    pub fn set_len(&mut self, len: usize) -> candle_core::Result<()> {
        self.try_set_len(len)?;
        self.current_seq_len = len;
        Ok(())
    }

    pub fn append(&mut self, src: &Tensor) -> Result<()> {
        let seq_len = src.dim(self.dim)?;
        // This doesn't seem very idiomatic but because the creation can fail, it's tricky to use
        // self.all_data.get_or_insert_with.
        if self.all_data.is_none() {
            let mut shape = src.dims().to_vec();
            shape[self.dim] = self.capacity_seq_len;
            let ad = Tensor::zeros(shape, src.dtype(), src.device())?;
            self.all_data = Some(ad);
        };

        // Expand kv cache
        if self.current_seq_len + seq_len > self.capacity_seq_len {
            let diff = self.current_seq_len + seq_len - self.capacity_seq_len;
            let n_blocks_needed = diff.div_ceil(NormalCache::CACHE_GROW_SIZE);
            self.capacity_seq_len += n_blocks_needed * NormalCache::CACHE_GROW_SIZE;
            if self.capacity_seq_len > self.max_seq_len {
                candle_core::bail!(
                    "kv-cache: requested capacity ({}) above max seq len ({})",
                    self.capacity_seq_len,
                    self.max_seq_len
                )
            }
            let mut shape = src.dims().to_vec();
            shape[self.dim] = self.capacity_seq_len;
            let ad = Tensor::zeros(shape, src.dtype(), src.device())?;
            ad.slice_set(self.all_data.as_ref().unwrap(), self.dim, 0)?;
            self.all_data = Some(ad);
        }

        let ad = self.all_data.as_mut().unwrap();

        // Encode before storage when a codec is installed. The encoded tensor
        // preserves shape + dtype (see `KvCacheCodec` contract), so slice_set
        // doesn't care whether we wrote plain or quantized values.
        let encoded_storage;
        let to_store: &Tensor = match &self.codec {
            Some(codec) => {
                encoded_storage = codec.encode(src)?;
                &encoded_storage
            }
            None => src,
        };
        ad.slice_set(to_store, self.dim, self.current_seq_len)?;
        self.current_seq_len += seq_len;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};
    use std::sync::Arc;

    use super::super::codec::PassthroughCodec;
    use super::SingleCache;

    fn make_src(values: &[f32]) -> candle_core::Result<Tensor> {
        Tensor::new(values.to_vec(), &Device::Cpu)?.reshape((1, 1, values.len(), 1))
    }

    /// Installing a PassthroughCodec must not change observed values —
    /// proves the codec dispatch path is wired correctly.
    #[test]
    fn passthrough_codec_roundtrip() -> candle_core::Result<()> {
        let mut cache = SingleCache::new(2, 8, 8);
        cache.set_codec(Arc::new(PassthroughCodec));

        cache.append(&make_src(&[0., 1., 2.])?)?;
        cache.append(&make_src(&[3., 4.])?)?;

        let out = cache.current_data()?.unwrap();
        assert_eq!(out.flatten_all()?.to_vec1::<f32>()?, vec![0., 1., 2., 3., 4.]);
        assert_eq!(cache.current_seq_len(), 5);

        Ok(())
    }

    /// Without a codec, behaviour must be bit-exact — guards against codec
    /// dispatch silently creeping into the `None` path.
    #[test]
    fn no_codec_default_matches_legacy_behaviour() -> candle_core::Result<()> {
        let mut cache = SingleCache::new(2, 8, 8);
        assert!(cache.codec.is_none());
        cache.append(&make_src(&[0., 1., 2.])?)?;
        let out = cache.current_data()?.unwrap();
        assert_eq!(out.flatten_all()?.to_vec1::<f32>()?, vec![0., 1., 2.]);
        Ok(())
    }
}
