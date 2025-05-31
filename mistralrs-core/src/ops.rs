use candle_core::{shape::Dim, DType, Result, Tensor, D};

#[cfg(feature = "cuda")]
use crate::cuda::ffi;

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct ArgSort {
    asc: bool,
    last_dim: usize,
    inplace: bool,
}

impl candle_core::CustomOp1 for ArgSort {
    fn name(&self) -> &'static str {
        "argsort"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        panic!("not implemented!")
    }

    #[allow(clippy::cast_possible_truncation)]
    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        storage: &candle_core::CudaStorage,
        layout: &candle_core::Layout,
    ) -> Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;
        use candle_core::cuda_backend::CudaStorageSlice;
        use candle_core::cuda_backend::WrapErr;
        let dev = storage.device();
        let elem_count = layout.shape().elem_count();
        let ncols = self.last_dim as i32;
        let nrows = elem_count as i32 / ncols;
        let dst = unsafe { dev.alloc::<u32>(elem_count) }.w()?;

        use std::ffi::c_void;

        let src = match &storage.slice {
            CudaStorageSlice::U8(inp) => inp.device_ptr(),
            CudaStorageSlice::U32(inp) => inp.device_ptr(),
            CudaStorageSlice::I64(inp) => inp.device_ptr(),
            CudaStorageSlice::BF16(inp) => inp.device_ptr(),
            CudaStorageSlice::F16(inp) => inp.device_ptr(),
            CudaStorageSlice::F32(inp) => inp.device_ptr(),
            CudaStorageSlice::F64(inp) => inp.device_ptr(),
            _ => candle_core::bail!("Unexpected dtype in asort"),
        };
        let src_ptr = *src as *const c_void;
        let dst_ptr = *dst.device_ptr() as *mut c_void;
        let stream = *dev.cu_stream() as i64;
        unsafe {
            if self.asc {
                match storage.dtype() {
                    candle_core::DType::U8 => {
                        ffi::asort_asc_u8(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                    }
                    candle_core::DType::U32 => {
                        ffi::asort_asc_u32(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                    }
                    candle_core::DType::I64 => {
                        ffi::asort_asc_i64(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                    }
                    candle_core::DType::BF16 => {
                        ffi::asort_asc_bf16(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                    }
                    candle_core::DType::F16 => {
                        ffi::asort_asc_f16(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                    }
                    candle_core::DType::F32 => {
                        ffi::asort_asc_f32(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                    }
                    candle_core::DType::F64 => {
                        ffi::asort_asc_f64(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                    }
                    _ => candle_core::bail!("Unexpected dtype in asort"),
                }
            } else {
                match storage.dtype() {
                    candle_core::DType::U8 => {
                        ffi::asort_desc_u8(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                    }
                    candle_core::DType::U32 => {
                        ffi::asort_desc_u32(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                    }
                    candle_core::DType::I64 => {
                        ffi::asort_desc_i64(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                    }
                    candle_core::DType::BF16 => {
                        ffi::asort_desc_bf16(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                    }
                    candle_core::DType::F16 => {
                        ffi::asort_desc_f16(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                    }
                    candle_core::DType::F32 => {
                        ffi::asort_desc_f32(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                    }
                    candle_core::DType::F64 => {
                        ffi::asort_desc_f64(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                    }
                    _ => candle_core::bail!("Unexpected dtype in asort"),
                }
            }
        }
        let dst_ret = candle_core::cuda_backend::CudaStorage {
            slice: CudaStorageSlice::U32(dst),
            device: dev.clone(),
        };
        Ok((dst_ret, layout.shape().clone()))
    }
}

#[allow(dead_code)]
pub trait ArgSortOp {
    fn arg_sort(&self, asc: bool) -> Result<Tensor>;
    fn sort(&self, asc: bool) -> Result<(Tensor, Tensor)>;
}

impl ArgSortOp for Tensor {
    /// Returns the indices that sort the tensor along the last dimension.
    ///
    /// If `asc` is `true`, sorting is in ascending order. Otherwise sorting is performed in
    /// descending order. The sort is unstable so there is no guarantees on the final order when it
    /// comes to ties.
    fn arg_sort(&self, asc: bool) -> Result<Tensor> {
        if !self.is_contiguous() {
            return Err(candle_core::Error::RequiresContiguous { op: "arg_sort" });
        }
        let last_dim = match self.dims().last() {
            Some(last_dim) => *last_dim,
            None => candle_core::bail!("empty last-dim in arg-sort"),
        };
        // No need for a backward pass for arg sort.
        self.apply_op1_no_bwd(&ArgSort {
            asc,
            last_dim,
            inplace: false,
        })
    }

    /// Sorts the tensor along the last dimension, returns the sorted tensor together with the
    /// sorted indexes.
    ///
    /// If `asc` is `true`, sorting is in ascending order. Otherwise sorting is performed in
    /// descending order. The sort is unstable so there is no guarantees on the final order when it
    /// comes to ties.
    fn sort(&self, asc: bool) -> Result<(Tensor, Tensor)> {
        if !self.is_contiguous() {
            return Err(candle_core::Error::RequiresContiguous { op: "arg_sort" });
        }
        let last_dim = match self.dims().last() {
            Some(last_dim) => *last_dim,
            None => candle_core::bail!("empty last-dim in arg-sort"),
        };
        let sorted = self.copy()?;

        let asort = sorted.apply_op1_no_bwd(&ArgSort {
            asc,
            last_dim,
            inplace: true,
        })?;

        Ok((sorted, asort))
    }
}

#[allow(dead_code)]
pub struct TopKOutput {
    pub values: Tensor,
    pub indices: Tensor,
}

pub trait TopKLastDimOp {
    /// Topk in the last dim. `values` retains a gradient but `indices` has none w.r.t self.
    /// This expects a contiguous tensor.
    /// Note: this implements torch.topk with sorted=True.
    fn topk(&self, topk: usize) -> Result<TopKOutput>;

    /// Topk in the last dim. `values` retains a gradient but `indices` has none w.r.t self.
    /// This expects a contiguous tensor.
    /// Note: this implements torch.topk with sorted=False.
    fn topk_unsorted(&self, topk: usize) -> Result<TopKOutput>;
}

impl TopKLastDimOp for Tensor {
    fn topk(&self, topk: usize) -> Result<TopKOutput> {
        // Sorted descending
        // #[cfg(feature = "cuda")]
        // let (values, sorted_indices) = self.sort(false)?;
        // #[cfg(not(feature = "cuda"))]
        let (values, sorted_indices) = self.sort_last_dim(false)?;
        let topk_indices = sorted_indices.narrow(D::Minus1, 0, topk)?.contiguous()?;
        let topk_values = values.narrow(D::Minus1, 0, topk)?.contiguous()?;
        Ok(TopKOutput {
            values: topk_values,
            indices: topk_indices,
        })
    }

    fn topk_unsorted(&self, topk: usize) -> Result<TopKOutput> {
        // Sorted descending
        let TopKOutput { values, indices } = self.topk(topk)?;
        // Reorder the indices ascending
        #[cfg(feature = "cuda")]
        let reorder_indices = indices.arg_sort(true)?;
        #[cfg(not(feature = "cuda"))]
        let reorder_indices = indices.arg_sort_last_dim(true)?;
        let topk_indices_unsorted = indices
            .to_dtype(DType::F32)?
            .gather(&reorder_indices, D::Minus1)?
            .to_dtype(DType::U32)?;
        let topk_values_unsorted = values.gather(&reorder_indices, D::Minus1)?;
        Ok(TopKOutput {
            values: topk_values_unsorted,
            indices: topk_indices_unsorted,
        })
    }
}

pub trait RepeatInterleaveOp {
    fn repeat_interleave<D: Dim>(&self, repeats: usize, dim: D) -> Result<Tensor>;
    fn repeat_interleave_flat(&self, repeats: Vec<u32>) -> Result<Tensor>;
}

impl RepeatInterleaveOp for Tensor {
    fn repeat_interleave<D: Dim>(&self, repeats: usize, dim: D) -> Result<Tensor> {
        let dim = dim.to_index(self.shape(), "repeat_interleave")?;
        let dim_elements = self.dim(dim)?;
        // For metal
        assert!(self.dtype().is_float());
        #[allow(clippy::cast_possible_truncation)]
        let indices = Tensor::new(
            (0..dim_elements)
                .flat_map(|i| vec![i as u32; repeats])
                .collect::<Vec<_>>(),
            self.device(),
        )?;
        self.index_select(&indices, dim)
    }

    fn repeat_interleave_flat(&self, repeats: Vec<u32>) -> Result<Tensor> {
        let xs = self.flatten_all()?;
        if repeats.len() != xs.dim(0)? {
            candle_core::bail!(
                "repeats ({}) must match flattened self length ({})",
                repeats.len(),
                xs.dim(0)?
            );
        }
        #[allow(clippy::cast_possible_truncation)]
        let indices = Tensor::new(
            (0..xs.dim(0)?)
                .flat_map(|i| vec![i as u32; repeats[i] as usize])
                .collect::<Vec<_>>(),
            xs.device(),
        )?;
        xs.index_select(&indices, 0)
    }
}

pub trait SplitOp {
    fn split<D: Dim>(&self, splits: &[usize], dim: D) -> Result<Vec<Tensor>>;
}

impl SplitOp for Tensor {
    fn split<D: Dim>(&self, splits: &[usize], dim: D) -> Result<Vec<Tensor>> {
        let dim = dim.to_index(self.shape(), "split")?;
        let mut split_res = Vec::new();
        let mut index = 0;
        for split in splits {
            split_res.push(self.narrow(dim, index, *split)?);
            index += *split;
        }
        Ok(split_res)
    }
}

#[allow(dead_code)]
pub trait BincountOp {
    fn bincount(&self, minlength: u32) -> Result<Vec<u32>>;
}

#[allow(dead_code)]
fn bincount(values: &[u32], minlength: u32) -> Vec<u32> {
    // let max_val = values.iter().max().copied().unwrap_or(0);
    // let result_len = (max_val + 1).max(minlength);
    // values.iter().fold(
    //     // Start with a histogram vector of zeros.
    //     vec![0u32; result_len as usize],
    //     // For each value, update the histogram.
    //     |mut histogram, &value| {
    //         histogram[value as usize] += 1;
    //         histogram
    //     },
    // )

    use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

    // Early return if there are no values.
    if values.is_empty() {
        return vec![0u32; minlength as usize];
    }

    // Compute the maximum value in parallel.
    // SAFETY: we know `values` is nonempty.
    let max_val = *values.par_iter().max().unwrap();

    // The histogram length must cover all observed values as well as `minlength`.
    let result_len = (max_val + 1).max(minlength) as usize;

    // Build per-thread histograms in parallel.
    // We use unsafe indexing to eliminate bounds checks in the inner loop.
    values
        .par_iter()
        .fold(
            || vec![0u32; result_len],
            |mut local_hist, &v| {
                // SAFETY: v is guaranteed to be <= max_val, so it is in bounds.
                unsafe {
                    *local_hist.get_unchecked_mut(v as usize) += 1;
                }
                local_hist
            },
        )
        // Merge the per-thread histograms in parallel.
        .reduce(
            || vec![0u32; result_len],
            |mut global_hist, local_hist| {
                for i in 0..result_len {
                    // SAFETY: we know local histogram is at least result_len, as is global_hist
                    unsafe {
                        *global_hist.get_unchecked_mut(i) += local_hist.get_unchecked(i);
                    }
                }
                global_hist
            },
        )
}

#[allow(dead_code)]
impl BincountOp for Tensor {
    fn bincount(&self, minlength: u32) -> Result<Vec<u32>> {
        let values = self.to_vec1::<u32>()?;

        Ok(bincount(&values, minlength))
    }
}

// https://github.com/mokeyish/candle-ext/blob/ca4547c803469bd51c00ce5eda2f18dd249c8f10/src/triangular.rs#L21
pub fn apply_triangular(xs: &Tensor, diagonal: isize, upper: bool) -> Result<Tensor> {
    let device = xs.device();
    let (l, s) = xs.dims2()?;
    let mut xs_tri = vec![];
    for i in 0..l as isize {
        for j in 0..s as isize {
            let cond = if upper {
                i + diagonal > j
            } else {
                i + diagonal < j
            };
            xs_tri.push(if cond { 0u8 } else { 1u8 });
        }
    }
    xs * Tensor::from_vec(xs_tri, (l, s), device)?.to_dtype(xs.dtype())?
}

mod tests {
    #[test]
    fn test_topk() {
        use crate::ops::{TopKLastDimOp, TopKOutput};
        use candle_core::Tensor;
        let device = candle_core::Device::Cpu;
        //  [[1, 3, 5],
        //   [2, 4, 6]]
        let x = Tensor::arange(1f32, 7f32, &device)
            .unwrap()
            .reshape((3, 2))
            .unwrap()
            .t()
            .unwrap()
            .contiguous()
            .unwrap();
        let TopKOutput { values, indices } = x.topk(2).unwrap();
        assert_eq!(
            x.to_vec2::<f32>().unwrap(),
            vec![vec![1f32, 3f32, 5f32], vec![2f32, 4f32, 6f32]]
        );
        assert_eq!(
            values.to_vec2::<f32>().unwrap(),
            vec![vec![5f32, 3f32], vec![6f32, 4f32]]
        );
        assert_eq!(
            indices.to_vec2::<u32>().unwrap(),
            vec![vec![2u32, 1u32], vec![2u32, 1u32]]
        );
    }

    #[test]
    fn test_repeat_interleave() -> candle_core::Result<()> {
        use crate::ops::RepeatInterleaveOp;
        use candle_core::{Device, Tensor};

        let input = Tensor::new(
            vec![vec![vec![1f32, 2., 3.], vec![4f32, 5., 6.]]],
            &Device::Cpu,
        )?;

        let repeat_interleaved = input.repeat_interleave(2, 2)?;
        assert_eq!(
            repeat_interleaved.to_vec3::<f32>()?,
            vec![vec![
                vec![1., 1., 2., 2., 3., 3.],
                vec![4., 4., 5., 5., 6., 6.]
            ]]
        );

        Ok(())
    }

    #[test]
    fn test_repeat_interleave_flat() -> candle_core::Result<()> {
        use crate::ops::RepeatInterleaveOp;
        use candle_core::{Device, Tensor};

        let input = Tensor::new(vec![1., 2., 3., 4.], &Device::Cpu)?;

        let repeat_interleaved = input.repeat_interleave_flat(vec![1u32, 2u32, 3u32, 4u32])?;
        assert_eq!(
            repeat_interleaved.to_vec1::<f64>()?,
            vec![1., 2., 2., 3., 3., 3., 4., 4., 4., 4.]
        );

        Ok(())
    }
}
