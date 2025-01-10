use candle_core::{DType, Device, Error, Result, Tensor, WithDType};
use float8::F8E4M3;
use safetensors::tensor as st;
use safetensors::tensor::SafeTensors;
use std::collections::HashMap;
use std::path::Path;

fn convert_slice<T: WithDType>(data: &[u8], shape: &[usize], device: &Device) -> Result<Tensor> {
    let size_in_bytes = T::DTYPE.size_in_bytes();
    let elem_count = data.len() / size_in_bytes;
    if (data.as_ptr() as usize) % size_in_bytes == 0 {
        // SAFETY This is safe because we just checked that this
        // was correctly aligned.
        let data: &[T] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const T, elem_count) };
        Tensor::from_slice(data, shape, device)
    } else {
        // XXX: We need to specify `T` here, otherwise the compiler will infer u8 because of the following cast
        // Making this vector too small to fit a full f16/f32/f64 weights, resulting in out-of-bounds access
        let mut c: Vec<T> = Vec::with_capacity(elem_count);
        // SAFETY: We just created c, so the allocated memory is necessarily
        // contiguous and non overlapping with the view's data.
        // We're downgrading the `c` pointer from T to u8, which removes alignment
        // constraints.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), c.as_mut_ptr() as *mut u8, data.len());
            c.set_len(elem_count)
        }
        Tensor::from_slice(&c, shape, device)
    }
}

fn convert_slice_with_cast<T: Sized + Copy, U: WithDType, F: Fn(T) -> Result<U>>(
    data: &[u8],
    shape: &[usize],
    device: &Device,
    conv: F,
) -> Result<Tensor> {
    let size_in_bytes = std::mem::size_of::<T>();
    let elem_count = data.len() / size_in_bytes;
    if (data.as_ptr() as usize) % size_in_bytes == 0 {
        // SAFETY This is safe because we just checked that this
        // was correctly aligned.
        let data: &[T] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const T, elem_count) };
        let data = data.iter().map(|t| conv(*t)).collect::<Result<Vec<_>>>()?;
        Tensor::from_vec(data, shape, device)
    } else {
        // XXX: We need to specify `T` here, otherwise the compiler will infer u8 because of the following cast
        // Making this vector too small to fit a full f16/f32/f64 weights, resulting in out-of-bounds access
        let mut c: Vec<T> = Vec::with_capacity(elem_count);
        // SAFETY: We just created c, so the allocated memory is necessarily
        // contiguous and non overlapping with the view's data.
        // We're downgrading the `c` pointer from T to u8, which removes alignment
        // constraints.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), c.as_mut_ptr() as *mut u8, data.len());
            c.set_len(elem_count)
        }
        let c = c.into_iter().map(conv).collect::<Result<Vec<_>>>()?;
        Tensor::from_vec(c, shape, device)
    }
}

fn convert_with_cast_<T: Sized + Copy, U: WithDType, F: Fn(T) -> Result<U>>(
    view: &st::TensorView<'_>,
    device: &Device,
    conv: F,
) -> Result<Tensor> {
    convert_slice_with_cast::<T, U, F>(view.data(), view.shape(), device, conv)
}

fn convert_<T: WithDType>(view: &st::TensorView<'_>, device: &Device) -> Result<Tensor> {
    convert_slice::<T>(view.data(), view.shape(), device)
}

pub trait Load {
    fn load(&self, device: &Device, dtype: Option<DType>) -> Result<Tensor>;
}

impl Load for st::TensorView<'_> {
    fn load(&self, device: &Device, dtype: Option<DType>) -> Result<Tensor> {
        convert(self, device, dtype)
    }
}

fn convert(
    view: &st::TensorView<'_>,
    device: &Device,
    cast_dtype: Option<DType>,
) -> Result<Tensor> {
    match (view.dtype(), cast_dtype) {
        (st::Dtype::U8, _) => convert_::<u8>(view, device),
        (st::Dtype::U16, _) => {
            let conv = |x| Ok(u32::from(x));
            convert_with_cast_::<u16, u32, _>(view, device, conv)
        }
        (st::Dtype::U32, _) => convert_::<u32>(view, device),
        (st::Dtype::I16, _) => convert_::<i16>(view, device),
        (st::Dtype::I32, _) => convert_::<i32>(view, device),
        (st::Dtype::I64, _) => convert_::<i64>(view, device),
        (st::Dtype::BF16, None | Some(DType::BF16)) => convert_::<half::bf16>(view, device),
        (st::Dtype::F16, None | Some(DType::F16)) => convert_::<half::f16>(view, device),
        (st::Dtype::F32, _) => convert_::<f32>(view, device),
        (st::Dtype::F64, _) => convert_::<f64>(view, device),
        (st::Dtype::F8_E4M3, _) => convert_::<F8E4M3>(view, device),

        (st::Dtype::BF16, Some(DType::F16)) => {
            let conv = |x: half::bf16| Ok(half::f16::from_f32(x.to_f32()));
            convert_with_cast_::<half::bf16, half::f16, _>(view, device, conv)
        }
        (st::Dtype::BF16, Some(DType::F32)) => {
            let conv = |x: half::bf16| Ok(x.to_f32());
            convert_with_cast_::<half::bf16, f32, _>(view, device, conv)
        }
        (st::Dtype::F16, Some(DType::BF16)) => {
            let conv = |x: half::f16| Ok(half::bf16::from_f32(x.to_f32()));
            convert_with_cast_::<half::f16, half::bf16, _>(view, device, conv)
        }
        (st::Dtype::F16, Some(DType::F32)) => {
            let conv = |x: half::f16| Ok(x.to_f32());
            convert_with_cast_::<half::f16, f32, _>(view, device, conv)
        }
        (dtype, _) => Err(Error::UnsupportedSafeTensorDtype(dtype)),
    }
}

#[derive(yoke::Yokeable)]
struct SafeTensors_<'a>(SafeTensors<'a>);

pub struct MmapedSafetensors {
    safetensors: Vec<yoke::Yoke<SafeTensors_<'static>, memmap2::Mmap>>,
    routing: Option<HashMap<String, usize>>,
}

impl MmapedSafetensors {
    /// Creates a wrapper around a memory mapped file and deserialize the safetensors header.
    ///
    /// # Safety
    ///
    /// The unsafe is inherited from [`memmap2::MmapOptions`].
    pub unsafe fn new<P: AsRef<Path>>(p: P) -> Result<Self> {
        let p = p.as_ref();
        let file = std::fs::File::open(p).map_err(|e| Error::from(e).with_path(p))?;
        let file = memmap2::MmapOptions::new()
            .map(&file)
            .map_err(|e| Error::from(e).with_path(p))?;
        let safetensors = yoke::Yoke::<SafeTensors_<'static>, memmap2::Mmap>::try_attach_to_cart(
            file,
            |data: &[u8]| {
                let st = safetensors::SafeTensors::deserialize(data)
                    .map_err(|e| Error::from(e).with_path(p))?;
                Ok::<_, Error>(SafeTensors_(st))
            },
        )?;
        Ok(Self {
            safetensors: vec![safetensors],
            routing: None,
        })
    }

    pub fn load(&self, name: &str, dev: &Device, dtype: Option<DType>) -> Result<Tensor> {
        self.get(name)?.load(dev, dtype)
    }

    pub fn tensors(&self) -> Vec<(String, st::TensorView<'_>)> {
        let mut tensors = vec![];
        for safetensors in self.safetensors.iter() {
            tensors.push(safetensors.get().0.tensors())
        }
        tensors.into_iter().flatten().collect()
    }

    pub fn get(&self, name: &str) -> Result<st::TensorView<'_>> {
        let index = match &self.routing {
            None => 0,
            Some(routing) => {
                let index = routing.get(name).ok_or_else(|| {
                    Error::CannotFindTensor {
                        path: name.to_string(),
                    }
                    .bt()
                })?;
                *index
            }
        };
        Ok(self.safetensors[index].get().0.tensor(name)?)
    }
}
