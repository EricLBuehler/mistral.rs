use candle_core::{DType, Device, Error, IndexOp, Result, Shape, Storage, Tensor, WithDType};
use candle_nn::var_builder::{Backend, SimpleBackend, VarBuilderArgs};
use float8::F8E4M3;
use regex::Regex;
use safetensors::tensor as st;
use safetensors::tensor::SafeTensors;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::ShardedVarBuilder;

const UQFF_DUMMY_TENSOR_MARKER: &str = "__mistralrs_uqff_dummy_tensor__";
const CPU_LOAD_PAGE_ADVICE_THRESHOLD: usize = 32 << 20;

fn streaming_cpu_load(device: &Device) -> bool {
    device.is_cpu()
        && crate::get_immediate_isq().is_some_and(|isq| {
            isq.ty.is_some() || isq.overrides.iter().any(|entry| entry.ty.is_some())
        })
}

#[cfg(target_os = "linux")]
fn advise_load_pages(pointer: *const u8, len: usize, advice: libc::c_int) {
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page <= 0 {
        return;
    }
    let page = page as usize;
    let start = (pointer as usize).div_ceil(page) * page;
    let end = (pointer as usize + len) / page * page;
    if end > start {
        unsafe {
            libc::madvise(start as *mut libc::c_void, end - start, advice);
        }
    }
}

fn cpu_weight_buffer<T>(elements: usize, streaming: bool) -> Vec<T> {
    let data: Vec<T> = Vec::with_capacity(elements);
    #[cfg(target_os = "linux")]
    if streaming && elements * std::mem::size_of::<T>() >= CPU_LOAD_PAGE_ADVICE_THRESHOLD {
        // THP compaction on these temporary buffers stalls SMMU invalidation on unified CUDA hosts.
        advise_load_pages(
            data.as_ptr().cast(),
            elements * std::mem::size_of::<T>(),
            libc::MADV_NOHUGEPAGE,
        );
    }
    #[cfg(not(target_os = "linux"))]
    let _ = streaming;
    data
}

fn release_loaded_pages(bytes: &[u8], streaming: bool) {
    #[cfg(target_os = "linux")]
    if streaming {
        // Only copied, read-only checkpoint spans reach here; PLE keeps its own live maps.
        advise_load_pages(bytes.as_ptr(), bytes.len(), libc::MADV_DONTNEED);
    }
    #[cfg(not(target_os = "linux"))]
    let _ = (bytes, streaming);
}

fn load_cpu_slices<T: WithDType + bytemuck::Pod>(
    slices: safetensors::slice::SliceIterator<'_>,
    shape: &[usize],
    streaming: bool,
) -> Result<Tensor> {
    let streaming = streaming && slices.remaining_byte_len() >= CPU_LOAD_PAGE_ADVICE_THRESHOLD;
    let mut data = cpu_weight_buffer::<T>(
        slices.remaining_byte_len() / std::mem::size_of::<T>(),
        streaming,
    );
    for slice in slices {
        match bytemuck::try_cast_slice(slice) {
            Ok(values) => data.extend_from_slice(values),
            Err(bytemuck::PodCastError::TargetAlignmentGreaterAndInputNotAligned) => data.extend(
                slice
                    .chunks_exact(std::mem::size_of::<T>())
                    .map(bytemuck::pod_read_unaligned::<T>),
            ),
            Err(error) => return Err(Error::msg(error)),
        }
        release_loaded_pages(slice, streaming);
    }
    Tensor::from_vec(data, shape, &Device::Cpu)
}

fn load_tensor_slices(
    slices: safetensors::slice::SliceIterator<'_>,
    shape: &[usize],
    source_dtype: DType,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    if device.is_cpu() {
        let streaming = streaming_cpu_load(device);
        match source_dtype {
            DType::F16 => {
                return load_cpu_slices::<half::f16>(slices, shape, streaming)?.to_dtype(dtype)
            }
            DType::BF16 => {
                return load_cpu_slices::<half::bf16>(slices, shape, streaming)?.to_dtype(dtype)
            }
            DType::F32 => return load_cpu_slices::<f32>(slices, shape, streaming)?.to_dtype(dtype),
            _ => {}
        }
    }
    let raw = collect_tensor_slices(slices);
    Tensor::from_raw_buffer(&raw, source_dtype, shape, device)?.to_dtype(dtype)
}

fn collect_tensor_slices<'a>(
    mut slices: safetensors::slice::SliceIterator<'a>,
) -> std::borrow::Cow<'a, [u8]> {
    let Some(first) = slices.next() else {
        return std::borrow::Cow::Borrowed(&[]);
    };
    let remaining = slices.remaining_byte_len();
    if remaining == 0 {
        return std::borrow::Cow::Borrowed(first);
    }
    let mut bytes = Vec::with_capacity(first.len() + remaining);
    bytes.extend_from_slice(first);
    for slice in slices {
        bytes.extend_from_slice(slice);
    }
    std::borrow::Cow::Owned(bytes)
}

fn marked_uqff_dummy_tensor(name: &str) -> Option<&str> {
    name.strip_prefix(UQFF_DUMMY_TENSOR_MARKER)
        .and_then(|path| path.strip_prefix('.'))
}

fn matches_dummy_regex(make_dummy_regexes: &Option<Arc<Vec<Regex>>>, path: &str) -> bool {
    make_dummy_regexes
        .as_ref()
        .is_some_and(|regexes| regexes.iter().any(|regex| regex.is_match(path)))
}

fn convert_slice<T: WithDType>(data: &[u8], shape: &[usize], device: &Device) -> Result<Tensor> {
    let size_in_bytes = T::DTYPE.size_in_bytes();
    let elem_count = data.len() / size_in_bytes;
    if (data.as_ptr() as usize).is_multiple_of(size_in_bytes) {
        // SAFETY This is safe because we just checked that this
        // was correctly aligned.
        let data: &[T] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const T, elem_count) };
        if streaming_cpu_load(device)
            && data.len() * size_in_bytes >= CPU_LOAD_PAGE_ADVICE_THRESHOLD
        {
            let mut owned = cpu_weight_buffer(elem_count, true);
            owned.extend_from_slice(data);
            Tensor::from_vec(owned, shape, device)
        } else {
            Tensor::from_slice(data, shape, device)
        }
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
    if (data.as_ptr() as usize).is_multiple_of(size_in_bytes) {
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
        (st::Dtype::F32, Some(DType::BF16)) => {
            let conv = |x: f32| Ok(half::bf16::from_f32(x));
            convert_with_cast_::<f32, half::bf16, _>(view, device, conv)
        }
        (st::Dtype::F32, Some(DType::F16)) => {
            let conv = |x: f32| Ok(half::f16::from_f32(x));
            convert_with_cast_::<f32, half::f16, _>(view, device, conv)
        }

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
        (st::Dtype::F6_E2M3, _)
        | (st::Dtype::F6_E3M2, _)
        | (st::Dtype::F4, _)
        | (st::Dtype::F8_E8M0, _) => {
            // For dummy types, we need to handle loading by creating a dummy tensor
            // Since these types don't have actual data representation, we'll create
            // a tensor that indicates it's a dummy type
            convert_dummy(view, device)
        }
        (dtype, _) => Err(Error::UnsupportedSafeTensorDtype(dtype)),
    }
}

fn convert_dummy(view: &st::TensorView<'_>, device: &Device) -> Result<Tensor> {
    // For dummy types, we'll create the appropriate storage variant that preserves
    // both the raw data and the correct dtype
    let (dtype, _dtype_name) = match view.dtype() {
        st::Dtype::F6_E2M3 => (DType::F6E2M3, "F6_E2M3 (MX6)"),
        st::Dtype::F6_E3M2 => (DType::F6E3M2, "F6_E3M2 (MX6)"),
        st::Dtype::F4 => (DType::F4, "F4 (MX4)"),
        st::Dtype::F8_E8M0 => (DType::F8E8M0, "F8_E8M0"),
        _ => unreachable!("convert_dummy called with non-dummy dtype"),
    };

    // Load the raw bytes
    let data = view.data();
    let shape = view.shape();

    // Create storage with the appropriate dummy type variant
    let storage = match device {
        Device::Cpu => {
            let cpu_storage = match dtype {
                DType::F6E2M3 => candle_core::cpu_backend::CpuStorage::F6E2M3(data.to_vec()),
                DType::F6E3M2 => candle_core::cpu_backend::CpuStorage::F6E3M2(data.to_vec()),
                DType::F4 => candle_core::cpu_backend::CpuStorage::F4(data.to_vec()),
                DType::F8E8M0 => candle_core::cpu_backend::CpuStorage::F8E8M0(data.to_vec()),
                _ => unreachable!(),
            };
            Storage::Cpu(cpu_storage)
        }
        #[cfg(feature = "cuda")]
        Device::Cuda(device) => {
            let mut slice = unsafe { device.alloc::<u8>(data.len())? };
            device.memcpy_htod(data, &mut slice)?;

            let slice = match dtype {
                DType::F6E2M3 => candle_core::cuda_backend::CudaStorageSlice::F6E2M3(slice),
                DType::F6E3M2 => candle_core::cuda_backend::CudaStorageSlice::F6E3M2(slice),
                DType::F4 => candle_core::cuda_backend::CudaStorageSlice::F4(slice),
                DType::F8E8M0 => candle_core::cuda_backend::CudaStorageSlice::F8E8M0(slice),
                _ => unreachable!(),
            };
            let storage = candle_core::cuda_backend::CudaStorage {
                slice,
                device: device.clone(),
            };
            Storage::Cuda(storage)
        }
        #[cfg(not(feature = "cuda"))]
        Device::Cuda(_) => {
            return Err(Error::Msg("CUDA support not compiled".to_string()));
        }
        #[cfg(feature = "metal")]
        Device::Metal(device) => {
            let buffer = device.new_buffer_with_data(data)?;

            let storage = candle_core::metal_backend::MetalStorage::new(
                buffer,
                device.clone(),
                data.len(),
                dtype,
            );
            Storage::Metal(storage)
        }
        #[cfg(not(feature = "metal"))]
        Device::Metal(_) => {
            return Err(Error::Msg("Metal support not compiled".to_string()));
        }
    };

    Ok(Tensor::from((storage, shape)))
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

    /// Creates a wrapper around multiple memory mapped file and deserialize the safetensors headers.
    ///
    /// If a tensor name appears in multiple files, the last entry is returned.
    ///
    /// # Safety
    ///
    /// The unsafe is inherited from [`memmap2::MmapOptions`].
    pub unsafe fn multi<P: AsRef<Path>>(paths: &[P]) -> Result<Self> {
        Self::multi_impl(paths, false)
    }

    /// Creates a wrapper around multiple memory mapped files and rejects duplicate tensor names.
    ///
    /// # Safety
    ///
    /// The unsafe is inherited from [`memmap2::MmapOptions`].
    pub unsafe fn multi_unique<P: AsRef<Path>>(paths: &[P]) -> Result<Self> {
        Self::multi_impl(paths, true)
    }

    unsafe fn multi_impl<P: AsRef<Path>>(paths: &[P], reject_duplicates: bool) -> Result<Self> {
        let mut routing: HashMap<String, usize> = HashMap::new();
        let mut safetensors = vec![];
        for (index, p) in paths.iter().enumerate() {
            let p = p.as_ref();
            let file = std::fs::File::open(p).map_err(|e| Error::from(e).with_path(p))?;
            let file = memmap2::MmapOptions::new()
                .map(&file)
                .map_err(|e| Error::from(e).with_path(p))?;
            let data = yoke::Yoke::<SafeTensors_<'static>, memmap2::Mmap>::try_attach_to_cart(
                file,
                |data: &[u8]| {
                    let st = safetensors::SafeTensors::deserialize(data)
                        .map_err(|e| Error::from(e).with_path(p))?;
                    Ok::<_, Error>(SafeTensors_(st))
                },
            )?;
            for k in data.get().0.names() {
                if let Some(previous_index) = routing.get(k).copied().filter(|_| reject_duplicates)
                {
                    let previous = paths[previous_index].as_ref();
                    candle_core::bail!(
                        "Duplicate tensor key `{k}` found in `{}` and `{}`.",
                        previous.display(),
                        p.display()
                    );
                }
                routing.insert(k.to_string(), index);
            }
            safetensors.push(data)
        }
        Ok(Self {
            safetensors,
            routing: Some(routing),
        })
    }

    pub fn load(&self, name: &str, dev: &Device, dtype: Option<DType>) -> Result<Tensor> {
        let view = self.get(name)?;
        let tensor = view.load(dev, dtype)?;
        release_loaded_pages(
            view.data(),
            streaming_cpu_load(dev) && view.data().len() >= CPU_LOAD_PAGE_ADVICE_THRESHOLD,
        );
        Ok(tensor)
    }

    pub fn tensors(&self) -> Vec<(String, st::TensorView<'_>)> {
        let mut tensors = vec![];
        for safetensors in self.safetensors.iter() {
            tensors.push(safetensors.get().0.tensors())
        }
        tensors.into_iter().flatten().collect()
    }

    pub(crate) fn tensors_by_file(&self) -> Vec<Vec<(String, st::TensorView<'_>)>> {
        self.safetensors
            .iter()
            .map(|safetensors| safetensors.get().0.tensors())
            .collect()
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

impl SimpleBackend for MmapedSafetensors {
    fn get(
        &self,
        s: Shape,
        name: &str,
        _: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> Result<Tensor> {
        let tensor = self.get_unchecked(name, dtype, dev)?;
        if tensor.shape() != &s {
            Err(candle_core::Error::UnexpectedShape {
                msg: format!("shape mismatch for {name}"),
                expected: s,
                got: tensor.shape().clone(),
            }
            .bt())?
        }
        Ok(tensor)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> Result<Tensor> {
        self.load(name, dev, Some(dtype))
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.get(name).is_ok()
    }
}

pub enum ShardedSafeTensors {
    Sharded {
        b: Arc<MmapedSafetensors>,
        make_dummy_regexes: Option<Arc<Vec<Regex>>>,
        predicate: Arc<dyn Fn(String) -> bool + Send + Sync + 'static>,
    },
    SimpleBackend {
        b: Box<dyn SimpleBackend + 'static>,
        make_dummy_regexes: Option<Arc<Vec<Regex>>>,
    },
}

pub fn full_tensor_name(vb: &ShardedVarBuilder, name: &str) -> String {
    let prefix = vb.prefix();
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}

pub fn is_uqff_dummy_tensor(vb: &ShardedVarBuilder, name: &str) -> bool {
    let path = full_tensor_name(vb, name);
    vb.root()
        .contains_tensor(&format!("{UQFF_DUMMY_TENSOR_MARKER}.{path}"))
}

impl ShardedSafeTensors {
    /// Initializes a `VarBuilder` that retrieves tensors stored in a collection of safetensors
    /// files and make them usable in a sharded way.
    ///
    /// - If `regexes` is specified, this will be used in `make_dummy_predicate` based on `.any`
    /// - Only include keys for which predicate evaluates to true.
    ///
    /// # Safety
    ///
    /// The unsafe is inherited from [`memmap2::MmapOptions`].
    pub unsafe fn sharded<P: AsRef<std::path::Path>>(
        paths: &[P],
        dtype: DType,
        dev: &Device,
        make_dummy_regexes: Option<Arc<Vec<Regex>>>,
        predicate: Arc<dyn Fn(String) -> bool + Send + Sync + 'static>,
    ) -> Result<ShardedVarBuilder> {
        let tensors = Arc::new(MmapedSafetensors::multi(paths)?);
        // mirror get()'s gating so tensor_shape never reports a tensor get() would refuse
        let shapes = tensors
            .tensors()
            .into_iter()
            .filter(|(name, _)| {
                !matches_dummy_regex(&make_dummy_regexes, name) && predicate(name.to_string())
            })
            .map(|(name, view)| (name, view.shape().to_vec()))
            .collect();
        let backend = ShardedSafeTensors::Sharded {
            b: tensors.clone(),
            make_dummy_regexes,
            predicate,
        };
        Ok(
            ShardedVarBuilder::from_varbuilder(VarBuilderArgs::new_with_args(backend, dtype, dev))
                .with_shapes(shapes)
                .with_raw_safetensors(tensors),
        )
    }
}

/// Tensor shape enumeration for in-memory backends, feeding the VB's shape index.
pub trait TensorShapes {
    fn tensor_shapes(&self) -> HashMap<String, Vec<usize>>;
}

impl TensorShapes for HashMap<String, Tensor> {
    fn tensor_shapes(&self) -> HashMap<String, Vec<usize>> {
        self.iter()
            .map(|(name, t)| (name.clone(), t.dims().to_vec()))
            .collect()
    }
}

impl TensorShapes for candle_nn::VarMap {
    fn tensor_shapes(&self) -> HashMap<String, Vec<usize>> {
        self.data()
            .lock()
            .unwrap()
            .iter()
            .map(|(name, var)| (name.clone(), var.as_tensor().dims().to_vec()))
            .collect()
    }
}

impl ShardedSafeTensors {
    pub fn wrap<B: SimpleBackend + TensorShapes + 'static>(
        backend: B,
        dtype: DType,
        dev: Device,
    ) -> ShardedVarBuilder {
        Self::wrap_with_dummy_regexes(backend, dtype, dev, None)
    }

    pub fn wrap_with_dummy_regexes<B: SimpleBackend + TensorShapes + 'static>(
        backend: B,
        dtype: DType,
        dev: Device,
        make_dummy_regexes: Option<Arc<Vec<Regex>>>,
    ) -> ShardedVarBuilder {
        let shapes = backend.tensor_shapes();
        ShardedVarBuilder::from_varbuilder(VarBuilderArgs::new_with_args(
            Self::SimpleBackend {
                b: Box::new(backend),
                make_dummy_regexes,
            },
            dtype,
            &dev,
        ))
        .with_shapes(shapes)
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Shard {
    Simple {
        dim: usize,
        rank: usize,
        world_size: usize,
    },
    Offset {
        dim: usize,
        offset: usize,
        len: usize,
    },
}

impl Shard {
    pub fn apply_to(&self, tensor: &Tensor) -> Result<Tensor> {
        match *self {
            Shard::Simple {
                dim,
                rank,
                world_size,
            } => {
                let size = tensor.dim(dim)?;
                let shape = tensor.dims().to_vec();

                if size % world_size != 0 {
                    return Err(Error::ShapeMismatchSplit {
                        shape: shape.into(),
                        dim,
                        n_parts: world_size,
                    });
                }
                let block_size = size / world_size;
                let start = rank * block_size;
                let stop = (rank + 1) * block_size;

                if dim == 0 {
                    tensor.i(start..stop)
                } else if dim == 1 {
                    tensor.i((.., start..stop))
                } else if dim == 2 {
                    tensor.i((.., .., start..stop))
                } else {
                    candle_core::bail!("Got sharded on dimensions != 0 or 1 or 2")
                }
            }
            Shard::Offset { dim, offset, len } => {
                let start = offset;
                let stop = start + len;

                if dim == 0 {
                    tensor.i(start..stop)
                } else if dim == 1 {
                    tensor.i((.., start..stop))
                } else if dim == 2 {
                    tensor.i((.., .., start..stop))
                } else {
                    candle_core::bail!("Got sharded on dimensions != 0 or 1 or 2")
                }
            }
        }
    }
}

impl Default for Shard {
    fn default() -> Self {
        Self::Simple {
            dim: 0,
            rank: 0,
            world_size: 1,
        }
    }
}

/// Get part of a tensor, typically used to do Tensor Parallelism sharding.
///
/// If the tensor is of size (1024, 1024).
///
/// `dim` corresponds to the dimension to slice into
/// `rank` is the rank of the current process
/// `world_size` is the total number of ranks in the process group
///
/// `get_sharded("tensor", 0, 0, 2)` means `tensor.i((..512))`
/// `get_sharded("tensor", 0, 1, 2)` means `tensor.i((512..))`
/// `get_sharded("tensor", 1, 0, 2)` means `tensor.i((.., ..512))`
impl Backend for ShardedSafeTensors {
    type Hints = Shard;

    fn get(
        &self,
        target_shape: Shape,
        path: &str,
        h: Self::Hints,
        dtype: DType,
        dev: &Device,
    ) -> Result<Tensor> {
        if let Shard::Simple { world_size: 1, .. } = &h {
            // There is no sharding to be applied here so we use the default backend to speed
            // things up.
            match self {
                Self::Sharded {
                    b,
                    make_dummy_regexes,
                    predicate,
                } => {
                    if let Some(make_dummy_regexes) = make_dummy_regexes {
                        if make_dummy_regexes.iter().any(|x| x.is_match(path)) {
                            return Err(Error::CannotFindTensor {
                                path: path.to_string(),
                            });
                        }
                    }
                    let should_include = predicate(path.to_string());
                    if !should_include {
                        return Err(Error::CannotFindTensor {
                            path: path.to_string(),
                        });
                    }

                    return SimpleBackend::get(
                        b.as_ref(),
                        target_shape,
                        path,
                        Default::default(),
                        dtype,
                        dev,
                    );
                }
                Self::SimpleBackend { b, .. } => {
                    return SimpleBackend::get(
                        b.as_ref(),
                        target_shape,
                        path,
                        Default::default(),
                        dtype,
                        dev,
                    )
                }
            }
        }

        let result = match h {
            Shard::Simple {
                dim,
                rank,
                world_size,
            } => {
                match self {
                    Self::Sharded {
                        b,
                        make_dummy_regexes,
                        predicate,
                    } => {
                        use safetensors::slice::IndexOp;

                        if let Some(make_dummy_regexes) = make_dummy_regexes {
                            if make_dummy_regexes.iter().any(|x| x.is_match(path)) {
                                return Err(Error::CannotFindTensor {
                                    path: path.to_string(),
                                });
                            }
                        }
                        let should_include = predicate(path.to_string());
                        if !should_include {
                            return Err(Error::CannotFindTensor {
                                path: path.to_string(),
                            });
                        }

                        let view = b.get(path)?;
                        let view_dtype = view.dtype();
                        let mut shape = view.shape().to_vec();
                        let size = shape[dim];

                        if size % world_size != 0 {
                            return Err(Error::ShapeMismatchSplit {
                                shape: shape.into(),
                                dim,
                                n_parts: world_size,
                            });
                        }
                        let block_size = size / world_size;
                        let start = rank * block_size;
                        let stop = (rank + 1) * block_size;

                        // Everything is expressed in tensor dimension
                        // bytes offsets is handled automatically for safetensors.

                        let iterator = if dim == 0 {
                            view.slice(start..stop).map_err(|_| {
                                Error::Msg(format!(
                                    "Cannot slice tensor {path} ({shape:?} along dim {dim} with {start}..{stop}"
                                ))
                            })?
                        } else if dim == 1 {
                            view.slice((.., start..stop)).map_err(|_| {
                                Error::Msg(format!(
                                    "Cannot slice tensor {path} ({shape:?} along dim {dim} with {start}..{stop}"
                                ))
                            })?
                        } else if dim == 2 {
                            view.slice((.., .., start..stop)).map_err(|_| {
                                Error::Msg(format!(
                                    "Cannot slice tensor {path} ({shape:?} along dim {dim} with {start}..{stop}"
                                ))
                            })?
                        } else {
                            candle_core::bail!("Got sharded on dimensions != 0 or 1 or 2")
                        };

                        shape[dim] = block_size;

                        let view_dtype: DType = view_dtype.try_into()?;
                        load_tensor_slices(iterator, &shape, view_dtype, dev, dtype)?
                    }
                    Self::SimpleBackend { b, .. } => {
                        let tensor = b.get(target_shape, path, Default::default(), dtype, dev)?;
                        h.apply_to(&tensor)?
                    }
                }
            }
            Shard::Offset { dim, offset, len } => {
                match self {
                    Self::Sharded {
                        b,
                        make_dummy_regexes,
                        predicate,
                    } => {
                        use safetensors::slice::IndexOp;

                        if let Some(make_dummy_regexes) = make_dummy_regexes {
                            if make_dummy_regexes.iter().any(|x| x.is_match(path)) {
                                return Err(Error::CannotFindTensor {
                                    path: path.to_string(),
                                });
                            }
                        }
                        let should_include = predicate(path.to_string());
                        if !should_include {
                            return Err(Error::CannotFindTensor {
                                path: path.to_string(),
                            });
                        }

                        let view = b.get(path)?;
                        let view_dtype = view.dtype();
                        let mut shape = view.shape().to_vec();

                        let start = offset;
                        let stop = start + len;

                        // Everything is expressed in tensor dimension
                        // bytes offsets is handled automatically for safetensors.

                        let iterator = if dim == 0 {
                            view.slice(start..stop).map_err(|_| {
                                Error::Msg(format!(
                                    "Cannot slice tensor {path} ({shape:?} along dim {dim} with {start}..{stop}"
                                ))
                            })?
                        } else if dim == 1 {
                            view.slice((.., start..stop)).map_err(|_| {
                                Error::Msg(format!(
                                    "Cannot slice tensor {path} ({shape:?} along dim {dim} with {start}..{stop}"
                                ))
                            })?
                        } else if dim == 2 {
                            view.slice((.., .., start..stop)).map_err(|_| {
                                Error::Msg(format!(
                                    "Cannot slice tensor {path} ({shape:?} along dim {dim} with {start}..{stop}"
                                ))
                            })?
                        } else {
                            candle_core::bail!("Got sharded on dimensions != 0 or 1 or 2")
                        };

                        shape[dim] = len;

                        let view_dtype: DType = view_dtype.try_into()?;
                        load_tensor_slices(iterator, &shape, view_dtype, dev, dtype)?
                    }
                    Self::SimpleBackend { b, .. } => {
                        let tensor = b.get(target_shape, path, Default::default(), dtype, dev)?;
                        h.apply_to(&tensor)?
                    }
                }
            }
        };

        result.contiguous()
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> Result<Tensor> {
        match self {
            Self::Sharded {
                b,
                make_dummy_regexes,
                predicate,
            } => {
                if let Some(make_dummy_regexes) = make_dummy_regexes {
                    if make_dummy_regexes.iter().any(|x| x.is_match(name)) {
                        return Err(Error::CannotFindTensor {
                            path: name.to_string(),
                        });
                    }
                }
                let should_include = predicate(name.to_string());
                if !should_include {
                    return Err(Error::CannotFindTensor {
                        path: name.to_string(),
                    });
                }
                <MmapedSafetensors as SimpleBackend>::get_unchecked(b.as_ref(), name, dtype, dev)
            }
            Self::SimpleBackend { b, .. } => b.as_ref().get_unchecked(name, dtype, dev),
        }
    }

    fn contains_tensor(&self, name: &str) -> bool {
        match self {
            Self::Sharded {
                b,
                make_dummy_regexes,
                predicate,
            } => {
                if let Some(path) = marked_uqff_dummy_tensor(name) {
                    return matches_dummy_regex(make_dummy_regexes, path);
                }
                if let Some(make_dummy_regexes) = make_dummy_regexes {
                    if make_dummy_regexes.iter().any(|x| x.is_match(name)) {
                        return false;
                    }
                }
                let should_include = predicate(name.to_string());
                if !should_include {
                    return false;
                }
                b.get(name).is_ok()
            }
            Self::SimpleBackend {
                b,
                make_dummy_regexes,
            } => {
                if let Some(path) = marked_uqff_dummy_tensor(name) {
                    return matches_dummy_regex(make_dummy_regexes, path);
                }
                b.as_ref().contains_tensor(name)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sliced_checkpoint_loading_preserves_offsets_shapes_and_dtypes() -> Result<()> {
        const SHAPE: (usize, usize, usize) = (4, 6, 8);
        let dir = tempfile::tempdir().map_err(Error::wrap)?;
        for dtype in [DType::F16, DType::BF16, DType::F32] {
            let path = dir.path().join(format!("{dtype:?}.safetensors"));
            let source = Tensor::from_vec(
                (0..SHAPE.0 * SHAPE.1 * SHAPE.2)
                    .map(|i| (i as f32 - 83.0) * 0.031)
                    .collect::<Vec<_>>(),
                SHAPE,
                &Device::Cpu,
            )?
            .to_dtype(dtype)?;
            candle_core::safetensors::save(&HashMap::from([("weight", source.clone())]), &path)?;
            let vb = unsafe {
                ShardedSafeTensors::sharded(
                    &[&path],
                    dtype,
                    &Device::Cpu,
                    None,
                    Arc::new(|_| true),
                )?
            };
            for dim in 0..3 {
                let half = source.dim(dim)? / 2;
                for (hint, offset, len) in [
                    (
                        Shard::Simple {
                            dim,
                            rank: 1,
                            world_size: 2,
                        },
                        half,
                        half,
                    ),
                    (
                        Shard::Offset {
                            dim,
                            offset: 1,
                            len: 2,
                        },
                        1,
                        2,
                    ),
                ] {
                    for target_dtype in [dtype, DType::F32] {
                        let expected = source
                            .narrow(dim, offset, len)?
                            .to_dtype(target_dtype)?
                            .force_contiguous()?;
                        let actual =
                            vb.get_with_hints_dtype(SHAPE, "weight", hint, target_dtype)?;
                        assert_eq!(actual.dims(), expected.dims());
                        assert_eq!(actual.dtype(), target_dtype);
                        assert_eq!(
                            actual
                                .to_dtype(DType::F32)?
                                .flatten_all()?
                                .to_vec1::<f32>()?,
                            expected
                                .to_dtype(DType::F32)?
                                .flatten_all()?
                                .to_vec1::<f32>()?,
                            "{dtype:?} {target_dtype:?} dim={dim} offset={offset}"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn unaligned_checkpoint_slices_preserve_float_values() -> Result<()> {
        use safetensors::slice::IndexOp;
        const ROWS: usize = 4;
        const COLS: usize = 8;
        for (dtype, view_dtype) in [
            (DType::F16, st::Dtype::F16),
            (DType::BF16, st::Dtype::BF16),
            (DType::F32, st::Dtype::F32),
        ] {
            let source = Tensor::from_vec(
                (0..ROWS * COLS)
                    .map(|i| (i as f32 - 13.0) / 7.0)
                    .collect::<Vec<_>>(),
                (ROWS, COLS),
                &Device::Cpu,
            )?
            .to_dtype(dtype)?;
            let values = source.flatten_all()?;
            let bytes = match dtype {
                DType::F16 => bytemuck::cast_slice(&values.to_vec1::<half::f16>()?).to_vec(),
                DType::BF16 => bytemuck::cast_slice(&values.to_vec1::<half::bf16>()?).to_vec(),
                DType::F32 => bytemuck::cast_slice(&values.to_vec1::<f32>()?).to_vec(),
                _ => unreachable!(),
            };
            let mut aligned = vec![0u32; (bytes.len() + 1).div_ceil(std::mem::size_of::<u32>())];
            let unaligned =
                &mut bytemuck::cast_slice_mut::<_, u8>(&mut aligned)[1..bytes.len() + 1];
            unaligned.copy_from_slice(&bytes);
            let view = st::TensorView::new(view_dtype, vec![ROWS, COLS], unaligned)?;
            let actual = load_tensor_slices(
                view.slice((.., 1..3)).map_err(Error::wrap)?,
                &[ROWS, 2],
                dtype,
                &Device::Cpu,
                DType::F32,
            )?;
            assert_eq!(
                actual.to_vec2::<f32>()?,
                source
                    .narrow(1, 1, 2)?
                    .to_dtype(DType::F32)?
                    .to_vec2::<f32>()?,
                "{dtype:?}"
            );
        }
        Ok(())
    }
}
