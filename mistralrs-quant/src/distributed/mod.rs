use std::{fmt::Debug, sync::Barrier};

use candle_core::{Result, Tensor};
pub use ops::{Comm, Id, SumAllReduce};
pub mod layers;
pub mod socket;

pub trait BarrierLike: Debug + Send + Sync {
    fn wait(&self) -> Result<()>;
}

impl BarrierLike for Barrier {
    fn wait(&self) -> Result<()> {
        Barrier::wait(self);
        Ok(())
    }
}

pub fn get_global_tp_size_from_devices() -> Result<usize> {
    #[cfg(feature = "cuda")]
    {
        use candle_core::cuda::WrapErr;
        candle_core::cuda::cudarc::driver::result::device::get_count()
            .w()
            .map(|x| x as usize)
    }
    #[cfg(not(feature = "cuda"))]
    Ok(1)
}

pub fn use_nccl() -> bool {
    (std::env::var("MISTRALRS_NO_NCCL").is_err()
        || std::env::var("MISTRALRS_NO_NCCL").is_ok_and(|x| x != "1"))
        && (cfg!(feature = "nccl") && cfg!(feature = "cuda"))
}

pub trait DistributedOperation {
    fn sum_all_reduce(&self, xs: &Tensor) -> Result<Tensor>;
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
mod ops {
    use std::{fmt::Debug, ops::Deref, sync::Arc};

    use candle_core::{
        backend::BackendStorage, cuda::cudarc, cuda_backend::WrapErr, CpuStorage, CustomOp1, DType,
        Device, Layout, Result, Shape, Tensor,
    };

    #[derive(Debug, Clone, Copy)]
    pub struct Id(cudarc::nccl::Id);

    impl Id {
        pub fn new() -> Self {
            let id = cudarc::nccl::Id::new().expect("Failed to create `Id`.");
            Self(id)
        }

        pub fn uninit(internal: [::core::ffi::c_char; 128usize]) -> Self {
            Self(cudarc::nccl::Id::uninit(internal))
        }

        pub fn internal(&self) -> &[::core::ffi::c_char; 128usize] {
            self.0.internal()
        }
    }

    #[derive(Debug)]
    pub struct Comm {
        comm: cudarc::nccl::Comm,
    }

    impl Comm {
        pub fn from_device(id: Id, dev: &Device, rank: usize, world_size: usize) -> Result<Self> {
            let device = dev.as_cuda_device()?.cuda_device();
            assert_eq!(rank, device.ordinal());
            Ok(Self {
                comm: cudarc::nccl::Comm::from_rank(device, rank, world_size, id.0)
                    .map_err(|e| e.0)
                    .expect("Failed to create `Comm`, error code"),
            })
        }
    }

    /// This is actually not safe: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/threadsafety.html
    unsafe impl Sync for Comm {}
    unsafe impl Send for Comm {}

    impl Deref for Comm {
        type Target = cudarc::nccl::Comm;

        fn deref(&self) -> &Self::Target {
            &self.comm
        }
    }

    #[derive(Clone, Debug)]
    pub struct SumAllReduce {
        comm: Arc<Comm>,
    }

    impl SumAllReduce {
        pub fn new(comm: &Arc<Comm>) -> Self {
            Self { comm: comm.clone() }
        }
    }

    impl super::DistributedOperation for SumAllReduce {
        fn sum_all_reduce(&self, xs: &Tensor) -> Result<Tensor> {
            xs.apply_op1_no_bwd(self)
        }
    }

    impl CustomOp1 for SumAllReduce {
        fn name(&self) -> &'static str {
            "SumAllReduce"
        }

        fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
            candle_core::bail!("SumAllReduce is never used on cpu")
        }

        fn cuda_fwd(
            &self,
            s: &candle_core::CudaStorage,
            l: &Layout,
        ) -> Result<(candle_core::CudaStorage, Shape)> {
            use cudarc::{driver::DeviceSlice, nccl::ReduceOp};
            use half::{bf16, f16};

            let elem_count = l.shape().elem_count();
            let dev = s.device().clone();
            let dst = match s.dtype() {
                DType::BF16 => {
                    let s = s.as_cuda_slice::<bf16>()?;
                    let s = match l.contiguous_offsets() {
                        Some((0, l)) if l == s.len() => s,
                        Some(_) | None => candle_core::bail!("input has to be contiguous"),
                    };
                    assert_eq!(dev.ordinal(), self.comm.rank());
                    assert!(elem_count > 0);
                    let mut dst = unsafe { dev.alloc::<bf16>(elem_count) }.w()?;
                    self.comm
                        .comm
                        .all_reduce(s, &mut dst, &ReduceOp::Sum)
                        .map_err(candle_core::Error::debug)?;
                    candle_core::CudaStorage::wrap_cuda_slice(dst, dev)
                }
                DType::F16 => {
                    let s = s.as_cuda_slice::<f16>()?;
                    let s = match l.contiguous_offsets() {
                        Some((0, l)) if l == s.len() => s,
                        Some(_) | None => candle_core::bail!("input has to be contiguous"),
                    };
                    let mut dst = unsafe { dev.alloc::<f16>(elem_count) }.w()?;
                    self.comm
                        .comm
                        .all_reduce(s, &mut dst, &ReduceOp::Sum)
                        .map_err(candle_core::Error::debug)?;
                    candle_core::CudaStorage::wrap_cuda_slice(dst, dev)
                }
                DType::F32 => {
                    let s = s.as_cuda_slice::<f32>()?;
                    let s = match l.contiguous_offsets() {
                        Some((0, l)) if l == s.len() => s,
                        Some(_) | None => candle_core::bail!("input has to be contiguous"),
                    };
                    let mut dst = unsafe { dev.alloc::<f32>(elem_count) }.w()?;
                    self.comm
                        .comm
                        .all_reduce(s, &mut dst, &ReduceOp::Sum)
                        .map_err(candle_core::Error::debug)?;
                    candle_core::CudaStorage::wrap_cuda_slice(dst, dev)
                }
                dtype => candle_core::bail!("unsupported dtype {dtype:?}"),
            };
            Ok((dst, l.shape().clone()))
        }
    }
}

#[cfg(feature = "ring")]
mod ops {
    use std::{
        collections::HashMap,
        env,
        io::{Read, Write},
        net::{TcpListener, TcpStream},
        sync::{Arc, Mutex},
        time::{Duration, Instant},
    };

    use candle_core::{
        backend::BackendStorage, CpuStorage, Device, Result, Storage, Tensor, WithDType,
    };

    #[derive(Debug, Clone, Copy)]
    pub struct Id;

    impl Default for Id {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Id {
        pub fn new() -> Self {
            Self
        }

        pub fn uninit(_internal: [::core::ffi::c_char; 128usize]) -> Self {
            Self
        }

        pub fn internal(&self) -> &[::core::ffi::c_char; 128usize] {
            static ZEROED_ID: [::core::ffi::c_char; 128] = [0; 128];
            &ZEROED_ID
        }
    }

    #[derive(Debug)]
    pub struct Comm;

    impl Comm {
        pub fn from_device(
            _id: Id,
            _dev: &Device,
            _rank: usize,
            _world_size: usize,
        ) -> Result<Self> {
            Ok(Self)
        }

        pub fn rank(&self) -> usize {
            env::var("RING_RANK").unwrap().parse().unwrap()
        }

        pub fn world_size(&self) -> usize {
            env::var("RING_WORLD_SIZE").unwrap().parse().unwrap()
        }
    }

    #[derive(Clone, Debug)]
    pub struct SumAllReduce {
        left: Arc<Mutex<TcpStream>>,
        right: Arc<Mutex<TcpStream>>,
        buffers: Arc<Mutex<HashMap<usize, Vec<u8>>>>,
    }

    impl SumAllReduce {
        pub fn new(_comm: &Arc<Comm>) -> Self {
            let cur_port: u16 = env::var("RING_PORT").unwrap().parse().unwrap();
            let right_port: u16 = env::var("RING_RIGHT").unwrap().parse().unwrap();

            let left_h = std::thread::spawn(move || {
                let left_listener = TcpListener::bind(format!("127.0.0.1:{cur_port}")).unwrap();

                let (left, _left_addr) = left_listener.accept().unwrap();
                left
            });
            let right_h = std::thread::spawn(move || {
                let start = Instant::now();
                loop {
                    let stream = TcpStream::connect(format!("127.0.0.1:{right_port}"));
                    if let Ok(stream) = stream {
                        return stream;
                    }
                    if start.elapsed() > Duration::from_secs(10) {
                        panic!("Failed to connect to right node due to timeout: over 10s");
                    }
                }
            });

            while !left_h.is_finished() || !right_h.is_finished() {}
            let left = left_h.join().unwrap();
            let right = right_h.join().unwrap();

            left.set_nodelay(true).unwrap();
            left.set_nonblocking(false).unwrap();
            right.set_nodelay(true).unwrap();
            right.set_nonblocking(false).unwrap();

            Self {
                left: Arc::new(Mutex::new(left)),
                right: Arc::new(Mutex::new(right)),
                buffers: Arc::new(Mutex::new(HashMap::new())),
            }
        }

        /// Send data to the right and receive data from the left.
        /// This received data is returned as a tensor and should be added to the original data.
        fn run<T: WithDType + Copy + std::ops::AddAssign>(
            &self,
            x: &Vec<T>,
            dims: &[usize],
            device: &Device,
        ) -> Result<Tensor> {
            let nbytes = x.len() * std::mem::size_of::<T>();

            // Clone the Arc references for use in spawned tasks
            let right = self.right.clone();
            let left = self.left.clone();

            // Send the data to the right
            {
                // Copy the data from x into a Vec<u8> for sending
                let data =
                    unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, nbytes) }.to_vec();

                let mut right_guard = right.lock().unwrap();
                right_guard
                    .write_all(&data)
                    .map_err(|e| candle_core::Error::msg(format!("write error: {:?}", e)))?;
            };

            // We save reallocation of buffers here.
            let mut buffers_guard = self.buffers.lock().unwrap();
            let buf = {
                buffers_guard.entry(nbytes).or_insert({
                    let mut buf = Vec::with_capacity(nbytes);
                    unsafe {
                        buf.set_len(nbytes);
                    }
                    buf
                })
            };

            // Receive the data from the left
            let buf = {
                let mut left_guard = left.lock().unwrap();
                left_guard
                    .read_exact(buf)
                    .map_err(|e| candle_core::Error::msg(format!("read error: {:?}", e)))?;
                buf
            };
            assert_ne!(buf.len(), 0);

            // Interpret the received bytes as a slice of T and add element-wise into x
            let received: &[T] =
                unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const T, x.len()) };

            Tensor::from_slice(&received, dims, device)
        }
    }

    impl super::DistributedOperation for SumAllReduce {
        fn sum_all_reduce(&self, xs: &Tensor) -> Result<Tensor> {
            let storage = xs.storage_and_layout().0;
            let cpu_storage = match &*storage {
                Storage::Cpu(storage) => storage,
                Storage::Cuda(storage) => &storage.to_cpu_storage()?,
                Storage::Metal(storage) => &storage.to_cpu_storage()?,
            };

            let delta = match cpu_storage {
                CpuStorage::BF16(x) => self.run(x, xs.dims(), xs.device())?,
                CpuStorage::F32(x) => self.run(x, xs.dims(), xs.device())?,
                CpuStorage::F16(x) => self.run(x, xs.dims(), xs.device())?,
                _ => candle_core::bail!("Unsupported dtype for ring backend"),
            };

            xs + delta
        }
    }
}

#[cfg(not(any(all(feature = "cuda", feature = "nccl"), feature = "ring")))]
mod ops {
    use std::sync::Arc;

    use candle_core::{Device, Result, Tensor};

    #[derive(Debug, Clone, Copy)]
    pub struct Id;

    impl Default for Id {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Id {
        pub fn new() -> Self {
            Self
        }

        pub fn uninit(_internal: [::core::ffi::c_char; 128usize]) -> Self {
            Self
        }

        pub fn internal(&self) -> &[::core::ffi::c_char; 128usize] {
            static ZEROED_ID: [::core::ffi::c_char; 128] = [0; 128];
            &ZEROED_ID
        }
    }

    #[derive(Debug)]
    pub struct Comm;

    impl Comm {
        pub fn from_device(
            _id: Id,
            _dev: &Device,
            _rank: usize,
            _world_size: usize,
        ) -> Result<Self> {
            Ok(Self)
        }

        pub fn rank(&self) -> usize {
            0
        }

        pub fn world_size(&self) -> usize {
            1
        }
    }

    #[derive(Clone, Debug)]
    pub struct SumAllReduce;

    impl SumAllReduce {
        pub fn new(_comm: &Arc<Comm>) -> Self {
            Self
        }
    }

    impl super::DistributedOperation for SumAllReduce {
        fn sum_all_reduce(&self, xs: &Tensor) -> Result<Tensor> {
            Ok(xs.clone())
        }
    }
}
