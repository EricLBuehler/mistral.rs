use std::{fmt::Debug, sync::Barrier};

use candle_core::Result;
pub use ops::{AllGather, Comm, Id, SumAllReduce};
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

    impl SumAllReduce {
        pub fn sum_all_reduce(&self, xs: &Tensor) -> Result<Tensor> {
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

    #[derive(Clone, Debug)]
    pub struct AllGather {
        comm: Arc<Comm>,
        dim: usize,
    }

    impl AllGather {
        pub fn new(comm: &Arc<Comm>, dim: usize) -> Self {
            Self {
                comm: comm.clone(),
                dim,
            }
        }
    }

    impl AllGather {
        pub fn all_gather(&self, xs: &Tensor) -> Result<Tensor> {
            xs.apply_op1_no_bwd(self)
        }
    }

    impl CustomOp1 for AllGather {
        fn name(&self) -> &'static str {
            "AllGather"
        }

        fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
            candle_core::bail!("AllGather is never used on cpu")
        }

        fn cuda_fwd(
            &self,
            s: &candle_core::CudaStorage,
            l: &Layout,
        ) -> Result<(candle_core::CudaStorage, Shape)> {
            use cudarc::driver::DeviceSlice;
            use half::{bf16, f16};

            let mut out_shape = l.shape().dims().to_vec();
            out_shape[self.dim] = out_shape[self.dim] * self.comm.world_size();
            let out_shape = Shape::from(out_shape);

            let elem_count = out_shape.elem_count();
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
                        .all_gather(s, &mut dst)
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
                        .all_gather(s, &mut dst)
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
                        .all_gather(s, &mut dst)
                        .map_err(candle_core::Error::debug)?;
                    candle_core::CudaStorage::wrap_cuda_slice(dst, dev)
                }
                dtype => candle_core::bail!("unsupported dtype {dtype:?}"),
            };
            Ok((dst, out_shape))
        }
    }
}

#[cfg(feature = "ring")]
mod ops {
    use std::{
        collections::HashMap,
        fmt::Debug,
        fs::File,
        sync::{Arc, Mutex, OnceLock},
        time::{Duration, Instant},
    };

    use serde::{Deserialize, Serialize};
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};

    use candle_core::{
        backend::BackendStorage, CpuStorage, Device, Result, Storage, Tensor, WithDType,
    };

    #[derive(Debug, Deserialize, Serialize)]
    struct RingConfig {
        port: u16,
        right_port: u16,
        right_ip: Option<String>,
        rank: usize,
        world_size: usize,
    }

    impl RingConfig {
        fn read() -> Self {
            let config_json = std::env::var("RING_CONFIG").expect("RING_CONFIG must be set");
            let config: RingConfig = serde_json::from_reader(
                &File::open(config_json).expect("Could not access Ring config JSON"),
            )
            .expect("Invalid JSON config");
            config
        }
    }

    // Lazily–initialized pair of TCP streams shared by every ring‑based collective op
    static LEFT_RIGHT_STREAMS: OnceLock<(Arc<Mutex<TcpStream>>, Arc<Mutex<TcpStream>>)> =
        OnceLock::new();

    fn get_ring_streams() -> (Arc<Mutex<TcpStream>>, Arc<Mutex<TcpStream>>) {
        LEFT_RIGHT_STREAMS
            .get_or_init(|| {
                let config = RingConfig::read();

                let cur_port = config.port;

                let right_ip = config.right_ip.unwrap_or("0.0.0.0".to_string());
                let right_port = config.right_port;

                let left_listener =
                    TcpListener::bind(format!("0.0.0.0:{cur_port}")).expect("bind left");

                let start = Instant::now();
                // Connect to the right neighbor using the provided IP
                let right = loop {
                    match TcpStream::connect(format!("{}:{}", right_ip, right_port)) {
                        Ok(s) => break s,
                        Err(_) if start.elapsed() > Duration::from_secs(10) => {
                            panic!("Failed to connect to right node due to 10-second timeout");
                        }
                        Err(_) => continue,
                    }
                };

                // Accept connection from the left neighbour
                let (left, _) = left_listener.accept().expect("accept left neighbour");

                left.set_nodelay(true).unwrap();
                left.set_nonblocking(false).unwrap();
                right.set_nodelay(true).unwrap();
                right.set_nonblocking(false).unwrap();

                (Arc::new(Mutex::new(left)), Arc::new(Mutex::new(right)))
            })
            .clone()
    }

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
    pub struct Comm {
        rank: usize,
        world_size: usize,
    }

    impl Comm {
        pub fn from_device(
            _id: Id,
            _dev: &Device,
            _rank: usize,
            _world_size: usize,
        ) -> Result<Self> {
            let config = RingConfig::read();

            Ok(Self {
                rank: config.rank,
                world_size: config.world_size,
            })
        }

        pub fn rank(&self) -> usize {
            self.rank
        }

        pub fn world_size(&self) -> usize {
            self.world_size
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
            let (left, right) = get_ring_streams();
            Self {
                left,
                right,
                buffers: Arc::new(Mutex::new(HashMap::new())),
            }
        }

        fn run<T: WithDType + Copy + std::ops::AddAssign>(
            &self,
            x: &Vec<T>,
            dims: &[usize],
            device: &Device,
        ) -> Result<Tensor> {
            let nbytes = x.len() * std::mem::size_of::<T>();
            // dbg!(nbytes);

            // --- ping‑pong to overlap latency ---------------------------------------
            // Clone the Arc references
            let right = self.right.clone();
            let left = self.left.clone();

            // View the local slice as bytes that can be written on the wire.
            let data_bytes = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, nbytes) };

            // Re‑use (or allocate) a receive buffer of identical size.
            let mut buffers_guard = self.buffers.lock().unwrap();
            let recv_buf = buffers_guard.entry(nbytes).or_insert_with(|| {
                let mut v = Vec::<u8>::with_capacity(nbytes);
                unsafe { v.set_len(nbytes) };
                v
            });

            // Lock both sockets once to avoid per‑call mutex overhead.
            let mut right_guard = right.lock().unwrap();
            let mut left_guard = left.lock().unwrap();

            // For the typical tensor size we see (~ 6 KiB) a single
            // write/read pair is faster than chunking because the extra
            // system‑call and loop overhead dominates.  Only fall back to the
            // chunked “ping‑pong” pipeline for larger transfers.
            if nbytes <= 8 * 1024 {
                // --- fast path: one shot ------------------------------------
                right_guard
                    .write_all(data_bytes)
                    .map_err(|e| candle_core::Error::msg(format!("write error: {:?}", e)))?;

                left_guard
                    .read_exact(recv_buf)
                    .map_err(|e| candle_core::Error::msg(format!("read error: {:?}", e)))?;
            } else {
                // --- slow path: chunked ping‑pong ---------------------------
                const CHUNK_SIZE: usize = 64 * 1024; // 64 KiB
                let mut offset = 0;

                while offset < nbytes {
                    let len = std::cmp::min(CHUNK_SIZE, nbytes - offset);

                    // send this chunk to the right neighbour
                    right_guard
                        .write_all(&data_bytes[offset..offset + len])
                        .map_err(|e| candle_core::Error::msg(format!("write error: {:?}", e)))?;

                    // receive the matching chunk from the left neighbour
                    left_guard
                        .read_exact(&mut recv_buf[offset..offset + len])
                        .map_err(|e| candle_core::Error::msg(format!("read error: {:?}", e)))?;

                    offset += len;
                }
            }

            drop(left_guard);
            drop(right_guard);
            // drop(buffers_guard);

            // -------------------------------------------------------------------------
            // Interpret the received bytes as a slice of T and add element‑wise into x
            let received: &[T] =
                unsafe { std::slice::from_raw_parts(recv_buf.as_ptr() as *const T, x.len()) };

            Tensor::from_slice(&received, dims, device)
        }
    }

    impl SumAllReduce {
        pub fn sum_all_reduce(&self, xs: &Tensor) -> Result<Tensor> {
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

    #[derive(Clone, Debug)]
    pub struct AllGather {
        left: Arc<Mutex<TcpStream>>,
        right: Arc<Mutex<TcpStream>>,
        buffers: Arc<Mutex<HashMap<usize, Vec<u8>>>>,
        dim: usize,
        world_size: usize,
        rank: usize,
    }

    impl AllGather {
        pub fn new(comm: &Arc<Comm>, dim: usize) -> Self {
            let (left, right) = get_ring_streams();
            Self {
                left,
                right,
                buffers: Arc::new(Mutex::new(HashMap::new())),
                dim,
                world_size: comm.world_size(),
                rank: comm.rank(),
            }
        }

        fn run<T: WithDType + Copy>(
            &self,
            x: &Vec<T>,
            dims: &[usize],
            device: &Device,
        ) -> Result<Tensor> {
            let elem_cnt = x.len();
            let nbytes = elem_cnt * std::mem::size_of::<T>();

            // Prepare output buffer that will hold slices from every rank.
            let mut out: Vec<T> = Vec::with_capacity(elem_cnt * self.world_size);
            unsafe { out.set_len(elem_cnt * self.world_size) };

            // Copy this rank’s slice into its final slot.
            let start = self.rank * elem_cnt;
            out[start..start + elem_cnt].copy_from_slice(x);

            let right = self.right.clone();
            let left = self.left.clone();
            let mut send_piece: &[T] = x;

            for step in 0..(self.world_size - 1) {
                // ---------- send to the right ----------
                let bytes =
                    unsafe { std::slice::from_raw_parts(send_piece.as_ptr() as *const u8, nbytes) };
                {
                    let mut rg = right.lock().unwrap();
                    rg.write_all(bytes)
                        .map_err(|e| candle_core::Error::msg(format!("write error: {:?}", e)))?;
                }

                // ---------- receive from the left ----------
                let mut bg = self.buffers.lock().unwrap();
                let buf = bg.entry(nbytes).or_insert_with(|| {
                    let mut v = Vec::with_capacity(nbytes);
                    unsafe { v.set_len(nbytes) };
                    v
                });
                {
                    let mut lg = left.lock().unwrap();
                    lg.read_exact(buf)
                        .map_err(|e| candle_core::Error::msg(format!("read error: {:?}", e)))?;
                }
                let recv_piece: &[T] =
                    unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const T, elem_cnt) };

                // Determine which global rank the received slice came from.
                let src_rank = (self.rank + self.world_size - step - 1) % self.world_size;
                let dst = src_rank * elem_cnt;
                out[dst..dst + elem_cnt].copy_from_slice(recv_piece);

                // Forward that slice in the next iteration.
                send_piece = recv_piece;
            }

            let mut out_dims = dims.to_vec();
            out_dims[self.dim] *= self.world_size;
            Tensor::from_slice(&out, out_dims, device)
        }

        pub fn all_gather(&self, xs: &Tensor) -> Result<Tensor> {
            let storage = xs.storage_and_layout().0;
            let cpu_storage = match &*storage {
                Storage::Cpu(s) => s,
                Storage::Cuda(s) => &s.to_cpu_storage()?,
                Storage::Metal(s) => &s.to_cpu_storage()?,
            };

            match cpu_storage {
                CpuStorage::BF16(x) => self.run(x, xs.dims(), xs.device()),
                CpuStorage::F32(x) => self.run(x, xs.dims(), xs.device()),
                CpuStorage::F16(x) => self.run(x, xs.dims(), xs.device()),
                _ => candle_core::bail!("Unsupported dtype for ring backend"),
            }
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

    impl SumAllReduce {
        pub fn sum_all_reduce(&self, xs: &Tensor) -> Result<Tensor> {
            Ok(xs.clone())
        }
    }

    #[derive(Clone, Debug)]
    pub struct AllGather;

    impl AllGather {
        pub fn new(_comm: &Arc<Comm>, _dim: usize) -> Self {
            Self
        }
    }

    impl AllGather {
        pub fn all_gather(&self, xs: &Tensor) -> Result<Tensor> {
            Ok(xs.clone())
        }
    }
}
