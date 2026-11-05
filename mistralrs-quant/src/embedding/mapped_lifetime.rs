use std::{
    any::Any,
    collections::HashSet,
    ffi::c_void,
    sync::{mpsc, Arc, Mutex, MutexGuard, OnceLock},
    thread::ThreadId,
};

use candle_core::{
    cuda::cudarc::driver::{sys, CudaEvent, CudaStream},
    Result, Tensor,
};

const USER_OBJECT_REFERENCES: u32 = 1;
const REAPER_NAME: &str = "cuda-mapped-release";

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct CaptureId {
    context: usize,
    id: u64,
}

#[derive(Debug)]
struct ReadCompletion {
    context: usize,
    stream: Arc<CudaStream>,
    thread: ThreadId,
    event: CudaEvent,
}

#[derive(Debug)]
pub(super) struct CudaMappedLifetime {
    _index: Tensor,
    _owner: Arc<dyn Any + Send + Sync>,
    reads: Mutex<Vec<ReadCompletion>>,
    captures: Mutex<HashSet<CaptureId>>,
}

impl CudaMappedLifetime {
    pub(super) fn new(index: Tensor, owner: Arc<dyn Any + Send + Sync>) -> Self {
        Self {
            _index: index,
            _owner: owner,
            reads: Mutex::new(Vec::new()),
            captures: Mutex::new(HashSet::new()),
        }
    }

    pub(super) fn read<'a>(
        self: &'a Arc<Self>,
        stream: &'a Arc<CudaStream>,
    ) -> Result<MappedRead<'a>> {
        stream
            .context()
            .bind_to_thread()
            .map_err(candle_core::Error::wrap)?;
        let mut status = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        let mut id = 0;
        let mut graph = std::ptr::null_mut();
        #[cfg(not(cuda_ge_13000))]
        let result = unsafe {
            sys::cuStreamGetCaptureInfo_v2(
                stream.cu_stream(),
                &mut status,
                &mut id,
                &mut graph,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        #[cfg(cuda_ge_13000)]
        let result = unsafe {
            sys::cuStreamGetCaptureInfo_v3(
                stream.cu_stream(),
                &mut status,
                &mut id,
                &mut graph,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        result.result().map_err(candle_core::Error::wrap)?;
        let context = Arc::as_ptr(stream.context()) as usize;
        match status {
            sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE => {
                self.retain_graph(graph, CaptureId { context, id })?;
                Ok(MappedRead { eager: None })
            }
            sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE => {
                let mut reads = self.reads.lock().unwrap();
                let thread = std::thread::current().id();
                let stream_id = stream.cu_stream() as usize;
                let index = match reads.iter().position(|read| {
                    read.context == context
                        && read.stream.cu_stream() as usize == stream_id
                        && read.thread == thread
                }) {
                    Some(index) => index,
                    None => {
                        reads.push(ReadCompletion {
                            context,
                            stream: stream.clone(),
                            thread,
                            event: stream
                                .context()
                                .new_event(None)
                                .map_err(candle_core::Error::wrap)?,
                        });
                        reads.len() - 1
                    }
                };
                Ok(MappedRead {
                    eager: Some(EagerRead {
                        reads,
                        index,
                        stream,
                    }),
                })
            }
            _ => {
                candle_core::bail!("mapped embedding gather cannot use an invalidated CUDA capture")
            }
        }
    }

    fn retain_graph(self: &Arc<Self>, graph: sys::CUgraph, capture: CaptureId) -> Result<()> {
        let mut captures = self.captures.lock().unwrap();
        if captures.contains(&capture) {
            return Ok(());
        }
        release_queue()?;
        let resource = Box::into_raw(Box::new(GraphResource {
            lifetime: self.clone(),
            capture,
        }));
        let mut object = std::ptr::null_mut();
        let created = unsafe {
            sys::cuUserObjectCreate(
                &mut object,
                resource.cast(),
                Some(release_graph_resource),
                USER_OBJECT_REFERENCES,
                sys::CUuserObject_flags::CU_USER_OBJECT_NO_DESTRUCTOR_SYNC as u32,
            )
        }
        .result();
        if let Err(error) = created {
            unsafe {
                drop(Box::from_raw(resource));
            }
            return Err(candle_core::Error::wrap(error));
        }
        let retained = unsafe {
            sys::cuGraphRetainUserObject(
                graph,
                object,
                USER_OBJECT_REFERENCES,
                sys::CUuserObjectRetain_flags::CU_GRAPH_USER_OBJECT_MOVE as u32,
            )
        }
        .result();
        if let Err(error) = retained {
            unsafe { sys::cuUserObjectRelease(object, USER_OBJECT_REFERENCES) }
                .result()
                .map_err(candle_core::Error::wrap)?;
            return Err(candle_core::Error::wrap(error));
        }
        captures.insert(capture);
        Ok(())
    }
}

impl Drop for CudaMappedLifetime {
    fn drop(&mut self) {
        for read in self.reads.get_mut().unwrap() {
            read.event.context().record_err(read.event.synchronize());
        }
    }
}

pub(super) struct MappedRead<'a> {
    eager: Option<EagerRead<'a>>,
}

struct EagerRead<'a> {
    reads: MutexGuard<'a, Vec<ReadCompletion>>,
    index: usize,
    stream: &'a CudaStream,
}

impl Drop for MappedRead<'_> {
    fn drop(&mut self) {
        if let Some(read) = &self.eager {
            read.stream
                .context()
                .record_err(read.reads[read.index].event.record(read.stream));
        }
    }
}

#[derive(Debug)]
struct GraphResource {
    lifetime: Arc<CudaMappedLifetime>,
    capture: CaptureId,
}

type ReleaseQueue = std::result::Result<mpsc::Sender<GraphResource>, String>;
static RELEASE_QUEUE: OnceLock<ReleaseQueue> = OnceLock::new();

fn release_queue() -> Result<&'static mpsc::Sender<GraphResource>> {
    RELEASE_QUEUE
        .get_or_init(|| {
            let (sender, receiver) = mpsc::channel::<GraphResource>();
            std::thread::Builder::new()
                .name(REAPER_NAME.to_owned())
                .spawn(move || {
                    for resource in receiver {
                        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            resource
                                .lifetime
                                .captures
                                .lock()
                                .unwrap()
                                .remove(&resource.capture);
                            drop(resource);
                        }));
                    }
                })
                .map_err(|error| error.to_string())?;
            Ok(sender)
        })
        .as_ref()
        .map_err(|error| candle_core::Error::msg(error.as_str()))
}

unsafe extern "C" fn release_graph_resource(resource: *mut c_void) {
    // CUDA user-object callbacks cannot call CUDA; Tensor drops run on the reaper thread.
    let resource = unsafe { Box::from_raw(resource.cast::<GraphResource>()) };
    RELEASE_QUEUE
        .get()
        .unwrap()
        .as_ref()
        .unwrap()
        .send(*resource)
        .unwrap();
}
