#![allow(dead_code)]

#[cfg(feature = "cuda")]
use candle_core::cuda::cudarc::driver::{CudaStream, DevicePtr};
use candle_core::{Device, Result};
use once_cell::sync::Lazy;
#[cfg(feature = "cuda")]
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[cfg(feature = "cuda")]
type CudaGraph = candle_core::cuda::cudarc::driver::safe::CudaGraph;

/// Global state for CUDA graph capture
pub struct CudaGraphState {
    /// Whether we're currently capturing
    capturing: bool,
    /// The stream we're capturing on
    #[cfg(feature = "cuda")]
    capture_stream: Option<Arc<CudaStream>>,
    /// Cached graphs by (batch_size, seq_len) for decode
    #[cfg(feature = "cuda")]
    cached_graphs: HashMap<(usize, usize), Arc<CudaGraph>>,
    /// Output tensors for cached graphs
    #[cfg(feature = "cuda")]
    cached_outputs: HashMap<(usize, usize), candle_core::Tensor>,
}

impl CudaGraphState {
    pub fn new() -> Self {
        Self {
            capturing: false,
            #[cfg(feature = "cuda")]
            capture_stream: None,
            #[cfg(feature = "cuda")]
            cached_graphs: HashMap::new(),
            #[cfg(feature = "cuda")]
            cached_outputs: HashMap::new(),
        }
    }

    /// Check if we should start capturing a graph
    pub fn should_capture(&self, is_prompt: bool, batch_size: usize, seq_len: usize) -> bool {
        #[cfg(feature = "cuda")]
        {
            // Only capture for decode (not prompt) and seq_len == 1
            !is_prompt && seq_len == 1 && !self.cached_graphs.contains_key(&(batch_size, seq_len))
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (is_prompt, batch_size, seq_len);
            false
        }
    }

    /// Check if we have a cached graph for this configuration
    pub fn has_cached_graph(&self, batch_size: usize, seq_len: usize) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.cached_graphs.contains_key(&(batch_size, seq_len))
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (batch_size, seq_len);
            false
        }
    }

    /// Start capturing a CUDA graph
    #[cfg(feature = "cuda")]
    pub fn start_capture(&mut self, device: &Device) -> Result<()> {
        if let Device::Cuda(cuda_device) = device {
            let stream = cuda_device.cuda_stream();
            stream.begin_capture()?;
            self.capturing = true;
            self.capture_stream = Some(stream.clone());
            Ok(())
        } else {
            candle_core::bail!("CUDA graph capture requested on non-CUDA device")
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub fn start_capture(&mut self, _device: &Device) -> Result<()> {
        candle_core::bail!("CUDA graphs not available without cuda feature")
    }

    /// End capture and store the graph
    #[cfg(feature = "cuda")]
    pub fn end_capture(&mut self, batch_size: usize, seq_len: usize) -> Result<()> {
        if let Some(stream) = &self.capture_stream {
            let graph = stream.end_capture()?;
            self.cached_graphs
                .insert((batch_size, seq_len), Arc::new(graph));
            self.capturing = false;
            self.capture_stream = None;
            Ok(())
        } else {
            candle_core::bail!("No active CUDA graph capture")
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub fn end_capture(&mut self, _batch_size: usize, _seq_len: usize) -> Result<()> {
        candle_core::bail!("CUDA graphs not available without cuda feature")
    }

    /// Launch a cached graph
    #[cfg(feature = "cuda")]
    pub fn launch_graph(&self, batch_size: usize, seq_len: usize) -> Result<bool> {
        if let Some(graph) = self.cached_graphs.get(&(batch_size, seq_len)) {
            graph.launch()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub fn launch_graph(&self, _batch_size: usize, _seq_len: usize) -> Result<bool> {
        Ok(false)
    }

    /// Clear all cached graphs (e.g., when model config changes)
    pub fn clear_cache(&mut self) {
        #[cfg(feature = "cuda")]
        {
            self.cached_graphs.clear();
        }
    }

    /// Check if we're currently capturing
    pub fn is_capturing(&self) -> bool {
        self.capturing
    }
}

/// Thread-safe global CUDA graph state
pub static CUDA_GRAPH_STATE: Lazy<Arc<Mutex<CudaGraphState>>> = 
    Lazy::new(|| Arc::new(Mutex::new(CudaGraphState::new())));