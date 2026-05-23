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

/// Information about a captured graph
#[cfg(feature = "cuda")]
pub struct CapturedGraph {
    graph: Arc<CudaGraph>,
    /// Pre-allocated output tensor that will be populated by graph execution
    output_tensor: candle_core::Tensor,
}

/// Global state for CUDA graph capture
pub struct CudaGraphState {
    /// Whether we're currently capturing
    capturing: bool,
    /// The stream we're capturing on
    #[cfg(feature = "cuda")]
    capture_stream: Option<Arc<CudaStream>>,
    /// Cached graphs by (batch_size, seq_len) for decode
    #[cfg(feature = "cuda")]
    cached_graphs: HashMap<(usize, usize), CapturedGraph>,
    /// Temporary output tensor used during capture
    #[cfg(feature = "cuda")]
    capture_output: Option<candle_core::Tensor>,
    /// Configuration for current capture
    #[cfg(feature = "cuda")]
    capture_config: Option<(usize, usize, usize)>, // (batch_size, seq_len, vocab_size)
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
            capture_output: None,
            #[cfg(feature = "cuda")]
            capture_config: None,
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
    pub fn start_capture(
        &mut self,
        device: &Device,
        batch_size: usize,
        seq_len: usize,
        vocab_size: usize,
    ) -> Result<()> {
        if let Device::Cuda(cuda_device) = device {
            let stream = cuda_device.cuda_stream();
            
            // Pre-allocate output tensor
            let output_tensor = candle_core::Tensor::zeros(
                &[batch_size, seq_len, vocab_size],
                candle_core::DType::F32,
                device,
            )?;
            
            stream.begin_capture()?;
            self.capturing = true;
            self.capture_stream = Some(stream.clone());
            self.capture_output = Some(output_tensor);
            self.capture_config = Some((batch_size, seq_len, vocab_size));
            Ok(())
        } else {
            candle_core::bail!("CUDA graph capture requested on non-CUDA device")
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub fn start_capture(
        &mut self,
        _device: &Device,
        _batch_size: usize,
        _seq_len: usize,
        _vocab_size: usize,
    ) -> Result<()> {
        candle_core::bail!("CUDA graphs not available without cuda feature")
    }

    /// End capture and store the graph
    #[cfg(feature = "cuda")]
    pub fn end_capture(&mut self, batch_size: usize, seq_len: usize) -> Result<()> {
        if let Some(stream) = &self.capture_stream {
            let graph = stream.end_capture()?;
            
            if let Some(output_tensor) = self.capture_output.take() {
                self.cached_graphs.insert(
                    (batch_size, seq_len),
                    CapturedGraph {
                        graph: Arc::new(graph),
                        output_tensor,
                    },
                );
            }
            
            self.capturing = false;
            self.capture_stream = None;
            self.capture_config = None;
            Ok(())
        } else {
            candle_core::bail!("No active CUDA graph capture")
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub fn end_capture(&mut self, _batch_size: usize, _seq_len: usize) -> Result<()> {
        candle_core::bail!("CUDA graphs not available without cuda feature")
    }

    /// Launch a cached graph and return the output tensor
    #[cfg(feature = "cuda")]
    pub fn launch_graph(
        &self,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Option<candle_core::Tensor>> {
        if let Some(captured) = self.cached_graphs.get(&(batch_size, seq_len)) {
            captured.graph.launch()?;
            // Return a clone of the output tensor reference
            // The graph execution will populate this tensor with the results
            Ok(Some(captured.output_tensor.clone()))
        } else {
            Ok(None)
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub fn launch_graph(
        &self,
        _batch_size: usize,
        _seq_len: usize,
    ) -> Result<Option<candle_core::Tensor>> {
        Ok(None)
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
    
    /// Get the output tensor being used for capture
    #[cfg(feature = "cuda")]
    pub fn get_capture_output(&self) -> Option<&candle_core::Tensor> {
        self.capture_output.as_ref()
    }
    
    #[cfg(not(feature = "cuda"))]
    pub fn get_capture_output(&self) -> Option<&candle_core::Tensor> {
        None
    }
}

/// Thread-safe global CUDA graph state
pub static CUDA_GRAPH_STATE: Lazy<Arc<Mutex<CudaGraphState>>> = 
    Lazy::new(|| Arc::new(Mutex::new(CudaGraphState::new())));