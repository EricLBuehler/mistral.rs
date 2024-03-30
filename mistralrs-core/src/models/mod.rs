use std::sync::{Arc, Mutex, MutexGuard};

use candle_core::Tensor;

use crate::get_mut_arcmutex;

pub(crate) mod quantized_llama;
