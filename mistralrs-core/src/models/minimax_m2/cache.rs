use candle_core::Tensor;

pub struct MinimaxCache {
    layer_cache: Vec<Tensor>,
}

impl MinimaxCache {
    pub fn get_linear(&self, layer_idx: usize) -> Option<Tensor> {
        self.layer_cache.get(layer_idx).map(|t| t.clone())
    }

    pub fn set_linear(&mut self, layer_idx: usize, tensor: Tensor) {
        self.layer_cache.insert(layer_idx, tensor);
    }
}
