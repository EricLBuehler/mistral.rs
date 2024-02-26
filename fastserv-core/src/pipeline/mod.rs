use candle_core::Tensor;

trait Pipeline {
    fn forward(&self, input_ids: &Tensor) -> Tensor;
    fn tokenize_prompt(&self, prompt: String) -> Tensor;
}