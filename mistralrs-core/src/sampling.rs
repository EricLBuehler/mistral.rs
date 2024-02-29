use serde::Deserialize;

#[derive(Clone, Deserialize)]
pub struct SamplingParams {
    pub temperature: Option<f64>,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    pub top_n_logprobs: usize,
    pub repeat_penalty: Option<f32>,
    pub stop_toks: Option<Vec<u32>>,
    pub max_len: Option<usize>,
}
