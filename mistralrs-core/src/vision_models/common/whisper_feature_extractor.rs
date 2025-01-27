pub struct WhisperFeatureExtractorConfig {
    pub feature_size: usize,
    pub sampling_rate: usize,
    pub hop_length: usize,
    pub chunk_length: usize,
    pub n_fft: usize,
    pub padding_value: f64,
}

impl Default for WhisperFeatureExtractorConfig {
    fn default() -> Self {
        Self {
            feature_size: 80,
            sampling_rate: 16000,
            hop_length: 160,
            chunk_length: 30,
            n_fft: 400,
            padding_value: 0.0,
        }
    }
}
