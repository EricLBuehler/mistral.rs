use mistralrs_quant::QuantizedConfig;

#[allow(dead_code)]
pub trait GdnConfig {
    fn hidden_size(&self) -> usize;
    fn rms_norm_eps(&self) -> f64;
    fn linear_conv_kernel_dim(&self) -> usize;
    fn linear_key_head_dim(&self) -> usize;
    fn linear_value_head_dim(&self) -> usize;
    fn linear_num_key_heads(&self) -> usize;
    fn linear_num_value_heads(&self) -> usize;
    fn quantization_config(&self) -> &Option<QuantizedConfig>;

    fn linear_key_dim(&self) -> usize {
        self.linear_num_key_heads() * self.linear_key_head_dim()
    }

    fn linear_value_dim(&self) -> usize {
        self.linear_num_value_heads() * self.linear_value_head_dim()
    }

    fn linear_conv_dim(&self) -> usize {
        self.linear_key_dim() * 2 + self.linear_value_dim()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GdnDims {
    pub hidden_size: usize,
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub conv_kernel_size: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub conv_dim: usize,
    pub v_per_group: usize,
}

impl GdnDims {
    pub fn new(cfg: &dyn GdnConfig) -> Self {
        let hidden_size = cfg.hidden_size();
        let num_k_heads = cfg.linear_num_key_heads();
        let num_v_heads = cfg.linear_num_value_heads();
        let head_k_dim = cfg.linear_key_head_dim();
        let head_v_dim = cfg.linear_value_head_dim();
        let conv_kernel_size = cfg.linear_conv_kernel_dim();
        let key_dim = num_k_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;
        let conv_dim = key_dim * 2 + value_dim;
        let v_per_group = num_v_heads / num_k_heads;

        Self {
            hidden_size,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_kernel_size,
            key_dim,
            value_dim,
            conv_dim,
            v_per_group,
        }
    }

    pub fn qkvz_out_dim(&self) -> usize {
        self.key_dim * 2 + self.value_dim * 2
    }

    pub fn ba_out_dim(&self) -> usize {
        self.num_v_heads * 2
    }
}
