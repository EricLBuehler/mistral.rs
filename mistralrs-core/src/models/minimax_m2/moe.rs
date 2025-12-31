#[derive(Clone)]
struct Mlp {
    gate_proj: Arc<dyn QuantMethod>,
    up_proj: Arc<dyn QuantMethod>,
    down_proj: Arc<dyn QuantMethod>,
    act_fn: Activation,
}

impl Mlp {
    fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
        i_size: usize,
    ) -> Result<Self> {
        let hidden_size = cfg.hidden_size;

        let gate_proj = ColumnParallelLayer::new(
            hidden_size,
            i_size,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("gate_proj"),
        )?;
        let up_proj = RowParallelLayer::new(
            hidden_size,
            i_size,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("up_proj"),
        )?;
        let down_proj = ColumnParallelLayer::new(
            i_size,
            hidden_size,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("down_proj"),
        )?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let original_dtype = xs.dtype();
        let mut xs = xs.clone();
        if let Some(t) = self.gate_proj.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        let mut current_hidden_states = MatMul
            .qmethod_matmul(&xs, &*self.gate_proj)?
            .apply(&self.act_fn)?;
        let rhs = MatMul.qmethod_matmul(&xs, &*self.up_proj)?;
        current_hidden_states = current_hidden_states.broadcast_mul(&rhs)?;
        let mut res = MatMul.qmethod_matmul(&current_hidden_states, &*self.down_proj)?;
        if self.gate_proj.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

/// MoE MLP layer for Qwen3 MoE
struct MoeMlp {
    gate: Linear,
    experts: MoEExperts,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
}

impl MoeMlp {
    fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        layer_device: Device,
        comm: &Arc<mistralrs_quant::Comm>,
        loading_isq: bool,
    ) -> Result<Self> {
        // Load gate
        let gate = layers::linear_no_bias(
            cfg.hidden_size,
            cfg.num_experts,
            vb.pp("gate").set_device(layer_device.clone()),
        )?;

        let moe_cfg = MoEExpertsConfig {
            num_experts: cfg.num_experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
            hidden_size: cfg.hidden_size,
            moe_intermediate_size: cfg.moe_intermediate_size,
        };

        // Load experts with automatic backend selection
        let experts = MoEExperts::new(
            &moe_cfg,
            vb,
            layer_device,
            comm,
            loading_isq,
            &cfg.quantization_config,
            cfg.hidden_act,
        )?;

        Ok(Self {
            gate,
            experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
            norm_topk_prob: cfg.norm_topk_prob,
        })
    }

    fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        let xs_flat = xs.reshape(((), hidden_dim))?;

        // Compute routing weights
        let router_logits = self.gate.forward(&xs_flat)?;
        let routing_weights =
            candle_nn::ops::softmax_last_dim(&router_logits.to_dtype(DType::F32)?)?;

        // Get top-k experts
        let topk_ids = routing_weights
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, self.num_experts_per_tok)?
            .contiguous()?;

        let mut topk_weights = routing_weights.gather(&topk_ids, D::Minus1)?;

        if self.norm_topk_prob {
            topk_weights = topk_weights.broadcast_div(&topk_weights.sum_keepdim(D::Minus1)?)?;
        }

        // Forward through experts
        let ys = self
            .experts
            .forward(xs, topk_weights, &topk_ids, is_prefill)?;

        ys.reshape((b_size, seq_len, hidden_dim))
    }

    fn gate(&self) -> &Linear {
        &self.gate
    }

    fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        self.experts.get_isq_layers()
    }
}

enum MoeOrMlp {
    Moe(MoeMlp),
    Mlp(Mlp),
}

impl MoeOrMlp {
    fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        match self {
            Self::Mlp(m) => m.forward(xs),
            Self::Moe(m) => m.forward(xs, is_prefill),
        }
    }
}
