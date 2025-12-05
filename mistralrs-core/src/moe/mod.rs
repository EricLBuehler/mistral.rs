use crate::cuda::moe;
use crate::layers::linear_no_bias;
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::linear::Linear;
use candle_nn::Module;
use mistralrs_quant::{Shard, ShardedVarBuilder, SumAllReduce};
use std::sync::Arc;

pub fn shard(dim: usize, rank: usize, world_size: usize) -> Shard {
    Shard::Simple {
        dim,
        rank,
        world_size,
    }
}

#[allow(dead_code)]
pub struct FusedMoe {
    gate: Linear,
    pub gate_up_w: Tensor,
    pub down_w: Tensor,
    w_size_n: usize,
    act: candle_nn::Activation,
    norm_topk_prob: bool,
    num_experts_per_tok: usize,
    all_reduce: SumAllReduce,
    world_size: usize,
    dtype: DType,
    /// If true, weights are in stacked format [E, K, N] instead of [E, N, K].
    /// Uses transposed kernel directly without any tensor modifications.
    stacked_format: bool,
}

pub struct MoEConfig {
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub hidden_size: usize,
    pub moe_intermediate_size: usize,
    pub norm_topk_prob: bool,
}

impl FusedMoe {
    /// Load MoE weights, automatically detecting the format:
    /// - Stacked format: `experts.gate_up_proj` [num_experts, hidden, inter*2] and
    ///   `experts.down_proj` [num_experts, inter, hidden]
    /// - Standard format: `experts.{i}.gate_proj`, `experts.{i}.up_proj`, `experts.{i}.down_proj`
    ///   per expert
    pub fn new(
        cfg: &MoEConfig,
        vb: ShardedVarBuilder,
        layer_device: Device,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let num_experts = cfg.num_experts;
        let experts_vb = vb.pp("experts").set_device(layer_device.clone());

        // Detect stacked format by checking for gate_up_proj tensor
        let is_stacked = experts_vb.contains_tensor("gate_up_proj");

        let gate = linear_no_bias(
            cfg.hidden_size,
            num_experts,
            vb.pp("gate").set_device(layer_device.clone()),
        )?;

        if is_stacked {
            // Stacked format: load pre-packed weights directly
            Self::load_stacked(cfg, gate, experts_vb, comm)
        } else {
            // Standard format: load and pack per-expert weights
            Self::load_standard(cfg, gate, experts_vb, comm)
        }
    }

    /// Load standard format weights (per-expert gate_proj, up_proj, down_proj)
    fn load_standard(
        cfg: &MoEConfig,
        gate: Linear,
        experts_vb: ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let num_experts = cfg.num_experts;
        let mut gate_up_experts = Vec::with_capacity(num_experts);
        let mut down_experts = Vec::with_capacity(num_experts);

        // Pack experts
        for i in 0..num_experts {
            let expert_vb = experts_vb.pp(i.to_string());
            let (gate_up_expert, down_expert) = {
                // n x k format
                let gate_expert = expert_vb.pp("gate_proj").get_with_hints(
                    (cfg.moe_intermediate_size, cfg.hidden_size),
                    "weight",
                    shard(0, comm.rank(), comm.world_size()),
                )?;
                let up_expert = expert_vb.pp("up_proj").get_with_hints(
                    (cfg.moe_intermediate_size, cfg.hidden_size),
                    "weight",
                    shard(0, comm.rank(), comm.world_size()),
                )?;
                let down_expert = expert_vb.pp("down_proj").get_with_hints(
                    (cfg.hidden_size, cfg.moe_intermediate_size),
                    "weight",
                    shard(1, comm.rank(), comm.world_size()),
                )?;
                // Pack gate_proj and up_proj
                let gate_up_expert = Tensor::cat(&[&gate_expert, &up_expert], 0)?;

                (gate_up_expert, down_expert)
            };

            gate_up_experts.push(gate_up_expert);
            down_experts.push(down_expert);
        }

        let gate_up_w = Tensor::stack(&gate_up_experts, 0)?;
        let down_w = Tensor::stack(&down_experts, 0)?;
        let world_size = comm.world_size();
        let w_size_n = gate_up_w.dim(1)? / 2;

        Ok(Self {
            gate,
            gate_up_w,
            down_w,
            w_size_n,
            act: candle_nn::Activation::Silu,
            norm_topk_prob: cfg.norm_topk_prob,
            num_experts_per_tok: cfg.num_experts_per_tok,
            all_reduce: SumAllReduce::new(comm),
            world_size,
            dtype: experts_vb.dtype(),
            stacked_format: false,
        })
    }

    /// Load stacked format weights (pre-packed gate_up_proj, down_proj).
    /// - gate_up_proj: [num_experts, hidden, inter*2]
    /// - down_proj: [num_experts, inter, hidden]
    ///
    /// No modifications are made to the tensors during loading.
    /// Uses a transposed kernel that handles this layout directly.
    fn load_stacked(
        cfg: &MoEConfig,
        gate: Linear,
        experts_vb: ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let num_experts = cfg.num_experts;

        // Load stacked gate_up_proj: [num_experts, hidden, inter*2]
        let gate_up_w = experts_vb.get_with_hints(
            (num_experts, cfg.hidden_size, cfg.moe_intermediate_size * 2),
            "gate_up_proj",
            shard(2, comm.rank(), comm.world_size()),
        )?;

        // Load stacked down_proj: [num_experts, inter, hidden]
        let down_w = experts_vb.get_with_hints(
            (num_experts, cfg.moe_intermediate_size, cfg.hidden_size),
            "down_proj",
            shard(1, comm.rank(), comm.world_size()),
        )?;

        let world_size = comm.world_size();
        let w_size_n = gate_up_w.dim(2)? / 2;

        Ok(Self {
            gate,
            gate_up_w,
            down_w,
            w_size_n,
            act: candle_nn::Activation::Silu,
            norm_topk_prob: cfg.norm_topk_prob,
            num_experts_per_tok: cfg.num_experts_per_tok,
            all_reduce: SumAllReduce::new(comm),
            world_size,
            dtype: experts_vb.dtype(),
            stacked_format: true,
        })
    }

    pub fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        let xs = xs.reshape(((), hidden_dim))?;
        let (num_tokens, hidden_dim) = xs.dims2()?;
        let router_logits = self.gate.forward(&xs)?;

        let routing_weights =
            candle_nn::ops::softmax_last_dim(&router_logits.to_dtype(DType::F32)?)?;

        let topk_ids = routing_weights
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, self.num_experts_per_tok)?
            .contiguous()?;

        let mut topk_weights = routing_weights.gather(&topk_ids, D::Minus1)?;

        if self.norm_topk_prob {
            topk_weights = topk_weights.broadcast_div(&topk_weights.sum_keepdim(D::Minus1)?)?;
        }

        let (expert_ids, sorted_token_ids) = if is_prefill {
            #[cfg(feature = "cuda")]
            {
                use crate::ops::ArgSortOp;
                topk_ids.flatten_all()?.sort(true)?
            }
            #[cfg(not(feature = "cuda"))]
            topk_ids.flatten_all()?.sort_last_dim(true)?
        } else {
            topk_ids.flatten_all()?.sort_last_dim(true)?
        };

        // Use appropriate kernel based on weight format
        let gate_up = if self.stacked_format {
            // Stacked format: [E, K, N] - use transposed kernel
            moe::moe_gemm_transposed(
                &xs,
                &self.gate_up_w,
                &None,
                &sorted_token_ids,
                &expert_ids,
                self.num_experts_per_tok,
                is_prefill,
            )?
        } else {
            // Standard format: [E, N, K]
            moe::moe_gemm(
                &xs,
                &self.gate_up_w,
                &None,
                &sorted_token_ids,
                &expert_ids,
                self.num_experts_per_tok,
                is_prefill,
            )?
        };

        let gate = gate_up
            .narrow(candle_core::D::Minus1, 0, self.w_size_n)?
            .contiguous()?;
        let up = gate_up
            .narrow(candle_core::D::Minus1, self.w_size_n, self.w_size_n)?
            .contiguous()?;

        //(M * top_k, N // 2)
        let down_inputs = (up * gate.apply(&self.act)?)?.reshape(((), self.w_size_n))?;

        // Use appropriate kernel based on weight format
        let mut ys = if self.stacked_format {
            // Stacked format: [E, K, N] - use transposed kernel
            moe::moe_gemm_transposed(
                &down_inputs,
                &self.down_w,
                &Some(topk_weights),
                &sorted_token_ids,
                &expert_ids,
                self.num_experts_per_tok,
                is_prefill,
            )?
        } else {
            // Standard format: [E, N, K]
            moe::moe_gemm(
                &down_inputs,
                &self.down_w,
                &Some(topk_weights),
                &sorted_token_ids,
                &expert_ids,
                self.num_experts_per_tok,
                is_prefill,
            )?
        }
        .reshape((num_tokens, (), hidden_dim))?
        .sum(D::Minus2)?;

        if self.world_size > 1 {
            ys = self.all_reduce.sum_all_reduce(&ys)?;
        }
        ys.reshape((b_size, seq_len, hidden_dim))
    }
}
