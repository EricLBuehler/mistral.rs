use std::sync::MutexGuard;

use candle_core::{Result, Tensor};
use candle_nn::{
    layer_norm::RmsNormNonQuantized, Activation, Embedding, Linear, Module, RmsNorm,
    RotaryEmbedding,
};

use crate::models::LayerCaches;

type NodeId = usize;

#[derive(Clone, Debug)]
pub enum NodeOperator {
    Embedding {
        op: Embedding,
    },
    RmsNorm {
        op: RmsNorm<RmsNormNonQuantized>,
        from: NodeId,
    },
    Linear {
        op: Linear,
        from: NodeId,
    },
    ReshapeRms {
        from: NodeId,
    },
    RoPE {
        op: RotaryEmbedding,
        q: NodeId,
        k: NodeId,
    },
    Matmul {
        l: NodeId,
        r: NodeId,
    },
    ReshapeAttn {
        num_heads: usize,
        from: NodeId,
    },
    Transpose12,
    Transpose23 {
        from: NodeId,
    },
    Scale {
        factor: f64,
    },
    ApplyAttentionMask {
        from: NodeId,
    },
    Softmax {
        from: NodeId,
    },
    ReshapeAttnOutput,
    Add {
        l: NodeId,
        r: NodeId,
    },
    Activation {
        op: Activation,
    },
    Mul {
        l: NodeId,
        r: NodeId,
    },
    Contiguous,
    UpdateKVCache {
        k: NodeId,
        q: NodeId,
        layer_idx: usize,
    },
    RepeatKV {
        num_kv_groups: usize,
        from: NodeId,
    },
}

#[derive(Debug)]
pub struct ComputationGraph {
    nodes: Vec<NodeOperator>,
}

impl ComputationGraph {
    pub fn empty() -> Self {
        Self { nodes: vec![] }
    }

    #[must_use]
    pub fn add_op(&mut self, op: NodeOperator) -> NodeId {
        self.nodes.push(op);
        self.nodes.len() - 1
    }

    /// Get ID of the latest node
    pub fn get_current_node_id(&self) -> NodeId {
        self.nodes.len() - 1
    }

    pub fn execute(
        &self,
        input: &Tensor,
        attention_mask: Option<Tensor>,
        cache: MutexGuard<'_, LayerCaches>,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
    ) -> Result<Tensor> {
        let mut x = input.clone();
        for op in &self.nodes {
            match op {
                NodeOperator::Embedding { op } => {}
                NodeOperator::RmsNorm { op, from } => {}
                NodeOperator::Linear { op, from } => {}
                NodeOperator::ReshapeRms { from } => {}
                NodeOperator::RoPE { op, q, k } => {}
                NodeOperator::Matmul { l, r } => {}
                NodeOperator::ReshapeAttn { num_heads, from } => {}
                NodeOperator::Transpose12 => {}
                NodeOperator::Transpose23 { from } => {}
                NodeOperator::Scale { factor } => {}
                NodeOperator::ApplyAttentionMask { from } => {}
                NodeOperator::Softmax { from } => {}
                NodeOperator::ReshapeAttnOutput => {}
                NodeOperator::Add { l, r } => {}
                NodeOperator::Activation { op } => {}
                NodeOperator::Mul { l, r } => {}
                NodeOperator::Contiguous => {}
                NodeOperator::UpdateKVCache { k, q, layer_idx } => {}
                NodeOperator::RepeatKV {
                    num_kv_groups,
                    from,
                } => {}
            };
        }
        Ok(x)
    }
}
