use std::{marker::PhantomData, sync::MutexGuard};

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
        num_heads: usize,
        head_dim: usize,
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
    ReshapeAttnOutput {
        hidden_size: usize,
    },
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
        v: NodeId,
        layer_idx: usize,
    },
    RepeatKV {
        num_kv_groups: usize,
        from: NodeId,
    },
    StartModel {
        inp: usize,
    },
}

#[derive(Debug)]
pub struct Constructing;
#[derive(Debug)]
pub struct Ready;

#[derive(Debug)]
pub struct ComputationGraph<State> {
    nodes: Vec<NodeOperator>,
    data: Vec<Option<Tensor>>,
    latest_bs: Option<usize>,
    _ghost: PhantomData<State>,
}

impl ComputationGraph<Constructing> {
    pub fn empty() -> Self {
        Self {
            nodes: vec![],
            data: vec![],
            latest_bs: None,
            _ghost: PhantomData,
        }
    }

    #[must_use]
    pub fn add_op(&mut self, op: NodeOperator) -> NodeId {
        self.nodes.push(op);
        self.nodes.len() - 1
    }

    pub fn finalize_graph(self) -> ComputationGraph<Ready> {
        let len = self.nodes.len();
        ComputationGraph {
            nodes: self.nodes,
            data: vec![None; len],
            latest_bs: None,
            _ghost: PhantomData,
        }
    }

    /// Get ID of the latest node
    pub fn get_current_node_id(&self) -> NodeId {
        self.nodes.len() - 1
    }
}

impl ComputationGraph<Ready> {
    pub fn execute(
        &mut self,
        input: &Tensor,
        attention_mask: Option<Tensor>,
        mut cache: MutexGuard<'_, LayerCaches>,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
    ) -> Result<Tensor> {
        let mut x = input.clone();
        for (i, op) in self.nodes.iter().enumerate() {
            let res = match op {
                NodeOperator::Embedding { op } => Some(op.forward(&x)?),
                NodeOperator::RmsNorm { op, from } => {
                    Some(op.forward(self.data[*from].as_ref().unwrap())?)
                }
                NodeOperator::Linear { op, from } => {
                    Some(op.forward(self.data[*from].as_ref().unwrap())?)
                }
                NodeOperator::ReshapeRms {
                    from,
                    num_heads,
                    head_dim,
                } => {
                    let v = &self.data[*from].as_ref().unwrap();
                    let (b_sz, q_len, _, _) = v.dims4()?;
                    Some(v.reshape((b_sz * q_len, *num_heads, *head_dim))?)
                }
                NodeOperator::RoPE { op, q, k } => {
                    let mut q = self.data[*q].as_ref().unwrap().clone();
                    let mut k = self.data[*k].as_ref().unwrap().clone();
                    let b_sz = q.dim(0).unwrap();
                    op.forward(seqlen_offsets, &start_offsets_kernel, &mut q, &mut k, b_sz)?;
                    None
                }
                NodeOperator::Matmul { l, r } => {
                    let l = self.data[*l].as_ref().unwrap();
                    let r = self.data[*r].as_ref().unwrap();
                    Some(l.matmul(r)?)
                }
                NodeOperator::ReshapeAttn { from } => {
                    let v = &self.data[*from].as_ref().unwrap();
                    let (b_sz_q_len, num_heads, head_dim) = v.dims3()?;
                    Some(v.reshape((
                        self.latest_bs.unwrap(),
                        b_sz_q_len / self.latest_bs.unwrap(),
                        num_heads,
                        head_dim,
                    ))?)
                }
                NodeOperator::Transpose12 => Some(x.transpose(1, 2)?),
                NodeOperator::Transpose23 { from } => {
                    Some(self.data[*from].as_ref().unwrap().transpose(2, 3)?)
                }
                NodeOperator::Scale { factor } => Some((x.clone() * (*factor))?),
                NodeOperator::ApplyAttentionMask { from } => Some(match &attention_mask {
                    None => x.clone(),
                    Some(mask) => self.data[*from].as_ref().unwrap().broadcast_add(mask)?,
                }),
                NodeOperator::Softmax { from } => Some(candle_nn::ops::softmax_last_dim(
                    self.data[*from].as_ref().unwrap(),
                )?),
                NodeOperator::ReshapeAttnOutput { hidden_size } => {
                    let (b_sz, q_len, _) = x.dims3()?;
                    Some(x.reshape((b_sz, q_len, *hidden_size))?)
                }
                NodeOperator::Add { l, r } => Some(
                    self.data[*l]
                        .as_ref()
                        .unwrap()
                        .add(self.data[*r].as_ref().unwrap())?,
                ),
                NodeOperator::Activation { op } => Some(op.forward(&x)?),
                NodeOperator::Mul { l, r } => Some(
                    self.data[*l]
                        .as_ref()
                        .unwrap()
                        .mul(self.data[*r].as_ref().unwrap())?,
                ),
                NodeOperator::Contiguous => Some(x.contiguous()?),
                NodeOperator::UpdateKVCache { k, v, layer_idx } => {
                    let kv_cache = &mut cache[*layer_idx];
                    let key_states = self.data[*k].as_ref().unwrap();
                    let value_states = self.data[*v].as_ref().unwrap();
                    let (key_states, value_states) = match &*kv_cache {
                        None => (key_states.clone(), value_states.clone()),
                        Some((prev_k, prev_v)) => {
                            let key_states = candle_nn::ops::kvconcat(prev_k, key_states, 2)?;
                            let value_states = candle_nn::ops::kvconcat(prev_v, value_states, 2)?;
                            (key_states, value_states)
                        }
                    };
                    *kv_cache = Some((key_states.clone(), value_states.clone()));
                    None
                }
                NodeOperator::RepeatKV {
                    num_kv_groups,
                    from,
                } => {
                    let n_rep = *num_kv_groups;
                    let xs = self.data[*from].as_ref().unwrap().clone();
                    if n_rep == 1 {
                        Some(xs)
                    } else {
                        let (b_sz, num_kv_heads, seq_len, head_dim) = xs.dims4()?;
                        Some(
                            xs.unsqueeze(2)?
                                .expand((b_sz, num_kv_heads, n_rep, seq_len, head_dim))?
                                .reshape((b_sz, num_kv_heads * n_rep, seq_len, head_dim))?,
                        )
                    }
                }
                NodeOperator::StartModel { inp } => {
                    self.latest_bs = Some(self.data[*inp].as_ref().unwrap().dim(0)?);
                    None
                }
            };
            if let Some(ref res) = res {
                x = res.clone();
            }
            self.data[i] = res;
        }
        self.latest_bs = None;
        Ok(x)
    }
}
