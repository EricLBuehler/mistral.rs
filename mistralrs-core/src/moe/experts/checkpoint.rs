use candle_core::{Result, Tensor};
use mistralrs_quant::{Shard, ShardedVarBuilder};
use std::sync::Arc;

use crate::moe::shard;

use super::config::{ExpertProj, MoEExpertsConfig};

/// Reads the experts checkpoint in any on-disk layout (combined `gate_up_proj` vs per-expert, detected once) and yields canonical ENK weights; the caller picks the root prefix.
pub(super) struct ExpertCheckpoint<'a> {
    pub(super) cfg: &'a MoEExpertsConfig,
    vb: ShardedVarBuilder,
    rank: usize,
    world_size: usize,
    layout: ExpertSourceLayout,
}

impl<'a> ExpertCheckpoint<'a> {
    pub(super) fn new(
        cfg: &'a MoEExpertsConfig,
        vb: ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let shape_of = |rel: &str| vb.tensor_shape(rel).map(|s| s.to_vec());
        let layout = ExpertSourceLayout::detect(&shape_of, ExpertProj::Gate).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "No known expert checkpoint layout under `{}`.",
                vb.prefix()
            ))
        })?;
        Ok(Self {
            cfg,
            vb,
            rank: comm.rank(),
            world_size: comm.world_size(),
            layout,
        })
    }

    /// Canonical ENK stack for one projection.
    pub(super) fn stacked_proj(&self, proj: ExpertProj) -> Result<Tensor> {
        let cfg = self.cfg;
        let num_experts = cfg.num_experts;
        match &self.layout {
            ExpertSourceLayout::Fused { .. } => match proj {
                ExpertProj::Gate | ExpertProj::Up => {
                    let inter_shard = cfg.moe_intermediate_size / self.world_size;
                    if !cfg.moe_intermediate_size.is_multiple_of(self.world_size) {
                        candle_core::bail!(
                            "Intermediate size {} is not divisible by world size {}.",
                            cfg.moe_intermediate_size,
                            self.world_size
                        );
                    }
                    let half_offset = match proj {
                        ExpertProj::Gate => 0,
                        ExpertProj::Up => cfg.moe_intermediate_size,
                        ExpertProj::Down => unreachable!(),
                    };
                    self.read_proj_offset(
                        (num_experts, cfg.moe_intermediate_size * 2, cfg.hidden_size),
                        (num_experts, cfg.hidden_size, cfg.moe_intermediate_size * 2),
                        "gate_up_proj",
                        1,
                        2,
                        half_offset + self.rank * inter_shard,
                        inter_shard,
                    )
                }
                ExpertProj::Down => self.read_proj(
                    (num_experts, cfg.hidden_size, cfg.moe_intermediate_size),
                    (num_experts, cfg.moe_intermediate_size, cfg.hidden_size),
                    "down_proj",
                    2,
                    1,
                ),
            },
            ExpertSourceLayout::PerExpert { names, .. } => {
                let name = proj.name_in(names);
                let (shape, shard_dim) = match proj {
                    ExpertProj::Gate | ExpertProj::Up => {
                        ((cfg.moe_intermediate_size, cfg.hidden_size), 0)
                    }
                    ExpertProj::Down => ((cfg.hidden_size, cfg.moe_intermediate_size), 1),
                };
                let slabs = (0..num_experts)
                    .map(|i| {
                        self.vb.pp(i.to_string()).pp(name).get_with_hints(
                            shape,
                            "weight",
                            shard(shard_dim, self.rank, self.world_size),
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                Tensor::stack(&slabs, 0)
            }
        }
    }

    /// Canonical ENK: gate_up [E, 2*inter, hidden], down [E, hidden, inter].
    pub(super) fn stacked_enk(&self) -> Result<(Tensor, Tensor)> {
        let cfg = self.cfg;
        let num_experts = cfg.num_experts;
        if let ExpertSourceLayout::Fused { .. } = self.layout {
            // gate_up concatenates gate and up along the inter dim, so a naive shard of that dim
            // would hand whole gate/up halves to different ranks. Shard each half separately.
            let gate_up = if self.world_size > 1 {
                let inter_shard = cfg.moe_intermediate_size / self.world_size;
                if !cfg.moe_intermediate_size.is_multiple_of(self.world_size) {
                    candle_core::bail!(
                        "Intermediate size {} is not divisible by world size {}.",
                        cfg.moe_intermediate_size,
                        self.world_size
                    );
                }
                let gate = self.read_proj_offset(
                    (num_experts, cfg.moe_intermediate_size * 2, cfg.hidden_size),
                    (num_experts, cfg.hidden_size, cfg.moe_intermediate_size * 2),
                    "gate_up_proj",
                    1,
                    2,
                    self.rank * inter_shard,
                    inter_shard,
                )?;
                let up = self.read_proj_offset(
                    (num_experts, cfg.moe_intermediate_size * 2, cfg.hidden_size),
                    (num_experts, cfg.hidden_size, cfg.moe_intermediate_size * 2),
                    "gate_up_proj",
                    1,
                    2,
                    cfg.moe_intermediate_size + self.rank * inter_shard,
                    inter_shard,
                )?;
                Tensor::cat(&[&gate, &up], 1)?
            } else {
                self.read_proj(
                    (num_experts, cfg.moe_intermediate_size * 2, cfg.hidden_size),
                    (num_experts, cfg.hidden_size, cfg.moe_intermediate_size * 2),
                    "gate_up_proj",
                    1,
                    2,
                )?
            };
            let down = self.read_proj(
                (num_experts, cfg.hidden_size, cfg.moe_intermediate_size),
                (num_experts, cfg.moe_intermediate_size, cfg.hidden_size),
                "down_proj",
                2,
                1,
            )?;
            Ok((gate_up, down))
        } else {
            // Per-expert nn.Linear weights [out, in]; stacking gives natural ENK directly.
            let mut gate_up_experts = Vec::with_capacity(num_experts);
            let mut down_experts = Vec::with_capacity(num_experts);
            let ExpertSourceLayout::PerExpert { names, .. } = self.layout else {
                unreachable!()
            };
            for i in 0..num_experts {
                let expert_vb = self.vb.pp(i.to_string());
                let gate = expert_vb.pp(names.gate).get_with_hints(
                    (cfg.moe_intermediate_size, cfg.hidden_size),
                    "weight",
                    shard(0, self.rank, self.world_size),
                )?;
                let up = expert_vb.pp(names.up).get_with_hints(
                    (cfg.moe_intermediate_size, cfg.hidden_size),
                    "weight",
                    shard(0, self.rank, self.world_size),
                )?;
                let down = expert_vb.pp(names.down).get_with_hints(
                    (cfg.hidden_size, cfg.moe_intermediate_size),
                    "weight",
                    shard(1, self.rank, self.world_size),
                )?;
                gate_up_experts.push(Tensor::cat(&[&gate, &up], 0)?);
                down_experts.push(down);
            }
            Ok((
                Tensor::stack(&gate_up_experts, 0)?,
                Tensor::stack(&down_experts, 0)?,
            ))
        }
    }

    fn fused_transposed(&self) -> bool {
        matches!(self.layout, ExpertSourceLayout::Fused { transposed, .. } if transposed)
    }

    /// Read a combined `[E, out, in]` projection; the detected layout fixes the orientation.
    fn read_proj(
        &self,
        canonical: (usize, usize, usize),
        transposed: (usize, usize, usize),
        name: &str,
        canonical_shard: usize,
        transposed_shard: usize,
    ) -> Result<Tensor> {
        if self.fused_transposed() {
            self.vb
                .get_with_hints(
                    transposed,
                    name,
                    shard(transposed_shard, self.rank, self.world_size),
                )
                .and_then(|t| t.transpose(1, 2)?.contiguous())
        } else {
            self.vb.get_with_hints(
                canonical,
                name,
                shard(canonical_shard, self.rank, self.world_size),
            )
        }
    }

    /// Like `read_proj`, but slicing an explicit offset range of the sharded dim.
    #[allow(clippy::too_many_arguments)]
    fn read_proj_offset(
        &self,
        canonical: (usize, usize, usize),
        transposed: (usize, usize, usize),
        name: &str,
        canonical_dim: usize,
        transposed_dim: usize,
        offset: usize,
        len: usize,
    ) -> Result<Tensor> {
        if self.fused_transposed() {
            self.vb
                .get_with_hints(
                    transposed,
                    name,
                    Shard::Offset {
                        dim: transposed_dim,
                        offset,
                        len,
                    },
                )
                .and_then(|t| t.transpose(1, 2)?.contiguous())
        } else {
            self.vb.get_with_hints(
                canonical,
                name,
                Shard::Offset {
                    dim: canonical_dim,
                    offset,
                    len,
                },
            )
        }
    }
}

/// On-disk expert layout, detected from tensor names and shapes alone. Any layout
/// [`ExpertCheckpoint`] reads must be representable here so calibration can re-read it.
pub(super) enum ExpertSourceLayout {
    /// Per-expert `{prefix}.{i}.{name}.weight` nn.Linear slabs.
    PerExpert {
        names: super::config::ExpertProjNames,
        count: usize,
    },
    /// Fused 3D `gate_up_proj`/`down_proj`; `transposed` = conv-A `[E, in, out]` orientation.
    Fused { transposed: bool, inter: usize },
}

impl ExpertSourceLayout {
    /// `shape_of` resolves names relative to the experts module.
    pub(super) fn detect(
        shape_of: &dyn Fn(&str) -> Option<Vec<usize>>,
        proj: ExpertProj,
    ) -> Option<Self> {
        for names in super::config::ExpertProjNames::KNOWN {
            let name = proj.name_in(&names);
            if shape_of(&format!("0.{name}.weight")).is_some() {
                let count = (0..)
                    .take_while(|i| shape_of(&format!("{i}.{name}.weight")).is_some())
                    .count();
                return Some(Self::PerExpert { names, count });
            }
        }

        let gu = shape_of("gate_up_proj")?;
        let dn = shape_of("down_proj")?;
        let (&[_, a, b], &[_, c, d]) = (&gu[..], &dn[..]) else {
            return None;
        };
        // Canonical ENK: gate_up [E, 2I, H], down [E, H, I]; transposed: [E, H, 2I] / [E, I, H].
        if a == 2 * d && b == c {
            Some(Self::Fused {
                transposed: false,
                inter: d,
            })
        } else if b == 2 * c && a == d {
            Some(Self::Fused {
                transposed: true,
                inter: c,
            })
        } else {
            None
        }
    }

    fn detect_in_map(
        shapes: &std::collections::HashMap<String, Vec<usize>>,
        prefix: &str,
        proj: ExpertProj,
    ) -> Option<Self> {
        Self::detect(&|rel| shapes.get(&format!("{prefix}.{rel}")).cloned(), proj)
    }

    /// Rebuild the canonical stacked `[E, out, in]` tensor for `proj` from the source.
    pub(super) fn read_stack(
        &self,
        source: &mistralrs_quant::safetensors::MmapedSafetensors,
        prefix: &str,
        proj: ExpertProj,
    ) -> Result<Tensor> {
        match self {
            Self::PerExpert { names, count } => {
                let name = proj.name_in(names);
                let slabs = (0..*count)
                    .map(|i| {
                        source.load(
                            &format!("{prefix}.{i}.{name}.weight"),
                            &candle_core::Device::Cpu,
                            None,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                Tensor::stack(&slabs, 0)
            }
            Self::Fused { transposed, inter } => {
                let tensor_name = match proj {
                    ExpertProj::Down => format!("{prefix}.down_proj"),
                    _ => format!("{prefix}.gate_up_proj"),
                };
                let mut t = source.load(&tensor_name, &candle_core::Device::Cpu, None)?;
                if *transposed {
                    t = t.transpose(1, 2)?.contiguous()?;
                }
                match proj {
                    ExpertProj::Gate => t.narrow(1, 0, *inter)?.contiguous(),
                    ExpertProj::Up => t.narrow(1, *inter, *inter)?.contiguous(),
                    ExpertProj::Down => Ok(t),
                }
            }
        }
    }
}

/// Whether `key` is a canonical expert key whose stack the source checkpoint can rebuild.
pub(crate) fn expert_stack_available(
    shapes: &std::collections::HashMap<String, Vec<usize>>,
    key: &str,
) -> bool {
    ExpertProj::split_canonical_key(key).is_some_and(|(prefix, proj)| {
        ExpertSourceLayout::detect_in_map(shapes, prefix, proj).is_some()
    })
}

/// Rebuild the canonical `[E, out, in]` stack for a tracked expert key from source weights.
pub(crate) fn rebuild_expert_stack(
    source: &mistralrs_quant::safetensors::MmapedSafetensors,
    shapes: &std::collections::HashMap<String, Vec<usize>>,
    key: &str,
) -> Result<Option<Tensor>> {
    let Some((prefix, proj)) = ExpertProj::split_canonical_key(key) else {
        return Ok(None);
    };
    let Some(layout) = ExpertSourceLayout::detect_in_map(shapes, prefix, proj) else {
        return Ok(None);
    };
    layout.read_stack(source, prefix, proj).map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use std::collections::HashMap;

    const E: usize = 4;
    const INTER: usize = 8;
    const HIDDEN: usize = 12;

    fn write_st(path: &std::path::Path, tensors: Vec<(String, Tensor)>) {
        candle_core::safetensors::save(&tensors.into_iter().collect(), path).unwrap();
    }

    fn open_source(
        path: &std::path::Path,
    ) -> (
        mistralrs_quant::safetensors::MmapedSafetensors,
        HashMap<String, Vec<usize>>,
    ) {
        let source = unsafe { mistralrs_quant::safetensors::MmapedSafetensors::new(path).unwrap() };
        let shapes = source
            .tensors()
            .into_iter()
            .map(|(name, view)| (name, view.shape().to_vec()))
            .collect();
        (source, shapes)
    }

    fn assert_close(a: &Tensor, b: &Tensor) {
        let diff = (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-6, "max diff {diff}");
    }

    #[test]
    fn expert_stack_per_expert_layout() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let file = dir.path().join("model.safetensors");
        let gate: Vec<Tensor> = (0..E)
            .map(|_| Tensor::randn(0f32, 1f32, (INTER, HIDDEN), &Device::Cpu).unwrap())
            .collect();
        let mut tensors = Vec::new();
        for (i, g) in gate.iter().enumerate() {
            tensors.push((
                format!("model.layers.0.mlp.experts.{i}.w1.weight"),
                g.clone(),
            ));
        }
        write_st(&file, tensors);
        let (source, shapes) = open_source(&file);

        let layout = ExpertSourceLayout::detect_in_map(
            &shapes,
            "model.layers.0.mlp.experts",
            ExpertProj::Gate,
        )
        .expect("per-expert layout");
        let stack = layout.read_stack(&source, "model.layers.0.mlp.experts", ExpertProj::Gate)?;
        assert_eq!(stack.dims(), [E, INTER, HIDDEN]);
        assert_close(&stack, &Tensor::stack(&gate, 0)?);
        Ok(())
    }

    #[test]
    fn expert_stack_fused_layouts() -> Result<()> {
        let gate_up = Tensor::randn(0f32, 1f32, (E, 2 * INTER, HIDDEN), &Device::Cpu)?;
        let down = Tensor::randn(0f32, 1f32, (E, HIDDEN, INTER), &Device::Cpu)?;

        for transposed in [false, true] {
            let dir = tempfile::tempdir()?;
            let file = dir.path().join("model.safetensors");
            let (gu_t, dn_t) = if transposed {
                (
                    gate_up.transpose(1, 2)?.contiguous()?,
                    down.transpose(1, 2)?.contiguous()?,
                )
            } else {
                (gate_up.clone(), down.clone())
            };
            write_st(
                &file,
                vec![
                    ("experts.gate_up_proj".to_string(), gu_t),
                    ("experts.down_proj".to_string(), dn_t),
                ],
            );
            let (source, shapes) = open_source(&file);

            let detect = |proj: ExpertProj| {
                ExpertSourceLayout::detect_in_map(&shapes, "experts", proj).expect("fused")
            };
            let gate = detect(ExpertProj::Gate).read_stack(&source, "experts", ExpertProj::Gate)?;
            let up = detect(ExpertProj::Up).read_stack(&source, "experts", ExpertProj::Up)?;
            let dn = detect(ExpertProj::Down).read_stack(&source, "experts", ExpertProj::Down)?;
            assert_close(&gate, &gate_up.narrow(1, 0, INTER)?.contiguous()?);
            assert_close(&up, &gate_up.narrow(1, INTER, INTER)?.contiguous()?);
            assert_close(&dn, &down);
        }
        Ok(())
    }

    fn loader_vb(path: &std::path::Path) -> ShardedVarBuilder {
        unsafe {
            mistralrs_quant::ShardedSafeTensors::sharded(
                &[path],
                candle_core::DType::F32,
                &candle_core::Device::Cpu,
                None,
                Arc::new(|_| true),
            )
            .unwrap()
        }
    }

    #[test]
    fn checkpoint_loads_all_layouts() -> Result<()> {
        let cfg = super::super::config::MoEExpertsConfig {
            num_experts: E,
            num_experts_per_tok: 2,
            hidden_size: HIDDEN,
            moe_intermediate_size: INTER,
        };
        let comm = Arc::new(
            mistralrs_quant::Comm::from_device(
                mistralrs_quant::Id::new(),
                &candle_core::Device::Cpu,
                0,
                1,
            )
            .unwrap(),
        );

        let gate_up = Tensor::randn(
            0f32,
            1f32,
            (E, 2 * INTER, HIDDEN),
            &candle_core::Device::Cpu,
        )?;
        let down = Tensor::randn(0f32, 1f32, (E, HIDDEN, INTER), &candle_core::Device::Cpu)?;

        // fused, both orientations
        for transposed in [false, true] {
            let dir = tempfile::tempdir()?;
            let file = dir.path().join("model.safetensors");
            let (gu_t, dn_t) = if transposed {
                (
                    gate_up.transpose(1, 2)?.contiguous()?,
                    down.transpose(1, 2)?.contiguous()?,
                )
            } else {
                (gate_up.clone(), down.clone())
            };
            write_st(
                &file,
                vec![
                    ("gate_up_proj".to_string(), gu_t),
                    ("down_proj".to_string(), dn_t),
                ],
            );
            let ckpt = ExpertCheckpoint::new(&cfg, loader_vb(&file), &comm)?;
            let (gu, dn) = ckpt.stacked_enk()?;
            assert_close(&gu, &gate_up);
            assert_close(&dn, &down);
            assert_close(
                &ckpt.stacked_proj(ExpertProj::Gate)?,
                &gate_up.narrow(1, 0, INTER)?,
            );
            assert_close(
                &ckpt.stacked_proj(ExpertProj::Up)?,
                &gate_up.narrow(1, INTER, INTER)?,
            );
            assert_close(&ckpt.stacked_proj(ExpertProj::Down)?, &down);
        }

        // per-expert, both naming families
        for names in super::super::config::ExpertProjNames::KNOWN {
            let dir = tempfile::tempdir()?;
            let file = dir.path().join("model.safetensors");
            let mut tensors = Vec::new();
            for i in 0..E {
                tensors.push((
                    format!("{i}.{}.weight", names.gate),
                    gate_up.get(i)?.narrow(0, 0, INTER)?.contiguous()?,
                ));
                tensors.push((
                    format!("{i}.{}.weight", names.up),
                    gate_up.get(i)?.narrow(0, INTER, INTER)?.contiguous()?,
                ));
                tensors.push((format!("{i}.{}.weight", names.down), down.get(i)?));
            }
            write_st(&file, tensors);
            let ckpt = ExpertCheckpoint::new(&cfg, loader_vb(&file), &comm)?;
            let (gu, dn) = ckpt.stacked_enk()?;
            assert_close(&gu, &gate_up);
            assert_close(&dn, &down);
            assert_close(
                &ckpt.stacked_proj(ExpertProj::Gate)?,
                &gate_up.narrow(1, 0, INTER)?,
            );
            assert_close(
                &ckpt.stacked_proj(ExpertProj::Up)?,
                &gate_up.narrow(1, INTER, INTER)?,
            );
            assert_close(&ckpt.stacked_proj(ExpertProj::Down)?, &down);
        }
        Ok(())
    }
}
