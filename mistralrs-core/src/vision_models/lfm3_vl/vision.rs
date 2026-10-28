use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Linear, Module};
use mistralrs_quant::{
    fused_split_glu, lfm3_spatial_conv::gated_spatial_conv, GluActivationType, ShardedVarBuilder,
};

use crate::{
    attention::{AttentionMask, SdpaParams},
    layers::{linear_no_bias, RmsNorm, Sdpa},
    pipeline::text_models_inputs_processor::FlashParams,
    utils::unvarbuilder::UnVarBuilder,
};

use super::config::{LayerType, VisionConfig};

struct VisionMlp {
    gate_up: Linear,
    down: Linear,
    intermediate_size: usize,
}

impl VisionMlp {
    fn new(config: &VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let shape = (config.intermediate_size, config.hidden_size);
        let gate = vb.pp("w1").get(shape, "weight")?;
        let up = vb.pp("w3").get(shape, "weight")?;
        Ok(Self {
            gate_up: Linear::new(Tensor::cat(&[gate, up], 0)?, None),
            down: linear_no_bias(config.intermediate_size, config.hidden_size, vb.pp("w2"))?,
            intermediate_size: config.intermediate_size,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate_up = self.gate_up.forward(xs)?;
        self.down.forward(&fused_split_glu(
            &gate_up,
            self.intermediate_size,
            GluActivationType::Silu,
        )?)
    }

    fn residual_tensors(&self, uvb: UnVarBuilder) {
        uvb.pp("w1").add_tensor(
            "weight",
            self.gate_up
                .weight()
                .narrow(0, 0, self.intermediate_size)
                .unwrap(),
        );
        uvb.pp("w3").add_tensor(
            "weight",
            self.gate_up
                .weight()
                .narrow(0, self.intermediate_size, self.intermediate_size)
                .unwrap(),
        );
        uvb.pp("w2").add(&self.down);
    }
}

struct VisionAttention {
    qkv: Linear,
    out_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

fn apply_rope(q: &Tensor, k: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<(Tensor, Tensor)> {
    let dtype = q.dtype();
    let (q, k) = mistralrs_quant::rotary::apply_rotary_qk_preselected(
        &q.to_dtype(DType::F32)?.transpose(0, 1)?.unsqueeze(0)?,
        &k.to_dtype(DType::F32)?.transpose(0, 1)?.unsqueeze(0)?,
        cos,
        sin,
        true,
    )?;
    Ok((
        q.to_dtype(dtype)?.squeeze(0)?.transpose(0, 1)?,
        k.to_dtype(dtype)?.squeeze(0)?.transpose(0, 1)?,
    ))
}

impl VisionAttention {
    fn new(config: &VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        let q = vb
            .pp("q_proj")
            .get((config.hidden_size, config.hidden_size), "weight")?;
        let kv_shape = (config.num_key_value_heads * head_dim, config.hidden_size);
        let k = vb.pp("k_proj").get(kv_shape, "weight")?;
        let v = vb.pp("v_proj").get(kv_shape, "weight")?;
        Ok(Self {
            qkv: Linear::new(Tensor::cat(&[q, k, v], 0)?, None),
            out_proj: linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("out_proj"))?,
            q_norm: RmsNorm::new(head_dim, config.norm_eps, vb.pp("q_layernorm"))?,
            k_norm: RmsNorm::new(head_dim, config.norm_eps, vb.pp("k_layernorm"))?,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim,
        })
    }

    #[allow(clippy::cast_precision_loss)]
    fn forward(
        &self,
        xs: &Tensor,
        frames: &[[usize; 2]],
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let tokens = xs.dim(0)?;
        let q_size = self.num_heads * self.head_dim;
        let kv_size = self.num_kv_heads * self.head_dim;
        let qkv = self.qkv.forward(xs)?;
        let q = self.q_norm.forward(&qkv.narrow(1, 0, q_size)?.reshape((
            tokens,
            self.num_heads,
            self.head_dim,
        ))?)?;
        let k = self
            .k_norm
            .forward(&qkv.narrow(1, q_size, kv_size)?.reshape((
                tokens,
                self.num_kv_heads,
                self.head_dim,
            ))?)?;
        let v = qkv.narrow(1, q_size + kv_size, kv_size)?.reshape((
            tokens,
            self.num_kv_heads,
            self.head_dim,
        ))?;
        let (q, k) = apply_rope(&q, &k, cos, sin)?;
        let mut outputs = Vec::new();
        let mut offset = 0;
        let mut start = 0;
        while start < frames.len() {
            let length = frames[start][0] * frames[start][1];
            let count = frames[start..]
                .iter()
                .take_while(|shape| shape[0] * shape[1] == length)
                .count();
            let q = q
                .narrow(0, offset, count * length)?
                .reshape((count, length, self.num_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            let k = k
                .narrow(0, offset, count * length)?
                .reshape((count, length, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            let v = v
                .narrow(0, offset, count * length)?
                .reshape((count, length, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            let output = Sdpa.run_attention(
                &q,
                &k,
                &v,
                &AttentionMask::None,
                Some(&FlashParams::empty(false)),
                &SdpaParams {
                    n_kv_groups: self.num_heads / self.num_kv_heads,
                    sliding_window: None,
                    softcap: None,
                    softmax_scale: 1.0 / (self.head_dim as f32).sqrt(),
                    sinks: None,
                },
            )?;
            outputs.push(output.transpose(1, 2)?.reshape((count * length, q_size))?);
            start += count;
            offset += count * length;
        }
        self.out_proj.forward(&Tensor::cat(&outputs, 0)?)
    }

    fn residual_tensors(&self, uvb: UnVarBuilder) {
        let q_size = self.num_heads * self.head_dim;
        let kv_size = self.num_kv_heads * self.head_dim;
        for (name, offset, size) in [
            ("q_proj", 0, q_size),
            ("k_proj", q_size, kv_size),
            ("v_proj", q_size + kv_size, kv_size),
        ] {
            uvb.pp(name)
                .add_tensor("weight", self.qkv.weight().narrow(0, offset, size).unwrap());
        }
        uvb.pp("out_proj").add(&self.out_proj);
        uvb.pp("q_layernorm").add(&self.q_norm);
        uvb.pp("k_layernorm").add(&self.k_norm);
    }
}

struct VisionConv {
    in_proj: Linear,
    spatial_conv: Tensor,
    out_proj: Linear,
    merge_size: usize,
}

#[cfg(test)]
fn to_raster(
    xs: &Tensor,
    batch: usize,
    height: usize,
    width: usize,
    merge: usize,
) -> Result<Tensor> {
    xs.reshape((
        batch,
        height / merge,
        width / merge,
        merge,
        merge,
        xs.dim(1)?,
    ))?
    .permute((0, 5, 1, 3, 2, 4))?
    .reshape((batch, (), height, width))?
    .contiguous()
}

#[cfg(test)]
fn from_raster(xs: &Tensor, merge: usize) -> Result<Tensor> {
    let (batch, channels, height, width) = xs.dims4()?;
    xs.reshape((batch, channels, height / merge, merge, width / merge, merge))?
        .permute((0, 2, 4, 3, 5, 1))?
        .reshape((batch * height * width, channels))
}

impl VisionConv {
    fn new(config: &VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            in_proj: linear_no_bias(config.hidden_size, 3 * config.hidden_size, vb.pp("in_proj"))?,
            spatial_conv: vb.pp("spatial_conv").get(
                (
                    config.hidden_size,
                    1,
                    config.conv_kernel_size,
                    config.conv_kernel_size,
                ),
                "weight",
            )?,
            out_proj: linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("out_proj"))?,
            merge_size: config.spatial_merge_size,
        })
    }

    fn forward(&self, xs: &Tensor, frames: &[[usize; 2]]) -> Result<Tensor> {
        let projected = self.in_proj.forward(xs)?;
        let mut outputs = Vec::new();
        let mut offset = 0;
        let mut start = 0;
        while start < frames.len() {
            let [height, width] = frames[start];
            let count = frames[start..]
                .iter()
                .take_while(|&&shape| shape == [height, width])
                .count();
            let frame = projected.narrow(0, offset, count * height * width)?;
            outputs.push(gated_spatial_conv(
                &frame,
                &self.spatial_conv,
                height,
                width,
                self.merge_size,
            )?);
            start += count;
            offset += count * height * width;
        }
        self.out_proj.forward(&Tensor::cat(&outputs, 0)?)
    }

    fn residual_tensors(&self, uvb: UnVarBuilder) {
        uvb.pp("in_proj").add(&self.in_proj);
        uvb.pp("spatial_conv")
            .add_tensor("weight", self.spatial_conv.clone());
        uvb.pp("out_proj").add(&self.out_proj);
    }
}

enum VisionOperator {
    Attention(VisionAttention),
    Conv(VisionConv),
}

struct VisionLayer {
    operator: VisionOperator,
    feed_forward: VisionMlp,
    operator_norm: RmsNorm,
    ffn_norm: RmsNorm,
}

impl VisionLayer {
    fn new(config: &VisionConfig, layer_type: LayerType, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            operator: match layer_type {
                LayerType::Attention => {
                    VisionOperator::Attention(VisionAttention::new(config, vb.pp("self_attn"))?)
                }
                LayerType::Conv => VisionOperator::Conv(VisionConv::new(config, vb.pp("conv"))?),
            },
            feed_forward: VisionMlp::new(config, vb.pp("feed_forward"))?,
            operator_norm: RmsNorm::new(
                config.hidden_size,
                config.norm_eps,
                vb.pp("operator_norm"),
            )?,
            ffn_norm: RmsNorm::new(config.hidden_size, config.norm_eps, vb.pp("ffn_norm"))?,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        frames: &[[usize; 2]],
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let normed = self.operator_norm.forward(xs)?;
        let output = match &self.operator {
            VisionOperator::Attention(attn) => attn.forward(&normed, frames, cos, sin)?,
            VisionOperator::Conv(conv) => conv.forward(&normed, frames)?,
        };
        let (residual, normed) = self.ffn_norm.forward_add_rms_norm(&output, xs)?;
        mistralrs_quant::elementwise::broadcast_add(&residual, &self.feed_forward.forward(&normed)?)
    }

    fn residual_tensors(&self, uvb: UnVarBuilder) {
        match &self.operator {
            VisionOperator::Attention(attn) => attn.residual_tensors(uvb.pp("self_attn")),
            VisionOperator::Conv(conv) => conv.residual_tensors(uvb.pp("conv")),
        }
        self.feed_forward.residual_tensors(uvb.pp("feed_forward"));
        uvb.pp("operator_norm").add(&self.operator_norm);
        uvb.pp("ffn_norm").add(&self.ffn_norm);
    }
}

pub(super) struct VisionModel {
    patchifier: Linear,
    embedding_norm: RmsNorm,
    layers: Vec<VisionLayer>,
    merge_size: usize,
    head_dim: usize,
    rope_theta: f64,
}

impl VisionModel {
    pub(super) fn new(config: &VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            patchifier: linear_no_bias(
                config.in_channels * config.temporal_patch_size * config.patch_size.pow(2),
                config.hidden_size,
                vb.pp("patchifier"),
            )?,
            embedding_norm: RmsNorm::new(
                config.hidden_size,
                config.norm_eps,
                vb.pp("embedding_norm"),
            )?,
            layers: config
                .layer_types
                .iter()
                .enumerate()
                .map(|(i, &layer_type)| VisionLayer::new(config, layer_type, vb.pp("layers").pp(i)))
                .collect::<Result<_>>()?,
            merge_size: config.spatial_merge_size,
            head_dim: config.hidden_size / config.num_attention_heads,
            rope_theta: config.rope_parameters.rope_theta,
        })
    }

    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    fn position_embeddings(
        &self,
        frames: &[[usize; 2]],
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let axis_dim = self.head_dim / 2;
        let mut frequencies = Vec::new();
        for &[height, width] in frames {
            for block_h in (0..height).step_by(self.merge_size) {
                for block_w in (0..width).step_by(self.merge_size) {
                    for h in block_h..block_h + self.merge_size {
                        for w in block_w..block_w + self.merge_size {
                            for coordinate in [h, w] {
                                for dim in (0..axis_dim).step_by(2) {
                                    frequencies.push(
                                        coordinate as f32
                                            / self.rope_theta.powf(dim as f64 / axis_dim as f64)
                                                as f32,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        let frequency = Tensor::from_vec(frequencies, ((), self.head_dim / 2), device)?;
        Ok((frequency.cos()?, frequency.sin()?))
    }

    pub(super) fn forward(&self, pixels: &Tensor, grid: &Tensor) -> Result<Tensor> {
        let mut frames = Vec::new();
        for shape in grid.to_device(&Device::Cpu)?.to_vec2::<u32>()? {
            let [time, height, width] = shape.as_slice() else {
                candle_core::bail!("LFM3-VL grid entries must have three dimensions");
            };
            frames.extend(std::iter::repeat_n(
                [*height as usize, *width as usize],
                *time as usize,
            ));
        }
        let pixels = pixels
            .to_device(self.patchifier.weight().device())?
            .to_dtype(self.patchifier.weight().dtype())?;
        let mut xs = self
            .embedding_norm
            .forward(&self.patchifier.forward(&pixels)?)?;
        let (cos, sin) = self.position_embeddings(&frames, xs.device())?;
        for layer in &self.layers {
            xs = layer.forward(&xs, &frames, &cos, &sin)?;
        }
        Ok(xs)
    }

    pub(super) fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.pp("patchifier").add(&self.patchifier);
        uvb.pp("embedding_norm").add(&self.embedding_norm);
        for (i, layer) in self.layers.iter().enumerate() {
            layer.residual_tensors(uvb.pp("layers").pp(i));
        }
        uvb.to_safetensors()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_block_raster_roundtrip_preserves_non_square_frames() -> Result<()> {
        let values = Tensor::arange(0f32, 96., &Device::Cpu)?.reshape((48, 2))?;
        let raster = to_raster(&values, 2, 4, 6, 2)?;
        let result = from_raster(&raster, 2)?;
        assert_eq!(values.to_vec2::<f32>()?, result.to_vec2::<f32>()?);
        let first_channel = raster.get(0)?.get(0)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(
            first_channel,
            vec![
                0., 2., 8., 10., 16., 18., 4., 6., 12., 14., 20., 22., 24., 26., 32., 34., 40.,
                42., 28., 30., 36., 38., 44., 46.
            ]
        );
        Ok(())
    }

    #[test]
    fn axial_rope_rotates_height_and_width_independently() -> Result<()> {
        let input = Tensor::new(&[[[1f32, 2., 3., 4., 5., 6., 7., 8.]]], &Device::Cpu)?;
        let angles = Tensor::new(
            &[[
                0f32,
                0.,
                std::f32::consts::FRAC_PI_2,
                std::f32::consts::FRAC_PI_2,
            ]],
            &Device::Cpu,
        )?;
        let result = apply_rope(&input, &input, &angles.cos()?, &angles.sin()?)?
            .0
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (actual, expected) in result.iter().zip([1., 2., -7., -8., 5., 6., 3., 4.]) {
            assert!((actual - expected).abs() < 1e-5);
        }
        Ok(())
    }
}
