//! Implementation of the Descript Audio Codec (DAC) model
//!
//! See: [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec)
//!
/// An efficient neural codec for compressing/decompressing audio
///
use candle_core::{IndexOp, Result, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, VarBuilder};

// Applies weight norm for inference by recomputing the weight tensor. This
// does not apply to training.
// https://pytorch.org/docs/stable/generated/torch.nn.utils.weight_norm.html
fn conv1d_weight_norm(
    in_c: usize,
    out_c: usize,
    kernel_size: usize,
    config: candle_nn::Conv1dConfig,
    vb: VarBuilder,
) -> Result<Conv1d> {
    let weight_g = vb.get((out_c, 1, 1), "weight_g")?;
    let weight_v = vb.get((out_c, in_c, kernel_size), "weight_v")?;
    let norm_v = weight_v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
    let weight = weight_v.broadcast_mul(&weight_g)?.broadcast_div(&norm_v)?;
    let bias = vb.get(out_c, "bias")?;
    Ok(Conv1d::new(weight, Some(bias), config))
}

fn conv_transpose1d_weight_norm(
    in_c: usize,
    out_c: usize,
    kernel_size: usize,
    bias: bool,
    config: candle_nn::ConvTranspose1dConfig,
    vb: VarBuilder,
) -> Result<ConvTranspose1d> {
    let weight_g = vb.get((in_c, 1, 1), "weight_g")?;
    let weight_v = vb.get((in_c, out_c, kernel_size), "weight_v")?;
    let norm_v = weight_v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
    let weight = weight_v.broadcast_mul(&weight_g)?.broadcast_div(&norm_v)?;
    let bias = if bias {
        Some(vb.get(out_c, "bias")?)
    } else {
        None
    };
    Ok(ConvTranspose1d::new(weight, bias, config))
}

#[derive(serde::Deserialize, Debug, Clone)]
pub struct Config {
    pub num_codebooks: usize,
    pub codebook_size: usize,
    pub latent_dim: usize,
}

impl Config {
    pub fn dia() -> Self {
        Self {
            num_codebooks: 9,
            codebook_size: 1024,
            latent_dim: 1024,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Snake1d {
    alpha: Tensor,
}

impl Snake1d {
    pub fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let alpha = vb.get((1, channels, 1), "alpha")?;
        Ok(Self { alpha })
    }
}

impl candle_core::Module for Snake1d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs_shape = xs.shape();
        let xs = xs.flatten_from(2)?;
        let sin = self.alpha.broadcast_mul(&xs)?.sin()?;
        let sin = (&sin * &sin)?;
        (xs + (&self.alpha + 1e-9)?.recip()?.broadcast_mul(&sin)?)?.reshape(xs_shape)
    }
}

#[derive(Debug, Clone)]
pub struct ResidualUnit {
    snake1: Snake1d,
    conv1: Conv1d,
    snake2: Snake1d,
    conv2: Conv1d,
}

impl ResidualUnit {
    pub fn new(dim: usize, dilation: usize, vb: VarBuilder) -> Result<Self> {
        let pad = ((7 - 1) * dilation) / 2;
        let vb = vb.pp("block");
        let snake1 = Snake1d::new(dim, vb.pp(0))?;
        let cfg1 = Conv1dConfig {
            dilation,
            padding: pad,
            ..Default::default()
        };
        let conv1 = conv1d_weight_norm(dim, dim, 7, cfg1, vb.pp(1))?;
        let snake2 = Snake1d::new(dim, vb.pp(2))?;
        let conv2 = conv1d_weight_norm(dim, dim, 1, Default::default(), vb.pp(3))?;
        Ok(Self {
            snake1,
            conv1,
            snake2,
            conv2,
        })
    }
}

impl candle_core::Module for ResidualUnit {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let ys = xs
            .apply(&self.snake1)?
            .apply(&self.conv1)?
            .apply(&self.snake2)?
            .apply(&self.conv2)?;
        let pad = (xs.dim(D::Minus1)? - ys.dim(D::Minus1)?) / 2;
        if pad > 0 {
            &ys + xs.narrow(D::Minus1, pad, ys.dim(D::Minus1)?)
        } else {
            ys + xs
        }
    }
}

#[derive(Debug, Clone)]
pub struct DecoderBlock {
    snake1: Snake1d,
    conv_tr1: ConvTranspose1d,
    res1: ResidualUnit,
    res2: ResidualUnit,
    res3: ResidualUnit,
}

impl DecoderBlock {
    pub fn new(in_dim: usize, out_dim: usize, stride: usize, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("block");
        let snake1 = Snake1d::new(in_dim, vb.pp(0))?;
        let cfg = ConvTranspose1dConfig {
            stride,
            padding: stride.div_ceil(2),
            ..Default::default()
        };
        let conv_tr1 =
            conv_transpose1d_weight_norm(in_dim, out_dim, 2 * stride, true, cfg, vb.pp(1))?;
        let res1 = ResidualUnit::new(out_dim, 1, vb.pp(2))?;
        let res2 = ResidualUnit::new(out_dim, 3, vb.pp(3))?;
        let res3 = ResidualUnit::new(out_dim, 9, vb.pp(4))?;
        Ok(Self {
            snake1,
            conv_tr1,
            res1,
            res2,
            res3,
        })
    }
}

impl candle_nn::Module for DecoderBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.snake1)?
            .apply(&self.conv_tr1)?
            .apply(&self.res1)?
            .apply(&self.res2)?
            .apply(&self.res3)
    }
}

#[derive(Debug, Clone)]
pub struct Decoder {
    conv1: Conv1d,
    blocks: Vec<DecoderBlock>,
    snake1: Snake1d,
    conv2: Conv1d,
}

impl Decoder {
    pub fn new(
        in_c: usize,
        mut channels: usize,
        rates: &[usize],
        d_out: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg1 = Conv1dConfig {
            padding: 3,
            ..Default::default()
        };
        let conv1 = conv1d_weight_norm(in_c, channels, 7, cfg1, vb.pp(0))?;
        let mut blocks = Vec::with_capacity(rates.len());
        for (idx, stride) in rates.iter().enumerate() {
            let block = DecoderBlock::new(channels, channels / 2, *stride, vb.pp(idx + 1))?;
            channels /= 2;
            blocks.push(block)
        }
        let snake1 = Snake1d::new(channels, vb.pp(rates.len() + 1))?;
        let conv2 = conv1d_weight_norm(channels, d_out, 7, cfg1, vb.pp(rates.len() + 2))?;
        Ok(Self {
            conv1,
            blocks,
            snake1,
            conv2,
        })
    }
}

impl candle_core::Module for Decoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.apply(&self.conv1)?;
        for block in self.blocks.iter() {
            xs = xs.apply(block)?
        }
        xs.apply(&self.snake1)?.apply(&self.conv2)
    }
}

#[allow(unused)]
#[derive(Clone, Debug)]
pub struct VectorQuantizer {
    in_proj: Conv1d,
    out_proj: Conv1d,
    codebook: candle_nn::Embedding,
}

impl VectorQuantizer {
    pub fn new(in_dim: usize, cb_size: usize, cb_dim: usize, vb: VarBuilder) -> Result<Self> {
        let in_proj = conv1d_weight_norm(in_dim, cb_dim, 1, Default::default(), vb.pp("in_proj"))?;
        let out_proj =
            conv1d_weight_norm(cb_dim, in_dim, 1, Default::default(), vb.pp("out_proj"))?;
        let codebook = candle_nn::embedding(cb_size, cb_dim, vb.pp("codebook"))?;
        Ok(Self {
            in_proj,
            out_proj,
            codebook,
        })
    }

    pub fn embed_code(&self, embed_id: &Tensor) -> Result<Tensor> {
        embed_id.apply(&self.codebook)
    }

    pub fn decode_code(&self, embed_id: &Tensor) -> Result<Tensor> {
        self.embed_code(embed_id)?.transpose(1, 2)
    }
}

#[derive(Clone, Debug)]
pub struct ResidualVectorQuantizer {
    quantizers: Vec<VectorQuantizer>,
}

impl ResidualVectorQuantizer {
    pub fn new(
        input_dim: usize,
        n_codebooks: usize,
        cb_size: usize,
        cb_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let vb = &vb.pp("quantizers");
        let quantizers = (0..n_codebooks)
            .map(|i| VectorQuantizer::new(input_dim, cb_size, cb_dim, vb.pp(i)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { quantizers })
    }

    #[allow(clippy::wrong_self_convention)]
    pub fn from_codes(&self, codes: &Tensor) -> Result<Tensor> {
        let mut sum = None;
        for (idx, quantizer) in self.quantizers.iter().enumerate() {
            let z_p_i = quantizer.decode_code(&codes.i((.., idx))?)?;
            let z_q_i = z_p_i.apply(&quantizer.out_proj)?;
            let s = match sum {
                None => z_q_i,
                Some(s) => (s + z_q_i)?,
            };
            sum = Some(s)
        }
        match sum {
            Some(s) => Ok(s),
            None => candle_core::bail!("empty codebooks"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    // pub encoder: Encoder,
    pub quantizer: ResidualVectorQuantizer,
    pub decoder: Decoder,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        // let encoder = Encoder::new(64, &[2, 4, 8, 8], cfg.latent_dim, vb.pp("encoder").pp("model"))?;
        let quantizer = ResidualVectorQuantizer::new(
            cfg.latent_dim,
            cfg.num_codebooks,
            cfg.codebook_size,
            8,
            vb.pp("quantizer"),
        )?;
        let decoder = Decoder::new(
            cfg.latent_dim,
            1536,
            &[8, 8, 4, 2],
            1,
            vb.pp("decoder").pp("model"),
        )?;
        Ok(Self {
            // encoder,
            decoder,
            quantizer,
        })
    }

    pub fn decode_codes(&self, audio_codes: &Tensor) -> Result<Tensor> {
        let audio_values = self.quantizer.from_codes(audio_codes)?;
        audio_values.apply(&self.decoder)
    }
}
