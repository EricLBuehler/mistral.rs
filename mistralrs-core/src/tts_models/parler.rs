// https://github.com/huggingface/parler-tts/blob/dcaed95e1cce6f616e3e1956f8d63f0f3f5dfe5f/parler_tts/modeling_parler_tts.py

use std::usize;

use candle_core::{IndexOp, Result, Tensor, D};
use candle_nn::{
    embedding, Activation, Conv1d, Conv1dConfig, ConvTranspose1dConfig, Embedding, Module,
    Sequential, VarBuilder,
};
use serde::Deserialize;

use crate::{
    common_models::t5::{T5Config, T5ForConditionalGeneration},
    layers::{l2_norm, wn_conv1d, wn_conv_transpose1d},
};

#[derive(Clone, Debug, Deserialize)]
pub struct ParlerDecoderConfig {
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub num_hidden_layers: usize,
    pub ffn_dim: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub num_cross_attention_key_value_heads: Option<usize>,
    pub activation_function: Activation,
    pub hidden_size: usize,
    pub scale_embedding: bool,
    pub num_codebooks: usize,
    /// Use RoPE or absolute positional embeddings
    pub rope_embeddings: bool,
    pub rope_theta: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ParlerDACConfig {
    pub num_codebooks: usize,
    /// kbps
    pub model_bitrate: usize,
    pub codebook_size: usize,
    pub latent_dim: usize,
    pub frame_rate: usize,
    pub sampling_rate: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ParlerConfig {
    pub audio_encoder: ParlerDACConfig,
    pub decoder: ParlerDecoderConfig,
    pub text_encoder: T5Config,
    pub is_encoder_decoder: bool,
    pub prompt_cross_attention: bool,
}

/// Implementation of VQ similar to Karpathy's repo:
/// https://github.com/karpathy/deep-vector-quantization
struct VectorQuantize {
    in_proj: Conv1d,
    out_proj: Conv1d,
    codebook: Embedding,
}

struct VectorQuantizeOutput {
    z_q: Tensor,
    indices: Tensor,
    z_e: Tensor,
}

impl VectorQuantize {
    fn new(
        input_dim: usize,
        codebook_size: usize,
        codebook_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            in_proj: wn_conv1d(
                input_dim,
                codebook_dim,
                1,
                Conv1dConfig::default(),
                vb.pp("in_proj"),
            )?,
            out_proj: wn_conv1d(
                codebook_dim,
                input_dim,
                1,
                Conv1dConfig::default(),
                vb.pp("out_proj"),
            )?,
            codebook: embedding(codebook_size, codebook_dim, vb.pp("codebook"))?,
        })
    }

    /// Quantized the input tensor using a fixed codebook and returns the corresponding codebook vectors
    fn forward(&self, xs: &Tensor) -> Result<VectorQuantizeOutput> {
        // Factorized codes (ViT-VQGAN) Project input into low-dimensional space
        let z_e = self.in_proj.forward(xs)?;
        let (z_q, indices) = self.decode_latents(&z_e)?;

        let z_q = self.out_proj.forward(&z_q)?;

        Ok(VectorQuantizeOutput { z_q, indices, z_e })
    }

    fn embed_code(&self, embed_id: &Tensor) -> Result<Tensor> {
        self.codebook.forward(embed_id)
    }

    fn decode_code(&self, embed_id: &Tensor) -> Result<Tensor> {
        self.embed_code(embed_id)?.transpose(1, 2)
    }

    fn decode_latents(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let encodings = xs.reshape(((), xs.dim(1)?))?;
        let codebook = self.codebook.embeddings();

        // torch F.normalize dim default is 1
        let encodings = l2_norm(&encodings, 1)?;
        let codebook = l2_norm(&codebook, 1)?;

        // Compute euclidean distance with codebook

        let dist = (encodings.powf(2.)?.sum_keepdim(1)?
            - 2. * encodings.matmul(&codebook.t()?)?
            + codebook.powf(2.)?.sum_keepdim(1)?.t()?)?;
        let indices = dist.neg()?.argmax(1)?.reshape((xs.dim(0)?, ()))?;

        Ok((self.decode_code(&indices)?, indices))
    }
}

/// Introduced in SoundStream: An end2end neural audio codec
/// https://arxiv.org/abs/2107.03312
struct ResidualVectorQuantize {
    quantizers: Vec<VectorQuantize>,
}

struct ResidualVectorQuantizeOutput {
    z_q: Tensor,
    codes: Tensor,
    latents: Tensor,
}

impl ResidualVectorQuantize {
    fn new(
        input_dim: usize,
        n_codebooks: usize,
        codebook_size: usize,
        codebook_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut quantizers = Vec::with_capacity(n_codebooks);
        let quantizers_vb = vb.pp("quantizers");
        let codebook_dim = vec![codebook_dim; n_codebooks];
        for i in 0..n_codebooks {
            quantizers.push(VectorQuantize::new(
                input_dim,
                codebook_size,
                codebook_dim[i],
                quantizers_vb.pp(i),
            )?)
        }

        Ok(Self { quantizers })
    }

    /// Quantized the input tensor using a fixed set of `n` codebooks and returns the corresponding codebook vectors
    fn forward(&self, z: &Tensor) -> Result<ResidualVectorQuantizeOutput> {
        let mut codebook_indices = Vec::new();
        let mut latents = Vec::new();

        let mut residual = z.clone();
        let mut z_q = None;

        for quantizer in self.quantizers.iter() {
            let VectorQuantizeOutput {
                z_q: z_q_i,
                indices: indices_i,
                z_e: z_e_i,
            } = quantizer.forward(&residual)?;

            // Original python code creates a mask which is all ones here...
            // UNLESS `n_quantizers` is something we can specify, this is ok to leave out

            residual = (residual - &z_q_i)?;
            z_q = if let Some(z_q) = z_q {
                Some((z_q + z_q_i)?)
            } else {
                Some(z_q_i)
            };

            codebook_indices.push(indices_i);
            latents.push(z_e_i);
        }

        let codes = Tensor::stack(&codebook_indices, 1)?;
        let latents = Tensor::cat(&latents, 1)?;

        Ok(ResidualVectorQuantizeOutput {
            z_q: z_q.unwrap(),
            codes,
            latents,
        })
    }

    /// Given the quantized codes, reconstruct the continuous representation.
    ///
    /// Takes quantized discrete representation
    fn from_codes(&self, codes: &Tensor) -> Result<Tensor> {
        let mut z_p = Vec::new();
        let mut z_q = None;
        let n_codebooks = codes.dim(1)?;
        for i in 0..n_codebooks {
            let z_p_i = self.quantizers[i].decode_code(&codes.i((.., i, ..))?)?;
            z_p.push(z_p_i.clone());

            let z_q_i = self.quantizers[i].out_proj.forward(&z_p_i)?;

            z_q = if let Some(z_q) = z_q {
                Some((z_q + z_q_i)?)
            } else {
                Some(z_q_i)
            };
        }

        Ok(z_q.unwrap())
    }
}

// https://github.com/descriptinc/descript-audio-codec/blob/c7cfc5d2647e26471dc394f95846a0830e7bec34/dac/nn/layers.py#L27
struct Snake1d {
    alpha: Tensor,
}

impl Snake1d {
    fn new(channels: usize, vb: &VarBuilder) -> Result<Self> {
        Ok(Self {
            alpha: vb.get((1, channels, 1), "alpha")?,
        })
    }
}

impl Module for Snake1d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let shape = xs.dims();
        let mut xs = xs.reshape((shape[0], shape[1], ()))?;
        xs = (&xs + (&self.alpha + 1e-9)?.recip()?)?
            .broadcast_mul(&self.alpha.broadcast_mul(&xs)?.sin()?)?;
        xs.reshape(shape)
    }
}

// https://github.com/descriptinc/descript-audio-codec/blob/c7cfc5d2647e26471dc394f95846a0830e7bec34/dac/model/dac.py#L24
struct ResidualUnit {
    block: Sequential,
}

impl ResidualUnit {
    fn new(dim: usize, dilation: usize, vb: &VarBuilder) -> Result<Self> {
        let pad = ((7 - 1) * dilation) / 2;
        let mut block = candle_nn::seq();
        block = block.add(Snake1d::new(dim, &vb.pp("block.0"))?);
        block = block.add(wn_conv1d(
            dim,
            dim,
            7,
            Conv1dConfig {
                padding: pad,
                dilation,
                ..Default::default()
            },
            vb.pp("block.1"),
        )?);
        block = block.add(Snake1d::new(dim, &vb.pp("block.2"))?);
        block = block.add(wn_conv1d(
            dim,
            dim,
            1,
            Conv1dConfig {
                padding: pad,
                dilation,
                ..Default::default()
            },
            vb.pp("block.3"),
        )?);
        Ok(Self { block })
    }
}

impl Module for ResidualUnit {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.block.forward(x)?;
        let pad = (x.dim(D::Minus1)? - y.dim(D::Minus1)?) / 2;
        let x = if pad > 0 {
            x.narrow(D::Minus1, pad, x.dim(D::Minus1)? - pad)?
        } else {
            x.clone()
        };
        x + y
    }
}

// From other library
// https://github.com/descriptinc/descript-audio-codec/blob/c7cfc5d2647e26471dc394f95846a0830e7bec34/dac/model/dac.py#L147
struct DAC {
    encoder: Sequential,
    decoder: Sequential,
    quantizer: ResidualVectorQuantize,
    hop_length: usize,
}

struct DACEncoding {
    z: Tensor,
    codes: Tensor,
    latents: Tensor,
}

impl DAC {
    const CODEBOOK_DIM: usize = 8;
    const ENCODER_DIM: usize = 64;
    const ENCODER_RATES: [usize; 4] = [2, 4, 8, 8];
    const DECODER_DIM: usize = 1536;
    const DECODER_RATES: [usize; 4] = [8, 8, 4, 2];
    const SAMPLE_RATE: usize = 44100;

    fn make_encoder(cfg: &ParlerDACConfig, vb: VarBuilder) -> Result<Sequential> {
        let mut encoder = candle_nn::seq();

        let block_vb = vb.pp("block");
        let mut block_i = 0;
        // Add first conv
        encoder = encoder.add(wn_conv1d(
            1,
            Self::ENCODER_DIM,
            7,
            Conv1dConfig {
                padding: 3,
                stride: 1,
                dilation: 1,
                groups: 1,
            },
            block_vb.pp(block_i),
        )?);
        block_i += 1;

        let mut d_model = Self::ENCODER_DIM;
        // Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in Self::ENCODER_RATES {
            d_model *= 2;

            let block_vb = block_vb.pp(block_i);

            // Make an encoder block
            let mut block = candle_nn::seq();
            block = block.add(ResidualUnit::new(d_model / 2, 1, &block_vb.pp(0))?);
            block = block.add(ResidualUnit::new(d_model / 2, 3, &block_vb.pp(1))?);
            block = block.add(ResidualUnit::new(d_model / 2, 9, &block_vb.pp(2))?);
            block = block.add(Snake1d::new(d_model / 2, &block_vb.pp(3))?);
            block = block.add(wn_conv1d(
                d_model / 2,
                d_model,
                2 * stride,
                Conv1dConfig {
                    stride,
                    padding: stride.div_ceil(2),
                    ..Default::default()
                },
                block_vb.pp(4),
            )?);
            block_i += 1;

            encoder = encoder.add(block);
        }

        // Create last conv
        encoder = encoder.add(Snake1d::new(d_model, &block_vb.pp(block_i))?);
        block_i += 1;
        encoder = encoder.add(wn_conv1d(
            d_model,
            cfg.latent_dim,
            3,
            Conv1dConfig {
                padding: 1,
                ..Default::default()
            },
            block_vb.pp(block_i),
        )?);
        // block_i += 1;

        Ok(encoder)
    }

    fn make_decoder(cfg: &ParlerDACConfig, vb: VarBuilder) -> Result<Sequential> {
        let mut decoder = candle_nn::seq();

        let block_vb = vb.pp("model");
        let mut block_i = 0;
        // Add first conv
        decoder = decoder.add(wn_conv1d(
            cfg.latent_dim,
            Self::DECODER_DIM,
            7,
            Conv1dConfig {
                padding: 3,
                stride: 1,
                dilation: 1,
                groups: 1,
            },
            block_vb.pp(block_i),
        )?);
        block_i += 1;

        let mut output_dim = usize::MAX;
        // Add upsampling + MRF blocks
        for (i, stride) in Self::ENCODER_RATES.into_iter().enumerate() {
            let input_dim = Self::DECODER_DIM / 2usize.pow(i as u32);
            output_dim = Self::DECODER_DIM / 2usize.pow(i as u32 + 1);

            let block_vb = block_vb.pp(block_i);

            // Make a decoder block
            let mut block = candle_nn::seq();
            block = block.add(Snake1d::new(input_dim, &block_vb.pp(0))?);
            block = block.add(wn_conv_transpose1d(
                input_dim,
                output_dim,
                2 * stride,
                ConvTranspose1dConfig {
                    stride,
                    padding: stride.div_ceil(2),
                    ..Default::default()
                },
                block_vb.pp(1),
            )?);
            block = block.add(ResidualUnit::new(output_dim, 1, &block_vb.pp(2))?);
            block = block.add(ResidualUnit::new(output_dim, 3, &block_vb.pp(3))?);
            block = block.add(ResidualUnit::new(output_dim, 9, &block_vb.pp(4))?);
            block_i += 1;

            decoder = decoder.add(block);
        }

        // Create last conv
        decoder = decoder.add(Snake1d::new(output_dim, &block_vb.pp(block_i))?);
        block_i += 1;
        decoder = decoder.add(wn_conv1d(
            output_dim,
            1,
            7,
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            block_vb.pp(block_i),
        )?);
        // block_i += 1;
        decoder = decoder.add(|x: &Tensor| x.tanh());

        Ok(decoder)
    }

    fn new(cfg: &ParlerDACConfig, vb: VarBuilder) -> Result<Self> {
        let hop_length = Self::ENCODER_RATES.into_iter().product::<usize>();

        let encoder = Self::make_encoder(cfg, vb.pp("encoder"))?;
        let decoder = Self::make_decoder(cfg, vb.pp("decoder"))?;
        let quantizer = ResidualVectorQuantize::new(
            cfg.latent_dim,
            cfg.num_codebooks,
            cfg.codebook_size,
            Self::CODEBOOK_DIM,
            vb.pp("quantizer"),
        )?;

        Ok(Self {
            encoder,
            decoder,
            quantizer,
            hop_length,
        })
    }

    fn preprocess(&self, audio_data: &Tensor, sample_rate: Option<usize>) -> Result<Tensor> {
        let sample_rate = sample_rate.unwrap_or(Self::SAMPLE_RATE);
        assert_eq!(sample_rate, Self::SAMPLE_RATE);

        let length = audio_data.dim(D::Minus1)?;
        let right_pad = length.div_ceil(self.hop_length) * self.hop_length - length;
        audio_data.pad_with_zeros(D::Minus1, 0, right_pad)
    }

    /// Encode given audio data and return quantized latent codes
    fn encode(&self, audio_data: &Tensor) -> Result<DACEncoding> {
        let z = self.encoder.forward(audio_data)?;
        let ResidualVectorQuantizeOutput {
            z_q: z,
            codes,
            latents,
        } = self.quantizer.forward(&z)?;
        Ok(DACEncoding { z, codes, latents })
    }

    /// Decode given latent codes and return audio data
    fn decode(&self, z: &Tensor) -> Result<Tensor> {
        self.decoder.forward(z)
    }
}

// Model from Parler source (wrapper)
// https://github.com/huggingface/parler-tts/blob/dcaed95e1cce6f616e3e1956f8d63f0f3f5dfe5f/parler_tts/dac_wrapper/modeling_dac.py#L12
struct DACModel {
    model: DAC,
}

impl DACModel {
    fn new(cfg: &ParlerDACConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            model: DAC::new(cfg, vb.pp("model"))?,
        })
    }

    /// Incodes input audio waveform into discrete codes.
    ///
    /// A stack of frames containing the discrete encoded codes for the input audio waveform.
    fn encode(
        &self,
        input_values: &Tensor,
        padding_mask: Option<&Tensor>,
        sample_rate: Option<usize>,
    ) -> Result<Tensor> {
        let (_, channels, input_len) = input_values.dims3()?;

        if ![1, 2].contains(&channels) {
            candle_core::bail!("Number of audio channels must be 1 or 2");
        }

        let audio_data = self.model.preprocess(&input_values, sample_rate)?;

        // TODO: for now, no chunk length. Add it?
        // https://github.com/huggingface/parler-tts/blob/dcaed95e1cce6f616e3e1956f8d63f0f3f5dfe5f/parler_tts/dac_wrapper/modeling_dac.py#L61-L66
        let chunk_length = input_len;
        let stride = input_len;

        let mut encoded_frames = Vec::new();

        let step = chunk_length - stride;
        if (input_len % stride) - step != 0 {
            candle_core::bail!("The input length is not properly padded for batched chunked decoding. Make sure to pad the input correctly.");
        }

        for offset in (0..input_len - step).step_by(stride) {
            let frame = audio_data.narrow(D::Minus1, offset, offset + chunk_length)?;

            let DACEncoding {
                z: _,
                codes: encoded_frame,
                latents: _,
            } = self.model.encode(&frame)?;
            encoded_frames.push(encoded_frame);
        }

        Tensor::stack(&encoded_frames, 0)
    }

    /// Decodes the given frames into an output audio waveform.
    /// Note that the output might be a bit bigger than the input. In that case, any extra steps at the end can be
    /// trimmed.
    fn decode(&self, audio_codes: &Tensor) -> Result<Tensor> {
        let audio_values = self.model.quantizer.from_codes(&audio_codes)?;
        self.model.decode(&audio_values)
    }
}

pub struct ParlerModel {}

impl ParlerModel {
    pub fn new(config: &ParlerConfig, vb: &VarBuilder) -> Result<Self> {
        let text_encoder =
            T5ForConditionalGeneration::load(vb.pp("text_encoder"), &config.text_encoder)?;
        todo!()
    }
}
