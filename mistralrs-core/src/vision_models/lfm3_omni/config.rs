use serde::{Deserialize, Serialize};

use super::{audio::AudioEncoderConfig, memory::ActivationMemory};

const DEPTH_HIDDEN_BUFFERS: usize = 16;
const DEPTH_INTERMEDIATE_BUFFERS: usize = 3;
const ATTENTION_FLOAT32_BUFFERS: usize = 2;
const GENERATION_HISTORY_COPIES: usize = 2;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct Config {
    #[serde(flatten)]
    pub vision: crate::vision_models::lfm3_vl::Config,
    #[serde(flatten)]
    pub speech: SpeechConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct SpeechConfig {
    pub audio_encoder_config: AudioEncoderConfig,
    pub depth_config: super::generation::DepthConfig,
    pub codec_config: super::codec::CodecConfig,
    #[serde(default)]
    pub audio_sampling: Option<super::generation::AudioSampling>,
    pub num_codebooks: usize,
    pub codebook_size: usize,
    pub audio_eos_code: u32,
    pub maskgit_schedule: Vec<usize>,
    pub interleaved_n_text: usize,
    pub interleaved_n_audio: usize,
    pub text_end_token_id: u32,
    pub audio_token_id: u32,
    pub audio_out_token_id: u32,
    pub audio_start_token_id: u32,
    pub mixed_start_token_id: u32,
}

impl Config {
    pub(crate) fn output_activation_memory(
        &self,
        frames: usize,
        batch_size: usize,
        dtype: candle_core::DType,
    ) -> ActivationMemory {
        self.speech.output_activation_memory(
            frames,
            batch_size,
            self.vision.text_config.hidden_size,
            dtype,
        )
    }
}

impl SpeechConfig {
    pub(crate) fn weight_size_in_bytes(
        &self,
        text_hidden: usize,
        dtype: candle_core::DType,
    ) -> usize {
        let input = self.audio_encoder_config.hidden_size;
        let projector =
            2 * input + input * text_hidden + text_hidden * text_hidden + 2 * text_hidden;
        let parameters = self.audio_encoder_config.parameter_count()
            + projector
            + self.output_parameter_count(text_hidden);
        let tensor_bytes = dtype.size_in_bytes();
        // The codec's synthesis window remains float32 when model weights use a smaller dtype.
        parameters * tensor_bytes
            + self.codec_config.stft_win_length
                * candle_core::DType::F32
                    .size_in_bytes()
                    .saturating_sub(tensor_bytes)
    }

    pub(crate) fn output_activation_memory(
        &self,
        frames: usize,
        batch_size: usize,
        text_hidden_size: usize,
        dtype: candle_core::DType,
    ) -> ActivationMemory {
        let depth = &self.depth_config;
        let codebooks = self.num_codebooks;
        let tensor_bytes = dtype.size_in_bytes();
        let float_bytes = candle_core::DType::F32.size_in_bytes();
        let depth_bytes = codebooks
            * ((DEPTH_HIDDEN_BUFFERS * depth.hidden_size
                + DEPTH_INTERMEDIATE_BUFFERS * depth.intermediate_size)
                * tensor_bytes
                + self.codebook_size * (tensor_bytes + float_bytes))
            + codebooks
                * codebooks
                * depth.num_attention_heads
                * (ATTENTION_FLOAT32_BUFFERS * float_bytes + tensor_bytes);
        let history = GENERATION_HISTORY_COPIES
            * frames
            * codebooks
            * candle_core::DType::U32.size_in_bytes();
        ActivationMemory {
            retained_bytes: batch_size * (history + text_hidden_size * tensor_bytes),
            transient_bytes: depth_bytes
                .max(self.codec_config.activation_memory_bytes(frames, dtype)),
        }
    }

    fn output_parameter_count(&self, text_hidden: usize) -> usize {
        self.depth_config
            .depth_parameter_count(text_hidden, self.num_codebooks, self.codebook_size)
            + self.num_codebooks * self.codebook_size * text_hidden
            + self.codec_config.parameter_count()
    }
}
