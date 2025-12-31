use mistralrs_quant::QuantizedConfig;
use serde::{Deserialize, Serialize};

use crate::{layers::Activation, serde_default_fn};

serde_default_fn!(bool, tie_word_embeddings, false);

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Config {
    pub(crate) attn_type_list: Vec<usize>,
    pub(crate) head_dim: Option<usize>,
    pub(crate) hidden_act: Activation,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) max_position_embeddings: usize,
    pub(crate) mtp_transformer_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_experts_per_tok: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) num_local_experts: usize,
    pub(crate) num_mtp_modules: usize,
    pub(crate) qk_norm_type: String,
    pub(crate) quantization_config: Option<QuantizedConfig>,
    pub(crate) rms_norm_eps: f64,
    pub(crate) rope_theta: f64,
    pub(crate) rotary_dim: usize,
    pub(crate) scoring_func: String,
    pub(crate) shared_intermediate_size: usize,
    #[serde(default = "tie_word_embeddings")]
    pub(crate) tie_word_embeddings: bool,
    pub(crate) use_qk_norm: bool,
    pub(crate) use_routing_bias: bool,
    pub(crate) vocab_size: usize,
}

impl Config {
    pub(crate) fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs::File,
        io::{Read, Write},
        path::PathBuf,
    };

    use super::*;

    fn get_file(file: &str) -> (File, PathBuf) {
        let temp_dir = std::env::temp_dir().join("minimax-m2");
        std::fs::create_dir_all(&temp_dir);
        let target = format!(
            "https://huggingface.co/MiniMaxAI/MiniMax-M2/resolve/main/{}",
            file
        );
        let response = reqwest::blocking::get(target).expect("Could not download minimax config");
        let file_path = temp_dir.join(file);

        let content = response.bytes().expect("Could not read response ");
        let mut file = File::create(&file_path).expect("Could not create target file");
        file.write_all(&content)
            .expect("Could not write file contents");
        file.flush().expect("Could not flush file contents");
        (file, file_path)
    }
}
