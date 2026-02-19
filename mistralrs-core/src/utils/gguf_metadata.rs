use akin::akin;
use anyhow::ensure;
use anyhow::Result;
use candle_core::quantized::gguf_file;
use candle_core::DType;
use std::collections::HashMap;
use std::fs;
use tracing::warn;

use crate::attention::ATTENTION_CHUNK_SIZE;
use crate::gguf::Content;
use crate::matformer::MatformerSliceConfig;
use crate::paged_attention::ModelConfigLike;
use crate::pipeline::AutoDeviceMapParams;
use crate::pipeline::DeviceMappedModelLoader;
use crate::GGUFArchitecture;

#[derive(Debug)]
pub struct ContentConfig {
    max_seq_len: usize,
    hidden_size: usize,
    num_attn_heads: usize,
    num_kv_heads: usize,
    num_layers: usize,
    key_length: Option<usize>,
    value_length: Option<usize>,
}

#[allow(clippy::cast_possible_truncation)]
impl<'a, R: std::io::Seek + std::io::Read> From<&Content<'a, R>> for ContentConfig {
    fn from(value: &Content<'a, R>) -> Self {
        let metadata = value.get_metadata();
        let arch = metadata["general.architecture"].to_string().unwrap();
        Self {
            max_seq_len: metadata[&format!("{arch}.context_length")]
                .to_u64()
                .unwrap() as usize,
            hidden_size: metadata[&format!("{arch}.embedding_length")]
                .to_u64()
                .unwrap() as usize,
            num_attn_heads: metadata[&format!("{arch}.attention.head_count")]
                .to_u64()
                .unwrap() as usize,
            num_kv_heads: metadata[&format!("{arch}.attention.head_count_kv")]
                .to_u64()
                .unwrap() as usize,
            num_layers: metadata[&format!("{arch}.block_count")].to_u64().unwrap() as usize,
            key_length: metadata
                .get(&format!("{arch}.attention.key_length"))
                .map(|x| x.to_u64().unwrap() as usize),
            value_length: metadata
                .get(&format!("{arch}.attention.value_length"))
                .map(|x| x.to_u64().unwrap() as usize),
        }
    }
}

impl ModelConfigLike for ContentConfig {
    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn num_attn_heads(&self) -> usize {
        self.num_attn_heads
    }
    fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }
    fn num_layers(&self) -> usize {
        self.num_layers
    }
    fn k_head_dim(&self) -> usize {
        self.key_length
            .unwrap_or(self.hidden_size / self.num_attn_heads)
    }
    fn v_head_dim(&self) -> usize {
        self.value_length
            .unwrap_or(self.hidden_size / self.num_attn_heads)
    }
}

pub struct ContentMetadata<'a> {
    pub path_prefix: &'a str,
    pub metadata: &'a HashMap<String, gguf_file::Value>,
}

impl ContentMetadata<'_> {
    // Retrieve a prop the struct needs by querying the metadata content:
    pub fn get_value<T: TryFromValue>(&self, field_name: &str) -> Result<T, anyhow::Error> {
        let prop_key = format!("{prefix}.{field_name}", prefix = self.path_prefix);
        let value = self.metadata.get(&prop_key).cloned();

        // Unwrap the inner value of the `Value` enum via trait method,
        // otherwise format error with prop key as context:
        value
            .try_value_into()
            .or_else(|e| anyhow::bail!("`{prop_key}` `{e}`"))
    }

    // Retrieve a prop the struct needs by querying the metadata content:
    pub fn get_option_value<T: TryFromValue>(
        &self,
        field_name: &str,
    ) -> Result<Option<T>, anyhow::Error> {
        let prop_key = format!("{prefix}.{field_name}", prefix = self.path_prefix);
        let value = self.metadata.get(&prop_key).cloned();

        // Unwrap the inner value of the `Value` enum via trait method,
        // otherwise format error with prop key as context:
        value
            .map(|v| {
                v.try_value_into()
                    .or_else(|e| anyhow::bail!("`{prop_key}` `{e}`"))
            })
            .map_or(Ok(None), |res| res.map(Some))
    }

    // Fail early - Catch all missing mandatory keys upfront:
    pub fn has_required_keys(&self, fields: &[&str]) -> Result<()> {
        let mut all_props_are_present = true;

        for field_name in fields {
            let prop_key = format!("{prefix}.{field_name}", prefix = self.path_prefix);

            if !self.metadata.contains_key(&prop_key) {
                all_props_are_present = false;
                warn!("Expected GGUF metadata to have key: `{prop_key}`");
            }
        }

        ensure!(all_props_are_present, "Tokenizer is missing required props");
        Ok(())
    }

    // Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#required
    pub fn verify_arch(&self, expected_arch: &str) -> Result<()> {
        let actual_arch: String = self
            .metadata
            .get("general.architecture")
            .cloned()
            .try_value_into()?;

        anyhow::ensure!(
            actual_arch == expected_arch,
            "Expected `{expected_arch}` architecture, got `{actual_arch}`."
        );

        Ok(())
    }

    pub fn verify_arch_any(&self, expected_arch: &[&str]) -> Result<()> {
        expected_arch
            .iter()
            .try_for_each(|arch| self.verify_arch(arch))
    }
}

// These traits below are a workaround for converting candles GGUF `Value` enum type wrapper.
// A better upstream approach would instead be to provide serialize/deserialize support?
pub trait TryFromValue {
    fn try_from_value(value: gguf_file::Value) -> Result<Self, candle_core::Error>
    where
        Self: Sized;
}

// Value wrapped types, each has a different conversion method:
// NOTE: Type conversion methods internally bail with "not a <into type> <input value>"
// https://docs.rs/candle-core/latest/candle_core/quantized/gguf_file/enum.Value.html#variants
akin! {
    let &types = [String, bool, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64];
    let &to_type = [
        value.to_string().cloned(),
        value.to_bool(),
        value.to_f32(),
        value.to_f64(),
        value.to_i8(),
        value.to_i16(),
        value.to_i32(),
        value.to_i64(),
        value.to_u8(),
        value.to_u16(),
        value.to_u32(),
        value.to_u64(),
    ];

    impl TryFromValue for *types {
        fn try_from_value(value: gguf_file::Value) -> Result<Self, candle_core::Error> {
            *to_type.or_else(|_| candle_core::bail!("value is not a `*types`"))
        }
    }
}

// Vec<Value> to Vec<T> from above types:
impl<T: TryFromValue> TryFromValue for Vec<T> {
    fn try_from_value(value_vec: gguf_file::Value) -> Result<Self, candle_core::Error> {
        value_vec
            .to_vec()
            .or_else(|_| candle_core::bail!("value is not a `Vec`"))?
            .clone()
            .into_iter()
            .map(|item| T::try_from_value(item))
            .collect()
    }
}

pub trait TryValueInto<T>: Sized {
    fn try_value_into(self) -> Result<T, candle_core::Error>;
}

impl<T: TryFromValue> TryValueInto<T> for gguf_file::Value {
    fn try_value_into(self) -> Result<T, candle_core::Error> {
        T::try_from_value(self)
    }
}

impl<T: TryFromValue> TryValueInto<T> for Option<gguf_file::Value> {
    fn try_value_into(self) -> Result<T, candle_core::Error> {
        match self {
            Some(value) => value.try_value_into(),
            None => candle_core::bail!("Expected `Option<gguf_file::Value>` to contain a value"),
        }
    }
}

macro_rules! tensor_info_size_in_bytes {
    ($t:expr) => {
        $t.shape.elem_count() / $t.ggml_dtype.block_size() * $t.ggml_dtype.type_size()
    };
    ($t:expr, $ty:expr) => {
        $t.shape.elem_count() * $ty.size_in_bytes()
    };
}

pub struct GgufDeviceMapLoaderInner<'a, 'f> {
    pub model: &'a Content<'f, fs::File>,
    pub arch: GGUFArchitecture,
}

impl DeviceMappedModelLoader for GgufDeviceMapLoaderInner<'_, '_> {
    fn mapped_max_act_size_elems(
        &self,
        _config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };
        let num_heads = self.model.get_metadata()[&format!("{}.attention.head_count", self.arch)]
            .to_u32()? as usize;
        Ok(max_batch_size * num_heads * max_seq_len.min(&ATTENTION_CHUNK_SIZE))
    }
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }

    fn non_mapped_size_in_bytes(
        &self,
        _config: &str,
        _dtype: DType,
        _weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let size_in_bytes = match self.arch {
            GGUFArchitecture::Llama | GGUFArchitecture::Mistral3 => {
                let token_embd = tensor_info_size_in_bytes!(
                    self.model.tensor_info("token_embd.weight")?,
                    DType::F32
                );
                let output_norm = tensor_info_size_in_bytes!(
                    self.model.tensor_info("output_norm.weight")?,
                    DType::F32
                );
                let output = if !self.model.has_tensor("output.weight") {
                    tensor_info_size_in_bytes!(self.model.tensor_info("token_embd.weight")?)
                } else {
                    tensor_info_size_in_bytes!(self.model.tensor_info("output.weight")?)
                };
                token_embd + output_norm + output
            }
            GGUFArchitecture::Phi2 => {
                let token_embd = tensor_info_size_in_bytes!(
                    self.model.tensor_info("token_embd.weight")?,
                    DType::F32
                );
                let output_norm =
                    tensor_info_size_in_bytes!(
                        self.model.tensor_info("output_norm.weight")?,
                        DType::F32
                    ) + tensor_info_size_in_bytes!(self.model.tensor_info("output_norm.bias")?);
                let output = if !self.model.has_tensor("output.weight") {
                    tensor_info_size_in_bytes!(self.model.tensor_info("token_embd.weight")?)
                } else {
                    tensor_info_size_in_bytes!(self.model.tensor_info("output.weight")?)
                };
                token_embd + output_norm + output
            }
            GGUFArchitecture::Phi3 => {
                let token_embd = tensor_info_size_in_bytes!(
                    self.model.tensor_info("token_embd.weight")?,
                    DType::F32
                );
                let output_norm = tensor_info_size_in_bytes!(
                    self.model.tensor_info("output_norm.weight")?,
                    DType::F32
                );
                let output = if !self.model.has_tensor("output.weight") {
                    tensor_info_size_in_bytes!(self.model.tensor_info("token_embd.weight")?)
                } else {
                    tensor_info_size_in_bytes!(self.model.tensor_info("output.weight")?)
                };
                token_embd + output_norm + output
            }
            GGUFArchitecture::Qwen2 | GGUFArchitecture::Qwen3 | GGUFArchitecture::Qwen3MoE => {
                let token_embd = tensor_info_size_in_bytes!(
                    self.model.tensor_info("token_embd.weight")?,
                    DType::F32
                );
                let output_norm = tensor_info_size_in_bytes!(
                    self.model.tensor_info("output_norm.weight")?,
                    DType::F32
                );
                let output = if !self.model.has_tensor("output.weight") {
                    tensor_info_size_in_bytes!(self.model.tensor_info("token_embd.weight")?)
                } else {
                    tensor_info_size_in_bytes!(self.model.tensor_info("output.weight")?)
                };
                token_embd + output_norm + output
            }
            GGUFArchitecture::Starcoder2 => {
                let token_embd = tensor_info_size_in_bytes!(
                    self.model.tensor_info("token_embd.weight")?,
                    DType::F32
                );
                let output_norm =
                    tensor_info_size_in_bytes!(
                        self.model.tensor_info("output_norm.weight")?,
                        DType::F32
                    ) + tensor_info_size_in_bytes!(self.model.tensor_info("output_norm.bias")?);
                let output = if !self.model.has_tensor("output.weight") {
                    tensor_info_size_in_bytes!(self.model.tensor_info("token_embd.weight")?)
                } else {
                    tensor_info_size_in_bytes!(self.model.tensor_info("output.weight")?)
                };
                token_embd + output_norm + output
            }
            _ => unimplemented!(),
        };
        Ok(size_in_bytes)
    }
    fn num_layers(&self, _config: &str) -> Result<usize> {
        Ok(self.model.get_metadata()[&format!("{}.block_count", self.arch)].to_u32()? as usize)
    }
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        _dtype: DType,
        _weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let size_in_bytes = match self.arch {
            GGUFArchitecture::Llama => {
                let attn_norm = tensor_info_size_in_bytes!(
                    self.model.tensor_info("blk.0.attn_norm.weight")?,
                    DType::F32
                );
                let ffn_norm = tensor_info_size_in_bytes!(
                    self.model.tensor_info("blk.0.ffn_norm.weight")?,
                    DType::F32
                );

                let attn_q =
                    tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.attn_q.weight")?);
                let attn_k =
                    tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.attn_k.weight")?);
                let attn_v =
                    tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.attn_v.weight")?);
                let attn_output = tensor_info_size_in_bytes!(self
                    .model
                    .tensor_info("blk.0.attn_output.weight")?);

                // MoE or Mlp
                #[allow(clippy::cast_possible_truncation)]
                let n_expert = self
                    .model
                    .get_metadata()
                    .get("expert_count")
                    .map(|x| x.to_u64().unwrap() as usize)
                    .unwrap_or(0);
                let moe_or_mlp = if n_expert <= 1 {
                    let ffn_gate = tensor_info_size_in_bytes!(self
                        .model
                        .tensor_info("blk.0.ffn_gate.weight")?);
                    let ffn_up = tensor_info_size_in_bytes!(self
                        .model
                        .tensor_info("blk.0.ffn_up.weight")?);
                    let ffn_down = tensor_info_size_in_bytes!(self
                        .model
                        .tensor_info("blk.0.ffn_down.weight")?);
                    ffn_gate + ffn_up + ffn_down
                } else {
                    let mut moe_count = 0;
                    moe_count += tensor_info_size_in_bytes!(self
                        .model
                        .tensor_info("blk.0.ffn_gate_inp.weight")?);
                    match self.model.tensor_info("blk.0.ffn_gate_exps.weight") {
                        Ok(feed_forward_gate_exps) => {
                            moe_count += tensor_info_size_in_bytes!(feed_forward_gate_exps);
                            moe_count += tensor_info_size_in_bytes!(self
                                .model
                                .tensor_info("blk.0.ffn_down_exps.weight")?);
                            moe_count += tensor_info_size_in_bytes!(self
                                .model
                                .tensor_info("blk.0.ffn_up_exps.weight")?);
                        }
                        Err(_) => {
                            for i in 0..n_expert {
                                moe_count += tensor_info_size_in_bytes!(self
                                    .model
                                    .tensor_info(&format!("blk.0.ffn_gate.{i}.weight"),)?);
                                moe_count += tensor_info_size_in_bytes!(self
                                    .model
                                    .tensor_info(&format!("blk.0.ffn_down.{i}.weight"),)?);
                                moe_count += tensor_info_size_in_bytes!(self
                                    .model
                                    .tensor_info(&format!("blk.0.ffn_up.{i}.weight"))?);
                            }
                        }
                    }

                    moe_count
                };
                attn_norm + ffn_norm + attn_q + attn_k + attn_v + attn_output + moe_or_mlp
            }
            GGUFArchitecture::Phi2 => {
                let attn_norm = tensor_info_size_in_bytes!(
                    self.model.tensor_info("blk.0.attn_norm.weight")?,
                    DType::F32
                ) + tensor_info_size_in_bytes!(self
                    .model
                    .tensor_info("blk.0.attn_norm.bias")?);

                let attn_qkv =
                    tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.attn_qkv.weight")?);
                let attn_output = tensor_info_size_in_bytes!(self
                    .model
                    .tensor_info("blk.0.attn_output.weight")?);

                let ffn_up =
                    tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.ffn_up.weight")?);
                let ffn_down =
                    tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.ffn_down.weight")?);

                attn_norm + attn_qkv + attn_output + ffn_up + ffn_down
            }
            GGUFArchitecture::Phi3 => {
                let attn_norm = tensor_info_size_in_bytes!(
                    self.model.tensor_info("blk.0.attn_norm.weight")?,
                    DType::F32
                );
                let ffn_norm = tensor_info_size_in_bytes!(
                    self.model.tensor_info("blk.0.ffn_norm.weight")?,
                    DType::F32
                );

                let attn_qkv =
                    tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.attn_qkv.weight")?);
                let attn_output = tensor_info_size_in_bytes!(self
                    .model
                    .tensor_info("blk.0.attn_output.weight")?);

                let ffn_up =
                    tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.ffn_up.weight")?);
                let ffn_down =
                    tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.ffn_down.weight")?);

                attn_norm + ffn_norm + attn_qkv + attn_output + ffn_up + ffn_down
            }
            GGUFArchitecture::Qwen2 | GGUFArchitecture::Qwen3 | GGUFArchitecture::Qwen3MoE => {
                let attn_norm = tensor_info_size_in_bytes!(
                    self.model.tensor_info("blk.0.attn_norm.weight")?,
                    DType::F32
                );
                let ffn_norm = tensor_info_size_in_bytes!(
                    self.model.tensor_info("blk.0.ffn_norm.weight")?,
                    DType::F32
                );

                let mut attn_q =
                    tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.attn_q.weight")?);
                if let GGUFArchitecture::Qwen2 = self.arch {
                    attn_q +=
                        tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.attn_q.bias")?);
                }
                let mut attn_k =
                    tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.attn_k.weight")?);
                if let GGUFArchitecture::Qwen2 = self.arch {
                    attn_k +=
                        tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.attn_k.bias")?);
                }

                let mut attn_v =
                    tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.attn_v.weight")?);
                if let GGUFArchitecture::Qwen2 = self.arch {
                    attn_v +=
                        tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.attn_v.bias")?);
                }

                let attn_output = tensor_info_size_in_bytes!(self
                    .model
                    .tensor_info("blk.0.attn_output.weight")?);

                let ffn_gate = if let GGUFArchitecture::Qwen3MoE = self.arch {
                    tensor_info_size_in_bytes!(self
                        .model
                        .tensor_info("blk.0.ffn_gate_exps.weight")?)
                } else {
                    tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.ffn_gate.weight")?)
                };

                let ffn_up = if let GGUFArchitecture::Qwen3MoE = self.arch {
                    tensor_info_size_in_bytes!(self
                        .model
                        .tensor_info("blk.0.ffn_up_exps.weight")?)
                } else {
                    tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.ffn_up.weight")?)
                };

                let ffn_down = if let GGUFArchitecture::Qwen3MoE = self.arch {
                    tensor_info_size_in_bytes!(self
                        .model
                        .tensor_info("blk.0.ffn_down_exps.weight")?)
                } else {
                    tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.ffn_down.weight")?)
                };

                attn_norm
                    + ffn_norm
                    + attn_q
                    + attn_k
                    + attn_v
                    + attn_output
                    + ffn_gate
                    + ffn_up
                    + ffn_down
            }
            GGUFArchitecture::Starcoder2 => {
                let attn_norm = tensor_info_size_in_bytes!(
                    self.model.tensor_info("blk.0.attn_norm.weight")?,
                    DType::F32
                ) + tensor_info_size_in_bytes!(self
                    .model
                    .tensor_info("blk.0.attn_norm.bias")?);
                let ffn_norm = tensor_info_size_in_bytes!(
                    self.model.tensor_info("blk.0.ffn_norm.weight")?,
                    DType::F32
                ) + tensor_info_size_in_bytes!(self
                    .model
                    .tensor_info("blk.0.ffn_norm.bias")?);

                let attn_q = tensor_info_size_in_bytes!(self
                    .model
                    .tensor_info("blk.0.attn_q.weight")?)
                    + tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.attn_q.bias")?);
                let attn_k = tensor_info_size_in_bytes!(self
                    .model
                    .tensor_info("blk.0.attn_k.weight")?)
                    + tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.attn_k.bias")?);
                let attn_v = tensor_info_size_in_bytes!(self
                    .model
                    .tensor_info("blk.0.attn_v.weight")?)
                    + tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.attn_v.bias")?);
                let attn_output = tensor_info_size_in_bytes!(self
                    .model
                    .tensor_info("blk.0.attn_output.weight")?)
                    + tensor_info_size_in_bytes!(self
                        .model
                        .tensor_info("blk.0.attn_output.bias")?);

                let ffn_up = tensor_info_size_in_bytes!(self
                    .model
                    .tensor_info("blk.0.ffn_up.weight")?)
                    + tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.ffn_up.bias")?);
                let ffn_down = tensor_info_size_in_bytes!(self
                    .model
                    .tensor_info("blk.0.ffn_down.weight")?)
                    + tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.ffn_down.bias")?);

                attn_norm + ffn_norm + attn_q + attn_k + attn_v + attn_output + ffn_up + ffn_down
            }
            GGUFArchitecture::Mistral3 => {
                let attn_norm = tensor_info_size_in_bytes!(
                    self.model.tensor_info("blk.0.attn_norm.weight")?,
                    DType::F32
                );

                let attn_q =
                    tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.attn_q.weight")?);
                let attn_k =
                    tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.attn_k.weight")?);
                let attn_v =
                    tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.attn_v.weight")?);

                let attn_output = tensor_info_size_in_bytes!(self
                    .model
                    .tensor_info("blk.0.attn_output.weight")?);

                let ffn_norm = tensor_info_size_in_bytes!(
                    self.model.tensor_info("blk.0.ffn_norm.weight")?,
                    DType::F32
                );
                let ffn_up =
                    tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.ffn_up.weight")?);
                let ffn_down =
                    tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.ffn_down.weight")?);
                let ffn_gate =
                    tensor_info_size_in_bytes!(self.model.tensor_info("blk.0.ffn_gate.weight")?);

                attn_norm
                    + attn_q
                    + attn_k
                    + attn_v
                    + attn_output
                    + ffn_norm
                    + ffn_up
                    + ffn_down
                    + ffn_gate
            }

            _ => unimplemented!(),
        };
        Ok(vec![size_in_bytes; self.num_layers(config)?])
    }
    fn model_config(&self, _config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let model_config_metadata: ContentConfig = self.model.into();
        Ok(Box::new(model_config_metadata))
    }
}
