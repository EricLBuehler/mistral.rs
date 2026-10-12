use std::{fs::File, io::BufReader, path::Path};

use anyhow::{Context, Result};
use candle_core::{
    pickle::{Object, PthTensors, Stack},
    DType, Device, Tensor, D,
};
use candle_nn::{Activation, LayerNorm, Linear, Module};
use serde::Deserialize;
use serde_json::Value;

const LAYER_NORM_EPS: f64 = 1e-5;
const NORMALIZE_EPS: f64 = 1e-12;
const MAX_LOGIT_SCALE: f64 = 100.0;
const DEFAULT_PROJECTION_DIM: usize = 512;

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct ClmConfig {
    pub base_model: String,
    pub encoder_pooling: String,
    pub embedding_dim: usize,
    pub checkpoints: Vec<String>,
}

impl ClmConfig {
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.encoder_pooling == "last-token",
            "CLM requires last-token encoder pooling"
        );
        anyhow::ensure!(
            self.checkpoints.len() == 1,
            "CLM requires exactly one projection checkpoint"
        );
        anyhow::ensure!(self.embedding_dim > 0, "CLM embedding_dim must be positive");
        Ok(())
    }
}

#[derive(Deserialize)]
struct HeadConfig {
    width: usize,
    depth: usize,
    hidden_size: usize,
    projection_dim: usize,
    #[serde(default = "default_activation")]
    activation: String,
    #[serde(default)]
    layernorm: bool,
    #[serde(default)]
    residual: bool,
    model: String,
}

fn default_activation() -> String {
    "gelu".to_string()
}

struct Head {
    input: Linear,
    hidden: Vec<(Linear, Option<LayerNorm>)>,
    output: Linear,
    activation: Activation,
    residual: bool,
}

impl Head {
    fn load(path: &Path, key: &str, cfg: &HeadConfig, device: &Device) -> Result<Self> {
        let tensors = PthTensors::new(path, Some(key))?;
        let get = |name: &str| -> Result<Tensor> {
            Ok(tensors
                .get(name)?
                .with_context(|| format!("missing {key}.{name}"))?
                .to_dtype(DType::F32)?
                .to_device(device)?)
        };
        let linear = |name: &str, input: usize, output: usize| -> Result<Linear> {
            let weight = get(&format!("{name}.weight"))?;
            let bias = get(&format!("{name}.bias"))?;
            anyhow::ensure!(
                weight.dims() == [output, input] && bias.dims() == [output],
                "invalid shape for {key}.{name}"
            );
            Ok(Linear::new(weight, Some(bias)))
        };
        let mut hidden = Vec::new();
        for i in 0..cfg.depth - 2 {
            let layer = linear(&format!("hidden.{i}"), cfg.width, cfg.width)?;
            let norm = if cfg.layernorm {
                let weight = get(&format!("norms.{i}.weight"))?;
                let bias = get(&format!("norms.{i}.bias"))?;
                anyhow::ensure!(
                    weight.dims() == [cfg.width] && bias.dims() == [cfg.width],
                    "invalid CLM layer norm shape"
                );
                Some(LayerNorm::new(weight, bias, LAYER_NORM_EPS))
            } else {
                None
            };
            hidden.push((layer, norm));
        }
        let activation = match cfg.activation.as_str() {
            "gelu" => Activation::Gelu,
            "relu" => Activation::Relu,
            "silu" => Activation::Silu,
            other => anyhow::bail!("unsupported CLM activation {other}"),
        };
        Ok(Self {
            input: linear("inp", cfg.hidden_size, cfg.width)?,
            hidden,
            output: linear("out", cfg.width, cfg.projection_dim)?,
            activation,
            residual: cfg.residual,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let mut xs = self.activation.forward(&self.input.forward(xs)?)?;
        for (linear, norm) in &self.hidden {
            let mut h = linear.forward(&xs)?;
            if let Some(norm) = norm {
                h = norm.forward(&h)?;
            }
            h = self.activation.forward(&h)?;
            xs = if self.residual { (xs + h)? } else { h };
        }
        let xs = self.output.forward(&xs)?;
        let norm = xs
            .sqr()?
            .sum_keepdim(D::Minus1)?
            .sqrt()?
            .clamp(NORMALIZE_EPS, f64::INFINITY)?;
        xs.broadcast_div(&norm)
    }
}

pub(crate) struct ClmHeads {
    state: Head,
    action: Head,
    scale: f64,
}

impl ClmHeads {
    pub fn projection_dim(&self) -> usize {
        self.state.output.weight().dims()[0]
    }

    pub fn load(path: &Path, config: &ClmConfig, device: &Device) -> Result<Self> {
        let mut zip = zip::ZipArchive::new(BufReader::new(File::open(path)?))?;
        let name = zip
            .file_names()
            .find(|name| name.ends_with("/data.pkl"))
            .context("CLM checkpoint has no data.pkl")?
            .to_string();
        let mut stack = Stack::empty();
        stack.read_loop(&mut BufReader::new(zip.by_name(&name)?))?;
        let root = stack.finalize()?;
        let mut cfg = pickle_value(field(&root, "cfg").context("CLM checkpoint has no cfg")?)?;
        cfg["projection_dim"] = field(&root, "projection_dim")
            .map(pickle_value)
            .transpose()?
            .or_else(|| cfg.get("projection_dim").cloned())
            .unwrap_or(Value::from(DEFAULT_PROJECTION_DIM));
        let cfg: HeadConfig = serde_json::from_value(cfg)?;
        anyhow::ensure!(
            cfg.model == config.base_model,
            "CLM heads are locked to encoder {}, not {}",
            cfg.model,
            config.base_model
        );
        anyhow::ensure!(
            cfg.hidden_size == config.embedding_dim,
            "CLM encoder width does not match the heads"
        );
        anyhow::ensure!(
            cfg.depth >= 2 && cfg.width > 0 && cfg.projection_dim > 0,
            "invalid CLM head dimensions"
        );
        let scale = match field(&root, "logit_scale") {
            Some(Object::Float(value)) => *value,
            _ => PthTensors::new(path, None)?
                .get("logit_scale")?
                .context("CLM checkpoint has no logit_scale")?
                .to_dtype(DType::F32)?
                .to_scalar::<f32>()? as f64,
        };
        anyhow::ensure!(scale.is_finite(), "invalid CLM logit_scale");
        Ok(Self {
            state: Head::load(path, "state_head", &cfg, device)?,
            action: Head::load(path, "action_head", &cfg, device)?,
            scale: scale.exp().min(MAX_LOGIT_SCALE),
        })
    }

    pub fn project(&self, embeddings: &Tensor, state: bool) -> candle_core::Result<Tensor> {
        let xs = embeddings.to_dtype(DType::F32)?;
        let norm = (xs.sqr()?.sum_keepdim(D::Minus1)?.sqrt()? + NORMALIZE_EPS)?;
        let xs = xs.broadcast_div(&norm)?;
        if state {
            self.state.forward(&xs)
        } else {
            self.action.forward(&xs)
        }
    }

    pub fn score(
        &self,
        state: &Tensor,
        actions: &Tensor,
        temperature: f64,
    ) -> candle_core::Result<Tensor> {
        (actions.matmul(&state.t()?)? * (self.scale / temperature))?
            .flatten_all()?
            .contiguous()
    }
}

fn field<'a>(object: &'a Object, key: &str) -> Option<&'a Object> {
    match object {
        Object::Dict(fields) => fields
            .iter()
            .find_map(|(k, v)| (k == &Object::Unicode(key.to_string())).then_some(v)),
        _ => None,
    }
}

fn pickle_value(object: &Object) -> Result<Value> {
    Ok(match object {
        Object::Int(x) => Value::from(*x),
        Object::Long(x) => Value::from(*x),
        Object::Float(x) => Value::from(*x),
        Object::Bool(x) => Value::from(*x),
        Object::Unicode(x) => Value::from(x.clone()),
        Object::None => Value::Null,
        Object::Dict(fields) => {
            let mut map = serde_json::Map::new();
            for (key, value) in fields {
                if let Object::Unicode(key) = key {
                    map.insert(key.clone(), pickle_value(value)?);
                }
            }
            Value::Object(map)
        }
        _ => anyhow::bail!("unsupported CLM checkpoint configuration value"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires CLM_CHECKPOINT pointing to the published CLM_v0.1-8B.pt"]
    fn published_clm_heads_match_pytorch() -> Result<()> {
        let checkpoint = std::env::var("CLM_CHECKPOINT")?;
        let config = ClmConfig {
            base_model: "Qwen/Qwen3-8B".to_string(),
            encoder_pooling: "last-token".to_string(),
            embedding_dim: 4096,
            checkpoints: vec![checkpoint.clone()],
        };
        let heads = ClmHeads::load(Path::new(&checkpoint), &config, &Device::Cpu)?;
        let values: Vec<f32> = (0_u16..4 * 4096)
            .map(|i| (f32::from(i % 97) - 48.0) / 48.0)
            .collect();
        let xs = Tensor::from_vec(values, (4, 4096), &Device::Cpu)?;
        let state = heads.project(&xs.narrow(0, 0, 1)?, true)?;
        let actions = heads.project(&xs.narrow(0, 1, 3)?, false)?;
        let actual = heads.score(&state, &actions, 1.0)?.to_vec1::<f32>()?;
        let expected: Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/clm/published-head-reference.json"
        ))?;
        for (actual, expected) in actual.iter().zip(expected["logits"].as_array().unwrap()) {
            assert!(
                (*actual as f64 - expected.as_f64().unwrap()).abs() < 2e-5,
                "{actual} != {expected}"
            );
        }
        Ok(())
    }
}
