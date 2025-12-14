use std::{collections::HashMap, fs};

use anyhow::Context;
use candle_core::{
    quantized::{
        gguf_file::{self, TensorInfo, Value},
        GgmlDType, QTensor,
    },
    Device, Result,
};
use indexmap::IndexMap;
use tracing::info;

use crate::DEBUG;

use super::GGUFArchitecture;

/// List of all GgmlDType variants from Candle.
/// This should be kept in sync with candle_core::quantized::GgmlDType.
/// If Candle adds new dtype variants, add them here to include in error messages.
/// Reference: candle-core/src/quantized/mod.rs in the Candle repository.
const KNOWN_DTYPES: &[GgmlDType] = &[
    GgmlDType::F32,
    GgmlDType::F16,
    GgmlDType::BF16,
    GgmlDType::Q4_0,
    GgmlDType::Q4_1,
    GgmlDType::Q5_0,
    GgmlDType::Q5_1,
    GgmlDType::Q8_0,
    GgmlDType::Q8_1,
    GgmlDType::Q2K,
    GgmlDType::Q3K,
    GgmlDType::Q4K,
    GgmlDType::Q5K,
    GgmlDType::Q6K,
    GgmlDType::Q8K,
    // Add newer ones here if Candle adds more
];

fn get_supported_gguf_dtypes() -> String {
    KNOWN_DTYPES
        .iter()
        .map(|dt| format!("{:?}", dt))
        .collect::<Vec<_>>()
        .join(", ")
}

fn parse_gguf_value(value: &Value) -> String {
    match value {
        Value::Array(vs) => vs
            .iter()
            .map(parse_gguf_value)
            .collect::<Vec<String>>()
            .join(", "),
        Value::Bool(b) => b.to_string(),
        Value::F32(x) => x.to_string(),
        Value::F64(x) => x.to_string(),
        Value::I8(x) => x.to_string(),
        Value::I16(x) => x.to_string(),
        Value::I32(x) => x.to_string(),
        Value::I64(x) => x.to_string(),
        Value::String(x) => x.to_string(),
        Value::U8(x) => x.to_string(),
        Value::U16(x) => x.to_string(),
        Value::U32(x) => x.to_string(),
        Value::U64(x) => x.to_string(),
    }
}

// Internal invariant: contents and readers must be paired.
/// This abstracts the files for a GGUF model and enables multiple files to be used.
pub struct Content<'a, R: std::io::Seek + std::io::Read> {
    contents: Vec<gguf_file::Content>,
    readers: &'a mut [&'a mut R],
    arch: GGUFArchitecture,
    all_metadata: HashMap<String, Value>,
}

impl<'a, R: std::io::Seek + std::io::Read> Content<'a, R> {
    /// Create a `Content` from a set of file readers.
    pub fn from_readers(readers: &'a mut [&'a mut R]) -> Result<Self> {
        let mut contents = Vec::new();
        let n_readers = readers.len();
        for (i, reader) in readers.iter_mut().enumerate() {
            match gguf_file::Content::read(reader) {
                Ok(c) => {
                    contents.push(c);
                }
                Err(e) => {
                    let error_msg = format!("{}", e);
                    if error_msg.contains("unknown dtype for tensor") {
                        {
                            candle_core::bail!(
                                "Critical failure loading model part {}\n\
                                Verify you are using a supported quantization type\n\
                                Supported types: {}\n\
                                Candle error: {}",
                                i,
                                get_supported_gguf_dtypes(),
                                e
                            );
                        }
                    }
                    candle_core::bail!(
                        "Critical failure loading model part {}!\n\
                        Check whether your current quantization format is supported: {}",
                        i,
                        e
                    );
                }
            }
        }
        let n_splits = contents
            .iter()
            .filter_map(|ct| {
                ct.metadata
                    .get("split.count")
                    .map(|val| val.to_u64().unwrap())
            })
            .fold(Vec::new(), |mut accum, x| {
                if !accum.contains(&x) {
                    accum.push(x);
                }
                accum
            });
        if n_splits.len() > 1 {
            candle_core::bail!("GGUF files have differing `split.count` values: {n_splits:?}. Perhaps the GGUF files do not match?");
        }
        #[allow(clippy::cast_possible_truncation)]
        if !n_splits.is_empty() && n_readers != n_splits[0] as usize {
            candle_core::bail!(
                "Number of GGUF files does not match the number of splits, expected {} files.",
                n_splits[0]
            );
        } else if n_splits.len() == 1 {
            info!("GGUF file has been split into {} shards", n_splits[0]);
        }

        let mut arch = None;
        for ct in &contents {
            if !ct.metadata.contains_key("general.architecture") {
                continue;
            }

            arch = Some(
                ct.metadata["general.architecture"]
                    .to_string()
                    .context("Model metadata should have declared an architecture")
                    .and_then(GGUFArchitecture::from_value)
                    .unwrap(),
            );
        }
        let arch = arch.expect("GGUF files must specify `general.architecture`");

        let mut all_metadata = HashMap::new();
        for content in &contents {
            all_metadata.extend(content.metadata.clone())
        }

        Ok(Self {
            contents,
            readers,
            arch,
            all_metadata,
        })
    }

    pub fn arch(&self) -> GGUFArchitecture {
        self.arch
    }

    /// Retrieve a tensor info, searching through each content.
    pub fn tensor_info(&self, name: &str) -> Result<&TensorInfo> {
        for ct in &self.contents {
            if let Some(tensor_info) = ct.tensor_infos.get(name) {
                return Ok(tensor_info);
            }
        }
        candle_core::bail!("Cannot find tensor info for {name}")
    }

    /// Retrieve a tensor, searching through each content.
    pub fn tensor(&mut self, name: &str, device: &Device) -> Result<QTensor> {
        for (ct, reader) in self.contents.iter().zip(self.readers.iter_mut()) {
            if let Some(tensor_info) = ct.tensor_infos.get(name) {
                return tensor_info.read(reader, ct.tensor_data_offset, device);
            }
        }
        candle_core::bail!("Cannot find tensor info for {name}")
    }

    /// Check for a tensor, searching through each content.
    pub fn has_tensor(&self, name: &str) -> bool {
        for ct in self.contents.iter() {
            if ct.tensor_infos.contains_key(name) {
                return true;
            }
        }
        false
    }

    /// Print metadata for these contents.
    /// This will also log tensor name, shape and dtype to `mistralrs_gguf_tensors.txt` is DEBUG is enabled.
    pub fn print_metadata(&self) -> anyhow::Result<()> {
        // Find the ct with general.architecture
        let mut keys = Vec::new();
        let mut metadatas = Vec::new();
        let mut tensors = Vec::new();
        for ct in &self.contents {
            keys.extend(ct.metadata.keys());
            metadatas.push(&ct.metadata);

            if DEBUG.load(std::sync::atomic::Ordering::Relaxed) {
                for (name, info) in &ct.tensor_infos {
                    tensors.push(format!(
                        "name = `{name}`, shape = {:?}, dtype = {:?}",
                        info.shape.clone(),
                        info.ggml_dtype
                    ));
                }
            }
        }

        info!("Model config:");
        keys.sort();
        let mut output_keys = IndexMap::new();
        for name in keys {
            if !name.contains("tokenizer") {
                for metadata in &metadatas {
                    if let Some(val) = metadata.get(name) {
                        output_keys.insert(name, parse_gguf_value(val));
                    }
                }
            }
        }
        for (name, val) in output_keys {
            println!("{name}: {val}")
        }

        if DEBUG.load(std::sync::atomic::Ordering::Relaxed) {
            fs::write(
                "mistralrs_gguf_tensors.txt",
                serde_json::to_string_pretty(&tensors).expect("Serialization failed."),
            )?;

            info!("Debug is enabled, wrote the names and information about each tensor to `mistralrs_gguf_tensors.txt`.");
        }

        anyhow::Ok(())
    }

    /// Get all metadatas
    pub fn get_metadata(&self) -> &HashMap<String, Value> {
        &self.all_metadata
    }
}
