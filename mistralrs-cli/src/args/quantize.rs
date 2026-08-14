//! Quantize command argument structs for UQFF generation

use clap::{Args, Subcommand, ValueEnum};
use mistralrs_core::{AutoDeviceMapParams, IsqOrganization, ModelDType, NormalLoaderType};
use std::path::PathBuf;

use super::{FormatOptions, ModelFormat};

/// Quantize model type selection (base models only, no adapter support)
#[derive(Subcommand, Clone)]
pub enum QuantizeModelType {
    /// Auto-detect model type (recommended)
    Auto {
        #[command(flatten)]
        model: QuantizeModelSourceOptions,

        #[command(flatten)]
        quantization: QuantizeQuantizationOptions,

        #[command(flatten)]
        device: QuantizeDeviceOptions,

        #[command(flatten)]
        output: QuantizeOutputOptions,

        #[command(flatten)]
        multimodal: QuantizeMultimodalOptions,
    },

    /// Text generation model with explicit architecture
    Text {
        #[command(flatten)]
        model: QuantizeModelSourceOptions,

        /// Model architecture (required for text models)
        #[arg(short = 'a', long, value_parser = parse_arch)]
        arch: Option<NormalLoaderType>,

        #[command(flatten)]
        quantization: QuantizeQuantizationOptions,

        #[command(flatten)]
        device: QuantizeDeviceOptions,

        #[command(flatten)]
        output: QuantizeOutputOptions,
    },

    /// Multimodal model
    Multimodal {
        #[command(flatten)]
        model: QuantizeModelSourceOptions,

        #[command(flatten)]
        quantization: QuantizeQuantizationOptions,

        #[command(flatten)]
        device: QuantizeDeviceOptions,

        #[command(flatten)]
        output: QuantizeOutputOptions,

        #[command(flatten)]
        multimodal: QuantizeMultimodalOptions,
    },

    /// Embedding model
    Embedding {
        #[command(flatten)]
        model: QuantizePlainModelSourceOptions,

        #[command(flatten)]
        quantization: QuantizeQuantizationOptions,

        #[command(flatten)]
        device: QuantizeDeviceOptions,

        #[command(flatten)]
        output: QuantizeOutputOptions,
    },
}

/// Model source options for quantization
#[derive(Args, Clone)]
pub struct QuantizeModelSourceOptions {
    /// Hugging Face model ID or local model directory; optional when `-f` names local files
    #[arg(short = 'm', long)]
    pub model_id: Option<String>,

    /// Path to local tokenizer.json file
    #[arg(short = 't', long)]
    pub tokenizer: Option<PathBuf>,

    /// Model data type
    #[arg(long, default_value = "auto", value_parser = parse_dtype)]
    pub dtype: ModelDType,

    #[command(flatten)]
    pub format: QuantizeFormatOptions,

    /// Select an input GGUF artifact by bit width or quant name
    #[arg(long, conflicts_with = "quantized_file")]
    pub quant: Option<String>,
}

#[derive(Args, Clone)]
pub struct QuantizePlainModelSourceOptions {
    /// Hugging Face model ID or local model directory
    #[arg(short = 'm', long)]
    pub model_id: String,

    /// Path to local tokenizer.json file
    #[arg(short = 't', long)]
    pub tokenizer: Option<PathBuf>,

    /// Model data type
    #[arg(long, default_value = "auto", value_parser = parse_dtype)]
    pub dtype: ModelDType,
}

#[derive(Args, Clone, Default)]
pub struct QuantizeFormatOptions {
    /// Input model format: plain (safetensors) or GGUF. Auto-detected from `-f` when omitted.
    #[arg(long, value_enum)]
    pub format: Option<QuantizeModelFormat>,

    /// GGUF filename(s), separated by semicolons for multiple files
    #[arg(short = 'f', long)]
    pub quantized_file: Option<String>,

    /// GGUF projector override; auto-selected when unambiguous (semicolon-separated for multiple)
    #[arg(long)]
    pub mmproj: Option<String>,

    /// Optional model ID overriding configuration, tokenizer, and processor assets for a GGUF model
    #[arg(long)]
    pub tok_model_id: Option<String>,

    #[doc(hidden)]
    #[arg(skip)]
    pub direct_file_only: bool,
}

impl QuantizeFormatOptions {
    fn normalize(&mut self) -> anyhow::Result<()> {
        let mut format = self.to_format_options();
        format.normalize()?;
        self.update_from_format_options(format)
    }

    fn derive_local_model_root(&mut self) -> anyhow::Result<String> {
        let mut format = self.to_format_options();
        let root = format.derive_local_model_root()?;
        self.update_from_format_options(format)?;
        Ok(root)
    }

    fn to_format_options(&self) -> FormatOptions {
        FormatOptions {
            format: self.format.map(Into::into),
            quantized_file: self.quantized_file.clone(),
            mmproj: self.mmproj.clone(),
            tok_model_id: self.tok_model_id.clone(),
            gqa: 1,
            direct_file_only: self.direct_file_only,
        }
    }

    fn update_from_format_options(&mut self, format: FormatOptions) -> anyhow::Result<()> {
        self.format = match format.format {
            None => None,
            Some(ModelFormat::Plain) => Some(QuantizeModelFormat::Plain),
            Some(ModelFormat::Gguf) => Some(QuantizeModelFormat::Gguf),
            Some(ModelFormat::Ggml) => {
                anyhow::bail!(
                    "GGML inputs cannot be requantized; use a compatible GGUF or safetensors source"
                )
            }
        };
        self.quantized_file = format.quantized_file;
        self.mmproj = format.mmproj;
        self.tok_model_id = format.tok_model_id;
        self.direct_file_only = format.direct_file_only;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum QuantizeModelFormat {
    #[default]
    Plain,
    Gguf,
}

impl From<QuantizeModelFormat> for ModelFormat {
    fn from(format: QuantizeModelFormat) -> Self {
        match format {
            QuantizeModelFormat::Plain => Self::Plain,
            QuantizeModelFormat::Gguf => Self::Gguf,
        }
    }
}

/// Quantization options for UQFF generation (ISQ-related only, no from_uqff)
#[derive(Args, Clone)]
pub struct QuantizeQuantizationOptions {
    /// Output UQFF quantization type(s). Multiple values can be comma-separated or specified
    /// via repeated --isq flags (e.g., "--isq q4k,q8_0" or "--isq q4k --isq q8_0").
    #[arg(long = "isq", required = true, value_delimiter = ',')]
    pub in_situ_quant: Vec<String>,

    /// ISQ organization strategy: default or moqe
    #[arg(long)]
    pub isq_organization: Option<IsqOrganization>,

    /// imatrix file for enhanced quantization
    #[arg(long)]
    pub imatrix: Option<PathBuf>,

    /// Calibration file for imatrix generation
    #[arg(long, conflicts_with = "imatrix")]
    pub calibration_file: Option<PathBuf>,
}

/// Device options for quantization
#[derive(Args, Clone)]
pub struct QuantizeDeviceOptions {
    /// Force CPU-only execution
    #[arg(long)]
    pub cpu: bool,

    /// Device layer mapping (format: ORD:NUM;... e.g., "0:10;1:20")
    #[arg(short = 'n', long, value_delimiter = ';')]
    pub device_layers: Option<Vec<String>>,

    /// Topology YAML file for device mapping
    #[arg(long)]
    pub topology: Option<PathBuf>,

    /// Custom Hugging Face cache directory
    #[arg(long)]
    pub hf_cache: Option<PathBuf>,

    /// Max sequence length for automatic device mapping
    #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN)]
    pub max_seq_len: usize,

    /// Max batch size for automatic device mapping
    #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE)]
    pub max_batch_size: usize,
}

/// Output options for UQFF generation
#[derive(Args, Clone)]
pub struct QuantizeOutputOptions {
    /// Output path: a `.uqff` file path (single ISQ) or a directory (auto-names files per ISQ type).
    /// Examples: `-o model/model-q4k.uqff` or `-o output/`
    #[arg(short = 'o', long = "output", required = true)]
    pub output_path: PathBuf,

    /// Skip README.md model card generation (generated by default in directory mode)
    #[arg(long)]
    pub no_readme: bool,

    /// Base model ID for the generated README (skips interactive prompt)
    #[arg(long)]
    pub uqff_base_model: Option<String>,

    /// HF repo ID for the generated README and upload hint (skips interactive prompt)
    #[arg(long)]
    pub uqff_repo_id: Option<String>,
}

/// Multimodal model options for quantization
#[derive(Args, Clone, Default)]
pub struct QuantizeMultimodalOptions {
    /// Maximum edge length for image resizing (aspect ratio preserved)
    #[arg(long)]
    pub max_edge: Option<u32>,

    /// Maximum number of images per request
    #[arg(long)]
    pub max_num_images: Option<usize>,

    /// Maximum image dimension for device mapping
    #[arg(long)]
    pub max_image_length: Option<usize>,
}

/// Default options for quantize command when no model type subcommand is specified.
/// These mirror the Auto variant's options and are used to construct QuantizeModelType::Auto.
#[derive(clap::Args, Clone)]
pub struct QuantizeDefaultOptions {
    /// Hugging Face model ID or local model directory; optional when `-f` names local files
    #[arg(short = 'm', long)]
    pub model_id: Option<String>,

    /// Path to local tokenizer.json file
    #[arg(short = 't', long)]
    pub tokenizer: Option<PathBuf>,

    /// Model data type
    #[arg(long, default_value = "auto", value_parser = parse_dtype)]
    pub dtype: ModelDType,

    #[command(flatten)]
    pub format: QuantizeFormatOptions,

    /// Select an input GGUF artifact by bit width or quant name
    #[arg(long, conflicts_with = "quantized_file")]
    pub quant: Option<String>,

    /// Output UQFF quantization type(s). Multiple values can be comma-separated or specified
    /// via repeated --isq flags (e.g., "--isq q4k,q8_0" or "--isq q4k --isq q8_0").
    #[arg(long = "isq", value_delimiter = ',')]
    pub in_situ_quant: Vec<String>,

    /// ISQ organization strategy: default or moqe
    #[arg(long)]
    pub isq_organization: Option<IsqOrganization>,

    /// imatrix file for enhanced quantization
    #[arg(long)]
    pub imatrix: Option<PathBuf>,

    /// Calibration file for imatrix generation
    #[arg(long, conflicts_with = "imatrix")]
    pub calibration_file: Option<PathBuf>,

    /// Force CPU-only execution
    #[arg(long)]
    pub cpu: bool,

    /// Device layer mapping (format: ORD:NUM;... e.g., "0:10;1:20")
    #[arg(short = 'n', long, value_delimiter = ';')]
    pub device_layers: Option<Vec<String>>,

    /// Topology YAML file for device mapping
    #[arg(long)]
    pub topology: Option<PathBuf>,

    /// Custom Hugging Face cache directory
    #[arg(long)]
    pub hf_cache: Option<PathBuf>,

    /// Max sequence length for automatic device mapping
    #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN)]
    pub max_seq_len: usize,

    /// Max batch size for automatic device mapping
    #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE)]
    pub max_batch_size: usize,

    /// Output path: a `.uqff` file path (single ISQ) or a directory (auto-names files per ISQ type).
    #[arg(short = 'o', long = "output")]
    pub output_path: Option<PathBuf>,

    /// Skip README.md model card generation (generated by default in directory mode)
    #[arg(long)]
    pub no_readme: bool,

    /// Base model ID for the generated README (skips interactive prompt)
    #[arg(long)]
    pub uqff_base_model: Option<String>,

    /// HF repo ID for the generated README and upload hint (skips interactive prompt)
    #[arg(long)]
    pub uqff_repo_id: Option<String>,

    /// Maximum edge length for image resizing (aspect ratio preserved)
    #[arg(long)]
    pub max_edge: Option<u32>,

    /// Maximum number of images per request
    #[arg(long)]
    pub max_num_images: Option<usize>,

    /// Maximum image dimension for device mapping
    #[arg(long)]
    pub max_image_length: Option<usize>,
}

impl QuantizeDefaultOptions {
    /// Convert default options into a QuantizeModelType::Auto variant.
    /// Returns an error if required fields are missing.
    pub fn into_quantize_model_type(self) -> anyhow::Result<QuantizeModelType> {
        if self.in_situ_quant.is_empty() {
            return Err(anyhow::anyhow!("--isq is required"));
        }
        let output_path = self
            .output_path
            .ok_or_else(|| anyhow::anyhow!("--output (-o) is required"))?;

        let mut model_type = QuantizeModelType::Auto {
            model: QuantizeModelSourceOptions {
                model_id: self.model_id,
                tokenizer: self.tokenizer,
                dtype: self.dtype,
                format: self.format,
                quant: self.quant,
            },
            quantization: QuantizeQuantizationOptions {
                in_situ_quant: self.in_situ_quant,
                isq_organization: self.isq_organization,
                imatrix: self.imatrix,
                calibration_file: self.calibration_file,
            },
            device: QuantizeDeviceOptions {
                cpu: self.cpu,
                device_layers: self.device_layers,
                topology: self.topology,
                hf_cache: self.hf_cache,
                max_seq_len: self.max_seq_len,
                max_batch_size: self.max_batch_size,
            },
            output: QuantizeOutputOptions {
                output_path,
                no_readme: self.no_readme,
                uqff_base_model: self.uqff_base_model,
                uqff_repo_id: self.uqff_repo_id,
            },
            multimodal: QuantizeMultimodalOptions {
                max_edge: self.max_edge,
                max_num_images: self.max_num_images,
                max_image_length: self.max_image_length,
            },
        };
        normalize_quantize_model_type(&mut model_type)?;
        Ok(model_type)
    }
}

/// Get the effective QuantizeModelType, using default options if no subcommand was provided.
/// Returns an error if no subcommand is provided and required fields are missing.
pub fn resolve_quantize_model_type(
    model_type: Option<QuantizeModelType>,
    default_options: QuantizeDefaultOptions,
) -> anyhow::Result<QuantizeModelType> {
    let mut model_type = match model_type {
        Some(model_type) => model_type,
        None => return default_options.into_quantize_model_type(),
    };
    normalize_quantize_model_type(&mut model_type)?;
    Ok(model_type)
}

fn normalize_quantize_model_type(model_type: &mut QuantizeModelType) -> anyhow::Result<()> {
    let model = match model_type {
        QuantizeModelType::Auto { model, .. }
        | QuantizeModelType::Text { model, .. }
        | QuantizeModelType::Multimodal { model, .. } => model,
        QuantizeModelType::Embedding { .. } => return Ok(()),
    };

    model.format.normalize()?;
    if model.quant.is_some() && model.format.quantized_file.is_some() {
        anyhow::bail!("`--quant` and `--quantized-file` are mutually exclusive");
    }
    if model.quant.is_some() && matches!(model.format.format, Some(QuantizeModelFormat::Plain)) {
        anyhow::bail!("`--quant` selects an input GGUF artifact and requires GGUF format");
    }
    let model_id = match model.model_id.take() {
        Some(model_id) => model_id,
        None => model.format.derive_local_model_root()?,
    };
    model.model_id = Some(model_id);
    Ok(())
}

fn parse_arch(s: &str) -> Result<NormalLoaderType, String> {
    s.parse()
}

fn parse_dtype(s: &str) -> Result<ModelDType, String> {
    s.parse()
}

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use clap::{CommandFactory, Parser};

    use super::*;
    use crate::args::{Cli, Command};

    fn resolve(args: &[&str]) -> anyhow::Result<QuantizeModelType> {
        let cli = Cli::try_parse_from(
            ["mistralrs", "quantize"]
                .into_iter()
                .chain(args.iter().copied()),
        )?;
        let Command::Quantize {
            model_type,
            default_quantize,
        } = cli.command
        else {
            unreachable!()
        };
        resolve_quantize_model_type(model_type, default_quantize)
    }

    #[test]
    fn direct_gguf_infers_format_and_local_root() {
        let root = std::env::temp_dir().join(format!("mistralrs-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let model_path = root.join("model.gguf");
        fs::write(&model_path, []).unwrap();

        let model_type = resolve(&[
            "-f",
            &model_path.to_string_lossy(),
            "--isq",
            "q4k",
            "-o",
            &root.join("model.uqff").to_string_lossy(),
        ])
        .unwrap();
        let QuantizeModelType::Auto {
            model,
            quantization,
            ..
        } = model_type
        else {
            panic!("expected auto quantize model")
        };
        assert_eq!(
            model.model_id.as_deref().map(Path::new),
            Some(root.as_path())
        );
        assert_eq!(model.format.format, Some(QuantizeModelFormat::Gguf));
        assert_eq!(model.format.quantized_file.as_deref(), Some("model.gguf"));
        assert!(model.format.direct_file_only);
        assert_eq!(quantization.in_situ_quant, ["q4k"]);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn source_quant_and_output_isq_remain_distinct() {
        let model_type = resolve(&[
            "-m",
            "org/model-GGUF",
            "--quant",
            "4",
            "--isq",
            "q8_0",
            "-o",
            "model.uqff",
        ])
        .unwrap();
        let QuantizeModelType::Auto {
            model,
            quantization,
            ..
        } = model_type
        else {
            panic!("expected auto quantize model")
        };
        assert_eq!(model.quant.as_deref(), Some("4"));
        assert_eq!(quantization.in_situ_quant, ["q8_0"]);
        assert!(model.format.quantized_file.is_none());
    }

    #[test]
    fn source_quant_rejects_an_exact_file() {
        let error = match resolve(&[
            "-m",
            "org/model-GGUF",
            "-f",
            "model.gguf",
            "--quant",
            "4",
            "--isq",
            "q8_0",
            "-o",
            "model.uqff",
        ]) {
            Ok(_) => panic!("expected source selection conflict"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("cannot be used with"));
    }

    #[test]
    fn help_only_advertises_supported_input_formats() {
        let mut command = Cli::command();
        let quantize = command
            .find_subcommand_mut("quantize")
            .expect("quantize subcommand");
        let help = quantize.render_long_help().to_string();
        let help_lower = help.to_ascii_lowercase();

        assert!(help_lower.contains("possible values: plain, gguf"));
        assert!(!help_lower.contains("ggml"));
        assert!(!help.contains("--gqa"));

        let embedding = quantize
            .find_subcommand_mut("embedding")
            .expect("embedding subcommand");
        let embedding_help = embedding.render_long_help().to_string();
        assert!(!embedding_help.contains("--format"));
        assert!(!embedding_help.contains("--quantized-file"));
        assert!(!embedding_help.contains("--quant"));
    }
}
