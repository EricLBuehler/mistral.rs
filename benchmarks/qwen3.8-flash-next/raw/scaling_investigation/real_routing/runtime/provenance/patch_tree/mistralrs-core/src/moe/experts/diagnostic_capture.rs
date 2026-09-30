use std::{
    collections::HashMap,
    fs::{self, File},
    path::PathBuf,
    sync::{Mutex, OnceLock},
    time::{SystemTime, UNIX_EPOCH},
};

use candle_core::{quantized::gguf_file, DType, Device, Result, Tensor};
use serde::Deserialize;
use serde_json::json;

use super::{MoEExperts, MoEExpertsBackendImpl, GROUPED_PREFILL_MIN_TOKENS};

const CAPTURE_DIRECTORY_ENV: &str = "MISTRALRS_MOE_CAPTURE_DIR";
const CAPTURE_LAYER_SUFFIX: &str = ".layers.8.mlp";
const MAX_CAPTURE_ROWS: usize = 64;
const MAX_SAMPLES_PER_SHAPE: usize = 8;
const MAX_CAPTURE_CASES: usize = 16;
const WEIGHTS_FILE: &str = "layer8.gguf";

static CAPTURE_DIRECTORY: OnceLock<Option<PathBuf>> = OnceLock::new();
static CAPTURE_STATE: OnceLock<Mutex<CaptureState>> = OnceLock::new();

type ShapeCounts = HashMap<(String, usize, usize), (usize, usize)>;

#[derive(Default)]
struct CaptureState {
    weights_written: bool,
    cases: HashMap<String, ShapeCounts>,
}

#[derive(Deserialize)]
struct CaptureControl {
    case: String,
    row_counts: Vec<usize>,
    stages: Vec<String>,
    max_per_shape: usize,
    every_n: usize,
    skip_per_shape: usize,
}

impl MoEExperts {
    pub(crate) fn capture_diagnostic(
        &self,
        prefix: &str,
        stage: &str,
        tensors: [&Tensor; 4],
    ) -> Result<()> {
        if !prefix.ends_with(CAPTURE_LAYER_SUFFIX) {
            return Ok(());
        }
        let Some(directory) = CAPTURE_DIRECTORY
            .get_or_init(|| std::env::var_os(CAPTURE_DIRECTORY_ENV).map(PathBuf::from))
        else {
            return Ok(());
        };
        let [xs, ids, weights, output] = tensors;
        let (batch, query_len, hidden) = xs.dims3()?;
        let rows = batch * query_len;
        if rows == 0 || rows > MAX_CAPTURE_ROWS {
            return Ok(());
        }
        let control = match fs::read(directory.join("control.json")) {
            Ok(control) => control,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error.into()),
        };
        let control: CaptureControl =
            serde_json::from_slice(&control).map_err(candle_core::Error::wrap)?;
        if !control.row_counts.contains(&rows) || !control.stages.iter().any(|value| value == stage)
        {
            return Ok(());
        }
        if control.case.is_empty()
            || !control
                .case
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'_')
            || control.max_per_shape == 0
            || control.max_per_shape > MAX_SAMPLES_PER_SHAPE
            || control.every_n == 0
        {
            candle_core::bail!("invalid MoE diagnostic control");
        }
        if crate::perf_flags::cuda_graphs_enabled() {
            candle_core::bail!("MoE diagnostic capture requires MISTRALRS_CUDA_GRAPHS=0");
        }
        if self.world_size != 1 || self.lora_site.is_some() || xs.dtype() != DType::BF16 {
            candle_core::bail!(
                "MoE diagnostic capture requires unsharded BF16 inputs without LoRA"
            );
        }
        if !matches!(
            self.act,
            crate::layers::Activation::Silu | crate::layers::Activation::Swish
        ) {
            candle_core::bail!("MoE diagnostic capture requires SiLU experts");
        }
        let MoEExpertsBackendImpl::Fast(experts) = &self.backend else {
            candle_core::bail!("MoE diagnostic capture requires the quantized Fast backend");
        };
        let mut state = CAPTURE_STATE.get_or_init(Default::default).lock().unwrap();
        if !state.cases.contains_key(&control.case) && state.cases.len() >= MAX_CAPTURE_CASES {
            candle_core::bail!("MoE diagnostic capture case limit reached");
        }
        let counts = state
            .cases
            .entry(control.case.clone())
            .or_default()
            .entry((stage.to_string(), batch, query_len))
            .or_default();
        counts.0 += 1;
        let visit = counts.0;
        if counts.1 >= control.max_per_shape
            || visit <= control.skip_per_shape
            || (visit - control.skip_per_shape - 1) % control.every_n != 0
        {
            return Ok(());
        }
        let sample_index = counts.1;
        counts.1 += 1;
        let base = format!(
            "{}_{}_b{batch}_q{query_len}_{sample_index:02}",
            control.case, stage
        );
        let gate = experts
            .fused_gate_proj
            .get_qtensor()
            .ok_or_else(|| candle_core::Error::msg("missing quantized gate"))?;
        let up = experts
            .fused_up_proj
            .get_qtensor()
            .ok_or_else(|| candle_core::Error::msg("missing quantized up"))?;
        let down = experts
            .fused_down_proj
            .get_qtensor()
            .ok_or_else(|| candle_core::Error::msg("missing quantized down"))?;
        if gate.dtype() != up.dtype() || !mistralrs_quant::supports_mmq(gate.dtype()) {
            candle_core::bail!("MoE diagnostic capture requires matching MMQ gate/up formats");
        }
        if !state.weights_written {
            let mut file = File::options()
                .write(true)
                .create_new(true)
                .open(directory.join(WEIGHTS_FILE))?;
            gguf_file::write(
                &mut file,
                &[],
                &[
                    ("blk.8.ffn_gate_exps.weight", gate.as_ref()),
                    ("blk.8.ffn_up_exps.weight", up.as_ref()),
                    ("blk.8.ffn_down_exps.weight", down.as_ref()),
                ],
            )?;
            state.weights_written = true;
        }
        let xs = xs.reshape((rows, hidden))?.to_device(&Device::Cpu)?;
        let ids = ids
            .reshape((rows, self.num_experts_per_tok))?
            .to_device(&Device::Cpu)?;
        let weights = weights
            .reshape((rows, self.num_experts_per_tok))?
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?;
        let output = output.reshape((rows, hidden))?.to_device(&Device::Cpu)?;
        let mut occupancy = vec![0usize; self.num_experts];
        for expert in ids.flatten_all()?.to_vec1::<u32>()? {
            occupancy[expert as usize] += 1;
        }
        let samples = HashMap::from([
            ("xs", xs),
            ("ids", ids),
            ("weights", weights),
            ("output", output),
        ]);
        let sample_file = format!("{base}.safetensors");
        candle_core::safetensors::save(&samples, directory.join(&sample_file))?;
        let captured_unix_seconds = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(candle_core::Error::wrap)?
            .as_secs_f64();
        let metadata = json!({
            "case": control.case, "layer": 8, "prefix": prefix, "stage": stage,
            "captured_unix_seconds": captured_unix_seconds,
            "batch": batch, "query_len": query_len, "rows": rows, "hidden": hidden,
            "topk": self.num_experts_per_tok, "experts": self.num_experts,
            "output_shape": [rows, hidden], "output_dtype": "BF16",
            "shape_visit": visit, "sample_index": sample_index, "derived": false,
            "tensor_file": sample_file, "weights_file": WEIGHTS_FILE,
            "gate_dtype": format!("{:?}", gate.dtype()),
            "up_dtype": format!("{:?}", up.dtype()),
            "down_dtype": format!("{:?}", down.dtype()),
            "source_dispatch": if query_len > 1 && rows >= GROUPED_PREFILL_MIN_TOKENS {
                "grouped_mmq"
            } else { "indexed_gemv" },
            "expert_occupancy": occupancy,
            "capture_scope": "diagnostic eager model execution; fixed MTP depth; synchronized capture is not a throughput measurement",
        });
        serde_json::to_writer_pretty(
            File::create(directory.join(format!("{base}.json")))?,
            &metadata,
        )
        .map_err(candle_core::Error::wrap)?;
        tracing::info!(case = %control.case, stage, rows, sample_index, "saved MoE diagnostic sample");
        Ok(())
    }
}
