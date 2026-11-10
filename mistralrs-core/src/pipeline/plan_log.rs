use std::fmt::Write;

use candle_core::{DType, Device};

use crate::attention::{cuda_flash_usable, flash_backend_supports_sdpa, flash_dtype_supported};
use crate::paged_attention::plan::{PrefixPrefillPlan, PrefixPrefillPlanInput};
use crate::paged_attention::{CacheConfig, ModelConfigLike};

// A typical prompt chunk, well inside every kernel's query-length limits.
const REPRESENTATIVE_PROMPT_LEN: usize = 512;
const PER_FORWARD: &str = "decided per forward";
const LABEL_WIDTH: usize = 13;
const LISTED_LAYERS: usize = 4;

/// One resolved choice plus why a faster one was passed over.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PlanChoice {
    pub choice: &'static str,
    pub reason: Option<&'static str>,
}

impl PlanChoice {
    fn fastest(choice: &'static str) -> Self {
        Self {
            choice,
            reason: None,
        }
    }

    fn fallback(choice: &'static str, reason: &'static str) -> Self {
        Self {
            choice,
            reason: Some(reason),
        }
    }

    fn render(&self) -> String {
        match self.reason {
            Some(reason) => format!("{} ({reason})", self.choice),
            None => self.choice.to_string(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AttentionRow {
    pub prefill: PlanChoice,
    pub decode: Option<PlanChoice>,
}

/// Every layer that resolved to the same attention path, in layer order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AttentionSpan {
    pub layers: Vec<usize>,
    pub row: AttentionRow,
}

impl AttentionSpan {
    fn describe_layers(&self, num_layers: usize) -> String {
        if self.layers.len() == num_layers {
            return format!("all {num_layers} layers");
        }
        let shown = self
            .layers
            .iter()
            .take(LISTED_LAYERS)
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let more = if self.layers.len() > LISTED_LAYERS {
            ", ..."
        } else {
            ""
        };
        format!(
            "{} of {num_layers} layers ({shown}{more})",
            self.layers.len()
        )
    }
}

pub(crate) struct RuntimePlanInput<'a> {
    pub device: &'a Device,
    pub activation_dtype: DType,
    pub weights: String,
    pub model: Option<&'a dyn ModelConfigLike>,
    pub cache: Option<&'a CacheConfig>,
    pub head_dim: usize,
    pub has_sliding_window: bool,
    pub packed_prefill: bool,
    pub batched_sampling: bool,
    pub has_proposer: bool,
    pub decode_graphs: Option<bool>,
}

pub(crate) struct RuntimePlan {
    num_layers: usize,
    attention: Vec<AttentionSpan>,
    kv_cache: String,
    weights: String,
    prefill: PlanChoice,
    sampling: PlanChoice,
    speculation: PlanChoice,
    decode_graphs: Option<PlanChoice>,
}

impl RuntimePlan {
    pub(crate) fn resolve(input: &RuntimePlanInput<'_>) -> Self {
        let num_layers = input.model.map_or(1, |model| model.num_layers());
        let (attention, kv_cache) = match (input.model, input.cache) {
            (Some(model), Some(cache)) => (
                paged_attention_spans(model, cache, input.device, input.activation_dtype),
                format!(
                    "paged, {} blocks of {} tokens, {:?}",
                    cache.num_gpu_blocks,
                    cache.block_size,
                    cache.cache_type.to_dtype(input.activation_dtype)
                ),
            ),
            _ => (
                vec![AttentionSpan {
                    layers: (0..num_layers).collect(),
                    row: AttentionRow {
                        prefill: contiguous_attention(input),
                        decode: None,
                    },
                }],
                "contiguous".to_string(),
            ),
        };
        let prefill = if input.packed_prefill {
            PlanChoice::fastest("packed across sequences")
        } else {
            PlanChoice::fastest("one forward per uniform batch")
        };
        let sampling = if !input.device.is_cuda() {
            PlanChoice::fallback("per sequence", "batched sampling needs CUDA")
        } else if input.batched_sampling {
            PlanChoice::fastest("batched on GPU when every sequence allows it")
        } else {
            PlanChoice::fallback("per sequence", "a speculative proposer owns the step")
        };
        let speculation = if input.has_proposer {
            PlanChoice::fastest("proposer attached")
        } else {
            PlanChoice::fastest("off")
        };
        let decode_graphs = input.decode_graphs.map(|enabled| {
            if enabled {
                PlanChoice::fastest("CUDA graphs")
            } else {
                PlanChoice::fallback("eager launches", "model or flags rule out CUDA graphs")
            }
        });
        Self {
            num_layers,
            attention,
            kv_cache,
            weights: format!(
                "{}, {:?} activations",
                input.weights, input.activation_dtype
            ),
            prefill,
            sampling,
            speculation,
            decode_graphs,
        }
    }

    pub(crate) fn render(&self) -> String {
        let mut out = String::from("Runtime plan:");
        let mut line = |label: &str, value: &str| {
            let _ = write!(out, "\n  {label:<LABEL_WIDTH$}{value}");
        };
        line("weights", &self.weights);
        line("kv cache", &self.kv_cache);
        for span in &self.attention {
            let layers = span.describe_layers(self.num_layers);
            let value = match &span.row.decode {
                Some(decode) => format!(
                    "{layers}: prefill {}, decode {}",
                    span.row.prefill.render(),
                    decode.render()
                ),
                None => format!("{layers}: {}", span.row.prefill.render()),
            };
            line("attention", &value);
        }
        line("prefill", &self.prefill.render());
        if let Some(decode_graphs) = &self.decode_graphs {
            line("decode", &decode_graphs.render());
        }
        line("sampling", &self.sampling.render());
        line("speculation", &self.speculation.render());
        out
    }
}

fn contiguous_attention(input: &RuntimePlanInput<'_>) -> PlanChoice {
    if input.device.is_cpu() {
        return PlanChoice::fastest("fused CPU attention");
    }
    if !input.device.is_cuda() {
        return PlanChoice::fastest("backend SDPA");
    }
    if cuda_flash_usable(input.device, input.activation_dtype)
        && flash_backend_supports_sdpa(input.head_dim, false, input.has_sliding_window)
    {
        return PlanChoice::fastest("flash-attn");
    }
    let reason = if !crate::using_flash_attn() {
        "built without flash-attn"
    } else if !flash_dtype_supported(input.activation_dtype) {
        "activations are not F16/BF16"
    } else {
        "head dim or sliding window outside flash-attn"
    };
    PlanChoice::fallback("eager SDPA", reason)
}

fn paged_attention_spans(
    model: &dyn ModelConfigLike,
    cache: &CacheConfig,
    device: &Device,
    dtype: DType,
) -> Vec<AttentionSpan> {
    let mut spans: Vec<AttentionSpan> = Vec::new();
    for layer_idx in 0..model.num_layers() {
        let row = paged_attention_row(model, cache, device, dtype, layer_idx);
        match spans.iter_mut().find(|span| span.row == row) {
            Some(span) => span.layers.push(layer_idx),
            None => spans.push(AttentionSpan {
                layers: vec![layer_idx],
                row,
            }),
        }
    }
    spans
}

fn paged_attention_row(
    model: &dyn ModelConfigLike,
    cache: &CacheConfig,
    device: &Device,
    dtype: DType,
    layer_idx: usize,
) -> AttentionRow {
    if !model.layer_has_paged_kv_cache(layer_idx) {
        return AttentionRow {
            prefill: PlanChoice::fastest("no KV cache"),
            decode: None,
        };
    }
    let declared = model.prefix_prefill_attention_features(layer_idx);
    let features = declared.unwrap_or_default();
    let has_sliding_window =
        features.has_sliding_window || model.sliding_window_for_layer(layer_idx).is_some();
    let decode = decode_choice(model, layer_idx, features, has_sliding_window);
    if declared.is_none() {
        return AttentionRow {
            prefill: PlanChoice::fastest(PER_FORWARD),
            decode,
        };
    }
    #[cfg(cuda_unix)]
    let fa3_supported = crate::flashinfer::fa3_device_num_sm(device).is_some()
        && model.k_head_dim_for_layer(layer_idx) == model.v_head_dim_for_layer(layer_idx);
    #[cfg(not(cuda_unix))]
    let fa3_supported = false;
    let plan_input = PrefixPrefillPlanInput {
        device_is_cuda: device.is_cuda(),
        dtype,
        cache_dtype: cache.cache_type.to_dtype(dtype),
        has_alibi: features.has_alibi,
        has_sinks: features.has_sinks,
        has_custom_mask: false,
        causality_known: true,
        head_size: model.k_head_dim_for_layer(layer_idx),
        has_softcap: features.has_softcap,
        has_sliding_window,
        query_layout_is_dense: true,
        query_len: REPRESENTATIVE_PROMPT_LEN,
        q_heads: model.num_attn_heads_for_layer(layer_idx),
        kv_heads: model.num_kv_heads_for_layer(layer_idx),
        writes_cache: model.uses_own_kv_cache_for_layer(layer_idx),
        is_causal: true,
        has_noncausal_mm_context: false,
        fa3_supported,
        block_size: cache.block_size,
        attention_backend: model.attention_backend_kind_for_layer(layer_idx),
    };
    let plan = PrefixPrefillPlan::choose(plan_input);
    let reason = matches!(plan, PrefixPrefillPlan::GatherSdpa)
        .then(|| PrefixPrefillPlan::flash_paged_blocker(plan_input))
        .flatten();
    let prefill = PlanChoice {
        choice: plan.label(),
        reason,
    };
    AttentionRow { prefill, decode }
}

#[cfg(any(feature = "cuda", feature = "metal", test))]
fn decode_choice(
    model: &dyn ModelConfigLike,
    layer_idx: usize,
    features: crate::paged_attention::PrefixPrefillAttentionFeatures,
    has_sliding_window: bool,
) -> Option<PlanChoice> {
    use crate::paged_attention::plan::{DecodePlan, DecodePlanInput};

    let plan = DecodePlan::choose(DecodePlanInput {
        attention_backend: model.attention_backend_kind_for_layer(layer_idx),
        head_size: model.k_head_dim_for_layer(layer_idx),
        has_alibi: features.has_alibi,
        has_sinks: features.has_sinks,
        has_sliding_window,
    });
    Some(match plan {
        #[cfg(cuda_unix)]
        Ok(DecodePlan::FlashInfer(_)) => PlanChoice::fastest("FlashInfer paged"),
        Ok(DecodePlan::PagedAttention) => PlanChoice::fastest("paged attention kernel"),
        Ok(DecodePlan::GatherSdpa) => PlanChoice::fallback(
            "gather + SDPA",
            "head dim or sliding window outside the paged decode kernels",
        ),
        Err(_) => PlanChoice::fastest(PER_FORWARD),
    })
}

#[cfg(not(any(feature = "cuda", feature = "metal", test)))]
fn decode_choice(
    _model: &dyn ModelConfigLike,
    _layer_idx: usize,
    _features: crate::paged_attention::PrefixPrefillAttentionFeatures,
    _has_sliding_window: bool,
) -> Option<PlanChoice> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paged_attention::{
        KvCacheLayout, ModelConfigMetadata, PagedCacheType, PrefixPrefillAttentionFeatures,
    };

    fn metadata(num_layers: usize) -> ModelConfigMetadata {
        ModelConfigMetadata {
            max_seq_len: 8192,
            num_layers,
            hidden_size: 1024,
            num_kv_heads: 2,
            num_attn_heads: 8,
            sliding_window: None,
            k_head_dim: 128,
            v_head_dim: 128,
            kv_cache_layout: KvCacheLayout::Standard,
        }
    }

    fn input<'a>(
        device: &'a Device,
        model: Option<&'a dyn ModelConfigLike>,
        cache: Option<&'a CacheConfig>,
    ) -> RuntimePlanInput<'a> {
        RuntimePlanInput {
            device,
            activation_dtype: DType::F32,
            weights: "normal (no adapters)".to_string(),
            model,
            cache,
            head_dim: 128,
            has_sliding_window: false,
            packed_prefill: false,
            batched_sampling: false,
            has_proposer: false,
            decode_graphs: None,
        }
    }

    #[test]
    fn contiguous_cpu_plan_names_the_fused_path() {
        let model = metadata(4);
        let plan = RuntimePlan::resolve(&input(&Device::Cpu, Some(&model), None));
        let text = plan.render();
        assert!(text.starts_with("Runtime plan:"));
        assert!(text.contains("kv cache     contiguous"));
        assert!(text.contains("attention    all 4 layers: fused CPU attention"));
        assert!(text.contains("sampling     per sequence (batched sampling needs CUDA)"));
        assert!(text.contains("speculation  off"));
        assert!(!text.contains("\n  decode "));
    }

    #[test]
    fn paged_layers_with_one_path_collapse_into_one_span() {
        let model = crate::paged_attention::with_attention_features(
            std::sync::Arc::new(metadata(6)),
            Some(PrefixPrefillAttentionFeatures::default()),
        );
        let cache = CacheConfig {
            block_size: 32,
            num_gpu_blocks: 64,
            cache_type: PagedCacheType::Auto,
            kv_cache_group_ids: vec![0],
        };
        let spans = paged_attention_spans(model.as_ref(), &cache, &Device::Cpu, DType::F32);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].layers, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(spans[0].describe_layers(6), "all 6 layers");
        assert_eq!(
            spans[0].describe_layers(24),
            "6 of 24 layers (0, 1, 2, 3, ...)"
        );
        assert_eq!(spans[0].row.prefill.choice, "gather + SDPA");
        assert!(spans[0].row.prefill.reason.is_some());
    }

    #[test]
    fn undeclared_features_are_not_guessed() {
        let model = metadata(2);
        let cache = CacheConfig {
            block_size: 32,
            num_gpu_blocks: 64,
            cache_type: PagedCacheType::Auto,
            kv_cache_group_ids: vec![0],
        };
        let spans = paged_attention_spans(&model, &cache, &Device::Cpu, DType::F32);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].row.prefill, PlanChoice::fastest(PER_FORWARD));
    }
}
