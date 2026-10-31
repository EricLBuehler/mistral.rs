use anyhow::Result;

#[cfg(feature = "cuda")]
use crate::pipeline::HostBacking;
use crate::pipeline::{LayerResource, ResourceSource};

use super::config::Config;

const TEXT_ROOT: &str = "model.language_model";
const ALTERNATE_TEXT_ROOT: &str = "language_model.model";
pub(crate) const ALTERNATE_TEXT_EMBEDDING: &str = "language_model.model.embed_tokens.weight";

pub(crate) fn text_model_prefix(has_alternate_embedding: bool) -> &'static str {
    if has_alternate_embedding {
        ALTERNATE_TEXT_ROOT
    } else {
        TEXT_ROOT
    }
}

pub(crate) fn layer_resources(
    config: &str,
    source: ResourceSource<'_>,
) -> Result<Vec<LayerResource>> {
    let cfg: Config = serde_json::from_str(config)?;
    let Some(ple) = cfg.text_config.ple()? else {
        return Ok(Vec::new());
    };
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (ple, source);
        Ok(Vec::new())
    }
    #[cfg(feature = "cuda")]
    {
        let archive = source.gguf();
        let raw = if archive.is_none() {
            source.safetensors()?
        } else {
            None
        };
        let root = text_model_prefix({
            source.weights.is_some_and(|vb| {
                crate::layers::contains_tensor_or_uqff(vb, ALTERNATE_TEXT_EMBEDDING)
            }) || raw
                .as_ref()
                .is_some_and(|raw| raw.get(ALTERNATE_TEXT_EMBEDDING).is_ok())
        });
        let prefix = format!(
            "{root}.layers.{}.ple.ple_embedding.ngram_embedding",
            ple.layer_idx
        );
        if source.weights.is_none() && source.paths.is_empty() {
            return Ok(vec![LayerResource {
                name: prefix,
                layer: ple.layer_idx,
                device_bytes: super::ple::planned_resident_table_bytes(&cfg.text_config, &ple),
                host_backing: None,
            }]);
        }
        let table = super::ple::table_source(raw, archive, &prefix, ple.head_dim())?;
        let metadata = table.metadata();
        let host_backing = match metadata.host_backing {
            mistralrs_quant::embedding::EmbeddingBacking::Owned => HostBacking::Owned,
            mistralrs_quant::embedding::EmbeddingBacking::FileMapped => HostBacking::FileMapped,
        };
        Ok(vec![LayerResource {
            name: prefix,
            layer: ple.layer_idx,
            device_bytes: metadata.device_bytes,
            host_backing: Some(host_backing),
        }])
    }
}
