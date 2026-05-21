use std::fmt;
use std::sync::OnceLock;

use candle_core::{DType, Result, Tensor};

const ENV_FLAG: &str = "MISTRALRS_MTP_TRACE";
const MAX_TOKENS: usize = 24;

pub(crate) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var(ENV_FLAG).is_ok_and(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
    })
}

pub(crate) fn log(args: fmt::Arguments<'_>) {
    if enabled() {
        tracing::info!(target: "mistralrs_core::speculative::trace", "{args}");
    }
}

pub(crate) fn tensor(tensor: &Tensor) -> String {
    format!(
        "shape={:?}, dtype={:?}, device={:?}",
        tensor.dims(),
        tensor.dtype(),
        tensor.device()
    )
}

pub(crate) fn tokens(tokens: &[u32]) -> String {
    if tokens.len() <= MAX_TOKENS {
        return format!("{tokens:?}");
    }
    format!(
        "{:?} ... (+{} more)",
        &tokens[..MAX_TOKENS],
        tokens.len() - MAX_TOKENS
    )
}

pub(crate) fn logits_topk(logits: &Tensor, top_k: usize, focus_tokens: &[u32]) -> Result<String> {
    let logits = logits.flatten_all()?.to_dtype(DType::F32)?;
    let values = logits.to_vec1::<f32>()?;
    let vocab = values.len();
    if vocab == 0 {
        return Ok("top=[] focus=[]".to_string());
    }
    let top_k = top_k.min(vocab);
    let top = topk_pairs(&values, top_k)
        .into_iter()
        .map(|(token, value)| format!("{token}:{value:.4}"))
        .collect::<Vec<_>>();

    let mut focus = Vec::new();
    for token in focus_tokens {
        let token_idx = *token as usize;
        let Some(value) = values.get(token_idx).copied() else {
            focus.push(format!("{token}:oob"));
            continue;
        };
        if let Some(rank) = top
            .iter()
            .position(|candidate| candidate.starts_with(&format!("{token}:")))
            .map(|idx| idx + 1)
        {
            focus.push(format!("{token}:{value:.4}@top{rank}"));
        } else {
            focus.push(format!("{token}:{value:.4}"));
        }
    }

    Ok(format!(
        "top{top_k}=[{}], focus=[{}]",
        top.join(", "),
        focus.join(", ")
    ))
}

pub(crate) fn topk_pairs(values: &[f32], top_k: usize) -> Vec<(u32, f32)> {
    let mut ranked = values
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, value)| (idx as u32, value))
        .collect::<Vec<_>>();
    ranked.sort_unstable_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    ranked.truncate(top_k.min(ranked.len()));
    ranked
}

pub(crate) fn mask_summary(mask: &Tensor, max_rows: usize) -> Result<String> {
    let dims = mask.dims().to_vec();
    let rank = mask.rank();
    let q_len = match rank {
        2 => dims[0],
        3 => dims[1],
        4 => dims[2],
        _ => {
            return Ok(format!("rank={rank}, dims={dims:?}, rows=unsupported"));
        }
    };
    if q_len == 0 {
        return Ok(format!("rank={rank}, dims={dims:?}, rows=[]"));
    }

    let mut rows = vec![0];
    if q_len > 1 && max_rows > 1 {
        rows.push(q_len - 1);
    }
    if q_len > 2 && max_rows > 2 {
        rows.push(q_len / 2);
        rows.sort_unstable();
        rows.dedup();
    }

    let mut summaries = Vec::with_capacity(rows.len());
    for row_idx in rows {
        let row = match rank {
            2 => mask.narrow(0, row_idx, 1)?,
            3 => mask.narrow(1, row_idx, 1)?,
            4 => mask.narrow(2, row_idx, 1)?,
            _ => unreachable!(),
        };
        let row = row.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        summaries.push(format!("row{row_idx}:{}", summarize_mask_row(&row)));
    }

    Ok(format!(
        "rank={rank}, dims={dims:?}, rows=[{}]",
        summaries.join("; ")
    ))
}

fn summarize_mask_row(row: &[f32]) -> String {
    if row.is_empty() {
        return "len=0".to_string();
    }

    let mut allowed_count = 0usize;
    let mut first_allowed = None;
    let mut last_allowed = None;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;

    for (idx, value) in row.iter().copied().enumerate() {
        min = min.min(value);
        max = max.max(value);
        if value > -1e4 {
            allowed_count += 1;
            first_allowed.get_or_insert(idx);
            last_allowed = Some(idx);
        }
    }

    let head = row
        .iter()
        .take(4)
        .map(|value| format_mask_value(*value))
        .collect::<Vec<_>>();
    let tail_start = row.len().saturating_sub(4);
    let tail = row[tail_start..]
        .iter()
        .map(|value| format_mask_value(*value))
        .collect::<Vec<_>>();

    format!(
        "len={}, allowed={allowed_count}, first={first_allowed:?}, last={last_allowed:?}, min={}, max={}, head=[{}], tail=[{}]",
        row.len(),
        format_mask_value(min),
        format_mask_value(max),
        head.join(", "),
        tail.join(", ")
    )
}

fn format_mask_value(value: f32) -> String {
    if value.is_infinite() {
        if value.is_sign_positive() {
            "inf".to_string()
        } else {
            "-inf".to_string()
        }
    } else if value.abs() >= 1e4 {
        format!("{value:.1e}")
    } else {
        format!("{value:.2}")
    }
}
