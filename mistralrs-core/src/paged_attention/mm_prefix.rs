use std::collections::HashMap;

use anyhow::Result;
use candle_core::{Device, Tensor};

pub(crate) fn make_ranges_tensor(
    seq_ids: &[usize],
    ranges_by_seq_id: &HashMap<usize, Vec<(usize, usize)>>,
    kv_window_starts: &[usize],
    kv_window_lens: &[usize],
    query_lens: &[usize],
) -> Result<Option<Tensor>> {
    if seq_ids.len() != kv_window_starts.len()
        || seq_ids.len() != kv_window_lens.len()
        || seq_ids.len() != query_lens.len()
    {
        anyhow::bail!("mm prefix range metadata length mismatch");
    }
    let ranges = seq_ids
        .iter()
        .zip(
            kv_window_starts
                .iter()
                .zip(kv_window_lens.iter().zip(query_lens.iter())),
        )
        .map(|(seq_id, (&window_start, (&window_len, &query_len)))| {
            let window_end = window_start + window_len;
            // Bidirectional override only fires for query rows inside a range, so ranges that
            // end before the query span (the last query_len positions of the window) are dead.
            let query_span_start = window_len.saturating_sub(query_len);
            ranges_by_seq_id
                .get(seq_id)
                .into_iter()
                .flatten()
                .filter_map(|&(start, end)| {
                    let local_start = start.max(window_start);
                    let local_end = end.min(window_end);
                    (local_start < local_end
                        && local_end.saturating_sub(window_start) > query_span_start)
                        .then(|| (local_start - window_start, local_end - window_start))
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    if ranges.iter().all(Vec::is_empty) {
        return Ok(None);
    }

    let max_ranges = ranges.iter().map(Vec::len).max().unwrap_or(1).max(1);
    let mut flattened = Vec::with_capacity(seq_ids.len() * max_ranges * 2);
    for seq_ranges in ranges {
        for (start, end) in seq_ranges.iter().copied() {
            flattened.push(i32::try_from(start).map_err(anyhow::Error::msg)?);
            flattened.push(i32::try_from(end).map_err(anyhow::Error::msg)?);
        }
        for _ in seq_ranges.len()..max_ranges {
            flattened.extend([0i32, 0]);
        }
    }

    Tensor::from_vec(flattened, (seq_ids.len(), max_ranges, 2), &Device::Cpu)
        .map(Some)
        .map_err(anyhow::Error::msg)
}
