//! `get_rope_index`: 3D mrope position-id assignment.
//!
//! Earlier stages reused the reference `position_ids`; the engine must compute them from
//! `input_ids` + `image_grid_thw`. This is the native transformers-5.13
//! `PaddleOCRVLModel.get_rope_index` (Qwen2-VL style), images-only (video out of scope for the
//! OCR path).
//!
//! Walk the token stream left to right. A text run of length L gets identical positions
//! `st..st+L` on all three (t,h,w) rows. An image block of merged grid `(llm_t, llm_h, llm_w)`
//! gets, from `base`: row0 = base + t_index, row1 = base + h_index, row2 = base + w_index,
//! where the three indices enumerate the merged grid in row-major (t,h,w) order. The running
//! `next` cursor is always `(max position emitted so far) + 1`, so the next segment starts there.
//!
//! Verified against the reference `position_ids[3,174]` (integer-exact) and a worked example.

use candle_core::{Device, Result, Tensor};

/// Compute `position_ids [3, seq]` (i64) and the decode `rope_delta`.
///
/// `input_ids`: the full token sequence. `image_grids`: one `(t, h, w)` **patch** grid per image
/// (i.e. `image_grid_thw`, before the 2x2 merge). `image_token_id`: the placeholder id (100295).
/// `merge`: `spatial_merge_size` (2). Images must appear in `input_ids` in the same order as
/// `image_grids`, each as a contiguous run of exactly `t*(h/merge)*(w/merge)` placeholder tokens.
///
/// `rope_delta = max(position) + 1 - seq` is what a decode step adds to `cache_position` to get
/// the next position on all three rows (text continuation past the prefill).
pub fn get_rope_index(
    input_ids: &[i64],
    image_grids: &[(usize, usize, usize)],
    image_token_id: i64,
    merge: usize,
    dev: &Device,
) -> Result<(Tensor, i64)> {
    let seq = input_ids.len();
    let mut rows: [Vec<i64>; 3] = [
        Vec::with_capacity(seq),
        Vec::with_capacity(seq),
        Vec::with_capacity(seq),
    ];
    let mut next: i64 = 0; // running (max emitted position) + 1 = start of the next segment
    let mut st = 0usize; // cursor into input_ids

    // helper: emit a text run of length `len` starting at `start` on all 3 rows.
    let emit_text = |rows: &mut [Vec<i64>; 3], start: i64, len: usize| {
        for k in 0..len as i64 {
            for row in rows.iter_mut() {
                row.push(start + k);
            }
        }
    };

    for &(t, h, w) in image_grids {
        // text before this image block: first image token at/after st marks its start.
        let ed = (st..seq)
            .find(|&i| input_ids[i] == image_token_id)
            .unwrap_or(seq);
        let text_len = ed - st;
        emit_text(&mut rows, next, text_len);
        next += text_len as i64;

        // image block: merged grid, row-major (t, h, w).
        let (llm_t, llm_h, llm_w) = (t, h / merge, w / merge);
        let base = next;
        for ti in 0..llm_t as i64 {
            for hi in 0..llm_h as i64 {
                for wi in 0..llm_w as i64 {
                    rows[0].push(base + ti);
                    rows[1].push(base + hi);
                    rows[2].push(base + wi);
                }
            }
        }
        let span = llm_t.max(llm_h).max(llm_w) as i64;
        next = base + span; // (max in block) + 1
        st = ed + llm_t * llm_h * llm_w;
    }

    // trailing text after the last image.
    if st < seq {
        emit_text(&mut rows, next, seq - st);
    }

    debug_assert_eq!(
        rows[0].len(),
        seq,
        "position_ids length must equal sequence length"
    );
    let max_pos = rows.iter().flatten().copied().max().unwrap_or(-1);
    let delta = max_pos + 1 - seq as i64;

    let flat: Vec<i64> = rows.into_iter().flatten().collect();
    let position_ids = Tensor::from_vec(flat, (3, seq), dev)?;
    Ok((position_ids, delta))
}

#[cfg(test)]
mod tests {
    use super::*;

    // The `MultimodalModel` decode step computes the next token's mrope position as
    // `offset + delta` (all 3 rows), offset = prefill length. This must equal the position a fresh
    // `get_rope_index` over `[prompt, one_more_text_token]` assigns to that appended token, i.e.
    // decode continues the prefill text cursor. Uses a grid with a non-zero `delta` (image rows
    // compress positions) so the test is not vacuous.
    #[test]
    fn decode_position_continues_prefill_cursor() {
        let dev = Device::Cpu;
        let img = 999i64;
        // grid (1,4,4), merge 2 -> merged (1,2,2) -> 4 image placeholder tokens.
        let grid = (1usize, 4usize, 4usize);
        let prompt: Vec<i64> = vec![10, 11, img, img, img, img, 12, 13];
        let (_pos, delta) = get_rope_index(&prompt, &[grid], img, 2, &dev).unwrap();
        assert!(
            delta < 0,
            "image block should compress positions -> negative delta"
        );

        let decode_p = prompt.len() as i64 + delta;

        let mut extended = prompt.clone();
        extended.push(42);
        let (pos_ext, _) = get_rope_index(&extended, &[grid], img, 2, &dev).unwrap();
        let last_col: Vec<i64> = pos_ext
            .narrow(1, prompt.len(), 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<i64>()
            .unwrap();
        assert_eq!(
            last_col,
            vec![decode_p; 3],
            "decode position must continue the cursor"
        );
    }
}
