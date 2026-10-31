use super::*;
use crate::vision_models::qwen4_exp::config::tests::flash_next_text_config;

#[cfg(feature = "cuda")]
#[test]
fn planned_table_matches_checkpoint_rows() {
    // The checkpoint pads the hashed rows up to a multiple of its shard count
    const CHECKPOINT_ROWS: usize = 320_001_536;
    const CHECKPOINT_SHARDS: usize = 128;
    let cfg = flash_next_text_config(4);
    let ple = cfg.ple().unwrap().unwrap();
    let row_bytes =
        ple.head_dim() / 2 + ple.head_dim() / mistralrs_quant::embedding::SYMMETRIC_GROUP * 2;
    let rows = planned_resident_table_bytes(&cfg, &ple) / row_bytes;
    assert_eq!(
        rows.div_ceil(CHECKPOINT_SHARDS) * CHECKPOINT_SHARDS,
        CHECKPOINT_ROWS
    );
}

#[test]
fn hashed_rows_match_reference() {
    let cfg = flash_next_text_config(4);
    let ple = cfg.ple().unwrap().unwrap();
    let hash = PleHash::new(&cfg, &ple, 0);
    assert_eq!(
        hash.multipliers,
        [23703573157769, 20109073645365, 8052911324071]
    );
    assert_eq!(&hash.vocab_sizes[..3], &[20000003, 20000023, 20000033]);
    assert_eq!(&hash.offsets[..3], &[0, 20000003, 40000026]);

    // Rows 0, 1, 7, 8 and 15 per token from `Qwen4ExpTextNGramEmbedding` with no history
    let tokens = [9707, 11, 1879, 248044, 553, 1024, 7, 248044, 248044, 42];
    let expected = [
        [16410909, 39682429, 152932897, 169641436, 300984276],
        [18158303, 36390029, 159566246, 175132467, 307699687],
        [6380558, 26411572, 148942556, 164226950, 312121804],
        [18827606, 27700912, 142052741, 163793416, 318943450],
        [12851972, 31952205, 159255184, 173644045, 312013235],
        [12082042, 20450345, 149364883, 165982112, 316678718],
        [17604418, 20085892, 154993589, 175674761, 313648512],
        [10204458, 27984170, 158727320, 162586317, 310194357],
        [9663979, 26558231, 151022725, 170054832, 300986548],
        [4064167, 21428206, 156147803, 175176707, 314347972],
    ];
    let mut rows = vec![0u64; hash.num_heads()];
    for (local, expected) in expected.iter().enumerate() {
        hash.rows(&tokens, local, &[None, None], &mut rows);
        let picked = [rows[0], rows[1], rows[7], rows[8], rows[15]];
        assert_eq!(&picked, expected, "token {local}");
    }

    // A chunk boundary: the history carries the context the reference sees in-sequence
    let mut split = vec![0u64; hash.num_heads()];
    hash.rows(&tokens[5..], 0, &[Some(248044), Some(553)], &mut split);
    hash.rows(&tokens, 5, &[None, None], &mut rows);
    assert_eq!(split, rows);
}
