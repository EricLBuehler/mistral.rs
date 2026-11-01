use std::collections::HashMap;

use super::{
    generation::{GenerationMarkers, GenerationMode, GenerationState},
    GeneratedAudioInput, GenerationCache, SequenceGeneration,
};

fn generation_with_codes(entries: &[(usize, &[u32])]) -> SequenceGeneration {
    SequenceGeneration {
        state: GenerationState::new(
            GenerationMode::Audio,
            GenerationMarkers {
                audio_start: 10,
                mixed_start: 11,
                text_end: 12,
                audio_eos: 9,
                text_slots: 2,
                audio_slots: 3,
            },
        ),
        frames: vec![],
        code_history: entries
            .iter()
            .map(|&(position, codes)| (position, codes.to_vec()))
            .collect(),
        audio_id: "audio_test".into(),
        stream: None,
    }
}

#[test]
fn generated_codes_replay_full_prompts_and_packed_chunks() {
    let cache = GenerationCache {
        hidden: HashMap::new(),
        sequences: [
            (
                1,
                generation_with_codes(&[(3, &[1, 2]), (5, &[3, 4]), (6, &[9, 9])]),
            ),
            (2, generation_with_codes(&[(2, &[5, 6]), (4, &[7, 8])])),
        ]
        .into_iter()
        .collect(),
    };
    let full = cache.audio_inputs(&[1], &[0], &[7], false);
    assert_eq!(
        full.iter()
            .map(|input| input.token_index)
            .collect::<Vec<_>>(),
        [3, 5, 6]
    );
    assert_eq!(full[2].codes, [9, 9]);
    let packed = cache.audio_inputs(&[2, 1], &[2, 4], &[3, 3], true);
    assert_eq!(
        packed,
        [
            GeneratedAudioInput {
                batch_index: 0,
                token_index: 0,
                codes: vec![5, 6]
            },
            GeneratedAudioInput {
                batch_index: 0,
                token_index: 2,
                codes: vec![7, 8]
            },
            GeneratedAudioInput {
                batch_index: 0,
                token_index: 4,
                codes: vec![3, 4]
            },
            GeneratedAudioInput {
                batch_index: 0,
                token_index: 5,
                codes: vec![9, 9]
            },
        ]
    );
    let decode = cache.audio_inputs(&[2, 1], &[4, 5], &[1, 1], false);
    assert_eq!(decode[0].batch_index, 0);
    assert_eq!(decode[1].batch_index, 1);
    assert_eq!(decode[1].token_index, 0);
    assert_eq!(decode[1].codes, [3, 4]);
    assert_eq!(cache.audio_inputs(&[1], &[0], &[7], false), full);
}
