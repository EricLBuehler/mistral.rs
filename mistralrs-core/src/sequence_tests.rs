use super::*;
use crate::tools::{
    state::required_tool_call_deadline_tokens, ToolCallFormat, ToolCallState, ToolChoice,
};
use crate::{Function, Tool, ToolType};
use rand::RngCore;
use std::collections::HashMap;
use tokio::sync::mpsc::channel;

#[test]
fn audio_hash_distinguishes_channel_count() {
    let mono = AudioInput {
        samples: vec![0.25, -0.5, 0.75, 1.0],
        sample_rate: 16_000,
        channels: 1,
    };
    let stereo = AudioInput {
        channels: 2,
        ..mono.clone()
    };
    assert_ne!(mono.to_mono(), stereo.to_mono());
    let audios = SequenceAudios::new(vec![mono.clone(), stereo, mono]);
    assert_ne!(audios.hashes()[0], audios.hashes()[1]);
    assert_eq!(audios.hashes()[0], audios.hashes()[2]);
}

#[test]
fn image_hash_distinguishes_geometry() {
    let bytes = vec![1, 2, 3, 4, 5, 6];
    let wide =
        image::DynamicImage::ImageRgb8(image::RgbImage::from_raw(2, 1, bytes.clone()).unwrap());
    let tall = image::DynamicImage::ImageRgb8(image::RgbImage::from_raw(1, 2, bytes).unwrap());

    assert_eq!(wide.as_bytes(), tall.as_bytes());
    let images = SequenceImages::new(vec![wide, tall]);
    assert_ne!(images.hashes()[0], images.hashes()[1]);
}

#[test]
fn image_hash_distinguishes_color_type() {
    let rgba = image::DynamicImage::ImageRgba8(
        image::RgbaImage::from_raw(1, 1, vec![1, 2, 3, 4]).unwrap(),
    );
    let luma_alpha = image::DynamicImage::ImageLumaA16(
        image::ImageBuffer::<image::LumaA<u16>, Vec<u16>>::from_raw(
            1,
            1,
            vec![u16::from_ne_bytes([1, 2]), u16::from_ne_bytes([3, 4])],
        )
        .unwrap(),
    );

    assert_eq!(rgba.as_bytes(), luma_alpha.as_bytes());
    assert_ne!(rgba.color(), luma_alpha.color());
    let images = SequenceImages::new(vec![rgba, luma_alpha]);
    assert_ne!(images.hashes()[0], images.hashes()[1]);
}

fn make_test_sequence() -> Sequence {
    make_test_sequence_with_seed(None)
}

fn test_logprobs(token: u32) -> Logprobs {
    Logprobs {
        token,
        logprob: 0.0,
        bytes: None,
        top_logprobs: None,
    }
}

fn test_logprobs_with_value(token: u32, logprob: f32) -> Logprobs {
    Logprobs {
        token,
        logprob,
        bytes: None,
        top_logprobs: Some(Vec::new()),
    }
}

fn test_usage() -> Usage {
    Usage {
        completion_tokens: 0,
        prompt_tokens: 0,
        total_tokens: 0,
        prompt_tokens_details: None,
        completion_tokens_details: None,
        avg_tok_per_sec: 0.0,
        avg_prompt_tok_per_sec: 0.0,
        avg_compl_tok_per_sec: 0.0,
        total_time_sec: 0.0,
        total_prompt_time_sec: 0.0,
        total_completion_time_sec: 0.0,
    }
}

fn test_streaming_chunk(index: usize, finish_reason: Option<&str>) -> ChunkChoice {
    ChunkChoice {
        finish_reason: finish_reason.map(str::to_string),
        stop_sequence: None,
        index,
        delta: crate::Delta {
            audio: None,
            content: Some(format!("choice {index}")),
            role: "assistant".to_string(),
            tool_calls: None,
            reasoning_content: None,
        },
        logprobs: None,
    }
}

fn make_test_sequence_with_seed(seed: Option<u64>) -> Sequence {
    let (tx, _rx) = channel(1);
    let sampler = Sampler::new(
        None,
        0,
        None,
        None,
        None,
        None,
        None,
        32,
        1.0,
        0.0,
        HashMap::new(),
        vec![],
    )
    .unwrap();
    let group = Arc::new(Mutex::new(SequenceGroup::new(1, false, true, None)));

    Sequence::new_waiting(crate::sequence::SequenceInit {
        tokens: vec![1, 2, 3, 4, 5, 6, 7, 8],
        prompt: "prompt".to_string(),
        id: 0,
        timestamp: 0,
        layers: 0,
        responder: tx,
        sampler,
        stop_tokens: vec![],
        stop_strings: vec![],
        max_len: None,
        return_logprobs: false,
        is_xlora: false,
        group,
        response_index: 0,
        creation_time: 0,
        recognizer: SequenceRecognizer::None,
        suffix: None,
        prefix: None,
        input_images: None,
        input_audios: None,
        input_videos: None,
        tool_call_state: None,
        image_gen_response_format: None,
        sequence_stepping_type: SeqStepType::PromptAndDecode,
        diffusion_params: None,
        image_gen_save_file: None,
        seq_preallocated_cache: None,
        return_raw_logits: false,
        ignore_eos: false,
        eos_tokens: vec![],
        sampling_seed: seed,
    })
}

#[tokio::test]
async fn remaining_choice_streams_after_another_choice_finishes() {
    let (tx, mut rx) = channel(8);
    let group = Arc::new(Mutex::new(SequenceGroup::new(2, true, true, None)));
    let mut first = make_test_sequence();
    first.responder = tx.clone();
    first.response_index = 0;
    first.group = group.clone();
    let mut second = make_test_sequence();
    second.responder = tx;
    second.response_index = 1;
    second.group = group.clone();

    group
        .lock()
        .await
        .chat_streaming_chunks
        .push(test_streaming_chunk(0, None));
    group
        .lock()
        .await
        .maybe_send_streaming_response(&first, "test".to_string(), None)
        .await
        .unwrap();
    assert!(rx.try_recv().is_err());

    group
        .lock()
        .await
        .chat_streaming_chunks
        .push(test_streaming_chunk(1, None));
    group
        .lock()
        .await
        .maybe_send_streaming_response(&second, "test".to_string(), None)
        .await
        .unwrap();
    let Response::Chunk(response) = rx.recv().await.unwrap() else {
        panic!("expected chat chunk");
    };
    assert_eq!(response.choices.len(), 2);

    group
        .lock()
        .await
        .chat_streaming_chunks
        .push(test_streaming_chunk(0, Some("stop")));
    group
        .lock()
        .await
        .maybe_send_streaming_response(&first, "test".to_string(), Some(test_usage()))
        .await
        .unwrap();
    let Response::Chunk(response) = rx.recv().await.unwrap() else {
        panic!("expected first terminal chunk");
    };
    assert_eq!(response.choices[0].index, 0);

    group
        .lock()
        .await
        .chat_streaming_chunks
        .push(test_streaming_chunk(1, Some("stop")));
    group
        .lock()
        .await
        .maybe_send_streaming_response(&second, "test".to_string(), Some(test_usage()))
        .await
        .unwrap();
    let Response::Chunk(response) = rx.recv().await.unwrap() else {
        panic!("expected remaining terminal chunk");
    };
    assert_eq!(response.choices[0].index, 1);
}

#[test]
fn seeded_sequence_owns_its_sampling_stream() {
    let fallback = Arc::new(std::sync::Mutex::new(Isaac64Rng::seed_from_u64(7)));
    let first = make_test_sequence_with_seed(Some(42));
    let second = make_test_sequence_with_seed(Some(42));

    assert!(!Arc::ptr_eq(&first.sampling_rng(&fallback), &fallback));
    assert_eq!(
        first.sampling_rng(&fallback).lock().unwrap().next_u64(),
        second.sampling_rng(&fallback).lock().unwrap().next_u64()
    );

    let unseeded = make_test_sequence();
    assert!(Arc::ptr_eq(&unseeded.sampling_rng(&fallback), &fallback));
}

#[test]
fn prompt_rate_uses_accumulated_chunk_time() {
    let mut seq = make_test_sequence();
    seq.caches.cache.push(None);
    seq.finish_prompt_timing(Duration::from_secs(1));
    seq.finish_prompt_timing(Duration::from_secs(1));

    assert_eq!(seq.timing.total_prompt_time, Some(2_000));
    assert_eq!(seq.timing.prompt_tok_per_sec, 4.0);
}

fn add_test_media(seq: &mut Sequence) {
    seq.multimodal = MultimodalData::new(
        Some(vec![image::DynamicImage::new_rgb8(1, 1)]),
        Some(vec![AudioInput {
            samples: vec![0.0],
            sample_rate: 16_000,
            channels: 1,
        }]),
        Some(vec![VideoInput::from_frames(
            vec![image::DynamicImage::new_rgb8(1, 1)],
            24.0,
            None,
        )]),
        None,
        None,
        None,
    );
}

#[test]
fn ignore_eos_preserves_explicit_stops_and_length_limit() {
    let eos_tokens = [42];
    let mut seq = make_test_sequence();
    seq.ignore_eos = true;

    assert_eq!(seq.effective_eos_tokens(&eos_tokens, false), None);
    assert_eq!(seq.is_done(42, Some(&eos_tokens), 1024), None);

    seq.stop_tokens.push(42);
    assert_eq!(
        seq.is_done(42, Some(&eos_tokens), 1024),
        Some(StopReason::StopTok(42))
    );

    seq.stop_tokens.clear();
    seq.stop_strings.push("done".to_string());
    assert!(matches!(
        seq.add_token(test_logprobs(7), b"done".to_vec(), None),
        Some(StopReason::StopString { .. })
    ));

    seq.stop_strings.clear();
    seq.max_len = Some(1);
    assert_eq!(
        seq.is_done(7, Some(&eos_tokens), 1024),
        Some(StopReason::Length(1))
    );
}

#[test]
fn constrained_token_stop_string_is_removed_before_final_output() {
    let mut seq = make_test_sequence();
    seq.stop_strings.push("STOP_BOUNDARY".to_string());

    let reason = seq.add_token(test_logprobs(7), b"GREEN STOP_BOUNDARY RED".to_vec(), None);

    assert!(matches!(reason, Some(StopReason::StopString { .. })));
    assert_eq!(seq.get_final_delta(), "GREEN ");
    assert!(!String::from_utf8_lossy(seq.completion_bytes()).contains("STOP_BOUNDARY"));
}

#[test]
fn reasoning_parser_only_observes_visible_stop_filtered_bytes() {
    let mut seq = make_test_sequence();
    seq.stop_strings.push("STOP_BOUNDARY".to_string());
    seq.enable_reasoning(crate::reasoning_parsers::TagReasoningContext::new_think_tags());

    let reason = seq.add_token(
        test_logprobs(7),
        b"visible STOP_BOUNDARY leaked".to_vec(),
        None,
    );
    seq.finalize_reasoning();

    assert!(matches!(reason, Some(StopReason::StopString { .. })));
    assert_eq!(seq.get_response_content().as_deref(), Some("visible "));
    assert_eq!(seq.get_final_delta(), "visible ");
}

#[test]
fn reasoning_usage_distinguishes_untracked_formats_from_zero_tokens() {
    let mut seq = make_test_sequence();
    seq.caches.cache.push(None);
    seq.tool_call_state = Some(
        ToolCallState::new(
            crate::tools::ToolChoice::Auto,
            None,
            Some(crate::tools::ToolCallFormat::Atem),
        )
        .unwrap(),
    );
    seq.add_token(
        test_logprobs(0),
        b" to=self<|message|>thinking".to_vec(),
        None,
    );
    assert!(seq.has_reasoning_state());
    seq.update_time_info();
    assert!(get_mut_group!(seq)
        .get_usage()
        .completion_tokens_details
        .is_none());

    let mut seq = make_test_sequence();
    seq.enable_reasoning(TagReasoningContext::new_think_tags());
    seq.caches.cache.push(None);
    seq.update_time_info();
    assert_eq!(
        get_mut_group!(seq)
            .get_usage()
            .completion_tokens_details
            .unwrap()
            .reasoning_tokens,
        0,
    );
}

#[test]
fn cross_token_stop_string_is_never_streamed() {
    let mut seq = make_test_sequence();
    seq.stop_strings.push("STOP_BOUNDARY".to_string());

    assert_eq!(
        seq.add_token(test_logprobs(7), b"GREEN STO".to_vec(), None),
        None
    );
    assert_eq!(seq.get_delta().unwrap(), Some("GREEN ".to_string()));

    let reason = seq.add_token(test_logprobs(8), b"P_BOUNDARY RED".to_vec(), None);
    assert!(matches!(reason, Some(StopReason::StopString { .. })));
    assert_eq!(seq.get_final_delta(), "");
    assert!(!String::from_utf8_lossy(seq.completion_bytes()).contains("STOP_BOUNDARY"));
}

#[test]
fn unmatched_stop_prefix_is_flushed_on_eos() {
    let mut seq = make_test_sequence();
    seq.stop_strings.push("STOP_BOUNDARY".to_string());

    assert_eq!(
        seq.add_token(test_logprobs(7), b"GREEN STO".to_vec(), None),
        None
    );
    assert_eq!(seq.get_delta().unwrap(), Some("GREEN ".to_string()));
    assert_eq!(
        seq.add_token(test_logprobs(42), b"<eos>".to_vec(), Some(StopReason::Eos)),
        Some(StopReason::Eos)
    );
    assert_eq!(seq.get_final_delta(), "STO");
    assert!(!String::from_utf8_lossy(seq.completion_bytes()).contains("<eos>"));
}

#[test]
fn stop_token_id_still_excludes_its_decoded_bytes() {
    let mut seq = make_test_sequence();

    assert_eq!(
        seq.add_token(
            test_logprobs(42),
            b"STOP_BOUNDARY".to_vec(),
            Some(StopReason::StopTok(42)),
        ),
        Some(StopReason::StopTok(42))
    );
    assert!(seq.completion_bytes().is_empty());
}

#[test]
fn earliest_stop_string_wins_independent_of_configuration_order() {
    let mut seq = make_test_sequence();
    seq.stop_strings = vec!["RED".to_string(), "STOP".to_string()];

    let reason = seq.add_token(test_logprobs(7), b"GREEN STOP RED".to_vec(), None);

    assert_eq!(
        reason,
        Some(StopReason::StopString {
            stop_string_idx: 1,
            completion_bytes_pos: 6,
        })
    );
    assert_eq!(seq.get_final_delta(), "GREEN ");
}

#[test]
fn stop_string_on_length_boundary_reports_stop() {
    let mut seq = make_test_sequence();
    seq.stop_strings.push("STOP_BOUNDARY".to_string());

    let reason = seq.add_token(
        test_logprobs(7),
        b"GREEN STOP_BOUNDARY RED".to_vec(),
        Some(StopReason::Length(1)),
    );

    assert!(matches!(reason, Some(StopReason::StopString { .. })));
    assert_eq!(reason.unwrap().to_string(), "stop");
    assert_eq!(seq.get_final_delta(), "GREEN ");
}

#[test]
fn tool_parser_observes_a_stop_delimiter_hidden_from_output() {
    let mut seq = make_test_sequence();
    let tools = vec![weather_tool()];
    seq.stop_strings.push("<|eot|>".to_string());
    seq.tool_call_state = Some(
        ToolCallState::new(ToolChoice::Auto, Some(&tools), Some(ToolCallFormat::Atem)).unwrap(),
    );
    let output = b" to=get_weather<|message|><atem:function_calls><atem:invoke name=\"get_weather\"></atem:invoke></atem:function_calls><|eot|>";

    let reason = seq.add_token(test_logprobs(7), output.to_vec(), None);
    let parsed = seq
        .tool_call_state
        .as_mut()
        .unwrap()
        .finalize_for_response("", None, None, None)
        .unwrap();

    assert!(matches!(reason, Some(StopReason::StopString { .. })));
    assert!(!String::from_utf8_lossy(seq.completion_bytes()).contains("<|eot|>"));
    assert_eq!(parsed.tool_calls.len(), 1);
    assert_eq!(parsed.tool_calls[0].function.name, "get_weather");
}

#[test]
fn post_add_terminal_transition_flushes_an_unmatched_stop_prefix() {
    let mut seq = make_test_sequence();
    seq.stop_strings.push("STOP_BOUNDARY".to_string());

    assert_eq!(
        seq.add_token(test_logprobs(7), b"GREEN STO".to_vec(), None),
        None
    );
    seq.flush_stop_pending_bytes();

    assert_eq!(seq.get_final_delta(), "GREEN STO");
}

#[test]
fn delayed_stop_prefix_emissions_preserve_source_logprobs() {
    let mut seq = make_test_sequence();
    seq.return_logprobs = true;
    seq.stream_logprobs = true;
    seq.stop_strings.push("STOP_BOUNDARY".to_string());

    assert_eq!(
        seq.add_token(test_logprobs_with_value(11, -1.0), b"STO".to_vec(), None,),
        None
    );
    assert!(seq.take_ready_streaming_emissions(false).is_empty());
    assert_eq!(
        seq.add_token(test_logprobs_with_value(12, -2.0), b"X".to_vec(), None,),
        None
    );

    let emissions = seq.take_ready_streaming_emissions(false);
    assert_eq!(emissions.len(), 2);
    assert_eq!(emissions[0].text, "STO");
    assert_eq!(emissions[0].bytes, b"STO");
    assert_eq!(emissions[0].logprobs.token, 11);
    assert_eq!(emissions[0].logprobs.logprob, -1.0);
    assert_eq!(emissions[1].text, "X");
    assert_eq!(emissions[1].bytes, b"X");
    assert_eq!(emissions[1].logprobs.token, 12);
    assert_eq!(emissions[1].logprobs.logprob, -2.0);
}

#[test]
fn split_utf8_streaming_emissions_preserve_text_and_source_logprobs() {
    let mut seq = make_test_sequence();
    seq.return_logprobs = true;
    seq.stream_logprobs = true;

    assert_eq!(
        seq.add_token(test_logprobs_with_value(11, -1.0), vec![0xc3], None),
        None
    );
    assert!(seq.take_ready_streaming_emissions(false).is_empty());
    assert_eq!(
        seq.add_token(test_logprobs_with_value(12, -2.0), vec![0xa9], None),
        None
    );

    let emissions = seq.take_ready_streaming_emissions(false);
    assert_eq!(emissions.len(), 2);
    assert_eq!(
        emissions
            .iter()
            .map(|emission| emission.text.as_str())
            .collect::<String>(),
        "é"
    );
    assert_eq!(emissions[0].text, "");
    assert_eq!(emissions[0].bytes, [0xc3]);
    assert_eq!(emissions[0].logprobs.token, 11);
    assert_eq!(emissions[0].logprobs.logprob, -1.0);
    assert_eq!(emissions[1].text, "é");
    assert_eq!(emissions[1].bytes, [0xa9]);
    assert_eq!(emissions[1].logprobs.token, 12);
    assert_eq!(emissions[1].logprobs.logprob, -2.0);
}

#[test]
fn engine_eos_override_applies_to_default_sequence() {
    let eos_tokens = [42];
    let seq = make_test_sequence();

    assert_eq!(
        seq.effective_eos_tokens(&eos_tokens, false),
        Some(eos_tokens.as_slice())
    );
    assert_eq!(seq.effective_eos_tokens(&eos_tokens, true), None);
}

fn weather_tool() -> Tool {
    Tool {
        tp: ToolType::Function,
        function: Function {
            description: None,
            name: "get_weather".to_string(),
            parameters: None,
            strict: None,
        },
    }
}

fn add_required_tool(seq: &mut Sequence) {
    let tool = weather_tool();
    seq.tool_call_state =
        Some(ToolCallState::new(ToolChoice::Required, Some(&[tool]), None).unwrap());
}

fn required_tool_call_should_force(seq: &mut Sequence, max_model_len: usize) -> bool {
    let generated = seq.generated_len();
    let max_generation_len = seq.max_generation_len(max_model_len);
    let (_, remaining, _) =
        ToolCallState::required_tool_call_deadline_status(generated, max_generation_len);
    seq.tool_call_state
        .as_mut()
        .and_then(|state| state.maybe_force_required_grammar(remaining, max_generation_len, false))
        .is_some()
}

#[test]
fn required_tool_call_deadline_clamps() {
    assert_eq!(required_tool_call_deadline_tokens(512), 1024);
    assert_eq!(required_tool_call_deadline_tokens(8192), 2048);
    assert_eq!(required_tool_call_deadline_tokens(32768), 4096);
}

fn completion_whitespace_sequence(streaming: bool, logprobs: bool) -> Sequence {
    let mut seq = make_test_sequence();
    seq.caches.cache.push(None);
    seq.get_mut_group().is_chat = false;
    seq.get_mut_group().is_streaming = streaming;
    seq.return_logprobs = logprobs;
    seq.stream_logprobs = streaming && logprobs;
    seq
}

#[test]
fn completion_whitespace_final_preserves_echo_and_usage() {
    const PREFIX: &str = "def answer():\n";
    const SUFFIX: &str = "\n# done";
    const BODY: &str = "return 1\n";
    for whitespace in ["    ", "\n\n", "\t", " ", "\u{2003}"] {
        let mut seq = completion_whitespace_sequence(false, false);
        let prompt_tokens = seq.prompt_tokens();
        seq.prefix = Some(PREFIX.to_string());
        seq.suffix = Some(SUFFIX.to_string());
        assert_eq!(
            seq.add_token(test_logprobs(11), whitespace.as_bytes().to_vec(), None),
            None
        );
        assert_eq!(
            seq.add_token(
                test_logprobs(12),
                BODY.as_bytes().to_vec(),
                Some(StopReason::Length(2)),
            ),
            Some(StopReason::Length(2))
        );

        let expected = format!("{whitespace}{BODY}");
        assert_eq!(seq.completion_text(), expected);
        assert_eq!(seq.get_final_delta(), expected);
        assert_eq!(seq.get_final_delta(), "");
        seq.add_completion_choice_to_group(CompletionChoice {
            finish_reason: "length".to_string(),
            index: 0,
            text: seq.completion_text(),
            logprobs: None,
        });

        let group = seq.get_mut_group();
        assert_eq!(
            group.get_completion_choices()[0].text,
            format!("{PREFIX}{expected}{SUFFIX}")
        );
        let usage = group.get_usage();
        assert_eq!(usage.prompt_tokens, prompt_tokens);
        assert_eq!(usage.completion_tokens, 2);
        assert_eq!(usage.total_tokens, prompt_tokens + 2);
        assert_eq!(seq.logprobs().len(), 2);
    }
}

#[test]
fn completion_whitespace_streaming_preserves_text_and_token_boundaries() {
    for logprobs in [false, true] {
        for leading in ["    ", "\n\n    "] {
            let mut seq = completion_whitespace_sequence(true, logprobs);
            let pieces = [leading, "if True:\n", "        return 1\n"];
            let mut text = String::new();
            for (idx, piece) in pieces.iter().enumerate() {
                let token = u32::try_from(idx).unwrap();
                let finalize = idx + 1 == pieces.len();
                let reason = finalize.then_some(StopReason::Length(pieces.len()));
                assert_eq!(
                    seq.add_token(
                        test_logprobs_with_value(token, -1.0),
                        piece.as_bytes().to_vec(),
                        reason,
                    ),
                    reason
                );
                let delta = if logprobs {
                    let emissions = seq.take_ready_streaming_emissions(finalize);
                    assert_eq!(emissions.len(), 1);
                    assert_eq!(emissions[0].bytes, piece.as_bytes());
                    assert_eq!(emissions[0].logprobs.token, token);
                    assert_eq!(emissions[0].logprobs.logprob, -1.0);
                    emissions[0].text.clone()
                } else if finalize {
                    seq.get_final_delta()
                } else {
                    assert_eq!(seq.peek_delta().unwrap().as_deref(), Some(*piece));
                    seq.get_delta().unwrap().unwrap()
                };
                assert_eq!(delta, *piece);
                text.push_str(&delta);
                seq.add_streaming_completion_chunk_choice_to_group(CompletionChunkChoice {
                    text: delta,
                    index: 0,
                    finish_reason: finalize.then(|| "length".to_string()),
                    logprobs: None,
                });
            }
            assert_eq!(text, pieces.concat());
            assert_eq!(text, seq.completion_text());
            let group = seq.get_mut_group();
            assert_eq!(
                group
                    .completion_streaming_chunks
                    .iter()
                    .map(|chunk| chunk.text.as_str())
                    .collect::<String>(),
                text
            );
            assert_eq!(group.get_usage().completion_tokens, pieces.len());
            assert_eq!(seq.logprobs().len(), pieces.len());
        }
    }
}

#[test]
fn completion_whitespace_chat_keeps_existing_initial_trim() {
    const FIRST: &str = "\n    hello";
    const NEXT: &str = " world\n";
    let mut final_seq = completion_whitespace_sequence(false, false);
    final_seq.get_mut_group().is_chat = true;
    assert_eq!(
        final_seq.add_token(test_logprobs(11), FIRST.as_bytes().to_vec(), None),
        None
    );
    assert_eq!(final_seq.completion_text(), "hello");
    assert_eq!(final_seq.get_final_delta(), "hello");
    for logprobs in [false, true] {
        let mut seq = completion_whitespace_sequence(true, logprobs);
        seq.get_mut_group().is_chat = true;
        assert_eq!(
            seq.add_token(test_logprobs(11), FIRST.as_bytes().to_vec(), None),
            None
        );
        let first = if logprobs {
            seq.take_ready_streaming_emissions(false)[0].text.clone()
        } else {
            seq.get_delta().unwrap().unwrap()
        };
        assert_eq!(first, "hello");
        assert_eq!(
            seq.add_token(
                test_logprobs(12),
                NEXT.as_bytes().to_vec(),
                Some(StopReason::Length(2)),
            ),
            Some(StopReason::Length(2))
        );
        let next = if logprobs {
            seq.take_ready_streaming_emissions(true)[0].text.clone()
        } else {
            seq.get_final_delta()
        };
        assert_eq!(next, NEXT);
        assert_eq!(seq.completion_text(), format!("hello{NEXT}"));
    }
}

#[test]
fn final_delta_flushes_incomplete_utf8() {
    let mut seq = make_test_sequence();
    seq.stream.completion_bytes = vec![b'a', 0xe2, 0x82];

    assert_eq!(seq.get_delta().unwrap(), Some("a".to_string()));
    assert_eq!(seq.stream.stream_idx, 1);
    assert_eq!(seq.get_final_delta(), "\u{fffd}");
    assert_eq!(seq.stream.stream_idx, seq.stream.completion_bytes.len());
}

#[test]
fn replacement_character_is_not_buffered() {
    let mut seq = make_test_sequence();
    seq.stream.completion_bytes = "\u{fffd}".as_bytes().to_vec();

    assert_eq!(seq.get_delta().unwrap(), Some("\u{fffd}".to_string()));
    assert_eq!(seq.stream.stream_idx, seq.stream.completion_bytes.len());
}

#[test]
fn invalid_bytes_do_not_consume_a_trailing_incomplete_character() {
    let mut seq = make_test_sequence();
    seq.stream.completion_bytes = vec![0xff, 0xe2];

    assert_eq!(seq.get_delta().unwrap(), Some("\u{fffd}".to_string()));
    assert_eq!(seq.stream.stream_idx, 1);

    seq.stream.completion_bytes.extend([0x82, 0xac]);
    assert_eq!(seq.get_delta().unwrap(), Some("\u{20ac}".to_string()));
    assert_eq!(seq.stream.stream_idx, seq.stream.completion_bytes.len());
}

#[test]
fn prefix_cache_hit_is_recorded_once_for_generation_sequences() {
    let mut seq = make_test_sequence();
    assert!(seq.record_prefix_cache_hit());
    assert!(!seq.record_prefix_cache_hit());

    seq.sequence_stepping_type = SeqStepType::OneShot;
    seq.prefill.prefix_cache_hit_recorded = false;
    assert!(!seq.record_prefix_cache_hit());
}

#[cfg(feature = "cuda")]
#[test]
fn sampling_logprob_is_retained_for_multi_choice_groups() {
    assert!(!SequenceGroup::new(1, false, true, None).sampling_logprob_required());
    assert!(SequenceGroup::new(2, false, true, None).sampling_logprob_required());
    assert!(!SequenceGroup::new(1, false, false, Some(1)).sampling_logprob_required());
    assert!(SequenceGroup::new(1, false, false, Some(2)).sampling_logprob_required());
}

#[test]
fn required_tool_call_forces_immediately_when_max_tokens_is_below_deadline() {
    let mut seq = make_test_sequence();
    add_required_tool(&mut seq);
    seq.set_max_len(512);

    assert!(required_tool_call_should_force(&mut seq, 8192));
}

#[test]
fn required_tool_call_forces_at_remaining_deadline() {
    let mut seq = make_test_sequence();
    add_required_tool(&mut seq);
    seq.set_max_len(2048);

    assert!(!required_tool_call_should_force(&mut seq, 8192));
    seq.tokens.extend(std::iter::repeat_n(1, 1024));
    assert!(required_tool_call_should_force(&mut seq, 8192));
}

#[test]
fn chunk_media_view_does_not_double_count_prefix_items() {
    let mut seq = make_test_sequence();
    seq.set_mm_features(vec![
        MultiModalFeature {
            kind: MultimodalKind::Image,
            item_range: 0..1,
            hashes: vec![123],
            offset: 0,
            length: 3,
            attention_policy: MultimodalAttentionPolicy::Causal,
            splittable: false,
        },
        MultiModalFeature {
            kind: MultimodalKind::Image,
            item_range: 1..2,
            hashes: vec![456],
            offset: 4,
            length: 3,
            attention_policy: MultimodalAttentionPolicy::Causal,
            splittable: false,
        },
        MultiModalFeature {
            kind: MultimodalKind::Audio,
            item_range: 0..1,
            hashes: vec![789],
            offset: 7,
            length: 1,
            attention_policy: MultimodalAttentionPolicy::Causal,
            splittable: false,
        },
    ]);

    let seq = seq.prefill_v2_normal(vec![], vec![7, 8], 4);

    assert_eq!(seq.prefix_cache_len(), 4);
    assert_eq!(
        seq.count_prefix_cached_mm_items_by_kind(MultimodalKind::Image),
        0
    );
    assert_eq!(
        seq.count_prefix_cached_mm_items_by_kind(MultimodalKind::Audio),
        0
    );
    assert_eq!(seq.count_prefix_cached_mm_items(), 0);
}

#[test]
fn usage_reports_cached_prompt_tokens() {
    let seq = make_test_sequence();
    let seq = seq.prefill_v2_normal(vec![], vec![7, 8], 4);
    seq.update_time_info();
    let usage = seq.get_mut_group().get_usage();
    assert_eq!(
        usage.prompt_tokens_details.map(|d| d.cached_tokens),
        Some(4)
    );
}

#[test]
fn usage_omits_prompt_details_without_prefix_cache() {
    let seq = make_test_sequence();
    let mut seq = seq.prefill_v2_normal(vec![], vec![7, 8], 0);
    seq.set_num_computed_tokens(seq.len());
    seq.update_time_info();
    assert!(seq
        .get_mut_group()
        .get_usage()
        .prompt_tokens_details
        .is_none());
}

#[test]
fn usage_does_not_treat_prefill_progress_as_cached_tokens() {
    let seq = make_test_sequence();
    let mut seq = seq.prefill_v2_normal(vec![], vec![1, 2, 3, 4, 5, 6, 7, 8], 4);
    seq.set_num_computed_tokens(seq.len());
    seq.update_time_info();

    let usage = seq.get_mut_group().get_usage();
    assert_eq!(
        usage
            .prompt_tokens_details
            .map(|details| details.cached_tokens),
        Some(4)
    );
}

#[test]
fn non_chunk_media_view_counts_prefix_items() {
    let mut seq = make_test_sequence();
    seq.set_mm_features(vec![MultiModalFeature {
        kind: MultimodalKind::Image,
        item_range: 0..1,
        hashes: vec![123],
        offset: 0,
        length: 3,
        attention_policy: MultimodalAttentionPolicy::Causal,
        splittable: false,
    }]);
    seq.set_prefix_cache_len(4);

    assert_eq!(seq.count_prefix_cached_mm_items(), 1);
    assert_eq!(
        seq.count_prefix_cached_mm_items_by_kind(MultimodalKind::Image),
        1
    );
}

#[test]
fn multimodal_prefix_placeholder_delimited_ranges_include_wrappers() {
    let tokens = vec![1, 10, 20, 20, 11, 2, 10, 30, 30, 30, 11, 3];
    let img = find_placeholder_delimited_ranges(&tokens, 20, 10, 11);
    let video = find_placeholder_delimited_ranges(&tokens, 30, 10, 11);
    let fallback = find_placeholder_delimited_ranges(&tokens, 2, 99, 100);

    assert_eq!(img, vec![(1, 4)]);
    assert_eq!(video, vec![(6, 5)]);
    assert_eq!(fallback, vec![(5, 1)]);
}

#[test]
fn multimodal_prefix_cache_len_clamps_inside_feature() {
    let features = vec![MultiModalFeature {
        kind: MultimodalKind::Image,
        item_range: 0..1,
        hashes: vec![123],
        offset: 31,
        length: 4,
        attention_policy: MultimodalAttentionPolicy::NonCausal,
        splittable: false,
    }];

    assert_eq!(clamp_prefix_cache_len_for_mm_features(32, 32, &features), 0);
    assert_eq!(
        clamp_prefix_cache_len_for_mm_features(35, 32, &features),
        35
    );
    assert_eq!(
        clamp_prefix_cache_len_for_mm_features(64, 32, &features),
        64
    );
}

#[test]
fn multimodal_prefix_cache_len_clamps_to_a_fixed_point() {
    let features = vec![
        MultiModalFeature {
            kind: MultimodalKind::Image,
            item_range: 0..1,
            hashes: vec![1],
            offset: 16,
            length: 24,
            attention_policy: MultimodalAttentionPolicy::Causal,
            splittable: false,
        },
        MultiModalFeature {
            kind: MultimodalKind::Image,
            item_range: 1..2,
            hashes: vec![2],
            offset: 40,
            length: 20,
            attention_policy: MultimodalAttentionPolicy::Causal,
            splittable: false,
        },
    ];

    assert_eq!(
        clamp_prefix_cache_len_for_mm_features(50, 16, &features),
        16
    );
}

#[test]
fn unsplittable_causal_feature_clips_sequence_prefix_cache() {
    let mut seq = make_test_sequence();
    seq.set_mm_features(vec![MultiModalFeature {
        kind: MultimodalKind::Image,
        item_range: 0..1,
        hashes: vec![123],
        offset: 31,
        length: 4,
        attention_policy: MultimodalAttentionPolicy::Causal,
        splittable: false,
    }]);
    seq.set_prefix_cache_len(32);

    seq.clip_prefix_cache_len_for_mm_features(32);

    assert_eq!(seq.prefix_cache_len(), 0);
}

#[test]
fn noncausal_prefix_clipping_keeps_its_original_scope() {
    let mut seq = make_test_sequence();
    seq.set_mm_features(vec![MultiModalFeature {
        kind: MultimodalKind::Image,
        item_range: 0..1,
        hashes: vec![123],
        offset: 31,
        length: 4,
        attention_policy: MultimodalAttentionPolicy::Causal,
        splittable: false,
    }]);
    seq.set_prefix_cache_len(32);

    seq.clip_prefix_cache_len_for_non_causal_mm_features(32);

    assert_eq!(seq.prefix_cache_len(), 32);
}

#[test]
fn prefix_clipping_reaches_a_fixed_point_across_adjacent_features() {
    let mut seq = make_test_sequence();
    seq.set_mm_features(vec![
        MultiModalFeature {
            kind: MultimodalKind::Image,
            item_range: 0..1,
            hashes: vec![1],
            offset: 16,
            length: 24,
            attention_policy: MultimodalAttentionPolicy::Causal,
            splittable: false,
        },
        MultiModalFeature {
            kind: MultimodalKind::Image,
            item_range: 1..2,
            hashes: vec![2],
            offset: 40,
            length: 20,
            attention_policy: MultimodalAttentionPolicy::Causal,
            splittable: false,
        },
    ]);
    seq.set_prefix_cache_len(50);

    seq.clip_prefix_cache_len_for_mm_features(16);

    assert_eq!(seq.prefix_cache_len(), 16);
}

#[test]
fn chunk_media_accessors_expose_only_the_active_kind() {
    let mut seq = make_test_sequence();
    add_test_media(&mut seq);
    seq.set_mm_features(vec![
        MultiModalFeature {
            kind: MultimodalKind::Image,
            item_range: 0..1,
            hashes: vec![1],
            offset: 0,
            length: 2,
            attention_policy: MultimodalAttentionPolicy::Causal,
            splittable: false,
        },
        MultiModalFeature {
            kind: MultimodalKind::Audio,
            item_range: 0..1,
            hashes: vec![2],
            offset: 4,
            length: 1,
            attention_policy: MultimodalAttentionPolicy::Causal,
            splittable: false,
        },
        MultiModalFeature {
            kind: MultimodalKind::Video,
            item_range: 0..1,
            hashes: vec![3],
            offset: 6,
            length: 1,
            attention_policy: MultimodalAttentionPolicy::Causal,
            splittable: false,
        },
    ]);
    seq.set_prefix_cache_len(4);
    seq.set_prefill_toks(vec![1, 2, 3, 4, 5]);

    assert!(!seq.has_images());
    assert!(seq.image_hashes().unwrap().is_empty());
    assert!(seq.take_images().is_none());
    assert!(seq.has_audios());
    assert_eq!(seq.audio_hashes().unwrap().len(), 1);
    assert_eq!(seq.take_audios().unwrap().len(), 1);
    assert!(!seq.has_videos());
    assert!(seq.video_hashes().unwrap().is_empty());
    assert!(seq.take_videos().is_none());
}

#[test]
fn chunk_media_accessors_use_active_local_item_offsets() {
    let mut seq = make_test_sequence();
    seq.multimodal = MultimodalData::new(
        Some(vec![
            image::DynamicImage::new_rgb8(1, 1),
            image::DynamicImage::new_rgb8(2, 1),
        ]),
        None,
        None,
        None,
        None,
        None,
    );
    seq.set_mm_features(vec![
        MultiModalFeature {
            kind: MultimodalKind::Image,
            item_range: 0..1,
            hashes: vec![1],
            offset: 0,
            length: 2,
            attention_policy: MultimodalAttentionPolicy::Causal,
            splittable: false,
        },
        MultiModalFeature {
            kind: MultimodalKind::Image,
            item_range: 1..2,
            hashes: vec![2],
            offset: 4,
            length: 2,
            attention_policy: MultimodalAttentionPolicy::Causal,
            splittable: false,
        },
    ]);
    seq.set_prefix_cache_len(4);
    seq.set_prefill_toks(vec![1, 2, 3, 4, 5, 6]);

    assert_eq!(seq.active_prompt_query_range(), Some(4..6));
    assert_eq!(seq.active_prompt_local_query_range(), Some(4..6));
    assert_eq!(seq.prompt_position_source_toks(), &[1, 2, 3, 4, 5, 6]);
    assert_eq!(
        seq.active_multimodal_item_range(MultimodalKind::Image),
        Some(1..2)
    );
    assert_eq!(
        seq.active_local_multimodal_item_range(MultimodalKind::Image, 2),
        Some(1..2)
    );
    assert_eq!(seq.image_hashes().unwrap().len(), 1);
    assert_eq!(seq.take_images().unwrap()[0].width(), 2);
    assert_eq!(seq.count_prefix_cached_mm_items(), 0);
}

#[test]
fn suffix_only_prefill_view_survives_scheduler_state_transition() {
    let mut seq = make_test_sequence();
    seq.multimodal = MultimodalData::new(
        Some(vec![
            image::DynamicImage::new_rgb8(1, 1),
            image::DynamicImage::new_rgb8(2, 1),
        ]),
        Some(vec![
            AudioInput {
                samples: vec![1.0],
                sample_rate: 16_000,
                channels: 1,
            },
            AudioInput {
                samples: vec![2.0],
                sample_rate: 16_000,
                channels: 1,
            },
        ]),
        Some(vec![
            VideoInput::from_frames(vec![image::DynamicImage::new_rgb8(1, 1)], 24.0, None),
            VideoInput::from_frames(vec![image::DynamicImage::new_rgb8(2, 1)], 24.0, None),
        ]),
        None,
        None,
        None,
    );
    seq.set_mm_features(vec![
        MultiModalFeature {
            kind: MultimodalKind::Image,
            item_range: 0..1,
            hashes: vec![1],
            offset: 0,
            length: 1,
            attention_policy: MultimodalAttentionPolicy::Causal,
            splittable: false,
        },
        MultiModalFeature {
            kind: MultimodalKind::Audio,
            item_range: 0..1,
            hashes: vec![2],
            offset: 1,
            length: 1,
            attention_policy: MultimodalAttentionPolicy::Causal,
            splittable: false,
        },
        MultiModalFeature {
            kind: MultimodalKind::Video,
            item_range: 0..1,
            hashes: vec![3],
            offset: 2,
            length: 1,
            attention_policy: MultimodalAttentionPolicy::Causal,
            splittable: false,
        },
        MultiModalFeature {
            kind: MultimodalKind::Image,
            item_range: 1..2,
            hashes: vec![4],
            offset: 3,
            length: 1,
            attention_policy: MultimodalAttentionPolicy::Causal,
            splittable: false,
        },
        MultiModalFeature {
            kind: MultimodalKind::Audio,
            item_range: 1..2,
            hashes: vec![5],
            offset: 4,
            length: 1,
            attention_policy: MultimodalAttentionPolicy::Causal,
            splittable: false,
        },
        MultiModalFeature {
            kind: MultimodalKind::Video,
            item_range: 1..2,
            hashes: vec![6],
            offset: 5,
            length: 1,
            attention_policy: MultimodalAttentionPolicy::Causal,
            splittable: false,
        },
    ]);
    seq.keep_num_images(1);
    seq.keep_num_audios(1);
    seq.keep_num_video_frames(1);
    let mut seq = seq.prefill_v2_normal(vec![], vec![4, 5, 6], 3);
    seq.set_state(SequenceState::Waiting);
    seq.set_state(SequenceState::RunningPrompt);

    assert!(seq.has_suffix_only_prefill_toks());
    assert_eq!(seq.active_prompt_query_range(), Some(3..6));
    assert_eq!(seq.active_prompt_local_query_range(), Some(0..3));
    assert_eq!(seq.prompt_position_source_toks(), &[1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(
        seq.active_multimodal_item_range(MultimodalKind::Image),
        Some(1..2)
    );
    assert_eq!(
        seq.active_local_multimodal_item_range(MultimodalKind::Image, 1),
        Some(0..1)
    );
    assert_eq!(seq.image_hashes().unwrap().len(), 1);
    assert_eq!(seq.audio_hashes().unwrap().len(), 1);
    assert_eq!(seq.video_hashes().unwrap().len(), 1);
    assert_eq!(seq.take_images().unwrap()[0].width(), 2);
    assert_eq!(seq.take_audios().unwrap()[0].samples, vec![2.0]);
    assert_eq!(seq.take_videos().unwrap()[0].frames[0].width(), 2);
    assert_eq!(seq.count_prefix_cached_mm_items(), 0);

    seq.reset_prefill_toks();
    assert!(!seq.has_suffix_only_prefill_toks());
    assert_eq!(seq.active_prompt_query_range(), None);
    assert!(!seq.is_chunked_prefill_view());
}

#[test]
fn video_prefix_retention_uses_frame_counts() {
    let mut seq = make_test_sequence();
    seq.multimodal = MultimodalData::new(
        None,
        None,
        Some(vec![VideoInput::from_frames(
            vec![
                image::DynamicImage::new_rgb8(1, 1),
                image::DynamicImage::new_rgb8(2, 1),
                image::DynamicImage::new_rgb8(3, 1),
            ],
            24.0,
            None,
        )]),
        None,
        None,
        None,
    );

    seq.keep_num_video_frames(2);

    let videos = seq.clone_videos().unwrap();
    assert_eq!(videos.len(), 1);
    assert_eq!(videos[0].frames.len(), 2);
    assert_eq!(videos[0].frames[0].width(), 2);
    assert_eq!(seq.multimodal.video_hashes().unwrap().len(), 3);
}

#[test]
fn video_retention_by_item_keeps_complete_videos() {
    let mut seq = make_test_sequence();
    seq.multimodal = MultimodalData::new(
        None,
        None,
        Some(vec![
            VideoInput::from_frames(vec![image::DynamicImage::new_rgb8(1, 1)], 24.0, None),
            VideoInput::from_frames(
                vec![
                    image::DynamicImage::new_rgb8(2, 1),
                    image::DynamicImage::new_rgb8(3, 1),
                ],
                24.0,
                None,
            ),
        ]),
        None,
        None,
        None,
    );

    seq.keep_num_videos(1);

    let videos = seq.clone_videos().unwrap();
    assert_eq!(videos.len(), 1);
    assert_eq!(videos[0].frames.len(), 2);
    assert_eq!(videos[0].frames[0].width(), 2);
}

#[test]
fn non_chunk_media_accessors_preserve_full_media() {
    let mut seq = make_test_sequence();
    add_test_media(&mut seq);

    assert_eq!(seq.image_hashes().unwrap().len(), 1);
    assert_eq!(seq.take_images().unwrap().len(), 1);
    assert_eq!(seq.audio_hashes().unwrap().len(), 1);
    assert_eq!(seq.take_audios().unwrap().len(), 1);
    assert_eq!(seq.video_hashes().unwrap().len(), 1);
    assert_eq!(seq.take_videos().unwrap().len(), 1);
}
