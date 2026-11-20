use super::*;
use crate::{
    paged_attention::{block_hash::MultimodalKind, PagedCacheType},
    sampler::{Logprobs, Sampler},
    scheduler::IMAGE_MODALITY,
    sequence::{SeqStepType, SequenceGroup, SequenceRecognizer},
    speculative::SpeculativePrefixReplay,
    AudioInput, VideoInput,
};
use tokio::sync::{mpsc::channel, Mutex as TokioMutex};

fn test_scheduler() -> PagedAttentionScheduler {
    PagedAttentionScheduler::new(
        PagedAttentionSchedulerConfig {
            max_num_seqs: 8,
            max_num_batched_tokens: 4096,
            max_prefill_chunk_tokens: 4096,
            max_decode_steps_before_prefill: 8,
        },
        CacheConfig {
            block_size: 8,
            num_gpu_blocks: 128,
            cache_type: PagedCacheType::Auto,
            kv_cache_group_ids: vec![0],
        },
    )
}

#[derive(Default)]
struct RecordingPrefixValidator {
    cached_tokens: Vec<usize>,
    validated_ids: Vec<usize>,
    committed_ids: Arc<Mutex<Vec<usize>>>,
    released_slots: Vec<(usize, usize)>,
}

impl PagedPrefixCacheValidator for RecordingPrefixValidator {
    fn validate_prefix_cache_hit(
        &mut self,
        seq: &Sequence,
        _block_hashes: &[BlockHash],
        cached_tokens: usize,
        _block_size: usize,
    ) -> candle_core::Result<PagedPrefixCacheValidation> {
        self.cached_tokens.push(cached_tokens);
        let sequence_id = *seq.id();
        self.validated_ids.push(sequence_id);
        let committed_ids = Arc::clone(&self.committed_ids);
        Ok(PagedPrefixCacheValidation::staged(
            cached_tokens,
            move |_| {
                get_mut_arcmutex!(committed_ids).push(sequence_id);
                Ok(())
            },
        ))
    }

    fn release_recurrent_state(
        &mut self,
        sequence_id: usize,
        slot_idx: usize,
    ) -> candle_core::Result<bool> {
        self.released_slots.push((sequence_id, slot_idx));
        Ok(true)
    }
}

#[derive(Default)]
struct FailingPrefixValidator {
    released_slots: Vec<(usize, usize)>,
}

impl PagedPrefixCacheValidator for FailingPrefixValidator {
    fn validate_prefix_cache_hit(
        &mut self,
        _seq: &Sequence,
        _block_hashes: &[BlockHash],
        _cached_tokens: usize,
        _block_size: usize,
    ) -> candle_core::Result<PagedPrefixCacheValidation> {
        candle_core::bail!("injected recurrent state reset failure")
    }

    fn release_recurrent_state(
        &mut self,
        sequence_id: usize,
        slot_idx: usize,
    ) -> candle_core::Result<bool> {
        self.released_slots.push((sequence_id, slot_idx));
        Ok(true)
    }
}

#[derive(Default)]
struct FailingCommitPrefixValidator {
    released_slots: Vec<(usize, usize)>,
}

impl PagedPrefixCacheValidator for FailingCommitPrefixValidator {
    fn validate_prefix_cache_hit(
        &mut self,
        _seq: &Sequence,
        _block_hashes: &[BlockHash],
        cached_tokens: usize,
        _block_size: usize,
    ) -> candle_core::Result<PagedPrefixCacheValidation> {
        Ok(PagedPrefixCacheValidation::staged(cached_tokens, |_| {
            candle_core::bail!("injected recurrent state commit failure")
        }))
    }

    fn release_recurrent_state(
        &mut self,
        sequence_id: usize,
        slot_idx: usize,
    ) -> candle_core::Result<bool> {
        self.released_slots.push((sequence_id, slot_idx));
        Ok(true)
    }
}

type TestSequenceMedia = (
    Option<Vec<image::DynamicImage>>,
    Option<Vec<AudioInput>>,
    Option<Vec<VideoInput>>,
);

fn test_sequence_with_media_sender_and_group(
    id: usize,
    len: usize,
    input_media: TestSequenceMedia,
    tx: tokio::sync::mpsc::Sender<Response>,
    group: Arc<TokioMutex<SequenceGroup>>,
) -> Arc<Mutex<Sequence>> {
    let (input_images, input_audios, input_videos) = input_media;
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
    let seq = Sequence::new_waiting(crate::sequence::SequenceInit {
        tokens: vec![1; len],
        prompt: "prompt".to_string(),
        id,
        timestamp: id as u128,
        layers: 1,
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
        input_images,
        input_audios,
        input_videos,
        tool_call_state: None,
        image_gen_response_format: None,
        sequence_stepping_type: SeqStepType::PromptAndDecode,
        diffusion_params: None,
        image_gen_save_file: None,
        seq_preallocated_cache: None,
        return_raw_logits: false,
        ignore_eos: false,
        eos_tokens: vec![],
        sampling_seed: None,
    });
    seq.set_state(SequenceState::RunningCompletion);
    Arc::new(Mutex::new(seq))
}

fn test_sequence_with_media_and_sender(
    id: usize,
    len: usize,
    input_images: Option<Vec<image::DynamicImage>>,
    input_audios: Option<Vec<AudioInput>>,
    input_videos: Option<Vec<VideoInput>>,
    tx: tokio::sync::mpsc::Sender<Response>,
) -> Arc<Mutex<Sequence>> {
    test_sequence_with_media_sender_and_group(
        id,
        len,
        (input_images, input_audios, input_videos),
        tx,
        Arc::new(TokioMutex::new(SequenceGroup::new(1, false, true, None))),
    )
}

fn test_sequence_with_media_and_receiver(
    id: usize,
    len: usize,
    input_images: Option<Vec<image::DynamicImage>>,
    input_audios: Option<Vec<AudioInput>>,
    input_videos: Option<Vec<VideoInput>>,
) -> (Arc<Mutex<Sequence>>, tokio::sync::mpsc::Receiver<Response>) {
    let (tx, rx) = channel(1);
    (
        test_sequence_with_media_and_sender(id, len, input_images, input_audios, input_videos, tx),
        rx,
    )
}

fn test_sequence_with_media(
    id: usize,
    len: usize,
    input_images: Option<Vec<image::DynamicImage>>,
    input_audios: Option<Vec<AudioInput>>,
    input_videos: Option<Vec<VideoInput>>,
) -> Arc<Mutex<Sequence>> {
    test_sequence_with_media_and_receiver(id, len, input_images, input_audios, input_videos).0
}

fn test_sequence_with_images(
    id: usize,
    len: usize,
    input_images: Option<Vec<image::DynamicImage>>,
) -> Arc<Mutex<Sequence>> {
    test_sequence_with_media(id, len, input_images, None, None)
}

fn test_sequence(id: usize, len: usize) -> Arc<Mutex<Sequence>> {
    test_sequence_with_images(id, len, None)
}

fn test_audio_sequence(id: usize, len: usize) -> Arc<Mutex<Sequence>> {
    test_sequence_with_media(
        id,
        len,
        None,
        Some(vec![AudioInput {
            samples: vec![0.0],
            sample_rate: 16_000,
            channels: 1,
        }]),
        None,
    )
}

fn test_video_sequence(id: usize, len: usize) -> Arc<Mutex<Sequence>> {
    test_sequence_with_media(
        id,
        len,
        None,
        None,
        Some(vec![VideoInput::from_frames(
            vec![image::DynamicImage::new_rgb8(1, 1)],
            24.0,
            None,
        )]),
    )
}

#[test]
fn block_hash_update_waits_for_a_full_block() {
    let mut scheduler = test_scheduler();
    let seq_id = 7;
    let revision = 3;
    let mut tokens = vec![1; scheduler.block_size - 1];

    assert_eq!(
        scheduler.block_hash_update(seq_id, tokens.len(), revision),
        Some(BlockHashUpdate::Rebuild)
    );
    scheduler.ensure_block_hashes(seq_id, &tokens, &[], None, revision);
    assert!(scheduler.seq_block_hashes[&seq_id].is_empty());
    assert_eq!(
        scheduler.block_hash_update(seq_id, tokens.len(), revision),
        None
    );

    tokens.push(1);
    assert_eq!(
        scheduler.block_hash_update(seq_id, tokens.len(), revision),
        Some(BlockHashUpdate::Append)
    );
    scheduler.ensure_block_hashes(seq_id, &tokens, &[], None, revision);
    assert_eq!(scheduler.seq_block_hashes[&seq_id].len(), 1);

    tokens.extend(std::iter::repeat_n(2, scheduler.block_size - 1));
    assert_eq!(
        scheduler.block_hash_update(seq_id, tokens.len(), revision),
        None
    );
}

#[test]
fn block_hash_update_appends_then_rebuilds_on_revision_change() {
    let mut scheduler = test_scheduler();
    let seq_id = 9;
    let revision = 4;
    let mut tokens = vec![1; scheduler.block_size];

    scheduler.ensure_block_hashes(seq_id, &tokens, &[], None, revision);
    let first_hashes = scheduler.seq_block_hashes[&seq_id].clone();
    tokens.extend(std::iter::repeat_n(2, scheduler.block_size));

    assert_eq!(
        scheduler.block_hash_update(seq_id, tokens.len(), revision),
        Some(BlockHashUpdate::Append)
    );
    scheduler.ensure_block_hashes(seq_id, &tokens, &[], None, revision);
    assert_eq!(
        scheduler.seq_block_hashes[&seq_id],
        compute_block_hashes(&tokens, scheduler.block_size, &[], &[])
    );
    assert_eq!(
        scheduler.block_hash_update(seq_id, tokens.len(), revision),
        None
    );

    tokens[0] = 100;
    let next_revision = revision + 1;
    assert_eq!(
        scheduler.block_hash_update(seq_id, tokens.len(), next_revision),
        Some(BlockHashUpdate::Rebuild)
    );
    scheduler.ensure_block_hashes(seq_id, &tokens, &[], None, next_revision);
    assert_ne!(scheduler.seq_block_hashes[&seq_id][0], first_hashes[0]);
    assert_eq!(
        scheduler.seq_block_hashes[&seq_id],
        compute_block_hashes(&tokens, scheduler.block_size, &[], &[])
    );
}

fn test_speech_sequence(
    id: usize,
    prompt_len: usize,
    generated_len: usize,
) -> Arc<Mutex<Sequence>> {
    let seq = test_sequence(id, prompt_len);
    {
        let mut seq = get_mut_arcmutex!(seq);
        seq.set_audio_output(Some(crate::ChatAudioRequest {
            mode: crate::ChatAudioMode::TextAudio,
            format: crate::ChatAudioFormat::Wav,
        }));
        for _ in 0..generated_len {
            seq.add_token(
                Logprobs {
                    token: 2,
                    logprob: 0.0,
                    bytes: None,
                    top_logprobs: None,
                },
                Vec::new(),
                None,
            );
        }
        let len = seq.len();
        seq.set_num_computed_tokens(len);
    }
    seq
}

#[test]
fn speech_decode_blocks_are_not_cached_on_preempt_finish_or_eager_cache() {
    for path in ["preempt", "finish", "eager"] {
        let mut scheduler = test_scheduler();
        let seq_id = 10;
        let seq = test_speech_sequence(seq_id, scheduler.block_size + 3, scheduler.block_size * 2);
        let tokens = get_mut_arcmutex!(seq).get_toks().to_vec();
        let hashes = compute_block_hashes(&tokens, scheduler.block_size, &[], &[]);
        assert!(get_mut_arcmutex!(scheduler.kv_cache_manager)
            .allocate_slots(seq_id, tokens.len(), &[])
            .is_some());
        match path {
            "preempt" => scheduler._preempt(seq),
            "finish" => {
                get_mut_arcmutex!(seq).set_state(SequenceState::Done(StopReason::Eos));
                scheduler.running.push_back(seq);
                scheduler.free_finished_sequence_groups();
            }
            "eager" => {
                scheduler.running.push_back(seq);
                let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
                scheduler.schedule(&logger, None);
            }
            _ => unreachable!(),
        }
        assert_eq!(
            get_mut_arcmutex!(scheduler.kv_cache_manager)
                .get_computed_blocks(&hashes, tokens.len() + 1)
                .num_computed_tokens,
            scheduler.block_size,
            "{path}"
        );
    }
}

#[test]
fn speech_replay_prefix_hits_stop_at_the_prompt_boundary() {
    let mut scheduler = test_scheduler();
    let seq = test_speech_sequence(0, scheduler.block_size + 3, scheduler.block_size * 2);
    let tokens = get_mut_arcmutex!(seq).get_toks().to_vec();
    let hashes = compute_block_hashes(&tokens, scheduler.block_size, &[], &[]);
    {
        let mut kv_mgr = get_mut_arcmutex!(scheduler.kv_cache_manager);
        assert!(kv_mgr.allocate_slots(99, tokens.len(), &[]).is_some());
        kv_mgr.cache_blocks(99, &hashes, tokens.len());
        kv_mgr.free(99);
        assert!(
            kv_mgr
                .get_computed_blocks(&hashes, tokens.len())
                .num_computed_tokens
                > scheduler.block_size
        );
    }
    {
        let mut seq = get_mut_arcmutex!(seq);
        seq.set_state(SequenceState::Waiting);
        seq.set_num_computed_tokens(0);
    }
    scheduler.waiting.push_back(seq);
    let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
    let output = scheduler.schedule(&logger, None);
    assert_eq!(output.num_cached_tokens, [scheduler.block_size]);
    assert_eq!(
        get_mut_arcmutex!(output.scheduled[0]).prefix_cache_len(),
        scheduler.block_size
    );
}

#[test]
fn preemption_caches_hash_appended_after_token_growth() {
    let mut scheduler = test_scheduler();
    let seq_id = 10;
    let seq = test_sequence(seq_id, scheduler.block_size);
    let initial_tokens = get_mut_arcmutex!(seq).get_toks().to_vec();
    scheduler.ensure_block_hashes(seq_id, &initial_tokens, &[], None, 0);

    {
        let mut seq = get_mut_arcmutex!(seq);
        for _ in 0..scheduler.block_size {
            seq.add_token(
                Logprobs {
                    token: 2,
                    logprob: 0.0,
                    bytes: None,
                    top_logprobs: None,
                },
                Vec::new(),
                None,
            );
        }
        let len = seq.len();
        seq.set_num_computed_tokens(len);
    }
    let tokens = get_mut_arcmutex!(seq).get_toks().to_vec();
    let expected_hashes = compute_block_hashes(&tokens, scheduler.block_size, &[], &[]);
    assert!(get_mut_arcmutex!(scheduler.kv_cache_manager)
        .allocate_slots(seq_id, tokens.len(), &[])
        .is_some());

    scheduler._preempt(seq);

    assert_eq!(scheduler.seq_block_hashes[&seq_id], expected_hashes);
    let kv_mgr = get_mut_arcmutex!(scheduler.kv_cache_manager);
    assert_eq!(
        kv_mgr
            .get_computed_blocks(&expected_hashes, tokens.len() + 1)
            .num_computed_tokens,
        tokens.len()
    );
}

#[test]
fn finished_done_caches_but_error_only_cleans_up() {
    let mut scheduler = test_scheduler();
    let done_id = 10;
    let error_id = 20;
    let done = test_sequence(done_id, scheduler.block_size);
    let error = test_sequence(error_id, scheduler.block_size);
    get_mut_arcmutex!(error).set_toks_and_reallocate(vec![2; scheduler.block_size], None);

    let done_tokens = get_mut_arcmutex!(done).get_toks().to_vec();
    let error_tokens = get_mut_arcmutex!(error).get_toks().to_vec();
    let done_hashes = compute_block_hashes(&done_tokens, scheduler.block_size, &[], &[]);
    let error_hashes = compute_block_hashes(&error_tokens, scheduler.block_size, &[], &[]);
    scheduler.ensure_block_hashes(done_id, &done_tokens, &[], None, 0);
    scheduler.ensure_block_hashes(
        error_id,
        &error_tokens,
        &[],
        None,
        get_mut_arcmutex!(error).block_hash_revision(),
    );
    {
        let mut kv_mgr = get_mut_arcmutex!(scheduler.kv_cache_manager);
        assert!(kv_mgr
            .allocate_slots(done_id, done_tokens.len(), &[])
            .is_some());
        assert!(kv_mgr
            .allocate_slots(error_id, error_tokens.len(), &[])
            .is_some());
    }
    {
        let mut seq = get_mut_arcmutex!(done);
        seq.set_num_computed_tokens(done_tokens.len());
        seq.set_state(SequenceState::Done(StopReason::Eos));
    }
    {
        let mut seq = get_mut_arcmutex!(error);
        seq.set_num_computed_tokens(error_tokens.len());
        seq.set_state(SequenceState::Error);
    }
    scheduler.waiting_counts.insert(done_id, 3);
    scheduler.waiting_counts.insert(error_id, 4);
    scheduler.running.push_back(done);
    scheduler.waiting.push_back(error);

    scheduler.free_finished_sequence_groups();

    assert!(scheduler.running.is_empty());
    assert!(scheduler.waiting.is_empty());
    for seq_id in [done_id, error_id] {
        assert!(!scheduler.seq_block_hashes.contains_key(&seq_id));
        assert!(!scheduler.seq_block_hash_revisions.contains_key(&seq_id));
        assert!(!scheduler.waiting_counts.contains_key(&seq_id));
    }
    let kv_mgr = get_mut_arcmutex!(scheduler.kv_cache_manager);
    assert!(!kv_mgr.has_request(done_id));
    assert!(!kv_mgr.has_request(error_id));
    assert_eq!(
        kv_mgr
            .get_computed_blocks(&done_hashes, done_tokens.len() + 1)
            .num_computed_tokens,
        done_tokens.len()
    );
    assert_eq!(
        kv_mgr
            .get_computed_blocks(&error_hashes, error_tokens.len() + 1)
            .num_computed_tokens,
        0
    );
}

#[test]
fn enabling_prefix_cache_reuses_hashes_built_while_disabled() {
    let mut scheduler = test_scheduler();
    let seq_id = 10;
    let tokens = vec![1; scheduler.block_size * 2];
    let hashes = compute_block_hashes(&tokens, scheduler.block_size, &[], &[]);
    {
        let mut kv_mgr = get_mut_arcmutex!(scheduler.kv_cache_manager);
        assert!(kv_mgr.allocate_slots(99, tokens.len(), &[]).is_some());
        kv_mgr.cache_blocks(99, &hashes, scheduler.block_size);
        kv_mgr.free(99);
    }

    scheduler.set_prefix_caching_enabled_sync(false);
    let seq = test_sequence(seq_id, tokens.len());
    get_mut_arcmutex!(seq).set_state(SequenceState::Waiting);
    scheduler.waiting.push_back(seq);
    let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);

    let disabled = scheduler.schedule(&logger, None);
    assert_eq!(disabled.num_cached_tokens, vec![0]);
    assert_eq!(scheduler.seq_block_hashes[&seq_id], hashes);
    let seq = disabled.scheduled[0].clone();
    scheduler.running.clear();
    scheduler._preempt(seq);
    assert_eq!(scheduler.seq_block_hashes[&seq_id], hashes);

    scheduler.set_prefix_caching_enabled_sync(true);
    let enabled = scheduler.schedule(&logger, None);
    assert_eq!(enabled.num_cached_tokens, vec![scheduler.block_size]);
    assert_eq!(scheduler.seq_block_hashes[&seq_id], hashes);
}

#[test]
fn preempted_prompt_validates_zero_token_recurrent_state() {
    let mut scheduler = test_scheduler();
    let seq = test_sequence(0, 4);
    {
        let mut seq = get_mut_arcmutex!(seq);
        seq.set_prefix_cache_len(4);
        seq.set_num_computed_tokens(4);
        seq.set_recurrent_state_idx(Some(7));
    }
    scheduler._preempt(seq.clone());

    {
        let seq = get_mut_arcmutex!(seq);
        assert_eq!(seq.getstate(), SequenceState::Waiting);
        assert_eq!(seq.prefix_cache_len(), 0);
        assert_eq!(seq.num_computed_tokens(), 0);
    }

    let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
    let mut validator = RecordingPrefixValidator::default();
    let output = scheduler.schedule(&logger, Some(&mut validator));

    assert_eq!(output.scheduled.len(), 1);
    assert_eq!(validator.cached_tokens, vec![0]);
}

#[test]
fn scheduler_output_reports_preempted_sequence_ids() {
    let mut scheduler = test_scheduler();
    scheduler.requires_uniform_completion_batch = true;
    scheduler.running.push_back(test_sequence(10, 4));
    scheduler.running.push_back(test_sequence(20, 7));

    let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
    let output = Scheduler::schedule(&mut scheduler, &logger, None);
    let SchedulerOutput::PagedAttention {
        output,
        preempted_sequence_ids,
    } = output
    else {
        panic!("paged scheduler returned a default scheduler output");
    };

    assert_eq!(preempted_sequence_ids, vec![20]);
    assert_eq!(output.scheduled.len(), 1);
    assert_eq!(*get_mut_arcmutex!(output.scheduled[0]).id(), 10);
}

#[test]
fn readmitted_prefix_restore_supersedes_deferred_speculative_release() {
    let mut scheduler = test_scheduler();
    let seq = test_sequence(20, 16);
    scheduler._preempt(seq);

    let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
    let mut validator = RecordingPrefixValidator::default();
    let committed_ids = Arc::clone(&validator.committed_ids);
    let SchedulerOutput::PagedAttention {
        output,
        preempted_sequence_ids,
    } = Scheduler::schedule(&mut scheduler, &logger, Some(&mut validator))
    else {
        panic!("paged scheduler returned a default scheduler output");
    };

    assert_eq!(&*get_mut_arcmutex!(committed_ids), &[20]);
    assert!(preempted_sequence_ids.is_empty());
    assert_eq!(output.scheduled.len(), 1);
    assert_eq!(*get_mut_arcmutex!(output.scheduled[0]).id(), 20);
}

#[test]
fn deferred_speculative_releases_follow_final_state_and_are_unique() {
    let mut scheduler = test_scheduler();
    scheduler.running.push_back(test_sequence(20, 4));
    scheduler.preempted_sequence_ids = vec![20, 30, 20, 30];

    assert_eq!(scheduler.take_deferred_speculative_releases(), vec![30]);
    assert!(scheduler.preempted_sequence_ids.is_empty());
}

#[test]
fn finished_sequence_ids_are_visible_before_cleanup() {
    let mut scheduler = test_scheduler();
    let live = test_sequence(10, 4);
    let finished = test_sequence(20, 4);
    get_mut_arcmutex!(finished).set_state(SequenceState::Done(StopReason::Eos));
    scheduler.running.push_back(live);
    scheduler.running.push_back(finished);

    assert_eq!(Scheduler::get_finished_sequence_ids(&scheduler), vec![20]);
    scheduler.free_finished_sequence_groups();
    assert!(Scheduler::get_finished_sequence_ids(&scheduler).is_empty());
}

#[test]
fn closed_response_group_cancels_waiting_prefill_and_decode() {
    let mut scheduler = test_scheduler();
    let initial_free_blocks = get_mut_arcmutex!(scheduler.kv_cache_manager).num_free_blocks();
    let (tx, rx) = channel(1);
    let group = Arc::new(TokioMutex::new(SequenceGroup::new(3, false, true, None)));

    let waiting = test_sequence_with_media_sender_and_group(
        10,
        4,
        (None, None, None),
        tx.clone(),
        group.clone(),
    );
    get_mut_arcmutex!(waiting).set_state(SequenceState::Waiting);
    get_mut_arcmutex!(waiting).set_recurrent_state_idx(Some(10));
    scheduler.waiting.push_back(waiting.clone());

    let prefill = test_sequence_with_media_sender_and_group(
        20,
        4,
        (None, None, None),
        tx.clone(),
        group.clone(),
    );
    get_mut_arcmutex!(prefill).set_state(SequenceState::RunningPrompt);
    get_mut_arcmutex!(prefill).set_recurrent_state_idx(Some(20));
    assert!(get_mut_arcmutex!(scheduler.kv_cache_manager)
        .allocate_slots(20, 4, &[])
        .is_some());
    scheduler.running.push_back(prefill.clone());

    let decode = test_sequence_with_media_sender_and_group(30, 4, (None, None, None), tx, group);
    get_mut_arcmutex!(decode).set_recurrent_state_idx(Some(30));
    assert!(get_mut_arcmutex!(scheduler.kv_cache_manager)
        .allocate_slots(30, 4, &[])
        .is_some());
    scheduler.running.push_back(decode.clone());
    assert_eq!(
        get_mut_arcmutex!(scheduler.kv_cache_manager).num_active_blocks(),
        2
    );

    drop(rx);
    Scheduler::cancel_closed_response_groups(&mut scheduler);

    for seq in [&waiting, &prefill, &decode] {
        assert_eq!(
            get_mut_arcmutex!(seq).getstate(),
            SequenceState::Done(StopReason::Canceled)
        );
    }
    assert_eq!(
        Scheduler::get_finished_sequence_ids(&scheduler),
        vec![20, 30, 10]
    );
    assert_eq!(
        Scheduler::get_finished_recurrent_slots(&scheduler),
        vec![(20, 20), (30, 30), (10, 10)]
    );
    assert!(!scheduler.can_continue_decode_batch(&[30]));

    scheduler.free_finished_sequence_groups();
    assert!(scheduler.waiting.is_empty());
    assert!(scheduler.running.is_empty());
    assert_eq!(
        get_mut_arcmutex!(scheduler.kv_cache_manager).num_free_blocks(),
        initial_free_blocks
    );
    assert_eq!(
        get_mut_arcmutex!(scheduler.kv_cache_manager).num_active_blocks(),
        0
    );
}

#[test]
fn closed_response_stops_resident_decode_continuation() {
    let mut scheduler = test_scheduler();
    let (seq, rx) = test_sequence_with_media_and_receiver(10, 4, None, None, None);
    scheduler.running.push_back(seq);
    assert!(scheduler.can_continue_decode_batch(&[10]));

    drop(rx);
    Scheduler::cancel_closed_response_groups(&mut scheduler);

    assert!(!scheduler.can_continue_decode_batch(&[10]));
}

#[test]
fn prefix_validator_observes_the_clamped_cache_boundary() {
    let mut scheduler = test_scheduler();
    let tokens = vec![1; 24];
    let features = vec![MultiModalFeature {
        kind: MultimodalKind::Image,
        item_range: 0..1,
        hashes: vec![1],
        offset: 12,
        length: 8,
        attention_policy: crate::paged_attention::block_hash::MultimodalAttentionPolicy::Causal,
        splittable: false,
    }];
    let hashes = compute_block_hashes(&tokens, scheduler.block_size, &features, &[]);
    {
        let mut kv_mgr = get_mut_arcmutex!(scheduler.kv_cache_manager);
        assert!(kv_mgr.allocate_slots(99, tokens.len(), &[]).is_some());
        kv_mgr.cache_blocks(99, &hashes, 16);
        kv_mgr.free(99);
    }

    let seq = test_sequence(0, tokens.len());
    {
        let mut seq = get_mut_arcmutex!(seq);
        seq.set_state(SequenceState::Waiting);
        seq.set_mm_features(features);
    }
    scheduler.waiting.push_back(seq);

    let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
    let mut validator = RecordingPrefixValidator::default();
    let output = scheduler.schedule(&logger, Some(&mut validator));

    assert_eq!(validator.cached_tokens, vec![8]);
    assert_eq!(output.num_cached_tokens, vec![8]);
}

#[test]
fn cache_pressure_discards_staged_prefix_admission_until_retry_succeeds() {
    let mut scheduler = PagedAttentionScheduler::new(
        PagedAttentionSchedulerConfig {
            max_num_seqs: 8,
            max_num_batched_tokens: 4096,
            max_prefill_chunk_tokens: 4096,
            max_decode_steps_before_prefill: 8,
        },
        CacheConfig {
            block_size: 8,
            num_gpu_blocks: 2,
            cache_type: PagedCacheType::Auto,
            kv_cache_group_ids: vec![0],
        },
    );
    assert!(get_mut_arcmutex!(scheduler.kv_cache_manager)
        .allocate_slots(99, 8, &[])
        .is_some());
    let seq = test_sequence(0, 8);
    {
        let mut seq = get_mut_arcmutex!(seq);
        seq.set_state(SequenceState::Waiting);
        seq.set_recurrent_state_idx(Some(7));
    }
    scheduler.waiting.push_back(seq);
    let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
    let mut validator = RecordingPrefixValidator::default();

    let blocked = scheduler.schedule(&logger, Some(&mut validator));

    assert!(blocked.scheduled.is_empty());
    assert_eq!(validator.validated_ids, vec![0]);
    assert!(get_mut_arcmutex!(validator.committed_ids).is_empty());
    assert_eq!(scheduler.waiting.len(), 1);

    get_mut_arcmutex!(scheduler.kv_cache_manager).free(99);
    let admitted = scheduler.schedule(&logger, Some(&mut validator));

    assert_eq!(admitted.scheduled.len(), 1);
    assert_eq!(validator.validated_ids, vec![0, 0]);
    assert_eq!(*get_mut_arcmutex!(validator.committed_ids), vec![0]);
}

#[test]
fn disabled_waiting_prompt_preemption_preserves_decode_state() {
    let mut scheduler = PagedAttentionScheduler::new(
        PagedAttentionSchedulerConfig {
            max_num_seqs: 8,
            max_num_batched_tokens: 4096,
            max_prefill_chunk_tokens: 4096,
            max_decode_steps_before_prefill: 8,
        },
        CacheConfig {
            block_size: 8,
            num_gpu_blocks: 5,
            cache_type: PagedCacheType::Auto,
            kv_cache_group_ids: vec![0],
        },
    );
    scheduler.decode_steps_since_prefill = scheduler.config.max_decode_steps_before_prefill;

    let completion = test_sequence(10, 7);
    get_mut_arcmutex!(completion).set_num_computed_tokens(7);
    assert!(get_mut_arcmutex!(scheduler.kv_cache_manager)
        .allocate_slots(10, 7, &[])
        .is_some());
    scheduler.running.push_back(completion.clone());

    let waiting = test_sequence(20, 32);
    get_mut_arcmutex!(waiting).set_state(SequenceState::Waiting);
    scheduler.waiting.push_back(waiting.clone());
    scheduler.waiting_counts.insert(20, WAITING_TIMEOUT);
    Scheduler::set_waiting_prompt_preemption_enabled(&mut scheduler, false);

    let blocks_before = get_mut_arcmutex!(scheduler.kv_cache_manager)
        .get_block_ids(10)
        .unwrap()
        .to_vec();
    let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
    let SchedulerOutput::PagedAttention {
        output,
        preempted_sequence_ids,
    } = Scheduler::schedule(&mut scheduler, &logger, None)
    else {
        panic!("paged scheduler returned a default scheduler output");
    };

    assert!(preempted_sequence_ids.is_empty());
    assert_eq!(output.scheduled.len(), 1);
    assert_eq!(*get_mut_arcmutex!(output.scheduled[0]).id(), 10);
    assert_eq!(get_mut_arcmutex!(completion).num_computed_tokens(), 7);
    assert_eq!(
        get_mut_arcmutex!(completion).getstate(),
        SequenceState::RunningCompletion
    );
    assert_eq!(
        get_mut_arcmutex!(waiting).getstate(),
        SequenceState::Waiting
    );
    assert_eq!(scheduler.waiting.len(), 1);
    assert_eq!(
        get_mut_arcmutex!(scheduler.kv_cache_manager)
            .get_block_ids(10)
            .unwrap(),
        blocks_before
    );
    assert_eq!(
        scheduler.waiting_counts.get(&20),
        Some(&(WAITING_TIMEOUT + 1))
    );

    Scheduler::set_waiting_prompt_preemption_enabled(&mut scheduler, true);
    let SchedulerOutput::PagedAttention {
        output,
        preempted_sequence_ids,
    } = Scheduler::schedule(&mut scheduler, &logger, None)
    else {
        panic!("paged scheduler returned a default scheduler output");
    };

    assert_eq!(preempted_sequence_ids, vec![10]);
    assert_eq!(output.scheduled.len(), 1);
    assert_eq!(*get_mut_arcmutex!(output.scheduled[0]).id(), 20);
    assert!(!scheduler.waiting_counts.contains_key(&20));
}

#[test]
fn modality_requeue_does_not_stage_or_commit_prefix_admission() {
    let mut scheduler = test_scheduler();
    scheduler.requires_uniform_media_batch = true;
    let text = test_sequence(0, 8);
    let image = test_sequence_with_images(1, 8, Some(vec![image::DynamicImage::new_rgb8(1, 1)]));
    for seq in [&text, &image] {
        get_mut_arcmutex!(seq).set_state(SequenceState::Waiting);
    }
    scheduler.waiting.push_back(text);
    scheduler.waiting.push_back(image);
    let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
    let mut validator = RecordingPrefixValidator::default();

    let text_batch = scheduler.schedule(&logger, Some(&mut validator));

    assert_eq!(text_batch.scheduled.len(), 1);
    assert_eq!(validator.validated_ids, vec![0]);
    assert_eq!(*get_mut_arcmutex!(validator.committed_ids), vec![0]);
    assert_eq!(scheduler.waiting.len(), 1);

    get_mut_arcmutex!(scheduler.kv_cache_manager).free(0);
    scheduler.running.clear();
    let image_batch = scheduler.schedule(&logger, Some(&mut validator));

    assert_eq!(image_batch.scheduled.len(), 1);
    assert_eq!(*get_mut_arcmutex!(image_batch.scheduled[0]).id(), 1);
    assert_eq!(validator.validated_ids, vec![0, 1]);
    assert_eq!(*get_mut_arcmutex!(validator.committed_ids), vec![0, 1]);
}

#[test]
fn ignored_prompt_finishes_before_modality_requeue() {
    let mut scheduler = test_scheduler();
    scheduler.requires_uniform_media_batch = true;

    let first = test_sequence(0, 4);
    get_mut_arcmutex!(first).set_state(SequenceState::Waiting);
    scheduler.waiting.push_back(first);

    let (oversized, mut receiver) = test_sequence_with_media_and_receiver(
        1,
        1_024,
        Some(vec![image::DynamicImage::new_rgb8(1, 1)]),
        None,
        None,
    );
    {
        let mut oversized = get_mut_arcmutex!(oversized);
        oversized.set_state(SequenceState::Waiting);
        oversized.set_recurrent_state_idx(Some(7));
    }
    scheduler.waiting.push_back(oversized.clone());
    scheduler.waiting_counts.insert(1, WAITING_TIMEOUT);

    let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
    let mut validator = RecordingPrefixValidator::default();
    let output = scheduler.schedule(&logger, Some(&mut validator));

    assert_eq!(output.scheduled.len(), 1);
    assert_eq!(*get_mut_arcmutex!(output.scheduled[0]).id(), 0);
    assert!(scheduler.waiting.is_empty());
    assert_eq!(
        get_mut_arcmutex!(oversized).getstate(),
        SequenceState::FinishedIgnored
    );
    assert_eq!(get_mut_arcmutex!(oversized).recurrent_state_idx(), None);
    assert_eq!(validator.released_slots, vec![(1, 7)]);
    assert!(matches!(
        receiver.try_recv(),
        Ok(Response::ValidationError(_))
    ));
}

#[test]
fn recurrent_prefix_failure_rejects_request_and_releases_slot() {
    let mut scheduler = test_scheduler();
    let (seq, mut receiver) = test_sequence_with_media_and_receiver(1, 16, None, None, None);
    {
        let mut seq = get_mut_arcmutex!(seq);
        seq.set_state(SequenceState::Waiting);
        seq.set_recurrent_state_idx(Some(7));
    }
    scheduler.waiting.push_back(seq.clone());

    let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
    let mut validator = FailingPrefixValidator::default();
    let output = scheduler.schedule(&logger, Some(&mut validator));

    assert!(output.scheduled.is_empty());
    assert!(scheduler.waiting.is_empty());
    assert!(scheduler.running.is_empty());
    assert_eq!(validator.released_slots, vec![(1, 7)]);
    assert_eq!(get_mut_arcmutex!(seq).recurrent_state_idx(), None);
    assert_eq!(
        get_mut_arcmutex!(seq).getstate(),
        SequenceState::FinishedIgnored
    );
    let response = receiver.try_recv().unwrap();
    assert!(matches!(response, Response::InternalError(_)));
}

#[test]
fn recurrent_prefix_commit_failure_frees_kv_and_releases_slot() {
    let mut scheduler = test_scheduler();
    let initial_free_blocks = get_mut_arcmutex!(scheduler.kv_cache_manager).num_free_blocks();
    let (seq, mut receiver) = test_sequence_with_media_and_receiver(1, 16, None, None, None);
    {
        let mut seq = get_mut_arcmutex!(seq);
        seq.set_state(SequenceState::Waiting);
        seq.set_recurrent_state_idx(Some(7));
    }
    scheduler.waiting.push_back(seq.clone());

    let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
    let mut validator = FailingCommitPrefixValidator::default();
    let output = scheduler.schedule(&logger, Some(&mut validator));

    assert!(output.scheduled.is_empty());
    assert!(scheduler.waiting.is_empty());
    assert!(scheduler.running.is_empty());
    assert_eq!(validator.released_slots, vec![(1, 7)]);
    assert_eq!(get_mut_arcmutex!(seq).recurrent_state_idx(), None);
    assert_eq!(
        get_mut_arcmutex!(scheduler.kv_cache_manager).num_free_blocks(),
        initial_free_blocks
    );
    assert!(matches!(
        receiver.try_recv(),
        Ok(Response::InternalError(_))
    ));
}

#[test]
fn ragged_completion_batch_keeps_all_sequences_running() {
    let mut scheduler = test_scheduler();
    scheduler.requires_uniform_completion_batch = false;
    scheduler.requires_uniform_media_batch = false;
    scheduler.running.push_back(test_sequence(0, 4));
    scheduler.running.push_back(test_sequence(1, 7));

    scheduler.enforce_completion_compatibility();

    assert_eq!(scheduler.running.len(), 2);
    assert!(scheduler.waiting.is_empty());
}

#[test]
fn ragged_completion_batch_separates_incompatible_media() {
    let mut scheduler = test_scheduler();
    scheduler.requires_uniform_completion_batch = false;
    scheduler.requires_uniform_media_batch = true;
    scheduler.running.push_back(test_sequence(0, 4));
    scheduler.running.push_back(test_sequence_with_images(
        1,
        7,
        Some(vec![image::DynamicImage::new_rgb8(1, 1)]),
    ));

    scheduler.enforce_completion_compatibility();

    assert_eq!(scheduler.running.len(), 1);
    assert_eq!(scheduler.waiting.len(), 1);
}

#[test]
fn completion_media_signature_survives_consumed_prompt_inputs() {
    let mut scheduler = test_scheduler();
    scheduler.requires_uniform_completion_batch = false;
    scheduler.requires_uniform_media_batch = true;
    scheduler.running.push_back(test_sequence(0, 4));
    let image = test_sequence_with_images(1, 7, Some(vec![image::DynamicImage::new_rgb8(1, 1)]));
    {
        let mut image = get_mut_arcmutex!(image);
        image.set_mm_features(vec![MultiModalFeature {
            kind: MultimodalKind::Image,
            item_range: 0..1,
            hashes: vec![1],
            offset: 0,
            length: 1,
            attention_policy: crate::paged_attention::block_hash::MultimodalAttentionPolicy::Causal,
            splittable: false,
        }]);
        image.multimodal.has_changed_prompt = true;
        assert_eq!(image.take_images().unwrap().len(), 1);
        assert!(!image.has_images());
        assert_eq!(modality_signature(&image), IMAGE_MODALITY);
    }
    scheduler.running.push_back(image);

    scheduler.enforce_completion_compatibility();

    assert_eq!(scheduler.running.len(), 1);
    assert_eq!(scheduler.waiting.len(), 1);
}

#[test]
fn empty_image_list_does_not_split_text_completion_batch() {
    let mut scheduler = test_scheduler();
    scheduler.requires_uniform_completion_batch = false;
    scheduler.running.push_back(test_sequence(0, 4));
    let prompt = test_sequence_with_images(1, 7, Some(vec![]));
    get_mut_arcmutex!(prompt).set_state(SequenceState::RunningPrompt);
    scheduler.running.push_back(prompt);

    scheduler.enforce_completion_compatibility();

    assert_eq!(scheduler.running.len(), 2);
    assert!(scheduler.waiting.is_empty());
}

#[test]
fn uniform_completion_batch_preempts_other_lengths() {
    let mut scheduler = test_scheduler();
    scheduler.requires_uniform_completion_batch = true;
    scheduler.running.push_back(test_sequence(0, 4));
    scheduler.running.push_back(test_sequence(1, 7));

    scheduler.enforce_completion_compatibility();

    assert_eq!(scheduler.running.len(), 1);
    assert_eq!(scheduler.waiting.len(), 1);
    assert_eq!(get_mut_arcmutex!(scheduler.running[0]).len(), 4);
}

#[test]
fn ragged_prompt_batch_keeps_compatible_sequences() {
    let mut scheduler = test_scheduler();
    let prompts = VecDeque::from([test_sequence(0, 4), test_sequence(1, 7)]);

    for seq in &prompts {
        get_mut_arcmutex!(seq).set_state(SequenceState::RunningPrompt);
    }
    let scheduled = scheduler.bucket_and_preempt_sequences(prompts, BatchKind::Prompt, false);

    assert_eq!(scheduled.len(), 2);
    assert!(scheduler.waiting.is_empty());
}

#[test]
fn ragged_prompt_batch_mixes_media_and_text_sequences() {
    let mut scheduler = test_scheduler();
    scheduler.requires_uniform_media_batch = false;
    let prompts = VecDeque::from([
        test_sequence(0, 4),
        test_sequence_with_images(1, 7, Some(vec![image::DynamicImage::new_rgb8(1, 1)])),
    ]);

    for seq in &prompts {
        get_mut_arcmutex!(seq).set_state(SequenceState::RunningPrompt);
    }
    let scheduled = scheduler.bucket_and_preempt_sequences(prompts, BatchKind::Prompt, false);

    assert_eq!(scheduled.len(), 2);
    assert!(scheduler.waiting.is_empty());
}

#[test]
fn ragged_prompt_batch_mixes_distinct_media_modalities() {
    let mut scheduler = test_scheduler();
    scheduler.requires_uniform_media_batch = false;
    let prompts = VecDeque::from([
        test_sequence_with_images(0, 4, Some(vec![image::DynamicImage::new_rgb8(1, 1)])),
        test_audio_sequence(1, 7),
        test_video_sequence(2, 5),
    ]);

    for seq in &prompts {
        get_mut_arcmutex!(seq).set_state(SequenceState::RunningPrompt);
    }
    let scheduled = scheduler.bucket_and_preempt_sequences(prompts, BatchKind::Prompt, false);

    assert_eq!(scheduled.len(), 3);
    assert!(scheduler.waiting.is_empty());
}

#[test]
fn uniform_prompt_batch_separates_media_and_text_sequences() {
    let mut scheduler = test_scheduler();
    let prompts = VecDeque::from([
        test_sequence(0, 4),
        test_sequence_with_images(1, 4, Some(vec![image::DynamicImage::new_rgb8(1, 1)])),
    ]);

    for seq in &prompts {
        get_mut_arcmutex!(seq).set_state(SequenceState::RunningPrompt);
    }
    let scheduled = scheduler.bucket_and_preempt_sequences(prompts, BatchKind::Prompt, true);

    assert_eq!(scheduled.len(), 1);
    assert_eq!(scheduler.waiting.len(), 1);
}

#[test]
fn ragged_prompt_batch_separates_incompatible_media() {
    let mut scheduler = test_scheduler();
    scheduler.requires_uniform_prompt_batch = false;
    scheduler.requires_uniform_media_batch = true;
    let prompts = VecDeque::from([
        test_sequence(0, 4),
        test_sequence_with_images(1, 7, Some(vec![image::DynamicImage::new_rgb8(1, 1)])),
    ]);

    for seq in &prompts {
        get_mut_arcmutex!(seq).set_state(SequenceState::RunningPrompt);
    }
    let scheduled = scheduler.bucket_and_preempt_sequences(prompts, BatchKind::Prompt, false);

    assert_eq!(scheduled.len(), 1);
    assert_eq!(scheduler.waiting.len(), 1);
}

#[test]
fn uniform_prompt_batch_separates_image_and_audio_sequences() {
    let mut scheduler = test_scheduler();
    let prompts = VecDeque::from([
        test_sequence_with_images(0, 4, Some(vec![image::DynamicImage::new_rgb8(1, 1)])),
        test_audio_sequence(1, 4),
    ]);

    for seq in &prompts {
        get_mut_arcmutex!(seq).set_state(SequenceState::RunningPrompt);
    }
    let scheduled = scheduler.bucket_and_preempt_sequences(prompts, BatchKind::Prompt, true);

    assert_eq!(scheduled.len(), 1);
    assert_eq!(scheduler.waiting.len(), 1);
}

#[test]
fn uniform_prompt_batch_separates_image_and_video_sequences() {
    let mut scheduler = test_scheduler();
    let prompts = VecDeque::from([
        test_sequence_with_images(0, 4, Some(vec![image::DynamicImage::new_rgb8(1, 1)])),
        test_video_sequence(1, 4),
    ]);

    for seq in &prompts {
        get_mut_arcmutex!(seq).set_state(SequenceState::RunningPrompt);
    }
    let scheduled = scheduler.bucket_and_preempt_sequences(prompts, BatchKind::Prompt, true);

    assert_eq!(scheduled.len(), 1);
    assert_eq!(scheduler.waiting.len(), 1);
}

#[test]
fn ragged_prompt_batch_keeps_unequal_lengths() {
    let mut scheduler = test_scheduler();
    scheduler.supports_packed_prefill = true;
    let prompts = VecDeque::from([
        test_sequence(0, 4),
        test_sequence(1, 300),
        test_sequence(2, 600),
    ]);

    for seq in &prompts {
        get_mut_arcmutex!(seq).set_state(SequenceState::RunningPrompt);
    }
    let scheduled = scheduler.bucket_and_preempt_sequences(prompts, BatchKind::Prompt, false);

    let scheduled_ids: Vec<_> = scheduled
        .iter()
        .map(|seq| *get_mut_arcmutex!(seq).id())
        .collect();
    assert_eq!(scheduled_ids, vec![0, 1, 2]);
    assert!(scheduler.waiting.is_empty());
}

#[test]
fn ragged_prompt_batch_preserves_fcfs_order() {
    let mut scheduler = test_scheduler();
    scheduler.supports_packed_prefill = true;
    scheduler.waiting.push_back(test_sequence(4, 8));
    let prompts = VecDeque::from([
        test_sequence(0, 300),
        test_sequence(1, 4),
        test_sequence(2, 600),
        test_sequence(3, 7),
    ]);

    for seq in &prompts {
        get_mut_arcmutex!(seq).set_state(SequenceState::RunningPrompt);
    }
    let scheduled = scheduler.bucket_and_preempt_sequences(prompts, BatchKind::Prompt, false);

    let scheduled_ids: Vec<_> = scheduled
        .iter()
        .map(|seq| *get_mut_arcmutex!(seq).id())
        .collect();
    assert_eq!(scheduled_ids, vec![0, 1, 2, 3]);
    let waiting_ids: Vec<_> = scheduler
        .waiting
        .iter()
        .map(|seq| *get_mut_arcmutex!(seq).id())
        .collect();
    assert_eq!(waiting_ids, vec![4]);
}

#[test]
fn padded_ragged_prompt_batch_bounds_padding() {
    let mut scheduler = test_scheduler();
    let prompts = VecDeque::from([test_sequence(0, 4), test_sequence(1, 300)]);

    for seq in &prompts {
        get_mut_arcmutex!(seq).set_state(SequenceState::RunningPrompt);
    }
    let scheduled = scheduler.bucket_and_preempt_sequences(prompts, BatchKind::Prompt, false);

    assert_eq!(scheduled.len(), 1);
    assert_eq!(scheduler.waiting.len(), 1);
    assert_eq!(get_mut_arcmutex!(scheduled[0]).len(), 4);
}

#[test]
fn cached_prompt_batch_requires_matching_prefix_offsets() {
    let mut scheduler = test_scheduler();
    let first = test_sequence(0, 100);
    let second = test_sequence(1, 132);
    get_mut_arcmutex!(first).set_prefix_cache_len(32);
    get_mut_arcmutex!(second).set_prefix_cache_len(64);
    let prompts = VecDeque::from([first, second]);

    for seq in &prompts {
        get_mut_arcmutex!(seq).set_state(SequenceState::RunningPrompt);
    }
    let scheduled = scheduler.bucket_and_preempt_sequences(prompts, BatchKind::Prompt, true);

    assert_eq!(scheduled.len(), 1);
    assert_eq!(scheduler.waiting.len(), 1);
}

#[test]
fn prompt_batch_separates_raw_logits_requests() {
    let mut scheduler = test_scheduler();
    let first = test_sequence(0, 4);
    let second = test_sequence(1, 4);
    get_mut_arcmutex!(second).return_raw_logits = true;
    let prompts = VecDeque::from([first, second]);

    for seq in &prompts {
        get_mut_arcmutex!(seq).set_state(SequenceState::RunningPrompt);
    }
    let scheduled = scheduler.bucket_and_preempt_sequences(prompts, BatchKind::Prompt, true);

    assert_eq!(scheduled.len(), 1);
    assert_eq!(scheduler.waiting.len(), 1);
}

#[test]
fn prompt_batch_singletonizes_raw_logits_requests() {
    let mut scheduler = test_scheduler();
    let first = test_sequence(0, 4);
    let second = test_sequence(1, 4);
    get_mut_arcmutex!(first).return_raw_logits = true;
    get_mut_arcmutex!(second).return_raw_logits = true;
    let prompts = VecDeque::from([first, second]);

    for seq in &prompts {
        get_mut_arcmutex!(seq).set_state(SequenceState::RunningPrompt);
    }
    let scheduled = scheduler.bucket_and_preempt_sequences(prompts, BatchKind::Prompt, true);

    assert_eq!(scheduled.len(), 1);
    assert_eq!(scheduler.waiting.len(), 1);
}

#[test]
fn raw_logits_prompt_bypasses_reusable_prefix_blocks() {
    let mut scheduler = test_scheduler();
    let tokens = vec![1; 16];
    let hashes = compute_block_hashes(&tokens, scheduler.block_size, &[], &[]);
    {
        let mut kv_mgr = get_mut_arcmutex!(scheduler.kv_cache_manager);
        assert!(kv_mgr.allocate_slots(99, tokens.len(), &[]).is_some());
        kv_mgr.cache_blocks(99, &hashes, scheduler.block_size);
        kv_mgr.free(99);
        assert_eq!(
            kv_mgr
                .get_computed_blocks(&hashes, tokens.len())
                .num_computed_tokens,
            scheduler.block_size
        );
    }

    let raw = test_sequence(0, tokens.len());
    get_mut_arcmutex!(raw).return_raw_logits = true;
    get_mut_arcmutex!(raw).set_state(SequenceState::Waiting);
    scheduler.waiting.push_back(raw);

    let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
    let output = scheduler.schedule(&logger, None);

    assert_eq!(output.num_cached_tokens, vec![0]);
    assert_eq!(get_mut_arcmutex!(output.scheduled[0]).prefix_cache_len(), 0);
    let kv_mgr = get_mut_arcmutex!(scheduler.kv_cache_manager);
    assert_eq!(
        kv_mgr
            .get_computed_blocks(&hashes, tokens.len())
            .num_computed_tokens,
        scheduler.block_size
    );
}

#[test]
fn preempted_prefix_cache_hit_is_counted_once() {
    let mut scheduler = test_scheduler();
    let tokens = vec![1; 16];
    let hashes = compute_block_hashes(&tokens, scheduler.block_size, &[], &[]);
    {
        let mut kv_mgr = get_mut_arcmutex!(scheduler.kv_cache_manager);
        assert!(kv_mgr.allocate_slots(99, tokens.len(), &[]).is_some());
        kv_mgr.cache_blocks(99, &hashes, scheduler.block_size);
        kv_mgr.free(99);
    }

    let seq = test_sequence(0, tokens.len());
    get_mut_arcmutex!(seq).set_state(SequenceState::Waiting);
    scheduler.waiting.push_back(seq);

    let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
    logger.add_new_sequence();
    let first = scheduler.schedule(&logger, None);

    assert_eq!(first.num_cached_tokens, vec![scheduler.block_size]);
    assert_eq!(logger.prefix_cache_stats(), (1, 1));

    let seq = first.scheduled[0].clone();
    scheduler.running.clear();
    scheduler._preempt(seq);
    let second = scheduler.schedule(&logger, None);

    assert_eq!(second.num_cached_tokens, vec![scheduler.block_size]);
    assert_eq!(logger.prefix_cache_stats(), (1, 1));
}

#[test]
fn underfilled_decode_gets_one_completion_turn_before_refill() {
    let mut scheduler = test_scheduler();
    let running = test_sequence(0, 4);
    get_mut_arcmutex!(running).set_state(SequenceState::RunningPrompt);
    get_mut_arcmutex!(running).set_num_computed_tokens(4);
    scheduler.running.push_back(running);

    let waiting = test_sequence(1, 7);
    get_mut_arcmutex!(waiting).set_state(SequenceState::Waiting);
    scheduler.waiting.push_back(waiting);

    let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
    let completion = scheduler.schedule(&logger, None);
    assert_eq!(completion.scheduled.len(), 1);
    assert!(!get_mut_arcmutex!(completion.scheduled[0]).is_prompt());
    assert_eq!(scheduler.waiting.len(), 1);

    let prompt = scheduler.schedule(&logger, None);

    assert_eq!(prompt.scheduled.len(), 1);
    assert!(get_mut_arcmutex!(prompt.scheduled[0]).is_prompt());
    assert!(scheduler.waiting.is_empty());
}

#[test]
fn running_prompt_tail_within_budget_yields_after_prefill() {
    let mut scheduler = test_scheduler();
    let completion = test_sequence(0, 8);
    get_mut_arcmutex!(completion).set_num_computed_tokens(8);
    scheduler.running.push_back(completion);

    let prompt = test_sequence(1, 12);
    {
        let mut prompt = get_mut_arcmutex!(prompt);
        prompt.set_state(SequenceState::RunningPrompt);
        prompt.set_prefix_cache_len(8);
        prompt.set_num_computed_tokens(8);
    }
    scheduler.running.push_back(prompt);

    let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
    let decode = scheduler.schedule(&logger, None);
    assert_eq!(decode.scheduled.len(), 1);
    assert_eq!(*get_mut_arcmutex!(decode.scheduled[0]).id(), 0);

    let prompt = scheduler.schedule(&logger, None);
    assert_eq!(prompt.scheduled.len(), 1);
    assert_eq!(*get_mut_arcmutex!(prompt.scheduled[0]).id(), 1);
    assert!(get_mut_arcmutex!(prompt.scheduled[0]).is_prompt());
}

#[test]
fn incompatible_prompt_tails_yield_between_batches() {
    let mut scheduler = test_scheduler();
    scheduler.config.max_num_batched_tokens = 8;
    scheduler.scheduler_visible_prompt_chunks = true;
    scheduler.running.push_back(test_sequence(0, 8));

    for (id, len) in [(1, 10), (2, 11)] {
        let prompt = test_sequence(id, len);
        {
            let mut prompt = get_mut_arcmutex!(prompt);
            prompt.set_state(SequenceState::RunningPrompt);
            prompt.set_prefix_cache_len(8);
            prompt.set_num_computed_tokens(8);
        }
        scheduler.running.push_back(prompt);
    }

    let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
    let decode = scheduler.schedule(&logger, None);
    assert_eq!(*get_mut_arcmutex!(decode.scheduled[0]).id(), 0);

    let first = scheduler.schedule(&logger, None);
    assert_eq!(first.scheduled.len(), 1);
    assert_eq!(*get_mut_arcmutex!(first.scheduled[0]).id(), 1);
    get_mut_arcmutex!(first.scheduled[0]).set_num_computed_tokens(10);

    let decode = scheduler.schedule(&logger, None);
    assert_eq!(*get_mut_arcmutex!(decode.scheduled[0]).id(), 0);

    let second = scheduler.schedule(&logger, None);
    assert_eq!(second.scheduled.len(), 1);
    assert_eq!(*get_mut_arcmutex!(second.scheduled[0]).id(), 2);
    assert!(get_mut_arcmutex!(second.scheduled[0]).is_prompt());
}

#[test]
fn chunked_prompt_buckets_rotate_without_discarding_partial_state() {
    let mut scheduler = test_scheduler();
    scheduler.config.max_num_batched_tokens = 4;
    scheduler.scheduler_visible_prompt_chunks = true;

    for (id, computed) in [(0, 4), (1, 0), (2, 4)] {
        let prompt = test_sequence(id, 12);
        {
            let mut prompt = get_mut_arcmutex!(prompt);
            prompt.set_state(SequenceState::RunningPrompt);
            prompt.set_prefix_cache_len(computed);
            prompt.set_num_computed_tokens(computed);
        }
        scheduler.running.push_back(prompt);
    }

    let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
    let first = scheduler.schedule(&logger, None);
    assert_eq!(
        first
            .scheduled
            .iter()
            .map(|seq| *get_mut_arcmutex!(seq).id())
            .collect::<Vec<_>>(),
        vec![0, 2]
    );
    for seq in &first.scheduled {
        let mut seq = get_mut_arcmutex!(seq);
        seq.set_prefix_cache_len(6);
        seq.set_num_computed_tokens(6);
    }

    let second = scheduler.schedule(&logger, None);
    assert_eq!(second.scheduled.len(), 1);
    assert_eq!(*get_mut_arcmutex!(second.scheduled[0]).id(), 1);
    assert_eq!(scheduler.running.len(), 3);
    assert!(scheduler.waiting.is_empty());
    assert_eq!(
        scheduler
            .running
            .iter()
            .map(|seq| get_mut_arcmutex!(seq).num_computed_tokens())
            .collect::<Vec<_>>(),
        vec![6, 0, 6]
    );

    {
        let mut seq = get_mut_arcmutex!(second.scheduled[0]);
        seq.set_prefix_cache_len(4);
        seq.set_num_computed_tokens(4);
    }
    let third = scheduler.schedule(&logger, None);
    assert_eq!(
        third
            .scheduled
            .iter()
            .map(|seq| *get_mut_arcmutex!(seq).id())
            .collect::<Vec<_>>(),
        vec![2, 0]
    );
}

#[test]
fn active_decodes_are_prioritized_over_new_prompts() {
    let mut scheduler = test_scheduler();
    scheduler.running.push_back(test_sequence(0, 4));

    let waiting = test_sequence(1, 7);
    get_mut_arcmutex!(waiting).set_state(SequenceState::Waiting);
    scheduler.waiting.push_back(waiting);

    let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
    let completion = scheduler.schedule(&logger, None);

    assert_eq!(completion.scheduled.len(), 1);
    assert!(!get_mut_arcmutex!(completion.scheduled[0]).is_prompt());
    assert_eq!(scheduler.waiting.len(), 1);
}

#[test]
fn completion_turn_survives_finished_prompt_cleanup() {
    let mut scheduler = test_scheduler();
    scheduler.running.push_back(test_sequence(0, 4));
    scheduler.decode_steps_since_prefill = 0;

    let waiting = test_sequence(1, 7);
    get_mut_arcmutex!(waiting).set_state(SequenceState::Waiting);
    scheduler.waiting.push_back(waiting);

    let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
    let completion = scheduler.schedule(&logger, None);

    assert_eq!(completion.scheduled.len(), 1);
    assert_eq!(*get_mut_arcmutex!(completion.scheduled[0]).id(), 0);
    assert_eq!(scheduler.waiting.len(), 1);
}

#[test]
fn completion_turn_without_running_sequences_does_not_delay_prompts() {
    let mut scheduler = test_scheduler();
    scheduler.decode_steps_since_prefill = 0;

    let waiting = test_sequence(1, 7);
    get_mut_arcmutex!(waiting).set_state(SequenceState::Waiting);
    scheduler.waiting.push_back(waiting);

    let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
    let prompt = scheduler.schedule(&logger, None);

    assert_eq!(prompt.scheduled.len(), 1);
    assert!(get_mut_arcmutex!(prompt.scheduled[0]).is_prompt());
    assert!(scheduler.waiting.is_empty());
}

#[test]
fn prompt_chunk_size_stays_within_the_batch_token_budget() {
    let mut scheduler = test_scheduler();
    scheduler.config.max_num_batched_tokens = 4096;

    assert_eq!(scheduler.prompt_chunk_size(0, false), None);
    assert_eq!(scheduler.prompt_chunk_size(1, false), Some(4096));
    assert_eq!(scheduler.prompt_chunk_size(8, false), Some(512));
    assert_eq!(scheduler.prompt_chunk_size(16, false), Some(256));
    assert_eq!(scheduler.prompt_chunk_size(7, false), Some(585));
    assert!(scheduler.prompt_chunk_size(7, false).unwrap() * 7 <= 4096);
}

#[test]
fn prompt_chunk_size_uses_the_prefill_latency_budget() {
    let mut scheduler = test_scheduler();
    scheduler.scheduler_visible_prompt_chunks = true;
    scheduler.config.max_num_batched_tokens = 4096;
    scheduler.config.max_prefill_chunk_tokens = 512;

    assert_eq!(scheduler.prompt_chunk_size(1, true), Some(512));
    assert_eq!(scheduler.prompt_chunk_size(8, true), Some(64));
    assert_eq!(scheduler.prompt_chunk_size(16, true), Some(32));
    assert_eq!(scheduler.prompt_chunk_size(16, false), Some(256));
}

#[test]
fn mixed_prefill_budget_stays_bounded_between_prompt_turns() {
    let mut scheduler = test_scheduler();
    scheduler.scheduler_visible_prompt_chunks = true;
    scheduler.config.max_num_batched_tokens = 4096;
    scheduler.config.max_prefill_chunk_tokens = 512;

    assert_eq!(scheduler.prefill_token_budget(true), 512);
    scheduler.decode_steps_since_prefill = 4;
    assert_eq!(scheduler.prefill_token_budget(true), 512);
    scheduler.decode_steps_since_prefill = 8;
    assert_eq!(scheduler.prefill_token_budget(true), 512);
    scheduler.decode_steps_since_prefill = usize::MAX;
    assert_eq!(scheduler.prefill_token_budget(true), 512);
}

#[test]
fn idle_prompt_backlog_uses_full_admission_budget() {
    let mut scheduler = test_scheduler();
    scheduler.scheduler_visible_prompt_chunks = true;
    scheduler.config.max_prefill_chunk_tokens = 512;
    let prompt = test_sequence(0, 1024);
    get_mut_arcmutex!(prompt).set_state(SequenceState::Waiting);
    scheduler.waiting.push_back(prompt);

    scheduler.update_prompt_admission_epoch();

    assert!(scheduler.prompt_admission_epoch);
    assert_eq!(scheduler.prefill_token_budget(true), 4096);
}

#[test]
fn established_decode_keeps_latency_bounded_prompt_budget() {
    let mut scheduler = test_scheduler();
    scheduler.scheduler_visible_prompt_chunks = true;
    scheduler.config.max_prefill_chunk_tokens = 512;
    scheduler.running.push_back(test_sequence(0, 4));
    let prompt = test_sequence(1, 1024);
    get_mut_arcmutex!(prompt).set_state(SequenceState::Waiting);
    scheduler.waiting.push_back(prompt);

    scheduler.update_prompt_admission_epoch();

    assert!(!scheduler.prompt_admission_epoch);
    assert_eq!(scheduler.prefill_token_budget(true), 512);
}

#[test]
fn admission_epoch_allows_one_decode_turn_between_prompt_batches() {
    let mut scheduler = test_scheduler();
    scheduler.scheduler_visible_prompt_chunks = true;
    scheduler.config.max_prefill_chunk_tokens = 512;
    scheduler.prompt_admission_epoch = true;
    scheduler.running.push_back(test_sequence(0, 4));
    let prompt = test_sequence(1, 8192);
    get_mut_arcmutex!(prompt).set_state(SequenceState::RunningPrompt);
    scheduler.running.push_back(prompt);

    assert!(scheduler.completion_is_due());
    scheduler.decode_steps_since_prefill = 1;
    assert!(!scheduler.completion_is_due());
}

#[test]
fn admission_epoch_clears_after_prompt_backlog_drains() {
    let mut scheduler = test_scheduler();
    scheduler.prompt_admission_epoch = true;
    scheduler.running.push_back(test_sequence(0, 4));

    scheduler.update_prompt_admission_epoch();

    assert!(scheduler.prompt_admission_epoch);
    scheduler.decode_steps_since_prefill = 1;
    scheduler.update_prompt_admission_epoch();

    assert!(!scheduler.prompt_admission_epoch);
}

#[test]
fn established_decode_long_prompt_retains_normal_cadence() {
    let mut scheduler = test_scheduler();
    scheduler.scheduler_visible_prompt_chunks = true;
    scheduler.config.max_prefill_chunk_tokens = 512;
    scheduler.running.push_back(test_sequence(0, 4));
    let prompt = test_sequence(1, 8192);
    get_mut_arcmutex!(prompt).set_state(SequenceState::RunningPrompt);
    scheduler.running.push_back(prompt);

    scheduler.decode_steps_since_prefill = 7;
    assert!(scheduler.completion_is_due());
    scheduler.decode_steps_since_prefill = 8;
    assert!(!scheduler.completion_is_due());
}

#[test]
fn prefill_latency_budget_does_not_change_atomic_prompt_paths() {
    let mut scheduler = test_scheduler();
    scheduler.config.max_num_batched_tokens = 4096;
    scheduler.config.max_prefill_chunk_tokens = 512;

    assert_eq!(scheduler.prompt_chunk_size(1, true), Some(4096));
}

#[test]
fn completion_priority_uses_the_prefill_latency_budget() {
    let mut scheduler = test_scheduler();
    scheduler.scheduler_visible_prompt_chunks = true;
    scheduler.config.max_num_batched_tokens = 4096;
    scheduler.config.max_prefill_chunk_tokens = 512;
    scheduler.running.push_back(test_sequence(0, 4));
    let prompt = test_sequence(1, 1024);
    get_mut_arcmutex!(prompt).set_state(SequenceState::RunningPrompt);
    scheduler.running.push_back(prompt.clone());

    assert!(scheduler.completion_is_due());

    get_mut_arcmutex!(prompt).set_num_computed_tokens(512);
    assert!(scheduler.completion_is_due());

    scheduler.decode_steps_since_prefill = 1;
    assert!(!scheduler.completion_is_due());
}

#[test]
fn prompt_batch_does_not_exceed_the_prefill_latency_budget() {
    let mut scheduler = test_scheduler();
    scheduler.scheduler_visible_prompt_chunks = true;
    scheduler.config.max_num_batched_tokens = 16;
    scheduler.config.max_prefill_chunk_tokens = 2;
    let completion = test_sequence(100, 8);
    get_mut_arcmutex!(completion).set_num_computed_tokens(8);
    scheduler.running.push_back(completion);
    let candidates = (0..3)
        .map(|id| {
            let seq = test_sequence(id, 8);
            get_mut_arcmutex!(seq).set_state(SequenceState::RunningPrompt);
            seq
        })
        .collect::<VecDeque<_>>();

    let batch = scheduler.select_prompt_batch(candidates);

    assert_eq!(batch.scheduled.len(), 2);
    assert_eq!(batch.chunk_size, Some(1));
    assert_eq!(
        batch
            .chunks
            .unwrap()
            .into_iter()
            .map(|chunk| chunk.end - chunk.start)
            .sum::<usize>(),
        2
    );
}

#[test]
fn pure_prefill_batch_uses_the_full_token_budget() {
    let mut scheduler = test_scheduler();
    scheduler.scheduler_visible_prompt_chunks = true;
    scheduler.config.max_num_batched_tokens = 16;
    scheduler.config.max_prefill_chunk_tokens = 2;
    let candidates = (0..4)
        .map(|id| {
            let seq = test_sequence(id, 8);
            get_mut_arcmutex!(seq).set_state(SequenceState::RunningPrompt);
            seq
        })
        .collect::<VecDeque<_>>();

    let batch = scheduler.select_prompt_batch(candidates);

    assert_eq!(batch.scheduled.len(), 4);
    assert_eq!(batch.chunk_size, Some(4));
    assert_eq!(
        batch
            .chunks
            .unwrap()
            .into_iter()
            .map(|chunk| chunk.end - chunk.start)
            .sum::<usize>(),
        16
    );
}

fn span_packing_scheduler() -> PagedAttentionScheduler {
    let mut scheduler = test_scheduler();
    scheduler.scheduler_visible_prompt_chunks = true;
    scheduler.supports_packed_prefill = true;
    scheduler.requires_uniform_prompt_batch = false;
    scheduler.config.max_num_seqs = 64;
    scheduler.config.max_prefill_chunk_tokens = 512;
    scheduler
}

fn span_packing_prompts(lengths: &[usize]) -> VecDeque<Arc<Mutex<Sequence>>> {
    lengths
        .iter()
        .enumerate()
        .map(|(id, &length)| {
            let seq = test_sequence(id, length);
            get_mut_arcmutex!(seq).set_state(SequenceState::RunningPrompt);
            seq
        })
        .collect()
}

#[test]
fn packed_prompt_spans_finish_whole_prompts_within_the_idle_budget() {
    let mut scheduler = span_packing_scheduler();
    let candidates = span_packing_prompts(&[1024; 64]);

    let batch = scheduler.select_prompt_batch(candidates);

    assert_eq!(batch.scheduled.len(), 4);
    assert_eq!(scheduler.next_prompt_sequence_id, Some(4));
    assert!(batch
        .chunks
        .unwrap()
        .iter()
        .all(|chunk| (chunk.start, chunk.end) == (0, 1024)));
}

#[test]
fn packed_prompt_spans_pack_ragged_cold_queries_without_padding() {
    let mut scheduler = span_packing_scheduler();
    scheduler.config.max_num_batched_tokens = 512;
    let batch = scheduler.select_prompt_batch(span_packing_prompts(&[128, 256, 128, 64]));

    assert_eq!(batch.scheduled.len(), 3);
    assert_eq!(scheduler.next_prompt_sequence_id, Some(3));
    assert_eq!(
        batch
            .chunks
            .unwrap()
            .iter()
            .map(|chunk| chunk.end - chunk.start)
            .sum::<usize>(),
        512
    );
}

#[test]
fn packed_prompt_spans_budget_padding_for_cached_queries() {
    let mut scheduler = span_packing_scheduler();
    scheduler.config.max_num_batched_tokens = 512;
    let candidates = span_packing_prompts(&[512, 256, 256]);
    for seq in &candidates {
        get_mut_arcmutex!(seq).set_num_computed_tokens(128);
    }

    let batch = scheduler.select_prompt_batch(candidates);

    assert_eq!(batch.scheduled.len(), 1);
    assert_eq!(scheduler.next_prompt_sequence_id, Some(1));
    assert_eq!(
        (
            batch.chunks.as_ref().unwrap()[0].start,
            batch.chunks.unwrap()[0].end
        ),
        (128, 512)
    );
}

#[test]
fn packed_prompt_spans_keep_the_mixed_budget_and_rotate_without_starvation() {
    let mut scheduler = span_packing_scheduler();
    scheduler.running.push_back(test_sequence(100, 8));
    let candidates = span_packing_prompts(&[1024; 63]);
    scheduler.running.extend(candidates.iter().cloned());
    let mut seen = Vec::new();

    for _ in 0..candidates.len() {
        let batch = scheduler.select_prompt_batch(candidates.clone());
        assert_eq!(batch.scheduled.len(), 1);
        assert_eq!(
            (
                batch.chunks.as_ref().unwrap()[0].start,
                batch.chunks.unwrap()[0].end
            ),
            (0, 512)
        );
        seen.push(*get_mut_arcmutex!(batch.scheduled[0]).id());
    }
    assert_eq!(seen, (0..63).collect::<Vec<_>>());
    assert!(scheduler.completion_is_due());
    scheduler.decode_steps_since_prefill = scheduler.config.max_decode_steps_before_prefill;
    assert!(!scheduler.completion_is_due());
}

#[test]
fn packed_prompt_spans_respect_the_admission_quantum() {
    let mut scheduler = span_packing_scheduler();
    scheduler.prompt_admission_epoch = true;
    scheduler.running.push_back(test_sequence(100, 8));
    let candidates = span_packing_prompts(&[1024; 63]);
    scheduler.running.extend(candidates.iter().cloned());

    let batch = scheduler.select_prompt_batch(candidates);

    assert_eq!(batch.scheduled.len(), 8);
    assert_eq!(scheduler.next_prompt_sequence_id, Some(8));
    assert!(batch
        .chunks
        .unwrap()
        .iter()
        .all(|chunk| (chunk.start, chunk.end) == (0, 512)));
    assert!(scheduler.completion_is_due());
    scheduler.decode_steps_since_prefill = 1;
    assert!(!scheduler.completion_is_due());
}

#[test]
fn packed_prompt_spans_do_not_mix_final_and_nonfinal_queries() {
    let mut scheduler = span_packing_scheduler();
    scheduler.config.max_num_batched_tokens = 512;
    let candidates = span_packing_prompts(&[1024, 128, 128]);

    let first = scheduler.select_prompt_batch(candidates.clone());
    let second = scheduler.select_prompt_batch(candidates);

    assert_eq!(first.scheduled.len(), 1);
    assert_eq!(first.chunks.unwrap()[0].end, 512);
    assert_eq!(second.scheduled.len(), 2);
    assert!(second.chunks.unwrap().iter().all(|chunk| chunk.end == 128));
}

#[test]
fn packed_prompt_spans_require_the_runtime_packing_capability() {
    let mut scheduler = span_packing_scheduler();
    scheduler.supports_packed_prefill = false;
    let batch = scheduler.select_prompt_batch(span_packing_prompts(&[1024; 64]));

    assert_eq!(batch.scheduled.len(), 64);
    assert_eq!(batch.chunk_size, Some(64));
    assert!(batch.chunks.unwrap().iter().all(|chunk| chunk.end == 64));
}

#[test]
fn packed_prompt_spans_keep_small_aligned_budgets_nonempty() {
    let mut scheduler = span_packing_scheduler();
    scheduler.config.max_num_batched_tokens = 16;
    scheduler.prompt_chunks_require_block_alignment = true;
    scheduler.block_size = 32;
    let candidates = span_packing_prompts(&[80; 3]);
    for seq in &candidates {
        get_mut_arcmutex!(seq).set_num_computed_tokens(31);
    }

    let batch = scheduler.select_prompt_batch(candidates);

    assert_eq!(batch.scheduled.len(), 3);
    assert!(batch
        .chunks
        .unwrap()
        .iter()
        .all(|chunk| (chunk.start, chunk.end) == (31, 32)));
}

#[test]
fn packed_prompt_spans_preserve_recurrent_and_multimodal_planning() {
    let mut scheduler = span_packing_scheduler();
    scheduler.prefill_has_per_sequence_state = true;
    let recurrent = scheduler.select_prompt_batch(span_packing_prompts(&[1024; 64]));
    assert_eq!(recurrent.scheduled.len(), 8);
    assert_eq!(recurrent.chunk_size, Some(512));

    let mut scheduler = span_packing_scheduler();
    let candidates = span_packing_prompts(&[1024; 64]);
    get_mut_arcmutex!(candidates[63]).set_mm_features(vec![MultiModalFeature {
        kind: MultimodalKind::Image,
        item_range: 0..1,
        hashes: vec![1],
        offset: 0,
        length: 32,
        attention_policy: crate::paged_attention::block_hash::MultimodalAttentionPolicy::Causal,
        splittable: false,
    }]);
    let multimodal = scheduler.select_prompt_batch(candidates);
    assert_eq!(multimodal.scheduled.len(), 64);
    assert_eq!(multimodal.chunk_size, Some(64));
}

#[test]
fn recurrent_prefill_preserves_efficient_sequence_width_and_rotates() {
    let mut scheduler = test_scheduler();
    scheduler.scheduler_visible_prompt_chunks = true;
    scheduler.prefill_has_per_sequence_state = true;
    scheduler.config.max_num_seqs = 16;
    scheduler.config.max_num_batched_tokens = 4096;
    scheduler.config.max_prefill_chunk_tokens = 512;
    let candidates = (0..16)
        .map(|id| {
            let seq = test_sequence(id, 8192);
            get_mut_arcmutex!(seq).set_state(SequenceState::RunningPrompt);
            seq
        })
        .collect::<VecDeque<_>>();

    let first = scheduler.select_prompt_batch(candidates.clone());
    assert_eq!(first.scheduled.len(), 8);
    assert_eq!(first.chunk_size, Some(512));
    assert_eq!(
        first
            .scheduled
            .iter()
            .map(|seq| *get_mut_arcmutex!(seq).id())
            .collect::<Vec<_>>(),
        (0..8).collect::<Vec<_>>()
    );

    let second = scheduler.select_prompt_batch(candidates);
    assert_eq!(second.scheduled.len(), 8);
    assert_eq!(second.chunk_size, Some(512));
    assert_eq!(
        second
            .scheduled
            .iter()
            .map(|seq| *get_mut_arcmutex!(seq).id())
            .collect::<Vec<_>>(),
        (8..16).collect::<Vec<_>>()
    );
}

#[test]
fn recurrent_prefill_with_decode_uses_one_sequence_quantum() {
    let mut scheduler = test_scheduler();
    scheduler.scheduler_visible_prompt_chunks = true;
    scheduler.prefill_has_per_sequence_state = true;
    scheduler.config.max_num_seqs = 16;
    scheduler.config.max_num_batched_tokens = 4096;
    scheduler.config.max_prefill_chunk_tokens = 512;
    scheduler.running.push_back(test_sequence(100, 8));
    let candidates = (0..16)
        .map(|id| {
            let seq = test_sequence(id, 8192);
            get_mut_arcmutex!(seq).set_state(SequenceState::RunningPrompt);
            seq
        })
        .collect::<VecDeque<_>>();

    let batch = scheduler.select_prompt_batch(candidates);

    assert_eq!(batch.scheduled.len(), 1);
    assert_eq!(batch.chunk_size, Some(512));
    assert_eq!(batch.chunks.unwrap()[0].end, 512);
}

#[test]
fn recurrent_short_prefills_share_the_mixed_token_budget() {
    let mut scheduler = test_scheduler();
    scheduler.scheduler_visible_prompt_chunks = true;
    scheduler.prompt_chunks_require_block_alignment = true;
    scheduler.prefill_has_per_sequence_state = true;
    scheduler.config.max_num_seqs = 64;
    scheduler.config.max_num_batched_tokens = 4096;
    scheduler.config.max_prefill_chunk_tokens = 512;
    scheduler.running.push_back(test_sequence(100, 8));
    let candidates = (0..16)
        .map(|id| {
            let seq = test_sequence(id, 134);
            get_mut_arcmutex!(seq).set_state(SequenceState::RunningPrompt);
            seq
        })
        .collect::<VecDeque<_>>();

    let batch = scheduler.select_prompt_batch(candidates);

    assert_eq!(batch.scheduled.len(), 4);
    assert_eq!(batch.chunk_size, Some(128));
    assert_eq!(
        batch
            .chunks
            .unwrap()
            .into_iter()
            .map(|chunk| chunk.end - chunk.start)
            .sum::<usize>(),
        512
    );
}

#[test]
fn recurrent_short_prefills_remain_latency_bounded_after_steady_decode() {
    let mut scheduler = test_scheduler();
    scheduler.scheduler_visible_prompt_chunks = true;
    scheduler.prompt_chunks_require_block_alignment = true;
    scheduler.prefill_has_per_sequence_state = true;
    scheduler.config.max_num_seqs = 64;
    scheduler.config.max_num_batched_tokens = 4096;
    scheduler.config.max_prefill_chunk_tokens = 512;
    scheduler.decode_steps_since_prefill = 8;
    scheduler.running.push_back(test_sequence(100, 8));
    let candidates = (0..32)
        .map(|id| {
            let seq = test_sequence(id, 134);
            get_mut_arcmutex!(seq).set_state(SequenceState::RunningPrompt);
            seq
        })
        .collect::<VecDeque<_>>();

    let batch = scheduler.select_prompt_batch(candidates);

    assert_eq!(batch.scheduled.len(), 4);
    assert_eq!(batch.chunk_size, Some(128));
    assert_eq!(
        batch
            .chunks
            .unwrap()
            .into_iter()
            .map(|chunk| chunk.end - chunk.start)
            .sum::<usize>(),
        512
    );
}

#[test]
fn admission_epoch_uses_the_full_budget_for_short_prefills() {
    let mut scheduler = test_scheduler();
    scheduler.scheduler_visible_prompt_chunks = true;
    scheduler.prompt_chunks_require_block_alignment = true;
    scheduler.prefill_has_per_sequence_state = true;
    scheduler.prompt_admission_epoch = true;
    scheduler.set_prefix_caching_enabled_sync(true);
    scheduler.config.max_num_seqs = 64;
    scheduler.config.max_num_batched_tokens = 4096;
    scheduler.config.max_prefill_chunk_tokens = 512;
    scheduler.running.push_back(test_sequence(100, 8));
    let candidates = (0..32)
        .map(|id| {
            let seq = test_sequence(id, 134);
            get_mut_arcmutex!(seq).set_state(SequenceState::RunningPrompt);
            seq
        })
        .collect::<VecDeque<_>>();

    let batch = scheduler.select_prompt_batch(candidates);

    assert_eq!(batch.scheduled.len(), 32);
    assert_eq!(batch.chunk_size, Some(128));
    assert!(batch
        .chunks
        .as_ref()
        .unwrap()
        .iter()
        .all(|chunk| (chunk.start, chunk.end) == (0, 128)));
    assert_eq!(
        batch
            .chunks
            .unwrap()
            .into_iter()
            .map(|chunk| chunk.end - chunk.start)
            .sum::<usize>(),
        4096
    );
}

#[test]
fn admission_epoch_packs_complete_short_prefills_without_prefix_caching() {
    let mut scheduler = test_scheduler();
    scheduler.scheduler_visible_prompt_chunks = true;
    scheduler.prompt_chunks_require_block_alignment = true;
    scheduler.prefill_has_per_sequence_state = true;
    scheduler.prompt_admission_epoch = true;
    scheduler.set_prefix_caching_enabled_sync(false);
    scheduler.config.max_num_seqs = 64;
    scheduler.config.max_num_batched_tokens = 4096;
    scheduler.config.max_prefill_chunk_tokens = 512;
    scheduler.running.push_back(test_sequence(100, 8));
    let candidates = (0..32)
        .map(|id| {
            let seq = test_sequence(id, 134);
            get_mut_arcmutex!(seq).set_state(SequenceState::RunningPrompt);
            seq
        })
        .collect::<VecDeque<_>>();

    let batch = scheduler.select_prompt_batch(candidates);

    assert_eq!(batch.scheduled.len(), 30);
    assert_eq!(batch.chunk_size, Some(134));
    assert!(batch
        .chunks
        .as_ref()
        .unwrap()
        .iter()
        .all(|chunk| (chunk.start, chunk.end) == (0, 134)));
    assert_eq!(
        batch
            .chunks
            .unwrap()
            .into_iter()
            .map(|chunk| chunk.end - chunk.start)
            .sum::<usize>(),
        4020
    );
}

#[test]
fn admission_epoch_advances_the_rotating_prompt_frontier() {
    let mut scheduler = test_scheduler();
    scheduler.scheduler_visible_prompt_chunks = true;
    scheduler.prompt_chunks_require_block_alignment = true;
    scheduler.prefill_has_per_sequence_state = true;
    scheduler.supports_packed_prefill = true;
    scheduler.requires_uniform_prompt_batch = false;
    scheduler.prompt_admission_epoch = true;
    scheduler.block_size = 32;
    scheduler.config.max_num_seqs = 64;
    scheduler.config.max_num_batched_tokens = 4096;
    scheduler.config.max_prefill_chunk_tokens = 512;
    scheduler.running.push_back(test_sequence(100, 8));

    let tail = test_sequence(0, 134);
    {
        let mut tail = get_mut_arcmutex!(tail);
        tail.set_state(SequenceState::RunningPrompt);
        tail.set_num_computed_tokens(128);
    }
    let unstarted = (1..=2)
        .map(|id| {
            let seq = test_sequence(id, 134);
            get_mut_arcmutex!(seq).set_state(SequenceState::RunningPrompt);
            seq
        })
        .collect::<Vec<_>>();
    let batch = scheduler.select_prompt_batch(VecDeque::from([
        tail,
        unstarted[0].clone(),
        unstarted[1].clone(),
    ]));
    assert_eq!(batch.scheduled.len(), 1);
    assert_eq!(*get_mut_arcmutex!(batch.scheduled[0]).id(), 0);
    assert_eq!(batch.chunks.unwrap()[0].start, 128);

    let batch =
        scheduler.select_prompt_batch(VecDeque::from([unstarted[0].clone(), unstarted[1].clone()]));
    assert_eq!(
        batch
            .scheduled
            .iter()
            .map(|seq| *get_mut_arcmutex!(seq).id())
            .collect::<Vec<_>>(),
        vec![1, 2]
    );
    assert!(batch
        .chunks
        .unwrap()
        .iter()
        .all(|chunk| (chunk.start, chunk.end) == (0, 128)));

    let short_tail = test_sequence(3, 1000);
    {
        let mut short_tail = get_mut_arcmutex!(short_tail);
        short_tail.set_state(SequenceState::RunningPrompt);
        short_tail.set_num_computed_tokens(992);
    }
    let advanced_long = test_sequence(4, 8192);
    {
        let mut advanced_long = get_mut_arcmutex!(advanced_long);
        advanced_long.set_state(SequenceState::RunningPrompt);
        advanced_long.set_num_computed_tokens(1024);
    }
    let batch = scheduler.select_prompt_batch(VecDeque::from([advanced_long, short_tail.clone()]));
    assert_eq!(batch.scheduled.len(), 1);
    assert_eq!(*get_mut_arcmutex!(batch.scheduled[0]).id(), 4);
}

#[test]
fn admission_epoch_does_not_bypass_a_prefix_hit() {
    let mut scheduler = test_scheduler();
    scheduler.scheduler_visible_prompt_chunks = true;
    scheduler.prompt_chunks_require_block_alignment = true;
    scheduler.prefill_has_per_sequence_state = true;
    scheduler.supports_packed_prefill = true;
    scheduler.requires_uniform_prompt_batch = false;
    scheduler.prompt_admission_epoch = true;
    scheduler.block_size = 32;
    scheduler.config.max_num_seqs = 64;
    scheduler.config.max_num_batched_tokens = 4096;
    scheduler.config.max_prefill_chunk_tokens = 512;
    scheduler.next_prompt_sequence_id = Some(0);

    let cached = test_sequence(0, 134);
    {
        let mut cached = get_mut_arcmutex!(cached);
        cached.set_state(SequenceState::RunningPrompt);
        cached.set_prefix_cache_len(96);
        cached.set_num_computed_tokens(96);
    }
    let cold = test_sequence(1, 134);
    get_mut_arcmutex!(cold).set_state(SequenceState::RunningPrompt);

    let batch = scheduler.select_prompt_batch(VecDeque::from([cached, cold]));

    assert_eq!(batch.scheduled.len(), 1);
    assert_eq!(*get_mut_arcmutex!(batch.scheduled[0]).id(), 0);
    assert_eq!(batch.chunks.unwrap()[0].start, 96);
    assert_eq!(scheduler.next_prompt_sequence_id, Some(1));
}

#[test]
fn recurrent_prompt_tails_use_their_actual_token_cost() {
    let mut scheduler = test_scheduler();
    scheduler.scheduler_visible_prompt_chunks = true;
    scheduler.prompt_chunks_require_block_alignment = true;
    scheduler.prefill_has_per_sequence_state = true;
    scheduler.config.max_num_seqs = 64;
    scheduler.config.max_num_batched_tokens = 4096;
    scheduler.config.max_prefill_chunk_tokens = 512;
    scheduler.running.push_back(test_sequence(100, 8));
    let candidates = (0..16)
        .map(|id| {
            let seq = test_sequence(id, 134);
            {
                let mut seq_guard = get_mut_arcmutex!(seq);
                seq_guard.set_state(SequenceState::RunningPrompt);
                seq_guard.set_num_computed_tokens(128);
            }
            seq
        })
        .collect::<VecDeque<_>>();

    let batch = scheduler.select_prompt_batch(candidates);

    assert_eq!(batch.scheduled.len(), 16);
}

#[test]
fn packed_recurrent_final_tails_can_be_ragged() {
    let mut scheduler = test_scheduler();
    scheduler.scheduler_visible_prompt_chunks = true;
    scheduler.prompt_chunks_require_block_alignment = true;
    scheduler.prefill_has_per_sequence_state = true;
    scheduler.supports_packed_prefill = true;
    scheduler.requires_uniform_prompt_batch = false;
    scheduler.block_size = 32;
    scheduler.config.max_num_seqs = 64;
    scheduler.config.max_num_batched_tokens = 4096;
    scheduler.config.max_prefill_chunk_tokens = 512;
    scheduler.running.push_back(test_sequence(100, 8));
    let candidates = (1..=12)
        .map(|tail_len| {
            let seq = test_sequence(tail_len, 128 + tail_len);
            {
                let mut seq_guard = get_mut_arcmutex!(seq);
                seq_guard.set_state(SequenceState::RunningPrompt);
                seq_guard.set_num_computed_tokens(128);
            }
            seq
        })
        .collect::<VecDeque<_>>();

    let batch = scheduler.select_prompt_batch(candidates);

    assert_eq!(batch.scheduled.len(), 12);
    assert_eq!(
        batch
            .chunks
            .unwrap()
            .into_iter()
            .map(|chunk| chunk.end - chunk.start)
            .collect::<Vec<_>>(),
        (1..=12).collect::<Vec<_>>()
    );
}

#[test]
fn token_budget_caps_prompt_admission() {
    let mut scheduler = test_scheduler();
    scheduler.config.max_num_batched_tokens = 3;
    for id in 0..5 {
        let seq = test_sequence(id, 4);
        get_mut_arcmutex!(seq).set_state(SequenceState::Waiting);
        scheduler.waiting.push_back(seq);
    }

    let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
    let prompt = scheduler.schedule(&logger, None);

    assert_eq!(prompt.scheduled.len(), 3);
    assert_eq!(prompt.prompt_chunk_size, Some(1));
    assert_eq!(scheduler.waiting.len(), 2);
}

#[test]
fn deferred_prompt_tail_runs_first_without_reallocating_kv() {
    let mut scheduler = test_scheduler();
    Scheduler::set_scheduler_visible_prompt_chunks(
        &mut scheduler,
        true,
        false,
        SpeculativePrefixCheckpointPolicy::default(),
    );
    for id in 0..3 {
        let seq = test_sequence(id, 16);
        get_mut_arcmutex!(seq).set_state(SequenceState::Waiting);
        scheduler.waiting.push_back(seq);
    }

    let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
    let mut first = scheduler.schedule(&logger, None);
    assert_eq!(
        first
            .scheduled
            .iter()
            .map(|seq| *get_mut_arcmutex!(seq).id())
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    let first_omitted = first.retain_prompt_prefix(1).unwrap();
    assert_eq!(first_omitted, 1);
    assert_eq!(first.scheduled.len(), 1);
    assert_eq!(first.num_cached_tokens.len(), 1);
    assert_eq!(first.scheduled_prompt_chunks.as_ref().unwrap().len(), 1);

    let kv_before = {
        let kv_manager = get_mut_arcmutex!(scheduler.kv_cache_manager);
        (0..3)
            .map(|id| kv_manager.get_block_ids(id).unwrap().to_vec())
            .collect::<Vec<_>>()
    };
    let states_before = scheduler
        .running
        .iter()
        .map(|seq| get_mut_arcmutex!(seq).getstate())
        .collect::<Vec<_>>();

    Scheduler::defer_prompt_tail(&mut scheduler, first_omitted);

    let kv_after_deferral = {
        let kv_manager = get_mut_arcmutex!(scheduler.kv_cache_manager);
        (0..3)
            .map(|id| kv_manager.get_block_ids(id).unwrap().to_vec())
            .collect::<Vec<_>>()
    };
    assert_eq!(kv_after_deferral, kv_before);
    assert_eq!(
        scheduler
            .running
            .iter()
            .map(|seq| get_mut_arcmutex!(seq).getstate())
            .collect::<Vec<_>>(),
        states_before
    );

    let next = scheduler.schedule(&logger, None);
    assert_eq!(
        next.scheduled
            .iter()
            .map(|seq| *get_mut_arcmutex!(seq).id())
            .collect::<Vec<_>>(),
        vec![1, 2, 0]
    );
    let kv_after_schedule = {
        let kv_manager = get_mut_arcmutex!(scheduler.kv_cache_manager);
        (0..3)
            .map(|id| kv_manager.get_block_ids(id).unwrap().to_vec())
            .collect::<Vec<_>>()
    };
    assert_eq!(kv_after_schedule, kv_before);
    assert_eq!(
        scheduler
            .running
            .iter()
            .map(|seq| get_mut_arcmutex!(seq).getstate())
            .collect::<Vec<_>>(),
        states_before
    );
}

#[test]
fn token_budget_fairly_rotates_completion_batches() {
    let mut scheduler = test_scheduler();
    scheduler.config.max_num_batched_tokens = 2;
    for id in 0..3 {
        let seq = test_sequence(id, 4);
        get_mut_arcmutex!(seq).set_num_computed_tokens(4);
        scheduler.running.push_back(seq);
    }

    let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
    let first = scheduler.schedule(&logger, None);
    let first_ids = first
        .scheduled
        .iter()
        .map(|seq| *get_mut_arcmutex!(seq).id())
        .collect::<Vec<_>>();
    assert_eq!(first_ids, vec![0, 1]);

    let second = scheduler.schedule(&logger, None);
    let second_ids = second
        .scheduled
        .iter()
        .map(|seq| *get_mut_arcmutex!(seq).id())
        .collect::<Vec<_>>();
    assert_eq!(second_ids, vec![2, 0]);
}

#[test]
fn completion_batches_bootstrap_new_rows_without_dropping_staged_rows() {
    let mut scheduler = test_scheduler();
    for id in 0..2 {
        let seq = test_sequence(id, 4);
        let mut seq_guard = get_mut_arcmutex!(seq);
        seq_guard.set_num_computed_tokens(4);
        seq_guard.set_staged_speculative(vec![10, 11], None);
        drop(seq_guard);
        scheduler.running.push_back(seq);
    }
    let newcomer = test_sequence(2, 4);
    get_mut_arcmutex!(newcomer).set_num_computed_tokens(4);
    scheduler.running.push_back(newcomer.clone());

    let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
    let staged = scheduler.schedule(&logger, None);
    let staged_ids = staged
        .scheduled
        .iter()
        .map(|seq| *get_mut_arcmutex!(seq).id())
        .collect::<Vec<_>>();
    assert_eq!(staged_ids, vec![0, 1]);
    assert_eq!(
        scheduler
            .running
            .iter()
            .map(|seq| get_mut_arcmutex!(seq).active_staged_speculative_len())
            .collect::<Vec<_>>(),
        vec![2, 2, 0]
    );

    let bootstrap = scheduler.schedule(&logger, None);
    assert_eq!(bootstrap.scheduled.len(), 1);
    assert_eq!(*get_mut_arcmutex!(bootstrap.scheduled[0]).id(), 2);
    get_mut_arcmutex!(newcomer).set_staged_speculative(vec![12, 13], None);

    let joined = scheduler.schedule(&logger, None);
    let joined_ids = joined
        .scheduled
        .iter()
        .map(|seq| *get_mut_arcmutex!(seq).id())
        .collect::<Vec<_>>();
    assert_eq!(joined_ids, vec![0, 1, 2]);
}

#[test]
fn completion_batches_rotate_to_interior_staged_widths() {
    let widths = [2, 2, 6, 2, 6, 2, 2, 2];
    let rows = widths
        .iter()
        .enumerate()
        .map(|(id, &width)| {
            let seq = test_sequence(id, 4);
            let mut guard = get_mut_arcmutex!(seq);
            guard.set_num_computed_tokens(4);
            guard.set_staged_speculative(vec![10; width], None);
            drop(guard);
            seq
        })
        .collect::<Vec<_>>();

    let (first, cursor) = PagedAttentionScheduler::completion_batch_indices(&rows, 0, 4096);
    assert_eq!(first, vec![0, 1, 3, 5, 6, 7]);
    let (second, cursor) = PagedAttentionScheduler::completion_batch_indices(&rows, cursor, 4096);
    assert_eq!(second, vec![2, 4]);
    let (third, _) = PagedAttentionScheduler::completion_batch_indices(&rows, cursor, 4096);
    assert_eq!(third, vec![3, 5, 6, 7, 0, 1]);
    assert_eq!(
        rows.iter()
            .map(|seq| get_mut_arcmutex!(seq).active_staged_speculative_len())
            .collect::<Vec<_>>(),
        widths
    );
}

#[test]
fn completion_batches_rotate_to_interior_rows_over_budget() {
    let rows = [1, 4, 1]
        .iter()
        .enumerate()
        .map(|(id, &uncomputed)| {
            let seq = test_sequence(id, 4);
            get_mut_arcmutex!(seq).set_num_computed_tokens(4 - uncomputed);
            seq
        })
        .collect::<Vec<_>>();

    let (first, cursor) = PagedAttentionScheduler::completion_batch_indices(&rows, 0, 2);
    assert_eq!(first, vec![0, 2]);
    let (second, cursor) = PagedAttentionScheduler::completion_batch_indices(&rows, cursor, 2);
    assert_eq!(second, vec![1]);
    let (third, _) = PagedAttentionScheduler::completion_batch_indices(&rows, cursor, 2);
    assert_eq!(third, vec![2, 0]);
}

#[test]
fn mixed_depth_batches_merge_after_their_next_proposals() {
    let mut scheduler = test_scheduler();
    for (id, width) in [2, 2, 6, 2, 6, 2].into_iter().enumerate() {
        let seq = test_sequence(id, 4);
        let mut guard = get_mut_arcmutex!(seq);
        guard.set_num_computed_tokens(4);
        guard.set_staged_speculative(vec![10; width], None);
        drop(guard);
        scheduler.running.push_back(seq);
    }
    let mut tuner = crate::speculative::autotuner::SpeculativeAutotuner::default();
    tuner.set_candidates(&[2, 3, 4, 6]);
    let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);

    for expected_ids in [vec![0, 1, 3, 5], vec![2, 4]] {
        let batch = scheduler.schedule(&logger, None);
        let ids = batch
            .scheduled
            .iter()
            .map(|seq| *get_mut_arcmutex!(seq).id())
            .collect::<Vec<_>>();
        assert_eq!(ids, expected_ids);
        let hint = get_mut_arcmutex!(batch.scheduled[0]).speculative_depth_hint();
        assert_eq!(hint, Some(2));
        let depth = tuner.choose_with_hint(&ids, 6, hint);
        assert_eq!(depth, 2);
        for seq in &batch.scheduled {
            get_mut_arcmutex!(seq).set_staged_speculative(vec![11; depth], None);
        }
    }

    let merged = scheduler.schedule(&logger, None);
    assert_eq!(merged.scheduled.len(), 6);
    for seq in &merged.scheduled {
        let seq = get_mut_arcmutex!(seq);
        assert_eq!(seq.active_staged_speculative_len(), 2);
        assert_eq!(seq.speculative_depth_hint(), None);
    }
}

#[test]
fn completion_depth_hint_requires_a_joint_batch_within_budget() {
    let mut scheduler = test_scheduler();
    scheduler.config.max_num_batched_tokens = 8;
    for (id, width) in [2, 6, 2].into_iter().enumerate() {
        let seq = test_sequence(id, 4);
        let mut guard = get_mut_arcmutex!(seq);
        guard.set_num_computed_tokens(4);
        guard.set_staged_speculative(vec![10; width], None);
        guard.set_speculative_depth_hint(Some(6));
        drop(guard);
        scheduler.running.push_back(seq);
    }
    let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);

    let batch = scheduler.schedule(&logger, None);
    assert_eq!(batch.scheduled.len(), 2);
    assert!(scheduler
        .running
        .iter()
        .all(|seq| { get_mut_arcmutex!(seq).speculative_depth_hint().is_none() }));
}

#[test]
fn matching_depths_split_by_token_cost_do_not_get_a_hint() {
    let rows = [0, 9]
        .into_iter()
        .enumerate()
        .map(|(id, computed)| {
            let seq = test_sequence(id, 10);
            let mut guard = get_mut_arcmutex!(seq);
            guard.set_num_computed_tokens(computed);
            guard.set_staged_speculative(vec![10; 2], None);
            drop(guard);
            seq
        })
        .collect::<Vec<_>>();
    let (selected, _) = PagedAttentionScheduler::completion_batch_indices(&rows, 0, 6);
    assert_eq!(selected, vec![0]);
    assert_eq!(
        PagedAttentionScheduler::completion_depth_hint(&rows, &selected, 6),
        None
    );
}

#[test]
fn completion_depth_hint_breaks_ties_toward_shallower_drafts() {
    let rows = [6, 2]
        .into_iter()
        .enumerate()
        .map(|(id, width)| {
            let seq = test_sequence(id, 4);
            get_mut_arcmutex!(seq).set_staged_speculative(vec![10; width], None);
            seq
        })
        .collect::<Vec<_>>();
    assert_eq!(
        PagedAttentionScheduler::completion_depth_hint(&rows, &[0], 32),
        Some(2)
    );
}

#[test]
fn resident_decode_continuation_requires_exact_live_batch() {
    let mut scheduler = test_scheduler();
    scheduler.running.push_back(test_sequence(0, 4));
    scheduler.running.push_back(test_sequence(1, 4));

    let cursor = scheduler.completion_cursor;
    let decode_steps = scheduler.decode_steps_since_prefill;
    assert!(scheduler.can_continue_decode_batch(&[0, 1]));
    assert!(!scheduler.can_continue_decode_batch(&[1, 0]));
    assert!(!scheduler.can_continue_decode_batch(&[0]));
    assert!(!scheduler.can_continue_decode_batch(&[0, 1, 2]));
    assert_eq!(scheduler.completion_cursor, cursor);
    assert_eq!(scheduler.decode_steps_since_prefill, decode_steps);

    get_mut_arcmutex!(scheduler.running[1]).set_state(SequenceState::Done(StopReason::Canceled));
    assert!(!scheduler.can_continue_decode_batch(&[0, 1]));
}

#[test]
fn resident_decode_continuation_respects_prompt_fairness() {
    let mut scheduler = test_scheduler();
    for id in 0..scheduler.config.max_num_seqs {
        scheduler.running.push_back(test_sequence(id, 4));
    }
    let waiting = test_sequence(scheduler.config.max_num_seqs, 4);
    get_mut_arcmutex!(waiting).set_state(SequenceState::Waiting);
    scheduler.waiting.push_back(waiting);
    scheduler.decode_steps_since_prefill = scheduler.config.max_decode_steps_before_prefill - 1;
    let sequence_ids = (0..scheduler.config.max_num_seqs).collect::<Vec<_>>();

    assert!(scheduler.can_continue_decode_batch(&sequence_ids));
    let before = scheduler.decode_steps_since_prefill;
    scheduler.record_decode_continuation();
    assert_eq!(scheduler.decode_steps_since_prefill, before + 1);
    assert!(!scheduler.can_continue_decode_batch(&sequence_ids));
}

#[test]
fn underfilled_resident_decode_yields_after_one_turn() {
    let mut scheduler = test_scheduler();
    scheduler.running.push_back(test_sequence(0, 4));
    let waiting = test_sequence(1, 4);
    get_mut_arcmutex!(waiting).set_state(SequenceState::Waiting);
    scheduler.waiting.push_back(waiting);
    scheduler.decode_steps_since_prefill = 1;

    let cursor = scheduler.completion_cursor;
    let steps = scheduler.decode_steps_since_prefill;
    assert!(!scheduler.can_continue_decode_batch(&[0]));
    assert_eq!(scheduler.completion_cursor, cursor);
    assert_eq!(scheduler.decode_steps_since_prefill, steps);
}

#[test]
fn kv_blocked_prompt_does_not_interrupt_resident_decode() {
    let mut scheduler = test_scheduler();
    scheduler.running.push_back(test_sequence(0, 4));
    let waiting = test_sequence(1, 4);
    get_mut_arcmutex!(waiting).set_state(SequenceState::Waiting);
    scheduler.waiting.push_back(waiting);
    scheduler.decode_steps_since_prefill = 1;
    {
        let mut kv_manager = get_mut_arcmutex!(scheduler.kv_cache_manager);
        let free_tokens = kv_manager.num_free_blocks() * scheduler.block_size;
        assert!(kv_manager.allocate_slots(99, free_tokens, &[]).is_some());
        assert_eq!(kv_manager.num_free_blocks(), 0);
    }

    assert!(scheduler.can_continue_decode_batch(&[0]));
}

#[test]
fn resident_decode_continuation_stops_for_pending_termination() {
    let mut scheduler = test_scheduler();
    scheduler.running.push_back(test_sequence(0, 4));

    assert!(scheduler.can_continue_decode_batch_inner(&[0], false));
    assert!(!scheduler.can_continue_decode_batch_inner(&[0], true));
}

#[test]
fn resident_decode_continuation_preserves_subset_rotation() {
    let mut scheduler = test_scheduler();
    scheduler.config.max_num_batched_tokens = 2;
    for id in 0..3 {
        let seq = test_sequence(id, 4);
        get_mut_arcmutex!(seq).set_num_computed_tokens(4);
        scheduler.running.push_back(seq);
    }

    assert!(scheduler.can_continue_decode_batch(&[0, 1]));
    assert_eq!(scheduler.completion_cursor, 0);
    scheduler.record_decode_continuation();
    assert_eq!(scheduler.completion_cursor, 2);
    assert!(!scheduler.can_continue_decode_batch(&[0, 1]));
    assert!(scheduler.can_continue_decode_batch(&[2, 0]));

    scheduler.record_decode_continuation();
    assert_eq!(scheduler.completion_cursor, 1);
    assert!(scheduler.can_continue_decode_batch(&[1, 2]));
}

#[test]
fn lazy_prompt_admission_materializes_only_the_selected_chunk() {
    let mut scheduler = test_scheduler();
    scheduler.scheduler_visible_prompt_chunks = true;
    scheduler.config.max_num_batched_tokens = 8;
    let seq = test_sequence(0, 32);
    get_mut_arcmutex!(seq).set_state(SequenceState::Waiting);
    scheduler.waiting.push_back(seq);
    let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);

    let output = scheduler.schedule(&logger, None);

    assert_eq!(output.scheduled_prompt_chunks.unwrap()[0].end, 8);
    let kv_mgr = get_mut_arcmutex!(scheduler.kv_cache_manager);
    assert_eq!(kv_mgr.num_blocks_for_request(0), 1);
    assert_eq!(kv_mgr.num_reserved_blocks(), 3);
    assert_eq!(kv_mgr.num_active_blocks() + kv_mgr.num_reserved_blocks(), 4);
}

#[test]
fn lazy_prompt_allocation_grows_with_each_chunk_frontier() {
    let mut scheduler = test_scheduler();
    scheduler.scheduler_visible_prompt_chunks = true;
    scheduler.config.max_num_batched_tokens = 8;
    let seq = test_sequence(0, 24);
    get_mut_arcmutex!(seq).set_state(SequenceState::Waiting);
    scheduler.waiting.push_back(seq.clone());
    let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);

    let first = scheduler.schedule(&logger, None);
    let first_chunk = first.scheduled_prompt_chunks.unwrap()[0];
    let first_blocks = get_mut_arcmutex!(scheduler.kv_cache_manager)
        .get_block_ids(0)
        .unwrap()
        .to_vec();
    assert_eq!((first_chunk.start, first_chunk.end), (0, 8));
    assert_eq!(first_blocks.len(), 1);
    assert_eq!(
        get_mut_arcmutex!(scheduler.kv_cache_manager).num_reserved_blocks(),
        2
    );

    {
        let mut seq = get_mut_arcmutex!(seq);
        seq.set_num_computed_tokens(first_chunk.end);
        assert_eq!(seq.prefix_cache_len(), 0);
    }
    let second = scheduler.schedule(&logger, None);
    let second_chunk = second.scheduled_prompt_chunks.unwrap()[0];
    let kv_mgr = get_mut_arcmutex!(scheduler.kv_cache_manager);
    let second_blocks = kv_mgr.get_block_ids(0).unwrap();
    assert_eq!((second_chunk.start, second_chunk.end), (8, 16));
    assert_eq!(second_blocks.len(), 2);
    assert_eq!(&second_blocks[..first_blocks.len()], first_blocks);
    assert_eq!(kv_mgr.num_reserved_blocks(), 1);
}

#[test]
fn lazy_prompt_reservation_pressure_preserves_decode_and_staged_validation() {
    let mut scheduler = PagedAttentionScheduler::new(
        PagedAttentionSchedulerConfig {
            max_num_seqs: 8,
            max_num_batched_tokens: 8,
            max_prefill_chunk_tokens: 8,
            max_decode_steps_before_prefill: 1,
        },
        CacheConfig {
            block_size: 8,
            num_gpu_blocks: 8,
            cache_type: PagedCacheType::Auto,
            kv_cache_group_ids: vec![0],
        },
    );
    scheduler.scheduler_visible_prompt_chunks = true;
    let completion = test_sequence(0, 8);
    get_mut_arcmutex!(completion).set_num_computed_tokens(8);
    assert!(get_mut_arcmutex!(scheduler.kv_cache_manager)
        .allocate_slots(0, 9, &[])
        .is_some());
    scheduler.running.push_back(completion);

    let prompt = test_sequence(1, 48);
    get_mut_arcmutex!(prompt).set_state(SequenceState::Waiting);
    scheduler.waiting.push_back(prompt);
    scheduler.decode_steps_since_prefill = scheduler.config.max_decode_steps_before_prefill;
    let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
    let mut validator = RecordingPrefixValidator::default();
    let committed_ids = Arc::clone(&validator.committed_ids);

    let SchedulerOutput::PagedAttention {
        output,
        preempted_sequence_ids,
    } = Scheduler::schedule(&mut scheduler, &logger, Some(&mut validator))
    else {
        panic!("paged scheduler returned a default scheduler output");
    };

    assert_eq!(output.scheduled.len(), 1);
    assert_eq!(*get_mut_arcmutex!(output.scheduled[0]).id(), 0);
    assert!(preempted_sequence_ids.is_empty());
    assert_eq!(scheduler.waiting.len(), 1);
    assert_eq!(validator.validated_ids, vec![1]);
    assert!(get_mut_arcmutex!(committed_ids).is_empty());
    let kv_mgr = get_mut_arcmutex!(scheduler.kv_cache_manager);
    assert!(!kv_mgr.has_request(1));
    assert_eq!(kv_mgr.num_reserved_blocks(), 0);
}

#[test]
fn canceling_lazy_prompt_releases_physical_and_reserved_blocks() {
    let mut scheduler = test_scheduler();
    scheduler.scheduler_visible_prompt_chunks = true;
    scheduler.config.max_num_batched_tokens = 8;
    let initial_free_blocks = get_mut_arcmutex!(scheduler.kv_cache_manager).num_free_blocks();
    let (seq, receiver) = test_sequence_with_media_and_receiver(0, 32, None, None, None);
    get_mut_arcmutex!(seq).set_state(SequenceState::Waiting);
    scheduler.waiting.push_back(seq.clone());
    let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);

    let output = scheduler.schedule(&logger, None);
    let chunk_end = output.scheduled_prompt_chunks.unwrap()[0].end;
    {
        let mut seq = get_mut_arcmutex!(seq);
        seq.set_num_computed_tokens(chunk_end);
        assert_eq!(seq.prefix_cache_len(), 0);
    }
    assert_eq!(
        get_mut_arcmutex!(scheduler.kv_cache_manager).num_reserved_blocks(),
        3
    );

    drop(receiver);
    Scheduler::cancel_closed_response_groups(&mut scheduler);
    scheduler.free_finished_sequence_groups();

    let kv_mgr = get_mut_arcmutex!(scheduler.kv_cache_manager);
    assert!(!kv_mgr.has_request(0));
    assert_eq!(kv_mgr.num_active_blocks(), 0);
    assert_eq!(kv_mgr.num_reserved_blocks(), 0);
    assert_eq!(kv_mgr.num_free_blocks(), initial_free_blocks);
}

#[test]
fn lazy_prompt_prefix_hit_pins_cache_and_materializes_only_the_suffix_chunk() {
    let mut scheduler = test_scheduler();
    scheduler.scheduler_visible_prompt_chunks = true;
    scheduler.config.max_num_batched_tokens = 4;
    let tokens = vec![1; 24];
    let hashes = compute_block_hashes(&tokens, scheduler.block_size, &[], &[]);
    {
        let mut kv_mgr = get_mut_arcmutex!(scheduler.kv_cache_manager);
        kv_mgr.allocate_slots(99, 8, &[]).unwrap();
        kv_mgr.cache_blocks(99, &hashes, 8);
        kv_mgr.free(99);
    }

    let seq = test_sequence(0, tokens.len());
    get_mut_arcmutex!(seq).set_state(SequenceState::Waiting);
    scheduler.waiting.push_back(seq.clone());
    let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);
    let mut validator = RecordingPrefixValidator::default();
    let committed_ids = Arc::clone(&validator.committed_ids);

    let output = scheduler.schedule(&logger, Some(&mut validator));

    assert_eq!(output.num_cached_tokens, vec![8]);
    let chunk = output.scheduled_prompt_chunks.unwrap()[0];
    assert_eq!((chunk.start, chunk.end), (8, 12));
    assert_eq!(validator.cached_tokens, vec![8]);
    assert_eq!(&*get_mut_arcmutex!(committed_ids), &[0]);
    let kv_mgr = get_mut_arcmutex!(scheduler.kv_cache_manager);
    assert_eq!(kv_mgr.num_blocks_for_request(0), 2);
    assert_eq!(kv_mgr.num_reserved_blocks(), 1);
    assert_eq!(kv_mgr.num_cached_blocks(0), 1);
    drop(kv_mgr);

    {
        let mut seq = get_mut_arcmutex!(seq);
        seq.set_num_computed_tokens(chunk.end);
        assert_eq!(seq.prefix_cache_len(), 8);
    }
    let output = scheduler.schedule(&logger, Some(&mut validator));
    assert_eq!(output.num_cached_tokens, vec![8]);
    let chunk = output.scheduled_prompt_chunks.unwrap()[0];
    assert_eq!((chunk.start, chunk.end), (12, 16));
    assert_eq!(get_mut_arcmutex!(seq).prefix_cache_len(), 8);
}

#[test]
fn lazy_prompt_still_rejects_a_full_request_over_capacity() {
    let mut scheduler = PagedAttentionScheduler::new(
        PagedAttentionSchedulerConfig {
            max_num_seqs: 8,
            max_num_batched_tokens: 8,
            max_prefill_chunk_tokens: 8,
            max_decode_steps_before_prefill: 8,
        },
        CacheConfig {
            block_size: 8,
            num_gpu_blocks: 4,
            cache_type: PagedCacheType::Auto,
            kv_cache_group_ids: vec![0],
        },
    );
    scheduler.scheduler_visible_prompt_chunks = true;
    let (seq, mut receiver) = test_sequence_with_media_and_receiver(0, 25, None, None, None);
    get_mut_arcmutex!(seq).set_state(SequenceState::Waiting);
    scheduler.waiting.push_back(seq.clone());
    let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);

    let output = scheduler.schedule(&logger, None);

    assert!(output.scheduled.is_empty());
    assert_eq!(
        get_mut_arcmutex!(seq).getstate(),
        SequenceState::FinishedIgnored
    );
    assert!(matches!(
        receiver.try_recv(),
        Ok(Response::ValidationError(_))
    ));
    let kv_mgr = get_mut_arcmutex!(scheduler.kv_cache_manager);
    assert_eq!(kv_mgr.num_active_blocks(), 0);
    assert_eq!(kv_mgr.num_reserved_blocks(), 0);
}

#[test]
fn scheduler_visible_prefill_uses_exact_long_prompt_ranges() {
    let mut scheduler = test_scheduler();
    scheduler.scheduler_visible_prompt_chunks = true;
    scheduler.config.max_num_batched_tokens = 4;
    let seq = test_sequence(0, 10);
    get_mut_arcmutex!(seq).set_state(SequenceState::Waiting);
    scheduler.waiting.push_back(seq.clone());
    let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);

    for expected in [(0, 4), (4, 8), (8, 10)] {
        let output = scheduler.schedule(&logger, None);
        let chunks = output.scheduled_prompt_chunks.unwrap();
        assert_eq!(output.scheduled.len(), 1);
        assert_eq!((chunks[0].start, chunks[0].end), expected);
        assert_eq!(chunks[0].end - chunks[0].start, expected.1 - expected.0);
        let mut seq = get_mut_arcmutex!(seq);
        seq.set_num_computed_tokens(chunks[0].end);
        assert_eq!(seq.prefix_cache_len(), 0);
    }
}

#[test]
fn hybrid_prefill_schedules_the_maximum_reusable_prefix_boundary() {
    let mut scheduler = test_scheduler();
    scheduler.scheduler_visible_prompt_chunks = true;
    scheduler.prompt_chunks_require_block_alignment = true;
    scheduler.config.max_num_batched_tokens = 32;
    let seq = test_sequence(0, 64);
    get_mut_arcmutex!(seq).set_state(SequenceState::Waiting);
    scheduler.waiting.push_back(seq.clone());
    let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);

    for expected in [(0, 32), (32, 56), (56, 64)] {
        let output = scheduler.schedule(&logger, None);
        let chunks = output.scheduled_prompt_chunks.unwrap();
        assert_eq!(output.scheduled.len(), 1);
        assert_eq!((chunks[0].start, chunks[0].end), expected);
        let mut seq = get_mut_arcmutex!(seq);
        seq.set_num_computed_tokens(chunks[0].end);
        assert_eq!(seq.prefix_cache_len(), 0);
    }
}

#[test]
fn hybrid_prefill_schedules_the_suffix_replay_boundary() {
    let mut scheduler = test_scheduler();
    scheduler.scheduler_visible_prompt_chunks = true;
    scheduler.prompt_chunks_require_block_alignment = true;
    scheduler.prefix_policy =
        SpeculativePrefixCheckpointPolicy::new(SpeculativePrefixReplay::Suffix(16), false);
    scheduler.config.max_num_batched_tokens = 32;
    let seq = test_sequence(0, 64);
    get_mut_arcmutex!(seq).set_state(SequenceState::Waiting);
    scheduler.waiting.push_back(seq.clone());
    let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);

    for expected in [(0, 32), (32, 40), (40, 64)] {
        let output = scheduler.schedule(&logger, None);
        let chunks = output.scheduled_prompt_chunks.unwrap();
        assert_eq!(output.scheduled.len(), 1);
        assert_eq!((chunks[0].start, chunks[0].end), expected);
        let mut seq = get_mut_arcmutex!(seq);
        seq.set_num_computed_tokens(chunks[0].end);
        assert_eq!(seq.prefix_cache_len(), 0);
    }
}

#[test]
fn text_auxiliary_policy_keeps_multimodal_suffix_replay() {
    let policy = SpeculativePrefixCheckpointPolicy::new(SpeculativePrefixReplay::Suffix(16), true);
    let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);

    let mut text_scheduler = test_scheduler();
    text_scheduler.scheduler_visible_prompt_chunks = true;
    text_scheduler.prompt_chunks_require_block_alignment = true;
    text_scheduler.prefix_policy = policy;
    text_scheduler.config.max_num_batched_tokens = 32;
    let text = test_sequence(0, 64);
    get_mut_arcmutex!(text).set_state(SequenceState::Waiting);
    text_scheduler.waiting.push_back(text.clone());
    let first = text_scheduler.schedule(&logger, None);
    let first_chunk = first.scheduled_prompt_chunks.unwrap()[0];
    get_mut_arcmutex!(text).set_num_computed_tokens(first_chunk.end);
    let second = text_scheduler.schedule(&logger, None);
    assert_eq!(second.scheduled_prompt_chunks.unwrap()[0].end, 56);

    let mut media_scheduler = test_scheduler();
    media_scheduler.scheduler_visible_prompt_chunks = true;
    media_scheduler.prompt_chunks_require_block_alignment = true;
    media_scheduler.prefix_policy = policy;
    media_scheduler.config.max_num_batched_tokens = 32;
    let media = test_sequence_with_images(1, 64, Some(vec![image::DynamicImage::new_rgb8(1, 1)]));
    assert_eq!(
        policy.replay_for(modality_signature(&*get_mut_arcmutex!(media))),
        SpeculativePrefixReplay::Suffix(16)
    );
    {
        let mut media = get_mut_arcmutex!(media);
        media.set_mm_features(vec![MultiModalFeature {
            kind: MultimodalKind::Image,
            item_range: 0..1,
            hashes: vec![1],
            offset: 0,
            length: 8,
            attention_policy: crate::paged_attention::block_hash::MultimodalAttentionPolicy::Causal,
            splittable: false,
        }]);
        media.set_state(SequenceState::Waiting);
    }
    media_scheduler.waiting.push_back(media.clone());
    let first = media_scheduler.schedule(&logger, None);
    let first_chunk = first.scheduled_prompt_chunks.unwrap()[0];
    get_mut_arcmutex!(media).set_num_computed_tokens(first_chunk.end);
    let second = media_scheduler.schedule(&logger, None);
    assert_eq!(second.scheduled_prompt_chunks.unwrap()[0].end, 40);
}

#[test]
fn partial_prefill_runs_after_the_decode_turn_budget() {
    let mut scheduler = test_scheduler();
    scheduler.scheduler_visible_prompt_chunks = true;
    scheduler.config.max_num_batched_tokens = 4;
    scheduler.config.max_decode_steps_before_prefill = 3;

    let completion = test_sequence(0, 8);
    get_mut_arcmutex!(completion).set_num_computed_tokens(8);
    scheduler.running.push_back(completion);
    let prompt = test_sequence(1, 12);
    {
        let mut prompt = get_mut_arcmutex!(prompt);
        prompt.set_state(SequenceState::RunningPrompt);
        prompt.set_prefix_cache_len(4);
        prompt.set_num_computed_tokens(4);
    }
    scheduler.running.push_back(prompt);
    let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);

    for _ in 0..scheduler.config.max_decode_steps_before_prefill {
        let output = scheduler.schedule(&logger, None);
        assert_eq!(output.scheduled.len(), 1);
        assert!(get_mut_arcmutex!(output.scheduled[0]).is_completion());
        assert!(output.scheduled_prompt_chunks.is_none());
    }

    let output = scheduler.schedule(&logger, None);
    assert_eq!(output.scheduled.len(), 1);
    assert!(get_mut_arcmutex!(output.scheduled[0]).is_prompt());
    let chunks = output.scheduled_prompt_chunks.unwrap();
    assert_eq!((chunks[0].start, chunks[0].end), (4, 8));
}

#[test]
fn partial_prefill_yields_to_decode_between_quanta() {
    let mut scheduler = test_scheduler();
    scheduler.scheduler_visible_prompt_chunks = true;
    scheduler.config.max_num_batched_tokens = 16;
    scheduler.config.max_prefill_chunk_tokens = 4;
    scheduler.config.max_decode_steps_before_prefill = 3;

    let completion = test_sequence(0, 8);
    get_mut_arcmutex!(completion).set_num_computed_tokens(8);
    scheduler.running.push_back(completion);
    let prompt = test_sequence(1, 12);
    {
        let mut prompt = get_mut_arcmutex!(prompt);
        prompt.set_state(SequenceState::RunningPrompt);
        prompt.set_prefix_cache_len(4);
        prompt.set_num_computed_tokens(4);
    }
    scheduler.running.push_back(prompt.clone());
    scheduler.decode_steps_since_prefill = scheduler.config.max_decode_steps_before_prefill;
    let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);

    let prefill = scheduler.schedule(&logger, None);
    let chunks = prefill.scheduled_prompt_chunks.unwrap();
    assert_eq!((chunks[0].start, chunks[0].end), (4, 8));
    get_mut_arcmutex!(prompt).set_num_computed_tokens(8);

    let decode = scheduler.schedule(&logger, None);
    assert_eq!(decode.scheduled.len(), 1);
    assert_eq!(*get_mut_arcmutex!(decode.scheduled[0]).id(), 0);
    assert!(decode.scheduled_prompt_chunks.is_none());
}

#[test]
fn unsupported_prompt_paths_remain_atomic() {
    let logger = IntervalLogger::new(std::time::Duration::from_secs(3600), None);

    let mut raw_scheduler = test_scheduler();
    raw_scheduler.scheduler_visible_prompt_chunks = true;
    raw_scheduler.config.max_num_batched_tokens = 4;
    let raw = test_sequence(0, 10);
    {
        let mut raw = get_mut_arcmutex!(raw);
        raw.return_raw_logits = true;
        raw.set_state(SequenceState::Waiting);
    }
    raw_scheduler.waiting.push_back(raw);
    let raw_output = raw_scheduler.schedule(&logger, None);
    assert!(raw_output.scheduled_prompt_chunks.is_none());
    assert!(raw_output.prompt_chunk_size.is_none());
    assert_eq!(
        get_mut_arcmutex!(raw_scheduler.kv_cache_manager).num_blocks_for_request(0),
        2
    );
    assert_eq!(
        get_mut_arcmutex!(raw_scheduler.kv_cache_manager).num_reserved_blocks(),
        0
    );

    let mut media_scheduler = test_scheduler();
    media_scheduler.scheduler_visible_prompt_chunks = true;
    media_scheduler.config.max_num_batched_tokens = 4;
    let media = test_sequence_with_images(1, 10, Some(vec![image::DynamicImage::new_rgb8(1, 1)]));
    get_mut_arcmutex!(media).set_state(SequenceState::Waiting);
    media_scheduler.waiting.push_back(media);
    let media_output = media_scheduler.schedule(&logger, None);
    assert!(media_output.scheduled_prompt_chunks.is_none());
    assert!(media_output.prompt_chunk_size.is_none());
    assert_eq!(
        get_mut_arcmutex!(media_scheduler.kv_cache_manager).num_blocks_for_request(1),
        2
    );
    assert_eq!(
        get_mut_arcmutex!(media_scheduler.kv_cache_manager).num_reserved_blocks(),
        0
    );
}
