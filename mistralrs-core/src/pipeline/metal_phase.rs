use std::{
    sync::{Mutex, OnceLock},
    time::Instant,
};

use candle_core::Device;

const METAL_PHASE_TIMINGS_ENV: &str = "MISTRALRS_METAL_PHASE_TIMINGS";

static ENABLED: OnceLock<bool> = OnceLock::new();
static STATE: Mutex<PhaseState> = Mutex::new(PhaseState {
    step: None,
    last_step_end: None,
});

struct PhaseState {
    step: Option<StepMarks>,
    last_step_end: Option<Instant>,
}

struct StepMarks {
    batch: usize,
    is_prompt: bool,
    entered: Instant,
    forward_start: Option<Instant>,
    encoded: Option<Instant>,
    readback: Option<Instant>,
}

fn enabled(device: &Device) -> bool {
    device.is_metal()
        && *ENABLED.get_or_init(|| {
            std::env::var(METAL_PHASE_TIMINGS_ENV)
                .ok()
                .is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        })
}

fn with_step(f: impl FnOnce(&mut StepMarks)) {
    if let Some(step) = STATE.lock().unwrap().step.as_mut() {
        f(step);
    }
}

pub(crate) fn begin(device: &Device, is_prompt: bool, batch: usize) {
    if !enabled(device) {
        return;
    }
    STATE.lock().unwrap().step = Some(StepMarks {
        batch,
        is_prompt,
        entered: Instant::now(),
        forward_start: None,
        encoded: None,
        readback: None,
    });
}

pub(crate) fn forward_start() {
    with_step(|step| step.forward_start = Some(Instant::now()));
}

/// `forward_inputs` returned: the CPU finished encoding, the GPU may still be running.
pub(crate) fn encoded() {
    with_step(|step| step.encoded = Some(Instant::now()));
}

/// Logits reached the host, so the GPU work for this step is done.
pub(crate) fn readback() {
    with_step(|step| step.readback = Some(Instant::now()));
}

pub(crate) fn finish() {
    let mut state = STATE.lock().unwrap();
    let Some(step) = state.step.take() else {
        return;
    };
    let now = Instant::now();
    let ms = |from: Option<Instant>, to: Option<Instant>| match (from, to) {
        (Some(from), Some(to)) => to.duration_since(from).as_secs_f64() * 1e3,
        _ => f64::NAN,
    };
    let gap_ms = ms(state.last_step_end, Some(step.entered));
    let pre_ms = ms(Some(step.entered), step.forward_start);
    let encode_ms = ms(step.forward_start, step.encoded);
    let readback_ms = ms(step.encoded, step.readback);
    let post_ms = ms(step.readback, Some(now));
    let total_ms = ms(Some(step.entered), Some(now));
    state.last_step_end = Some(now);
    tracing::info!(
        batch = step.batch,
        is_prompt = step.is_prompt,
        gap_ms,
        pre_ms,
        encode_ms,
        readback_ms,
        post_ms,
        total_ms,
        "Metal phase timing"
    );
}
