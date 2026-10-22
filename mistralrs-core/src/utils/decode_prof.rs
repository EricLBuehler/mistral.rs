//! Per-phase decode timing.
//!
//! `perf` cannot attribute a CPU decode step on this host: the decode kernels run inside the
//! barrier pool, and their frames inline into `call_trampoline`, so the interesting part of a
//! step collapses into its caller. This splits a step into named phases instead, which is what
//! makes an A/B readable - a change to one phase moves that phase's line rather than being
//! lost in a 15% band around the total.
//!
//! Off, entering a phase is one relaxed atomic load. On, it is that plus two clock reads.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

/// The phase vocabulary is fixed: a decode step is measured in these and nothing else.
#[derive(Clone, Copy)]
#[repr(usize)]
pub enum Phase {
    /// Attention projections, q/k/v and o. Weight-bound.
    AttnProj = 0,
    /// KV cache read and the attention op itself. Cache-bound, unlike every other phase.
    Attention = 1,
    /// Layer norms and the residual accumulation.
    Norms = 2,
    Router = 3,
    /// The routed experts.
    Routed = 4,
    Shared = 5,
    /// Mixing the routed and shared results back together.
    Combine = 6,
    Embed = 7,
    LmHead = 8,
}

const PHASE_COUNT: usize = 9;

const PHASE_NAMES: [&str; PHASE_COUNT] = [
    "attn_proj",
    "attention",
    "norms",
    "router",
    "routed",
    "shared",
    "combine",
    "embed",
    "lm_head",
];

static ENABLED: AtomicBool = AtomicBool::new(false);
static STEPS: AtomicU64 = AtomicU64::new(0);
static NANOS: [AtomicU64; PHASE_COUNT] = [const { AtomicU64::new(0) }; PHASE_COUNT];

pub struct PhaseGuard(Option<(Phase, Instant)>);

impl Drop for PhaseGuard {
    fn drop(&mut self) {
        if let Some((phase, started)) = self.0 {
            // Saturating rather than truncating: a phase cannot outlast the process in any run
            // that reaches the report, but the counter is u64 and `as_nanos` is u128.
            let elapsed = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            NANOS[phase as usize].fetch_add(elapsed, Ordering::Relaxed);
        }
    }
}

#[inline]
pub fn profile(phase: Phase) -> PhaseGuard {
    if ENABLED.load(Ordering::Relaxed) {
        PhaseGuard(Some((phase, Instant::now())))
    } else {
        PhaseGuard(None)
    }
}

pub fn enable() {
    ENABLED.store(true, Ordering::Relaxed);
}

/// Marks one decode step, so a periodic report can divide by it.
#[inline]
pub fn step() {
    if ENABLED.load(Ordering::Relaxed) {
        STEPS.fetch_add(1, Ordering::Relaxed);
    }
}

/// `MISTRALRS_DECODE_PROF=<seconds>` turns the profiler on and prints a phase table that often,
/// as a delta over the interval in ms a decode step. That is what makes a live server
/// profileable without the benchmark harness, which needs a checkpoint its fixture knows.
// Same lossless-in-practice conversion as `report`: counters over a few hundred steps.
#[allow(clippy::cast_precision_loss)]
pub fn enable_from_env() {
    let Some(interval) = std::env::var("MISTRALRS_DECODE_PROF")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|s| *s > 0)
    else {
        return;
    };
    enable();
    std::thread::spawn(move || {
        let mut previous = [0u64; PHASE_COUNT];
        let mut previous_steps = 0u64;
        let mut previous_at = Instant::now();
        loop {
            std::thread::sleep(std::time::Duration::from_secs(interval));
            let steps = STEPS.load(Ordering::Relaxed);
            let done = steps.saturating_sub(previous_steps);
            if done == 0 {
                continue;
            }
            let at = Instant::now();
            let wall = (at - previous_at).as_secs_f64();
            let nanos: Vec<u64> = NANOS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
            let deltas: Vec<u64> = nanos
                .iter()
                .zip(&previous)
                .map(|(now, before)| now.saturating_sub(*before))
                .collect();
            let total: u64 = deltas.iter().sum();
            println!(
                "[decode-prof] {done} steps in {wall:.1} s ({:.2} tok/s), {:.1} ms of work a step",
                done as f64 / wall,
                total as f64 / done as f64 / 1e6
            );
            for (name, ns) in PHASE_NAMES.iter().zip(&deltas) {
                let ms = *ns as f64 / done as f64 / 1e6;
                let share = if total == 0 {
                    0.0
                } else {
                    100.0 * *ns as f64 / total as f64
                };
                println!("[decode-prof]   {name:<10} {ms:7.2} ms  {share:5.1}%");
            }
            previous.copy_from_slice(&nanos);
            previous_steps = steps;
            previous_at = at;
        }
    });
}

pub fn reset() {
    for counter in &NANOS {
        counter.store(0, Ordering::Relaxed);
    }
}

/// Prints the phase table and returns the total attributed time.
///
/// `steps` is how many decode steps were timed, so the numbers come out per step. The gap
/// between `attributed` and the wall clock the caller measured is the instrumentation's own
/// cost plus whatever runs outside an instrumented region, which is worth seeing rather than
/// assuming it away.
// The counters are nanosecond totals over a few hundred steps, so they are far inside the
// 52-bit range f64 represents exactly; the conversion is lossless in practice either way.
#[allow(clippy::cast_precision_loss)]
pub fn report(steps: u64, wall: std::time::Duration) -> f64 {
    let nanos: Vec<u64> = NANOS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let total: u64 = nanos.iter().sum();
    let attributed = total as f64 / steps as f64 / 1e6;
    let wall_ms = wall.as_secs_f64() * 1e3 / steps as f64;

    println!("[bench] phases over {steps} steps (ms/step, share of attributed):");
    for (name, ns) in PHASE_NAMES.iter().zip(&nanos) {
        let ms = *ns as f64 / steps as f64 / 1e6;
        let share = if total == 0 {
            0.0
        } else {
            100.0 * *ns as f64 / total as f64
        };
        println!("[bench]   {name:<10} {ms:7.2} ms  {share:5.1}%");
    }
    println!(
        "[bench]   {:<10} {attributed:7.2} ms  (wall {wall_ms:.2} ms/step, \
         unattributed {:.2} ms)",
        "attributed",
        wall_ms - attributed
    );
    attributed
}
