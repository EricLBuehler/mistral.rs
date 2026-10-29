use super::*;
use candle_core::cuda_backend::cudarc::driver::{result, sys, CudaGraph};
use candle_core::Var;
use std::time::Instant;

const HEAD_DIM: usize = 128;
const NORM_EPS: f64 = 1.0e-6;
const OUTPUT_TOLERANCE: f32 = 2.0e-3;
const REPLAY_STEPS: usize = 37;
const CACHE_MULTIPLIER: usize = 2;
const TIMING_CALLS: usize = 256;
const TIMING_ROUNDS: usize = 3;

#[derive(Clone, Copy)]
struct Geometry {
    batch: usize,
    capacity: usize,
    key_heads: usize,
    value_heads: usize,
    tiled: bool,
}

struct Inputs {
    geometry: Geometry,
    mixed: Var,
    a: Var,
    b: Var,
    gate: Var,
    a_log: Tensor,
    dt_bias: Tensor,
    weight: Tensor,
    slots: Var,
    activation: GdnGateActivation,
}

struct State {
    values: Var,
    keys: Tensor,
    deltas: Tensor,
    decays: Tensor,
    cursor: Var,
}

fn pattern(len: usize, salt: usize, scale: f32, offset: f32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let value = ((index.wrapping_mul(37) + salt.wrapping_mul(17)) % 257) as f32;
            (value / 128.0 - 1.0) * scale + offset
        })
        .collect()
}

fn flat(tensor: &Tensor) -> Result<Vec<f32>> {
    tensor
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()
}

fn assert_close(label: &str, actual: &Tensor, expected: &Tensor, tolerance: f32) -> Result<()> {
    let actual = flat(actual)?;
    let expected = flat(expected)?;
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            actual.is_finite() && expected.is_finite() && (actual - expected).abs() <= tolerance,
            "{label}: element {index}, actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }
    Ok(())
}

impl Inputs {
    fn new(geometry: Geometry, activation: GdnGateActivation, device: &Device) -> Result<Self> {
        let Geometry {
            batch,
            key_heads,
            value_heads,
            ..
        } = geometry;
        let conv = (2 * key_heads + value_heads) * HEAD_DIM;
        let inputs = Self {
            geometry,
            mixed: Var::zeros((batch, 1, conv), DType::BF16, device)?,
            a: Var::zeros((batch, 1, value_heads), DType::BF16, device)?,
            b: Var::zeros((batch, 1, value_heads), DType::BF16, device)?,
            gate: Var::zeros((batch, 1, value_heads, HEAD_DIM), DType::BF16, device)?,
            a_log: Tensor::from_vec(pattern(value_heads, 31, 0.05, -0.2), value_heads, device)?,
            dt_bias: Tensor::from_vec(pattern(value_heads, 32, 0.1, 0.3), value_heads, device)?,
            weight: Tensor::from_vec(pattern(HEAD_DIM, 33, 0.08, 1.0), HEAD_DIM, device)?
                .to_dtype(DType::BF16)?,
            slots: Var::from_vec((0..batch as u32).collect::<Vec<_>>(), batch, device)?,
            activation,
        };
        inputs.update(0)?;
        Ok(inputs)
    }

    fn update(&self, step: usize) -> Result<()> {
        for (variable, salt, scale, offset) in [
            (&self.mixed, 41, 0.08, 0.01),
            (&self.a, 42, 0.18, -0.04),
            (&self.b, 43, 0.2, 0.1),
            (&self.gate, 44, 0.3, 0.02),
        ] {
            let value = Tensor::from_vec(
                pattern(variable.elem_count(), salt + step, scale, offset),
                variable.shape(),
                variable.device(),
            )?
            .to_dtype(DType::BF16)?;
            variable.set(&value)?;
        }
        Ok(())
    }

    fn eager(&self, state: &State) -> Result<Tensor> {
        let Geometry {
            batch,
            key_heads,
            value_heads,
            tiled,
            ..
        } = self.geometry;
        let mut values = state.values.as_tensor().clone();
        let raw = fused_decode_recurrence_cuda(FusedDecodeRecurrence {
            mixed_qkv: &self.mixed,
            b: &self.b,
            a: &self.a,
            a_log: &self.a_log,
            dt_bias: &self.dt_bias,
            state: &mut values,
            batch_size: batch,
            num_k_heads: key_heads,
            num_v_heads: value_heads,
            head_k_dim: HEAD_DIM,
            head_v_dim: HEAD_DIM,
            tiled_v_heads: tiled,
            state_layout: RecurrentStateLayout::GdnValueMajor,
            slots: GdnStateSlots::Pooled(&self.slots),
        })?
        .reshape((batch, 1, value_heads, HEAD_DIM))?;
        rmsnorm_gated_cuda(&raw, &self.gate, &self.weight, NORM_EPS, self.activation)
    }

    fn deferred(&self, state: &State) -> Result<Tensor> {
        let Geometry {
            key_heads,
            value_heads,
            tiled,
            ..
        } = self.geometry;
        deferred_recurrence_rmsnorm_gate_cuda(GdnDeferredRecurrence {
            mixed_qkv: &self.mixed,
            b: &self.b,
            a: &self.a,
            a_log: &self.a_log,
            dt_bias: &self.dt_bias,
            state_pool: &state.values,
            active_slots: &self.slots,
            deferred_key: &state.keys,
            deferred_delta: &state.deltas,
            deferred_decay: &state.decays,
            deferred_cursor: &state.cursor,
            gate: &self.gate,
            norm_weight: &self.weight,
            norm_eps: NORM_EPS,
            gate_activation: self.activation,
            num_k_heads: key_heads,
            num_v_heads: value_heads,
            head_k_dim: HEAD_DIM,
            head_v_dim: HEAD_DIM,
            tiled_v_heads: tiled,
            state_layout: RecurrentStateLayout::GdnValueMajor,
            quantization: None,
        })?
        .into_tensor()
    }

    fn flush(&self, state: &State, slots: &Tensor) -> Result<()> {
        let Geometry {
            key_heads,
            value_heads,
            tiled,
            ..
        } = self.geometry;
        flush_deferred_state_cuda(GdnDeferredStateFlush {
            state_pool: &state.values,
            active_slots: slots,
            deferred_key: &state.keys,
            deferred_delta: &state.deltas,
            deferred_decay: &state.decays,
            deferred_cursor: &state.cursor,
            num_k_heads: key_heads,
            num_v_heads: value_heads,
            head_k_dim: HEAD_DIM,
            head_v_dim: HEAD_DIM,
            tiled_v_heads: tiled,
            state_layout: RecurrentStateLayout::GdnValueMajor,
        })
    }
}

impl State {
    fn new(geometry: Geometry, device: &Device) -> Result<Self> {
        let Geometry {
            capacity,
            key_heads,
            value_heads,
            ..
        } = geometry;
        Ok(Self {
            values: Var::from_vec(
                pattern(capacity * value_heads * HEAD_DIM * HEAD_DIM, 51, 0.01, 0.0),
                (capacity, value_heads, HEAD_DIM, HEAD_DIM),
                device,
            )?,
            keys: Tensor::zeros(
                (capacity, GDN_DEFERRED_STATE_DEPTH, key_heads, HEAD_DIM),
                DType::F32,
                device,
            )?,
            deltas: Tensor::zeros(
                (capacity, GDN_DEFERRED_STATE_DEPTH, value_heads, HEAD_DIM),
                DType::F32,
                device,
            )?,
            decays: Tensor::zeros(
                (capacity, GDN_DEFERRED_STATE_DEPTH, value_heads),
                DType::F32,
                device,
            )?,
            cursor: Var::zeros(capacity, DType::U32, device)?,
        })
    }

    fn reset_slot(&self, slot: usize) -> Result<()> {
        let mut values = flat(&self.values)?;
        let row = self.values.elem_count() / self.values.dim(0)?;
        values[slot * row..(slot + 1) * row].fill(0.0);
        self.values.set(&Tensor::from_vec(
            values,
            self.values.shape(),
            self.values.device(),
        )?)?;
        let mut cursor = self.cursor.to_device(&Device::Cpu)?.to_vec1::<u32>()?;
        cursor[slot] = 0;
        self.cursor.set(&Tensor::from_vec(
            cursor,
            self.cursor.shape(),
            self.cursor.device(),
        )?)
    }
}

fn capture<T>(device: &Device, launch: impl FnOnce() -> Result<T>) -> Result<(T, CudaGraph)> {
    let stream = device.as_cuda_device()?.cuda_stream();
    device.synchronize()?;
    let tracking = stream.context().is_event_tracking();
    if tracking {
        unsafe { stream.context().disable_event_tracking() };
    }
    if let Err(error) =
        stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
    {
        if tracking {
            unsafe { stream.context().enable_event_tracking() };
        }
        return Err(candle_core::Error::wrap(error));
    }
    let output = launch();
    let graph = stream.end_capture(
        sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
    );
    if tracking {
        unsafe { stream.context().enable_event_tracking() };
    }
    Ok((output?, graph.map_err(candle_core::Error::wrap)?.unwrap()))
}

#[test]
#[ignore = "requires a CUDA device"]
fn deferred_gate_replay_preserves_slots_and_snapshots_cuda() -> Result<()> {
    const BATCH: usize = 8;
    const CAPACITY: usize = 11;
    let device = Device::new_cuda_with_stream(0)?;
    let _htod_cache = device.as_cuda_device()?.enable_cuda_graph_htod_cache();
    for activation in [GdnGateActivation::Silu, GdnGateActivation::Sigmoid] {
        for tiled in [false, true] {
            let geometry = Geometry {
                batch: BATCH,
                capacity: CAPACITY,
                key_heads: 2,
                value_heads: 6,
                tiled,
            };
            let inputs = Inputs::new(geometry, activation, &device)?;
            let eager = State::new(geometry, &device)?;
            let deferred = State::new(geometry, &device)?;
            let initial = eager.values.copy()?;
            let all_slots =
                Tensor::from_vec((0..CAPACITY as u32).collect::<Vec<_>>(), CAPACITY, &device)?;
            let _ = inputs.eager(&eager)?;
            let _ = inputs.deferred(&deferred)?;
            eager.values.set(&initial)?;
            deferred.values.set(&initial)?;
            deferred
                .cursor
                .set(&Tensor::zeros(CAPACITY, DType::U32, &device)?)?;
            let (eager_output, eager_graph) = capture(&device, || inputs.eager(&eager))?;
            let (deferred_output, deferred_graph) =
                capture(&device, || inputs.deferred(&deferred))?;
            for step in 0..REPLAY_STEPS {
                inputs.update(step + 1)?;
                let active = [BATCH, 4, 1, BATCH, 3][step % 5];
                let slots = (0..BATCH)
                    .map(|row| {
                        if row < active {
                            ((row + step) % (CAPACITY - 1)) as u32
                        } else {
                            GDN_PAD_SLOT
                        }
                    })
                    .collect::<Vec<_>>();
                inputs
                    .slots
                    .set(&Tensor::from_vec(slots.clone(), BATCH, &device)?)?;
                eager_graph.launch().map_err(candle_core::Error::wrap)?;
                deferred_graph.launch().map_err(candle_core::Error::wrap)?;
                assert_close(
                    "changed-input graph output",
                    &deferred_output,
                    &eager_output,
                    OUTPUT_TOLERANCE,
                )?;
                for (row, slot) in slots.iter().enumerate() {
                    if *slot == GDN_PAD_SLOT {
                        assert!(flat(&deferred_output.narrow(0, row, 1)?)?
                            .iter()
                            .all(|value| *value == 0.0));
                    }
                }
                if step % 7 == 2 {
                    let slot = slots[0] as usize;
                    let selected = Tensor::from_vec(vec![slot as u32], 1, &device)?;
                    inputs.flush(&deferred, &selected)?;
                    let snapshot = deferred.values.narrow(0, slot, 1)?.copy()?;
                    assert_close(
                        "selected snapshot",
                        &snapshot,
                        &eager.values.narrow(0, slot, 1)?,
                        0.0,
                    )?;
                }
                if step % 9 == 5 {
                    let slot = slots[0] as usize;
                    eager.reset_slot(slot)?;
                    deferred.reset_slot(slot)?;
                }
                if step % 11 == 10 || step + 1 == REPLAY_STEPS {
                    inputs.flush(&deferred, &all_slots)?;
                    assert_close(
                        "materialized FP32 state",
                        &deferred.values,
                        &eager.values,
                        0.0,
                    )?;
                    assert_eq!(
                        deferred.cursor.to_device(&Device::Cpu)?.to_vec1::<u32>()?,
                        vec![0; CAPACITY]
                    );
                }
                assert_close(
                    "inactive slot",
                    &deferred.values.narrow(0, CAPACITY - 1, 1)?,
                    &initial.narrow(0, CAPACITY - 1, 1)?,
                    0.0,
                )?;
            }
        }
    }
    Ok(())
}

fn timed_graph(device: &Device, graph: &CudaGraph, calls: usize) -> Result<f64> {
    let repeats = TIMING_CALLS.div_ceil(calls).max(2);
    graph.launch().map_err(candle_core::Error::wrap)?;
    device.synchronize()?;
    let start = Instant::now();
    for _ in 0..repeats {
        graph.launch().map_err(candle_core::Error::wrap)?;
    }
    device.synchronize()?;
    Ok(start.elapsed().as_secs_f64() * 1.0e6 / (repeats * calls) as f64)
}

#[test]
#[ignore = "CUDA benchmark; run with --nocapture"]
fn benchmark_deferred_decode_crossover() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    let cuda = device.as_cuda_device()?;
    let _htod_cache = cuda.enable_cuda_graph_htod_cache();
    let l2_bytes = unsafe {
        result::device::get_attribute(
            cuda.cuda_stream().context().cu_device(),
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
        )
    }
    .map_err(candle_core::Error::wrap)? as usize;
    assert!(l2_bytes > 0);
    println!("deferred GDN probe: l2_bytes={l2_bytes}, state=FP32, input=BF16, gate=sigmoid");
    for (key_heads, value_heads) in [(8, 32), (16, 48), (16, 64)] {
        for batch in [1, 2, 4, 8, 16] {
            let geometry = Geometry {
                batch,
                capacity: batch,
                key_heads,
                value_heads,
                tiled: false,
            };
            let inputs = Inputs::new(geometry, GdnGateActivation::Sigmoid, &device)?;
            let state_bytes =
                batch * value_heads * HEAD_DIM * HEAD_DIM * std::mem::size_of::<f32>();
            for rotating in [false, true] {
                let sets = if rotating {
                    (l2_bytes * CACHE_MULTIPLIER) / state_bytes + 1
                } else {
                    1
                };
                let eager = (0..sets)
                    .map(|_| State::new(geometry, &device))
                    .collect::<Result<Vec<_>>>()?;
                let deferred = (0..sets)
                    .map(|_| State::new(geometry, &device))
                    .collect::<Result<Vec<_>>>()?;
                for state in &eager {
                    let _ = inputs.eager(state)?;
                }
                for state in &deferred {
                    let _ = inputs.deferred(state)?;
                    inputs.flush(state, &inputs.slots)?;
                }
                let short_flush = key_heads == 16 && value_heads == 48 && matches!(batch, 4 | 8);
                for period in if short_flush {
                    vec![GDN_DEFERRED_STATE_DEPTH, 2, 1]
                } else {
                    vec![GDN_DEFERRED_STATE_DEPTH]
                } {
                    let (eager_outputs, eager_graph) = capture(&device, || {
                        let mut outputs = Vec::with_capacity(period * sets);
                        for _ in 0..period {
                            for state in &eager {
                                outputs.push(inputs.eager(state)?);
                            }
                        }
                        Ok(outputs)
                    })?;
                    let (deferred_outputs, deferred_graph) = capture(&device, || {
                        let mut outputs = Vec::with_capacity(period * sets);
                        for _ in 0..period {
                            for state in &deferred {
                                outputs.push(inputs.deferred(state)?);
                            }
                        }
                        if period != GDN_DEFERRED_STATE_DEPTH {
                            for state in &deferred {
                                inputs.flush(state, &inputs.slots)?;
                            }
                        }
                        Ok(outputs)
                    })?;
                    let mut eager_us = Vec::with_capacity(TIMING_ROUNDS);
                    let mut deferred_us = Vec::with_capacity(TIMING_ROUNDS);
                    for round in 0..TIMING_ROUNDS {
                        if round % 2 == 0 {
                            eager_us.push(timed_graph(&device, &eager_graph, period * sets)?);
                            deferred_us.push(timed_graph(&device, &deferred_graph, period * sets)?);
                        } else {
                            deferred_us.push(timed_graph(&device, &deferred_graph, period * sets)?);
                            eager_us.push(timed_graph(&device, &eager_graph, period * sets)?);
                        }
                    }
                    eager_us.sort_by(f64::total_cmp);
                    deferred_us.sort_by(f64::total_cmp);
                    let eager_us = eager_us[TIMING_ROUNDS / 2];
                    let deferred_us = deferred_us[TIMING_ROUNDS / 2];
                    println!("cache={} batch={batch} hk={key_heads} hv={value_heads} sets={sets} working_bytes={} period={period} eager_us={eager_us:.3} deferred_us={deferred_us:.3} speedup={:.4}", if rotating { "rotating" } else { "warm" }, state_bytes * sets, eager_us / deferred_us);
                    drop((eager_outputs, deferred_outputs));
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "CUDA benchmark; run with --nocapture"]
fn benchmark_empty_deferred_flush_overhead() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    let cuda = device.as_cuda_device()?;
    let _htod_cache = cuda.enable_cuda_graph_htod_cache();
    let l2_bytes = unsafe {
        result::device::get_attribute(
            cuda.cuda_stream().context().cu_device(),
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
        )
    }
    .map_err(candle_core::Error::wrap)? as usize;
    for batch in [1, 2, 4] {
        let geometry = Geometry {
            batch,
            capacity: batch,
            key_heads: 16,
            value_heads: 48,
            tiled: false,
        };
        let inputs = Inputs::new(geometry, GdnGateActivation::Sigmoid, &device)?;
        let state_bytes =
            batch * geometry.value_heads * HEAD_DIM * HEAD_DIM * std::mem::size_of::<f32>();
        for rotating in [false, true] {
            let sets = if rotating {
                l2_bytes * CACHE_MULTIPLIER / state_bytes + 1
            } else {
                1
            };
            let states = (0..sets)
                .map(|_| State::new(geometry, &device))
                .collect::<Result<Vec<_>>>()?;
            for state in &states {
                let _ = inputs.eager(state)?;
                inputs.flush(state, &inputs.slots)?;
            }
            let (eager_outputs, eager_graph) = capture(&device, || {
                let mut outputs = Vec::with_capacity(GDN_DEFERRED_STATE_DEPTH * sets);
                for _ in 0..GDN_DEFERRED_STATE_DEPTH {
                    for state in &states {
                        outputs.push(inputs.eager(state)?);
                    }
                }
                Ok(outputs)
            })?;
            let (flush_outputs, flush_graph) = capture(&device, || {
                let mut outputs = Vec::with_capacity(GDN_DEFERRED_STATE_DEPTH * sets);
                for _ in 0..GDN_DEFERRED_STATE_DEPTH {
                    for state in &states {
                        inputs.flush(state, &inputs.slots)?;
                        outputs.push(inputs.eager(state)?);
                    }
                }
                Ok(outputs)
            })?;
            let mut eager_us = Vec::with_capacity(TIMING_ROUNDS);
            let mut flush_us = Vec::with_capacity(TIMING_ROUNDS);
            for round in 0..TIMING_ROUNDS {
                if round % 2 == 0 {
                    eager_us.push(timed_graph(
                        &device,
                        &eager_graph,
                        GDN_DEFERRED_STATE_DEPTH * sets,
                    )?);
                    flush_us.push(timed_graph(
                        &device,
                        &flush_graph,
                        GDN_DEFERRED_STATE_DEPTH * sets,
                    )?);
                } else {
                    flush_us.push(timed_graph(
                        &device,
                        &flush_graph,
                        GDN_DEFERRED_STATE_DEPTH * sets,
                    )?);
                    eager_us.push(timed_graph(
                        &device,
                        &eager_graph,
                        GDN_DEFERRED_STATE_DEPTH * sets,
                    )?);
                }
            }
            eager_us.sort_by(f64::total_cmp);
            flush_us.sort_by(f64::total_cmp);
            let eager_us = eager_us[TIMING_ROUNDS / 2];
            let flush_us = flush_us[TIMING_ROUNDS / 2];
            println!("empty-flush cache={} batch={batch} sets={sets} eager_us={eager_us:.3} flush_then_eager_us={flush_us:.3} extra_us={:.3}", if rotating { "rotating" } else { "warm" }, flush_us - eager_us);
            drop((eager_outputs, flush_outputs));
        }
    }
    Ok(())
}
