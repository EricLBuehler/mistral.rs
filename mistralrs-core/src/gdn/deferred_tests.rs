use super::*;
use candle_core::cuda_backend::cudarc::driver::{sys, CudaGraph};
use candle_core::{Device, Var};
use candle_nn::Linear;
use mistralrs_quant::{QuantMethodConfig, UnquantLinear};

use crate::kv_cache::{
    HybridCache, HybridCacheConfig, HybridLayerType, RecurrentLayerConfig, RecurrentStateSpec,
};

const HEAD_DIM: usize = 128;
const HIDDEN: usize = 8;
const KEY_HEADS: usize = 2;
const VALUE_HEADS: usize = 6;
const CONV_WIDTH: usize = 4;
const MAX_BATCH: usize = 8;
const MAX_SEQUENCE: usize = 32;
const NORM_EPS: f64 = 1.0e-6;
const OUTPUT_TOLERANCE: f32 = 2.0e-3;
const STATE_TOLERANCE: f32 = 1.0e-6;

fn tensor(shape: impl Into<candle_core::Shape>, phase: usize, device: &Device) -> Result<Tensor> {
    let shape = shape.into();
    let values = (0..shape.elem_count())
        .map(|idx| {
            (((idx.wrapping_mul(37) + phase.wrapping_mul(17)) % 257) as f32 / 128.0 - 1.0) * 0.08
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(values, shape, device)
}

fn linear(rows: usize, cols: usize, device: &Device) -> Result<Arc<dyn QuantMethod>> {
    Ok(Arc::new(UnquantLinear::new(
        QuantMethodConfig::Unquantized(Linear::new(
            tensor((rows, cols), 0, device)?.to_dtype(DType::BF16)?,
            None,
        )),
    )?))
}

fn layer(device: &Device, tiled: bool, activation: GdnGateActivation) -> Result<GatedDeltaNet> {
    let dims = GdnDims {
        hidden_size: HIDDEN,
        num_k_heads: KEY_HEADS,
        num_v_heads: VALUE_HEADS,
        head_k_dim: HEAD_DIM,
        head_v_dim: HEAD_DIM,
        conv_kernel_size: CONV_WIDTH,
        key_dim: KEY_HEADS * HEAD_DIM,
        value_dim: VALUE_HEADS * HEAD_DIM,
        conv_dim: (2 * KEY_HEADS + VALUE_HEADS) * HEAD_DIM,
        v_per_group: VALUE_HEADS / KEY_HEADS,
        v_head_layout: if tiled {
            GdnVHeadLayout::Tiled
        } else {
            GdnVHeadLayout::Grouped
        },
    };
    Ok(GatedDeltaNet {
        input_proj: GdnInputProjection::Split {
            in_proj_qkv: linear(dims.conv_dim, HIDDEN, device)?,
            in_proj_z: linear(dims.value_dim, HIDDEN, device)?,
            in_proj_b: linear(VALUE_HEADS, HIDDEN, device)?,
            in_proj_a: linear(VALUE_HEADS, HIDDEN, device)?,
            merged_qkv_z: None,
            merged_b_a: None,
        },
        conv1d_weight: tensor((dims.conv_dim, 1, CONV_WIDTH), 1, device)?.to_dtype(DType::BF16)?,
        dt_bias: Tensor::full(0.1f32, VALUE_HEADS, device)?,
        a_log: Tensor::full(-0.2f32, VALUE_HEADS, device)?,
        norm: RmsNormGated::from_parts(
            Tensor::ones(HEAD_DIM, DType::BF16, device)?,
            NORM_EPS,
            activation,
        ),
        out_proj: linear(HIDDEN, dims.value_dim, device)?,
        out_proj_input_shard: None,
        dims,
    })
}

fn cache(gdn: &GatedDeltaNet, device: &Device, deferred: bool) -> Result<HybridCache> {
    let mut cache = HybridCache::new(
        HybridCacheConfig {
            layer_types: vec![HybridLayerType::Recurrent],
            max_seq_len: MAX_SEQUENCE,
            recurrent: RecurrentLayerConfig {
                conv_dim: gdn.dims.conv_dim,
                conv_width: CONV_WIDTH,
                state: RecurrentStateSpec::Gdn {
                    heads: VALUE_HEADS,
                    key_dim: HEAD_DIM,
                    value_dim: HEAD_DIM,
                },
                recurrent_dtype: Some(DType::F32),
            },
        },
        DType::BF16,
        std::slice::from_ref(device),
    )?;
    if deferred {
        assert!(crate::gdn::reserve_deferred_decode_state(
            &mut cache,
            [(0, gdn)],
            DType::BF16
        )?);
    }
    for sequence in 0..MAX_BATCH {
        assert_eq!(cache.allocate_seq(sequence)?, sequence);
    }
    Ok(cache)
}

fn pool(cache: &HybridCache) -> &RecurrentStatePool {
    cache.get(0).unwrap().as_recurrent_pool().unwrap()
}

struct Inputs {
    x: Tensor,
    qkv: Var,
    z: Var,
    b: Var,
    a: Var,
    slots: Tensor,
    batch: usize,
    tokens: usize,
}

impl Inputs {
    fn new(gdn: &GatedDeltaNet, batch: usize, tokens: usize, device: &Device) -> Result<Self> {
        let inputs = Self {
            x: Tensor::zeros((batch, tokens, HIDDEN), DType::BF16, device)?,
            qkv: Var::zeros((batch, tokens, gdn.dims.conv_dim), DType::BF16, device)?,
            z: Var::zeros((batch, tokens, VALUE_HEADS, HEAD_DIM), DType::BF16, device)?,
            b: Var::zeros((batch, tokens, VALUE_HEADS), DType::BF16, device)?,
            a: Var::zeros((batch, tokens, VALUE_HEADS), DType::BF16, device)?,
            slots: Tensor::from_vec((0..batch as u32).collect::<Vec<_>>(), batch, device)?,
            batch,
            tokens,
        };
        inputs.update(0)?;
        Ok(inputs)
    }

    fn update(&self, step: usize) -> Result<()> {
        for (idx, value) in [&self.qkv, &self.z, &self.b, &self.a]
            .into_iter()
            .enumerate()
        {
            value
                .set(&tensor(value.shape(), step + idx, value.device())?.to_dtype(DType::BF16)?)?;
        }
        Ok(())
    }

    fn projection(&self, gdn: &GatedDeltaNet) -> Result<GdnProjection> {
        GdnProjection::from_split(
            self.qkv.as_tensor().clone(),
            self.z.as_tensor().clone(),
            self.b.as_tensor().clone(),
            self.a.as_tensor().clone(),
            &gdn.dims,
            self.batch,
            self.tokens,
        )
    }

    fn forward(
        &self,
        gdn: &GatedDeltaNet,
        state: &mut GdnLayerCache,
        kind: RecurrentBatchKind,
    ) -> Result<Tensor> {
        gdn.forward_projected_with_context(
            &self.x,
            self.projection(gdn)?,
            state,
            GdnForwardContext {
                batch_kind: kind,
                checkpoint_lanes: 1,
                transition_checkpoints: false,
                stash_out: None,
            },
        )
    }

    fn pooled(
        &self,
        gdn: &GatedDeltaNet,
        cache: &HybridCache,
        kind: RecurrentBatchKind,
    ) -> Result<Tensor> {
        let mut state = GdnLayerCache::checkout(pool(cache), &self.slots, kind)?;
        self.forward(gdn, &mut state, kind)
    }

    fn gathered_prefill(&self, gdn: &GatedDeltaNet, cache: &mut HybridCache) -> Result<Tensor> {
        let p = pool(cache);
        let mut state = GdnLayerCache::gathered(
            p.gather_conv_state(&self.slots)?,
            p.gather_recurrent_state(&self.slots)?,
            p.state_layout(),
        );
        let output = self.forward(gdn, &mut state, RecurrentBatchKind::Prefill)?;
        let p = cache.get_mut(0).unwrap().as_recurrent_pool_mut().unwrap();
        state.commit(p, &self.slots, None)?;
        Ok(output)
    }
}

fn assert_close(actual: &Tensor, expected: &Tensor, tolerance: f32) -> Result<()> {
    let flat = |value: &Tensor| {
        value
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()
    };
    let actual = flat(actual)?;
    let expected = flat(expected)?;
    assert_eq!(actual.len(), expected.len());
    for (idx, (a, b)) in actual.into_iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= tolerance,
            "index {idx}: {a} vs {b}"
        );
    }
    Ok(())
}

fn capture(
    device: &Device,
    launch: impl FnOnce() -> Result<Tensor>,
) -> Result<(Tensor, CudaGraph)> {
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
#[ignore = "requires CUDA"]
fn deferred_layer_cached_fallback_prefill_and_rollback_match_eager() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let _htod_cache = device.as_cuda_device()?.enable_cuda_graph_htod_cache();
    for (tiled, activation) in [false, true].into_iter().flat_map(|tiled| {
        [GdnGateActivation::Silu, GdnGateActivation::Sigmoid]
            .into_iter()
            .map(move |activation| (tiled, activation))
    }) {
        let gdn = layer(&device, tiled, activation)?;
        let mut deferred = cache(&gdn, &device, true)?;
        let mut eager = cache(&gdn, &device, false)?;
        assert!(!gdn.deferred_decode_supported(pool(&deferred), DType::F16));
        let mut graphs = Vec::new();
        for batch in [1, 2, 4, MAX_BATCH] {
            let inputs = Inputs::new(&gdn, batch, 1, &device)?;
            let _ = inputs.pooled(&gdn, &deferred, RecurrentBatchKind::Decode)?;
            for slot in 0..MAX_BATCH {
                deferred.reset_seq(slot, slot)?;
            }
            let (output, graph) = capture(&device, || {
                inputs.pooled(&gdn, &deferred, RecurrentBatchKind::Decode)
            })?;
            graphs.push((inputs, output, graph));
        }
        for slot in 0..MAX_BATCH {
            deferred.reset_seq(slot, slot)?;
        }
        for (step, graph_idx) in [3, 3, 0, 3, 1, 2, 3, 3, 0, 3, 1, 3].into_iter().enumerate() {
            let (inputs, output, graph) = &graphs[graph_idx];
            inputs.update(step + 1)?;
            graph.launch().map_err(candle_core::Error::wrap)?;
            let expected = inputs.pooled(&gdn, &eager, RecurrentBatchKind::Decode)?;
            assert_close(output, &expected, OUTPUT_TOLERANCE)?;
            if !crate::cuda::gdn::deferred_decode_supported(&device, inputs.batch, VALUE_HEADS)? {
                let cursors = pool(&deferred)
                    .deferred_state()
                    .unwrap()
                    .pending_rows
                    .to_vec1::<u32>()?;
                assert!(cursors[..inputs.batch].iter().all(|&cursor| cursor == 0));
                assert_close(
                    &pool(&deferred).recurrent_state.narrow(0, 0, inputs.batch)?,
                    &pool(&eager).recurrent_state.narrow(0, 0, inputs.batch)?,
                    STATE_TOLERANCE,
                )?;
            }
        }
        let pending = pool(&deferred)
            .deferred_state()
            .unwrap()
            .pending_rows
            .to_vec1::<u32>()?;
        assert!(pending.iter().any(|&cursor| cursor > 0));
        let saved = deferred.snapshot_recurrent_checkpoint_state(0)?;
        let expected_saved = eager.snapshot_recurrent_checkpoint_state(0)?;
        let (inputs, output, graph) = &graphs[3];
        graph.launch().map_err(candle_core::Error::wrap)?;
        assert_close(
            output,
            &inputs.pooled(&gdn, &eager, RecurrentBatchKind::Decode)?,
            OUTPUT_TOLERANCE,
        )?;
        deferred.restore_recurrent_checkpoint_state(0, &saved)?;
        eager.restore_recurrent_checkpoint_state(0, &expected_saved)?;
        let speculative = Inputs::new(&gdn, 2, 1, &device)?;
        assert_close(
            &speculative.pooled(&gdn, &deferred, RecurrentBatchKind::SpeculativeDecode)?,
            &speculative.pooled(&gdn, &eager, RecurrentBatchKind::SpeculativeDecode)?,
            OUTPUT_TOLERANCE,
        )?;
        let prefill = Inputs::new(&gdn, 2, 3, &device)?;
        assert_close(
            &prefill.pooled(&gdn, &deferred, RecurrentBatchKind::Prefill)?,
            &prefill.pooled(&gdn, &eager, RecurrentBatchKind::Prefill)?,
            OUTPUT_TOLERANCE,
        )?;
        graph.launch().map_err(candle_core::Error::wrap)?;
        let _ = inputs.pooled(&gdn, &eager, RecurrentBatchKind::Decode)?;
        assert_close(
            &prefill.gathered_prefill(&gdn, &mut deferred)?,
            &prefill.gathered_prefill(&gdn, &mut eager)?,
            OUTPUT_TOLERANCE,
        )?;
        for slot in 0..MAX_BATCH {
            let snapshots = deferred.snapshot_recurrent_state(slot, slot)?;
            let expected = eager.snapshot_recurrent_state(slot, slot)?;
            assert_close(
                &snapshots[0].recurrent_state,
                &expected[0].recurrent_state,
                STATE_TOLERANCE,
            )?;
            assert_close(&snapshots[0].conv_state, &expected[0].conv_state, 0.0)?;
        }
        graph.launch().map_err(candle_core::Error::wrap)?;
        let _ = inputs.pooled(&gdn, &eager, RecurrentBatchKind::Decode)?;
        assert!(deferred.release_seq(0, 0)?);
        assert!(eager.release_seq(0, 0)?);
        assert_eq!(deferred.allocate_seq(99)?, 0);
        assert_eq!(eager.allocate_seq(99)?, 0);
        assert_eq!(
            pool(&deferred)
                .deferred_state()
                .unwrap()
                .pending_rows
                .to_vec1::<u32>()?[0],
            0
        );
        graph.launch().map_err(candle_core::Error::wrap)?;
        assert_close(
            output,
            &inputs.pooled(&gdn, &eager, RecurrentBatchKind::Decode)?,
            OUTPUT_TOLERANCE,
        )?;
        device.synchronize()?;
        drop(graphs);
    }
    Ok(())
}
