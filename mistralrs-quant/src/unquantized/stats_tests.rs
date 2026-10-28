use super::*;

const INPUT_DIM: usize = 6;
const OUTPUT_DIM: usize = 5;
const INPUT_SCALE: f32 = 32.0;
const STATS_TOLERANCE: f32 = 1e-6;

#[test]
#[ignore = "requires a CUDA device"]
fn small_batch_gemv_records_each_forward_once() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    for dtype in [DType::BF16, DType::F16, DType::F32] {
        let weight = Tensor::ones((OUTPUT_DIM, INPUT_DIM), dtype, &device)?;
        let layer = <UnquantLinear as QuantMethod>::new(QuantMethodConfig::Unquantized(
            Linear::new(weight.clone(), None),
        ))?;
        let untracked = Tensor::ones((1, INPUT_DIM), dtype, &device)?;
        let _ = layer.forward(&untracked)?;
        assert!(layer.stats_snapshot().is_none());
        layer.begin_track_stats()?;
        assert_eq!(layer.stats_snapshot(), Some((0, 0)));
        let mut row_count = 0;
        let mut sum_squares = [0f32; INPUT_DIM];
        let shapes = [
            vec![1, INPUT_DIM],
            vec![1, 2, INPUT_DIM],
            vec![2, 2, 2, INPUT_DIM],
            vec![1, 9, INPUT_DIM],
        ];
        for (call, shape) in shapes.iter().enumerate() {
            let rows = shape.iter().product::<usize>() / INPUT_DIM;
            let values = (0..rows * INPUT_DIM)
                .map(|index| {
                    ((index / INPUT_DIM + 1) * (index % INPUT_DIM + 1)) as f32 / INPUT_SCALE
                })
                .collect::<Vec<_>>();
            for row in values.as_chunks::<INPUT_DIM>().0 {
                for (sum, value) in sum_squares.iter_mut().zip(row) {
                    *sum += value * value;
                }
            }
            let input = Tensor::from_vec(values, shape.as_slice(), &device)?.to_dtype(dtype)?;
            assert_eq!(
                crate::gemv::should_use_gemv(&input, &weight),
                rows <= crate::gemv::MAX_GEMV_BATCH_SIZE
            );
            let output = layer.forward(&input)?;
            let mut output_shape = shape.clone();
            *output_shape.last_mut().unwrap() = OUTPUT_DIM;
            assert_eq!(output.dims(), output_shape);
            assert!(output
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .all(|value| value.is_finite()));
            row_count += rows;
            assert_eq!(layer.stats_snapshot(), Some((call + 1, row_count)));
        }
        let actual = layer.end_track_stats()?.to_vec1::<f32>()?;
        assert!(layer.stats_snapshot().is_none());
        for (column, (actual, sum)) in actual.iter().zip(sum_squares).enumerate() {
            let expected = sum / row_count as f32 * shapes.len() as f32;
            assert!(
                (actual - expected).abs() <= STATS_TOLERANCE,
                "dtype={dtype:?} column={column} actual={actual} expected={expected}"
            );
        }
        let _ = layer.forward(&untracked)?;
        assert!(layer.stats_snapshot().is_none());
    }
    device.synchronize()?;
    Ok(())
}
