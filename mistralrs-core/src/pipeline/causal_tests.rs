use super::{validate_speculative_graph_tensor_metadata, SpeculativeGraphTensorMetadata};
use candle_core::{DType, DeviceLocation};

fn metadata(
    shape: &[usize],
    strides: &[usize],
    contiguous: bool,
    dtype: DType,
) -> SpeculativeGraphTensorMetadata {
    SpeculativeGraphTensorMetadata {
        shape: shape.to_vec(),
        strides: strides.to_vec(),
        contiguous,
        dtype,
        device: DeviceLocation::Cpu,
    }
}

#[test]
fn validates_tensor_count_and_metadata() {
    let expected = vec![metadata(&[4, 8, 16], &[128, 16, 1], true, DType::BF16)];
    assert!(validate_speculative_graph_tensor_metadata(&expected, &expected).is_ok());
    assert!(validate_speculative_graph_tensor_metadata(&expected, &[]).is_err());
    assert!(validate_speculative_graph_tensor_metadata(
        &expected,
        &[metadata(&[4, 8, 17], &[136, 17, 1], true, DType::BF16)]
    )
    .is_err());
    assert!(validate_speculative_graph_tensor_metadata(
        &expected,
        &[metadata(&[4, 8, 16], &[128, 16, 1], true, DType::F16)]
    )
    .is_err());
    assert!(validate_speculative_graph_tensor_metadata(
        &expected,
        &[metadata(&[4, 8, 16], &[1, 64, 4], false, DType::BF16)]
    )
    .is_err());
    assert!(validate_speculative_graph_tensor_metadata(
        &expected,
        &[metadata(&[4, 8, 16], &[128, 16, 1], false, DType::BF16)]
    )
    .is_err());
}
