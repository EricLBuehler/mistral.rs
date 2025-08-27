use candle_core::{from_storage_no_op, DType, MetalStorage, Result, Storage, Tensor};

/// blocks is (a,b,c,d)
/// scales is (a,b,c)
///
/// (a,b) are the prefix, d*c*2 is the output last dim.
pub fn mxfp4_unpack(blocks: &Tensor, scales: &Tensor, out_ty: DType) -> Result<Tensor> {
    let (a, b, c, d) = blocks.dims4()?;
    let rows_total = a * b * c;

    assert!(blocks.is_contiguous());
    assert!(scales.is_contiguous());
    assert_eq!(blocks.layout().start_offset(), 0);
    assert_eq!(scales.layout().start_offset(), 0);

    #[cfg(feature = "metal")]
    {
        use candle_core::{backend::BackendStorage, Shape};

        let b_s = blocks.storage_and_layout().0;
        let Storage::Metal(b_s) = &*b_s else {
            candle_core::bail!("expected metal")
        };
        let s_s = scales.storage_and_layout().0;
        let Storage::Metal(s_s) = &*s_s else {
            candle_core::bail!("expected metal")
        };

        let device = b_s.device();
        let command_buffer = device.command_buffer()?;
        command_buffer.set_label("mxfp4-unpack");

        let output = device.new_buffer(a * b * c * d * 2, out_ty, "mxfp4-unpack-output")?;

        crate::metal_kernels::call_unpack_mxfp4(
            device.device(),
            &command_buffer,
            &crate::metal_kernels::Kernels::new(),
            out_ty,
            b_s.buffer(),
            s_s.buffer(),
            &output,
            rows_total,
            d,
        )
        .map_err(candle_core::Error::wrap)?;

        let output_shape = Shape::from_dims(&[a, b, c * d * 2]);
        let output = from_storage_no_op(
            Storage::Metal(MetalStorage::new(
                output,
                device.clone(),
                a * b * c * d * 2,
                out_ty,
            )),
            output_shape,
            false,
        );

        Ok(output)
    }
    #[cfg(not(feature = "metal"))]
    {
        todo!()
    }
}
