use candle_core::{DType, Device, Result, Tensor};
use image::{DynamicImage, GenericImageView};

/// Output is (c, h, w)
pub(crate) fn image_to_pixels(image: &DynamicImage, device: &Device) -> Result<Tensor> {
    let (w, h) = image.dimensions();
    let n_channels = n_channels(image);
    let data = match n_channels {
        1 => image.to_luma8().into_raw(),
        2 => image.to_luma_alpha8().into_raw(),
        3 => image.to_rgb8().into_raw(),
        4 => image.to_rgba8().into_raw(),
        _ => candle_core::bail!("Unsupported channel count {n_channels}"),
    };
    let data = Tensor::from_vec(data, (h as usize, w as usize, n_channels), device)?;
    data.permute((2, 0, 1))?.to_dtype(DType::F32)
}

pub(crate) fn n_channels(image: &DynamicImage) -> usize {
    image.color().channel_count() as usize
}
