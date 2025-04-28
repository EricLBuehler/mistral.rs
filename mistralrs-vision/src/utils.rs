use candle_core::{DType, Device, Result, Tensor};
use image::{DynamicImage, GenericImageView};

/// Output is (c, h, w)
pub(crate) fn image_to_pixels(image: &DynamicImage, device: &Device) -> Result<Tensor> {
    let (w, h) = image.dimensions();
    let data = image.to_rgb8().into_raw();
    let data = Tensor::from_vec(data, (h as usize, w as usize, n_channels(image)), device)?;
    data.permute((2, 0, 1))?.to_dtype(DType::F32)
}

pub(crate) fn n_channels(image: &DynamicImage) -> usize {
    image.color().channel_count() as usize
}
