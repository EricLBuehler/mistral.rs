use crate::utils::{get_pixel_data, n_channels};
use candle_core::{DType, Device, Result, Tensor};
use image::{DynamicImage, GenericImageView};

use crate::ImageTransform;

/// Convert an image to a tensor. This converts the data from being in `[0, 255]` to `[0.0, 1.0]`.
/// The tensor's shape is (channels, height, width).
pub struct ToTensor;

impl ToTensor {
    fn to_tensor(device: &Device, channels: usize, data: Vec<Vec<Vec<u8>>>) -> Result<Tensor> {
        let mut accum = Vec::new();
        for row in data {
            let mut row_accum = Vec::new();
            for item in row {
                row_accum.push(
                    Tensor::from_slice(&item[..channels], (1, channels), &Device::Cpu)?
                        .to_dtype(DType::F32)?,
                )
            }
            let row = Tensor::cat(&row_accum, 0)?;
            accum.push(row.t()?.unsqueeze(1)?);
        }
        let t = Tensor::cat(&accum, 1)?.to_device(device)?;
        // Rescale to between 0 and 1
        t / 255.0f64
    }
}

impl ImageTransform for ToTensor {
    type Input = DynamicImage;
    type Output = Tensor;
    fn map(&self, x: &Self::Input, device: &Device) -> Result<Self::Output> {
        let num_channels = n_channels(x);
        let data = get_pixel_data(
            num_channels,
            x.to_rgba8(),
            x.dimensions().1 as usize,
            x.dimensions().0 as usize,
        );
        Self::to_tensor(device, num_channels, data)
    }
}

/// Normalize the image data based on the mean and standard deviation.
/// The value is computed as follows:
/// `
/// x[channel]=(x[channel - mean[channel]) / std[channel]
/// `
///
/// Expects an input tensor of shape (channels, height, width).
pub struct Normalize {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
}

impl ImageTransform for Normalize {
    type Input = Tensor;
    type Output = Self::Input;

    fn map(&self, x: &Self::Input, _: &Device) -> Result<Self::Output> {
        let num_channels = x.dim(0)?;
        if self.mean.len() != num_channels || self.std.len() != num_channels {
            candle_core::bail!("Num channels must match number of mean and std.");
        }
        let mut accum = Vec::new();
        for (i, channel) in x.chunk(num_channels, 0)?.iter().enumerate() {
            accum.push(((channel - self.mean[i])? / self.std[i])?);
        }
        Tensor::cat(&accum, 0)
    }
}

/// Resize the image via nearest interpolation.
pub struct InterpolateResize {
    pub target_w: usize,
    pub target_h: usize,
}

impl ImageTransform for InterpolateResize {
    type Input = Tensor;
    type Output = Self::Input;

    fn map(&self, x: &Self::Input, _: &Device) -> Result<Self::Output> {
        x.unsqueeze(0)?
            .interpolate2d(self.target_h, self.target_w)?
            .squeeze(0)
    }
}

mod tests {
    #[test]
    fn test_to_tensor() {
        use candle_core::Device;
        use image::{ColorType, DynamicImage};

        use crate::ImageTransform;

        use super::ToTensor;

        let image = DynamicImage::new(4, 5, ColorType::Rgb8);
        let res = ToTensor.map(&image, &Device::Cpu).unwrap();
        assert_eq!(res.dims(), &[3, 5, 4])
    }

    #[test]
    fn test_normalize() {
        use crate::{ImageTransform, Normalize};
        use candle_core::{DType, Device, Tensor};

        let image = Tensor::zeros((3, 5, 4), DType::U8, &Device::Cpu).unwrap();
        let res = Normalize {
            mean: vec![0.5, 0.5, 0.5],
            std: vec![0.5, 0.5, 0.5],
        }
        .map(&image, &Device::Cpu)
        .unwrap();
        assert_eq!(res.dims(), &[3, 5, 4])
    }
}
