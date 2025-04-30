use crate::utils::image_to_pixels;
use candle_core::{Device, Result, Tensor, D};
use image::DynamicImage;

use crate::ImageTransform;

/// Convert an image to a tensor. This converts the data from being in `[0, 255]` to `[0.0, 1.0]`.
/// The tensor's shape is (channels, height, width).
pub struct ToTensor;

impl ImageTransform for ToTensor {
    type Input = DynamicImage;
    type Output = Tensor;
    fn map(&self, x: &Self::Input, device: &Device) -> Result<Self::Output> {
        image_to_pixels(x, device)? / 255.
    }
}

/// Convert an image to a tensor without normalizing to `[0.0, 1.0]`.
/// The tensor's shape is (channels, height, width).
pub struct ToTensorNoNorm;

impl ImageTransform for ToTensorNoNorm {
    type Input = DynamicImage;
    type Output = Tensor;
    fn map(&self, x: &Self::Input, device: &Device) -> Result<Self::Output> {
        image_to_pixels(x, device)
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
        let num_channels = x.dim(D::Minus(3))?;
        if self.mean.len() != num_channels || self.std.len() != num_channels {
            candle_core::bail!(
                "Num channels ({}) must match number of mean ({}) and std ({}).",
                num_channels,
                self.mean.len(),
                self.std.len()
            );
        }
        let mut accum = Vec::new();
        for (i, channel) in x.chunk(num_channels, D::Minus(3))?.iter().enumerate() {
            accum.push(((channel - self.mean[i])? / self.std[i])?);
        }
        Tensor::cat(&accum, D::Minus(3))
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

impl<T: ImageTransform<Input = E, Output = E>, E: Clone> ImageTransform for Option<T> {
    type Input = T::Input;
    type Output = T::Output;

    fn map(&self, x: &T::Input, dev: &Device) -> Result<T::Output> {
        if let Some(this) = self {
            this.map(x, dev)
        } else {
            Ok(x.clone())
        }
    }
}

/// Multiply the pixe values by the provided factor.
///
/// Each pixel value is calculated as follows: x = x * factor
pub struct Rescale {
    pub factor: Option<f64>,
}

impl ImageTransform for Rescale {
    type Input = Tensor;
    type Output = Self::Input;

    fn map(&self, x: &Self::Input, _: &Device) -> Result<Self::Output> {
        if let Some(factor) = self.factor {
            x * factor
        } else {
            Ok(x.clone())
        }
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
