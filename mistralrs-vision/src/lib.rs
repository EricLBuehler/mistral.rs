//! This crate provides vision utilities for mistral.rs inspired by torchvision.
//! In particular, it represents transformations on some `Self` type which are applied
//! sequentially.
//!
//! ## Example
//! ```rust
//! use candle_core::Device;
//! use image::{ColorType, DynamicImage};
//! use mistralrs_vision::{ApplyTransforms, Normalize, ToTensor, Transforms};
//!
//! let image = DynamicImage::new(3, 4, ColorType::Rgb8);
//! let transforms = Transforms {
//!     input: &ToTensor,
//!     inner_transforms: &[&Normalize {
//!         mean: vec![0.5, 0.5, 0.5],
//!         std: vec![0.5, 0.5, 0.5],
//!     }],
//! };
//! let transformed = image.apply(transforms, &Device::Cpu).unwrap();
//! assert_eq!(transformed.dims(), &[3, 4, 3]);
//! ```

use candle_core::{Device, Result, Tensor};
use image::DynamicImage;
mod ops;
mod pad;
mod transforms;
pub(crate) mod utils;

pub use ops::{get_resize_image_size, make_pixel_mask, pad};
pub use pad::{pad_to_max_edge, pad_to_max_image_size};
pub use transforms::{InterpolateResize, Normalize, Rescale, ToTensor, ToTensorNoNorm};

/// A transform over an image. The input may vary but the output is always a Tensor.
pub trait ImageTransform {
    type Input;
    type Output;

    fn map(&self, x: &Self::Input, device: &Device) -> Result<Self::Output>;
}

/// Transforms to apply, starting with the `input` and then with each transform in
/// `inner_transforms` applied sequentially
pub struct Transforms<'a> {
    pub input: &'a dyn ImageTransform<Input = DynamicImage, Output = Tensor>,
    pub inner_transforms: &'a [&'a dyn ImageTransform<Input = Tensor, Output = Tensor>],
}

/// Transforms, with each of `inner_transforms` applied sequentially
pub struct TensorTransforms<'a> {
    pub inner_transforms: &'a [&'a dyn ImageTransform<Input = Tensor, Output = Tensor>],
}

/// Application of transforms to the Self type.
pub trait ApplyTransforms<'a> {
    fn apply(&self, transforms: Transforms<'a>, device: &Device) -> Result<Tensor>;
}

impl<'a> ApplyTransforms<'a> for DynamicImage {
    fn apply(&self, transforms: Transforms<'a>, device: &Device) -> Result<Tensor> {
        let mut res = transforms.input.map(self, device)?;
        for transform in transforms.inner_transforms {
            res = transform.map(&res, device)?;
        }
        Ok(res)
    }
}

/// Application of transforms to the Self type.
pub trait ApplyTensorTransforms<'a> {
    fn apply(&self, transforms: TensorTransforms<'a>, device: &Device) -> Result<Tensor>;
}

impl<'a> ApplyTensorTransforms<'a> for Tensor {
    fn apply(&self, transforms: TensorTransforms<'a>, device: &Device) -> Result<Tensor> {
        let mut res = self.clone();
        for transform in transforms.inner_transforms {
            res = transform.map(&res, device)?;
        }
        Ok(res)
    }
}
