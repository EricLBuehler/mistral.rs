use candle_core::{Device, Result, Tensor};
use image::DynamicImage;
mod transforms;
pub(crate) mod utils;
pub use transforms::{InterpolateResize, Normalize, ToTensor};

pub trait ImageTransform {
    type Input;
    type Output;

    fn map(&self, x: &Self::Input, device: &Device) -> Result<Self::Output>;
}

pub struct Transforms<'a> {
    pub input: &'a dyn ImageTransform<Input = DynamicImage, Output = Tensor>,
    pub inner_transforms: &'a [&'a dyn ImageTransform<Input = Tensor, Output = Tensor>],
}

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
