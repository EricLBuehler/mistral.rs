use candle_core::Device;
use image::{ColorType, DynamicImage};
use mistralrs_vision::{ApplyTransforms, Normalize, ToTensor, Transforms};

#[test]
fn normalize() {
    let image = DynamicImage::new(3, 4, ColorType::Rgb8);
    let transforms = Transforms {
        input: &ToTensor,
        inner_transforms: &[&Normalize {
            mean: vec![0.5, 0.5, 0.5],
            std: vec![0.5, 0.5, 0.5],
        }],
    };
    let transformed = (&image).apply(transforms, &Device::Cpu).unwrap();
    assert_eq!(transformed.dims(), &[3, 4, 3]);
}
