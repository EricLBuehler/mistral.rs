use candle_core::Device;
use image::{ColorType, DynamicImage};
use mistralrs_vision::{ApplyTransforms, InterpolateResize, Normalize, ToTensor, Transforms};

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
    let transformed = image.apply(transforms, &Device::Cpu).unwrap();
    assert_eq!(transformed.dims(), &[3, 4, 3]);
}

#[test]
fn normalize_and_interpolate_resize() {
    let image = DynamicImage::new(300, 400, ColorType::Rgb8);
    let transforms = Transforms {
        input: &ToTensor,
        inner_transforms: &[
            &Normalize {
                mean: vec![0.5, 0.5, 0.5],
                std: vec![0.5, 0.5, 0.5],
            },
            &InterpolateResize {
                target_h: 336,
                target_w: 336,
            },
        ],
    };
    let transformed = image.apply(transforms, &Device::Cpu).unwrap();
    assert_eq!(transformed.dims(), &[3, 336, 336]);
}
