#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::{Device, Result, Tensor};
use image::{imageops::FilterType, DynamicImage, GenericImageView, ImageBuffer, Pixel, Rgb};

use crate::pipeline::InputsProcessor;

pub(crate) struct NormalizationMetadata {
    pub(crate) image_mean: [f32; 3],
    pub(crate) image_std: [f32; 3],
}

pub(crate) struct PreprocessedImages {
    pub(crate) pixel_values: Tensor,
    pub(crate) pixel_attention_mask: Tensor,
}

pub(crate) fn empty_image(h: usize, w: usize) -> Vec<Vec<Rgb<u8>>> {
    vec![vec![Rgb::from([0u8, 0, 0]); w]; h]
}

pub(crate) fn get_pixel_data(image: &DynamicImage, h: usize, w: usize) -> Vec<Vec<Rgb<u8>>> {
    let mut pixel_data = empty_image(h, w);
    image
        .pixels()
        .for_each(|(x, y, pixel)| pixel_data[y as usize][x as usize] = pixel.to_rgb());
    pixel_data
}

pub(crate) fn from_pixel_data(data: Vec<Vec<Rgb<u8>>>, h: usize, w: usize) -> DynamicImage {
    let mut flat_data: Vec<u8> = Vec::with_capacity(w * h * 4);
    for row in data {
        for pixel in row {
            flat_data.extend_from_slice(&pixel.0);
        }
    }

    let img_buffer =
        ImageBuffer::from_vec(w as u32, h as u32, flat_data).expect("Unable to create ImageBuffer");

    DynamicImage::ImageRgb8(img_buffer)
}

pub(crate) fn resize(image: &DynamicImage, w: u32, h: u32, filter: FilterType) -> DynamicImage {
    image.resize(w, h, filter)
}

/// Output is: (3, height, width)
pub(crate) fn make_pixel_values(image: &DynamicImage, device: &Device) -> Result<Tensor> {
    let data = get_pixel_data(
        image,
        image.dimensions().1 as usize,
        image.dimensions().0 as usize,
    );
    let mut accum = Vec::new();
    for row in data {
        let mut row_accum = Vec::new();
        for item in row {
            let [r, g, b] = item.0;
            row_accum.push(Tensor::from_slice(&[r, g, b], (1, 3), device)?)
        }
        accum.push(Tensor::cat(&row_accum, 0)?.reshape((3, ()))?.unsqueeze(0)?);
    }
    Tensor::cat(&accum, 0)
}

pub trait ImagePreProcessor: InputsProcessor {
    const DEFAULT_MEAN: [f32; 3];
    const DEFAULT_STD: [f32; 3];

    #[allow(clippy::too_many_arguments)]
    fn preprocess(
        &self,
        images: Vec<DynamicImage>,
        do_resize: bool,
        rescale: Option<f32>,
        normalize: Option<NormalizationMetadata>,
        do_pad: bool,
        filter: FilterType,
        device: &Device,
    ) -> Result<PreprocessedImages>;

    /// Crops the image to the given size using a center crop. Note that if the image is too small, it will be padded.
    /// The returned image is always of size (height, width)
    #[allow(dead_code)]
    fn center_crop(&self, image: &DynamicImage, height: u32, width: u32) -> DynamicImage {
        let (orig_width, orig_height) = image.dimensions();
        let top = (orig_height as i32 - height as i32) / 2;
        let bottom = top + height as i32;
        let left = (orig_width as i32 - width as i32) / 2;
        let right = left + width as i32;

        // Check if cropped area is within image boundaries
        if top >= 0 && bottom <= height as i32 && left >= 0 && right <= width as i32 {
            return image.crop_imm(left as u32, top as u32, width, height);
        }

        let new_height = height.max(orig_height);
        let new_width = width.max(orig_width);

        // Pad the image...
        let top_pad = (new_height - orig_height) / 2;
        let bottom_pad = top_pad + orig_height;
        let left_pad = (new_width - orig_width) / 2;
        let right_pad = left_pad + orig_width;

        let (new_width, new_height) = (new_width as usize, new_height as usize);
        let mut pixel_data = get_pixel_data(image, new_height, new_width);

        let y_range = ((top + top_pad as i32).max(0) as usize)
            ..((new_height as i32).min(bottom + bottom_pad as i32) as usize);
        let x_range = ((left + left_pad as i32).max(0) as usize)
            ..((new_width as i32).min(right + right_pad as i32) as usize);
        pixel_data = pixel_data[y_range][x_range].to_vec();

        from_pixel_data(pixel_data, new_height, new_width)
    }

    /// Map an image's pixel channels.
    fn map_image<F>(&self, image: &DynamicImage, mut f: F) -> DynamicImage
    where
        F: FnMut(u8) -> u8,
    {
        let (w, h) = image.dimensions();
        let mut data = get_pixel_data(image, h as usize, w as usize);
        data.iter_mut().for_each(|row| {
            for c in row {
                #[allow(clippy::redundant_closure)]
                c.apply_without_alpha(|x| f(x));
            }
        });
        from_pixel_data(data, h as usize, w as usize)
    }

    /// Map an image's pixel channels and while providing the image channel
    fn map_image_channels<F>(&self, image: &DynamicImage, mut f: F) -> DynamicImage
    where
        F: FnMut(u8, usize) -> u8,
    {
        let (w, h) = image.dimensions();
        let mut data = get_pixel_data(image, h as usize, w as usize);
        data.iter_mut().for_each(|row| {
            for c in row {
                for (channel, x) in c.channels_mut().iter_mut().enumerate() {
                    *x = f(*x, channel);
                }
            }
        });
        from_pixel_data(data, h as usize, w as usize)
    }

    /// Rescale image by `scale`
    fn rescale(&self, image: &DynamicImage, scale: f32) -> DynamicImage {
        self.map_image(image, |x| (x as f32 * scale) as u8)
    }

    /// Normalizes the image using the standard distribution specified by `mean` and `std`
    /// for each channel.
    ///
    /// (image-mean)/std
    fn normalize(&self, image: &DynamicImage, mean: [f32; 3], std: [f32; 3]) -> DynamicImage {
        self.map_image_channels(image, |x, channel| {
            ((x as f32 - mean[channel]) / std[channel]) as u8
        })
    }
}
