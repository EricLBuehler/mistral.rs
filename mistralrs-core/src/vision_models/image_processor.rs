#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::HashMap;

use candle_core::{Device, Result, Tensor};
use image::{imageops::FilterType, DynamicImage, GenericImageView, Pixel, RgbImage, RgbaImage};

use crate::pipeline::InputsProcessor;

use super::preprocessor_config::{PreProcessorConfig, ToFilter};

pub(crate) struct PreprocessedImages {
    pub(crate) pixel_values: Tensor,
    pub(crate) pixel_attention_mask: Tensor,
}

pub(crate) fn empty_image(h: usize, w: usize) -> Vec<Vec<Vec<u8>>> {
    vec![vec![vec![]; w]; h]
}

pub(crate) fn get_pixel_data(image: &DynamicImage, h: usize, w: usize) -> Vec<Vec<Vec<u8>>> {
    let mut pixel_data = empty_image(h, w);
    let n_channels = image.color().channel_count() as usize;
    image.pixels().for_each(|(x, y, pixel)| {
        pixel_data[y as usize][x as usize] = pixel.channels()[..n_channels].to_vec()
    });
    pixel_data
}

pub(crate) fn from_pixel_data(mut data: Vec<Vec<Vec<u8>>>, h: usize, w: usize) -> DynamicImage {
    let channels = data[0][0].len();
    let cur_h = data.len();
    let cur_w = data[0].len();
    let delta_h = h - cur_h;
    let delta_w = w - cur_w;

    let mut flat_data: Vec<u8> = Vec::with_capacity(w * h * channels);
    data.extend(vec![vec![vec![0; channels]; cur_w]; delta_h]);
    for mut row in data {
        row.extend(vec![vec![0; channels]; delta_w]);
        for pixel in row {
            flat_data.extend_from_slice(&pixel);
        }
    }

    if channels == 3 {
        DynamicImage::ImageRgb8(
            RgbImage::from_raw(w as u32, h as u32, flat_data).expect("Unable to create RgbaImage"),
        )
    } else {
        DynamicImage::ImageRgba8(
            RgbaImage::from_raw(w as u32, h as u32, flat_data).expect("Unable to create RgbaImage"),
        )
    }
}

pub(crate) fn resize(
    image: &DynamicImage,
    size: &HashMap<String, u32>,
    resample: Option<usize>,
) -> Result<DynamicImage> {
    let (h, w) = if size.contains_key("shortest_edge") && size.contains_key("longest_edge") {
        let (mut w, mut h) = image.dimensions();

        let min_len = size["shortest_edge"];
        let max_len = size["longest_edge"];
        let aspect_ratio = w as f32 / h as f32;

        if w >= h && w > max_len {
            w = max_len;
            h = (w as f32 / aspect_ratio) as u32;
        } else if h > w && h > max_len {
            h = max_len;
            w = (h as f32 * aspect_ratio) as u32;
        }
        h = h.max(min_len);
        w = w.max(min_len);
        (h, w)
    } else if size.contains_key("height") && size.contains_key("width") {
        (size["height"], size["width"])
    } else {
        candle_core::bail!("Size must contain either both keys `shortest_edge` and `longest_edge` or `height` and `width`")
    };
    let filter: FilterType = resample.to_filter()?;
    Ok(image.resize(w, h, filter))
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
            row_accum.push(Tensor::from_slice(&item, (1, item.len()), &Device::Cpu)?)
        }
        let row = Tensor::cat(&row_accum, 0)?;
        accum.push(row.reshape((row.dim(1)?, ()))?.unsqueeze(1)?);
    }
    Tensor::cat(&accum, 1)?.to_device(device)
}

pub trait ImagePreProcessor: InputsProcessor {
    const DEFAULT_MEAN: [f64; 3];
    const DEFAULT_STD: [f64; 3];

    /// Preprocess the images.
    ///
    /// - `resize` specifies the (w,h) of the target and should be paired with `filter`.
    /// - `filter` filter type for resizing.
    /// - `rescale` multiplies by the scale.
    /// - `normalize` normalizes the image by the mean and std dev (if none, uses default mean/std).
    /// - `do_pad` pads the images to the one with the highest dimensions and will create a pixel attention mask.
    ///   Be sure to set this to `true` if the images differ in dimensions
    /// - `pad_to` pads the images to the specified dimension. This must be greater than or equal to the maximum
    ///   size of a specified image.
    #[allow(clippy::too_many_arguments)]
    fn preprocess(
        &self,
        images: Vec<DynamicImage>,
        config: &PreProcessorConfig,
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
                c.iter_mut().for_each(|x| {
                    *x = f(*x);
                });
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
                for (channel, x) in c.iter_mut().enumerate() {
                    *x = f(*x, channel);
                }
            }
        });
        from_pixel_data(data, h as usize, w as usize)
    }

    /// Rescale image by `scale`
    fn rescale(&self, image: &DynamicImage, scale: f64) -> DynamicImage {
        self.map_image(image, |x| (x as f64 * scale) as u8)
    }

    /// Normalizes the image using the standard distribution specified by `mean` and `std`
    /// for each channel.
    ///
    /// (image-mean)/std
    fn normalize(&self, image: &DynamicImage, mean: [f64; 3], std: [f64; 3]) -> DynamicImage {
        self.map_image_channels(image, |x, channel| {
            ((x as f64 - mean.get(channel).unwrap_or(&1.)) / std.get(channel).unwrap_or(&1.)) as u8
        })
    }
}
