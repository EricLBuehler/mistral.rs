use image::{DynamicImage, Pixel};

pub(crate) fn empty_image(h: usize, w: usize) -> Vec<Vec<Vec<u8>>> {
    vec![vec![vec![]; w]; h]
}

pub(crate) fn get_pixel_data(
    n_channels: usize,
    pixels: image::ImageBuffer<image::Rgba<u8>, Vec<u8>>,
    h: usize,
    w: usize,
) -> Vec<Vec<Vec<u8>>> {
    let mut pixel_data = empty_image(h, w);
    let pixels = pixels.pixels();
    for (n, pixel) in pixels.into_iter().enumerate() {
        let x = n % w;
        let y = n / h;
        pixel_data[y][x] = pixel.channels()[..n_channels].to_vec()
    }
    pixel_data
}

pub(crate) fn n_channels(image: &DynamicImage) -> usize {
    image.color().channel_count() as usize
}
