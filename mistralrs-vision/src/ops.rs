use candle_core::{Result, Tensor};

/// Pad an image of shape (c, h, w) to (c, max_h, max_w) by padding with zeros on the right and bottom.
pub fn pad(image: &Tensor, max_h: usize, max_w: usize) -> Result<Tensor> {
    let (c, h, w) = image.dims3()?;
    let new_image = Tensor::zeros((c, max_h, max_w), image.dtype(), image.device())?;
    new_image.slice_assign(&[0..c, 0..h, 0..w], image)
}

/// Generate pixel mask of shape (c, max_h, max_w). 1 indicates valid pixel, 0 indicates padding.
///
/// The input tensor is of shape (c, max_h, max_w) and the output mask is the same shape and
/// represents where pixels are. The mask shape is in the top left corner is passed as `h` and `w`.
pub fn make_pixel_mask(image: &Tensor, h: usize, w: usize) -> Result<Tensor> {
    let (_c, max_h, max_w) = image.dims3()?;
    let mask = Tensor::ones((h, w), image.dtype(), image.device())?;
    let zeros = Tensor::zeros((max_h, max_w), image.dtype(), image.device())?;
    // TODO(EricLBuehler): https://github.com/huggingface/candle/pull/2223 will make this nicer
    zeros.slice_assign(&[0..h, 0..w], &mask)
}

/// Given the image sizes (h, w) and the minimum and maximum lengths, calculate the image dimensions
/// which will preserve aspect ration while respecing the minimum and maximum lengths.
pub fn get_resize_image_size(
    (h, w): (usize, usize),
    (min_len, max_len): (usize, usize),
) -> (usize, usize) {
    let aspect_ratio = w as f64 / h as f64;

    let (new_h, new_w) = if w >= h && w > max_len {
        ((max_len as f64 / aspect_ratio) as usize, max_len)
    } else if h > w && h > max_len {
        (max_len, (max_len as f64 * aspect_ratio) as usize)
    } else {
        (h, w)
    };
    (new_h.max(min_len), new_w.max(min_len))
}
