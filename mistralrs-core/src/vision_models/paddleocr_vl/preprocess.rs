//! Image preprocessing: `smart_resize` + rescale/normalize/patchify, mirroring the native
//! transformers-5.13 `PaddleOCRVLImageProcessor` (a *fast*, torchvision-backed processor).
//!
//! RESIZE CAVEAT: the HF reference `pixel_values` are produced by the reference's torchvision
//! BICUBIC + antialias resize (uint8 in -> uint8 out). That is NOT byte-reproducible by the
//! `image` crate's `FilterType::CatmullRom` (mistral.rs's production resampler for `resample=3`),
//! nor by PIL BICUBIC. So the full production path is verified for shape/grid and its resize
//! divergence is measured (informational); the exact `pixel_values` parity is proven on the
//! reference's post-resize pixels through `normalize_patchify`. Token-parity through the
//! production path is the real gate.

use candle_core::{Device, Result, Tensor};
use image::{imageops::FilterType, DynamicImage, GenericImageView};

pub const PATCH: usize = 14;
pub const MERGE: usize = 2;
pub const FACTOR: usize = PATCH * MERGE; // 28: the multiple H and W are snapped to
pub const MIN_PIXELS: usize = 144 * 28 * 28; // 112896 (preprocessor_config.json)
pub const MAX_PIXELS: usize = 1280 * 28 * 28; // 1003520
const RESCALE: f64 = 0.00392156862745098; // exact 1/255 constant from config
const MEAN: f64 = 0.5;
const STD: f64 = 0.5;

/// Native `smart_resize` with the OCR-task pixel bounds. Maps `(h, w)` to the nearest multiples of
/// `FACTOR` with total pixels clamped to `[MIN_PIXELS, MAX_PIXELS]`, aspect preserved.
pub fn smart_resize(height: usize, width: usize) -> (usize, usize) {
    smart_resize_bounded(height, width, MIN_PIXELS, MAX_PIXELS)
}

/// Pixel-bound-parameterized `smart_resize` (spotting passes a larger `max_pixels`; that per-task
/// override lives in the pipeline layer, not the HF processor).
/// Python `round()` is banker's rounding, matched by `f64::round_ties_even`; `floor`/`ceil` are exact.
pub fn smart_resize_bounded(
    height: usize,
    width: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> (usize, usize) {
    let f = FACTOR as f64;
    let (mut h, mut w) = (height as f64, width as f64);
    if h < f {
        w = (w * f / h).round_ties_even();
        h = f;
    }
    if w < f {
        h = (h * f / w).round_ties_even();
        w = f;
    }
    let mut h_bar = (h / f).round_ties_even() * f;
    let mut w_bar = (w / f).round_ties_even() * f;
    let (min_px, max_px) = (min_pixels as f64, max_pixels as f64);
    if h_bar * w_bar > max_px {
        let beta = (h * w / max_px).sqrt();
        h_bar = (h / beta / f).floor() * f;
        w_bar = (w / beta / f).floor() * f;
    } else if h_bar * w_bar < min_px {
        let beta = (min_px / (h * w)).sqrt();
        h_bar = (h * beta / f).ceil() * f;
        w_bar = (w * beta / f).ceil() * f;
    }
    (h_bar as usize, w_bar as usize)
}

/// `resized`: f32 `[3, H, W]`, values 0..255 (channel-first). Returns `pixel_values`
/// `[grid_h*grid_w, 3, PATCH, PATCH]`, patch index row-major over `(grid_h, grid_w)`.
///
/// rescale (`*1/255`) then normalize (`(x-mean)/std`) as two affines (torch's op order), then the
/// SigLIP-style patchify: split H->(gh,P) and W->(gw,P), group patch dims first. Row-major
/// `[C, gh, P, gw, P]` -> permute -> `[gh, gw, C, P, P]` matches the native processor's
/// `permute(0,1,4,6,3,2,5,7)` (with the batch/grid_t/tps singletons dropped).
pub fn normalize_patchify(resized: &Tensor) -> Result<Tensor> {
    let (c, h, w) = resized.dims3()?;
    let (gh, gw) = (h / PATCH, w / PATCH);
    let x = resized
        .affine(RESCALE, 0.0)? // rescale: v = x/255
        .affine(1.0 / STD, -MEAN / STD)?; // normalize: (v-mean)/std
    x.reshape((c, gh, PATCH, gw, PATCH))?
        .permute((1, 3, 0, 2, 4))?
        .contiguous()?
        .reshape((gh * gw, c, PATCH, PATCH))
}

/// Full production preprocessing: decode -> `smart_resize` -> `image`-crate CatmullRom resize ->
/// `normalize_patchify`. Returns (`pixel_values [N,3,14,14]`, `grid_thw = (t=1, gh, gw)`).
/// The CatmullRom resize is mistral.rs's `resample=3` path and is NOT byte-exact vs the HF
/// reference; this is the path that must yield token-parity.
pub fn preprocess_image(path: &str, dev: &Device) -> Result<(Tensor, (usize, usize, usize))> {
    let img = image::open(path).map_err(candle_core::Error::wrap)?;
    preprocess_decoded(&img, dev)
}

/// Same as `preprocess_image` but on an already-decoded image (the inputs_processor hands us
/// `DynamicImage`s, not paths). The resize+normalize path is byte-identical to `preprocess_image`.
pub fn preprocess_decoded(
    img: &DynamicImage,
    dev: &Device,
) -> Result<(Tensor, (usize, usize, usize))> {
    let (w0, h0) = img.dimensions();
    let (h_bar, w_bar) = smart_resize(h0 as usize, w0 as usize);
    let resized = img
        .resize_exact(w_bar as u32, h_bar as u32, FilterType::CatmullRom)
        .to_rgb8();
    let buf: Vec<f32> = resized.as_raw().iter().map(|&v| v as f32).collect();
    let chw = Tensor::from_vec(buf, (h_bar, w_bar, 3), dev)? // [H,W,3]
        .permute((2, 0, 1))? // [3,H,W]
        .contiguous()?;
    let px = normalize_patchify(&chw)?;
    Ok((px, (1, h_bar / PATCH, w_bar / PATCH)))
}
