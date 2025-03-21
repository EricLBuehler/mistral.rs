use image::{
    imageops::FilterType, DynamicImage, GenericImage, GenericImageView, ImageBuffer, Rgb, Rgba,
};

fn resize_image_to_max_edge(img: &DynamicImage, max_edge: u32) -> DynamicImage {
    // Get the original dimensions of the image
    let (width, height) = img.dimensions();

    // Calculate the scaling factor
    let scale = if width > height {
        max_edge as f32 / width as f32
    } else {
        max_edge as f32 / height as f32
    };

    // New dimensions
    let new_width = (width as f32 * scale) as u32;
    let new_height = (height as f32 * scale) as u32;

    // Resize the image
    img.resize_exact(new_width, new_height, FilterType::Lanczos3)
}

/// 1) Resize the images to the maximum edge length - preserving aspect ratio
/// 2) Pad all the images with black padding.
pub fn pad_to_max_edge(images: &[DynamicImage], max_edge: u32) -> Vec<DynamicImage> {
    let mut new_images = Vec::new();
    for image in images {
        new_images.push(resize_image_to_max_edge(image, max_edge));
    }

    let mut max_height = 0;
    let mut max_width = 0;
    for image in &new_images {
        let (w, h) = image.dimensions();
        if w > max_width {
            max_width = w;
        }
        if h > max_height {
            max_height = h;
        }
    }

    for image in &mut new_images {
        match image {
            DynamicImage::ImageRgb8(rgb8) => {
                let mut padded_image = ImageBuffer::from_pixel(max_width, max_height, Rgb([0; 3]));

                padded_image
                    .copy_from(rgb8, 0, 0)
                    .expect("Failed to copy image");

                *rgb8 = padded_image;
            }
            DynamicImage::ImageRgba8(rgba8) => {
                let mut padded_image = ImageBuffer::from_pixel(max_width, max_height, Rgba([0; 4]));

                padded_image
                    .copy_from(rgba8, 0, 0)
                    .expect("Failed to copy image");

                *rgba8 = padded_image;
            }
            _ => panic!("rgb8 or rgba8 are the only supported image types"),
        }
    }

    new_images
}

pub fn pad_to_max_image_size(mut images: Vec<DynamicImage>) -> Vec<DynamicImage> {
    let mut max_height = 0;
    let mut max_width = 0;
    for image in &images {
        let (w, h) = image.dimensions();
        if w > max_width {
            max_width = w;
        }
        if h > max_height {
            max_height = h;
        }
    }

    for image in &mut images {
        match image {
            DynamicImage::ImageRgb8(rgb8) => {
                let mut padded_image = ImageBuffer::from_pixel(max_width, max_height, Rgb([0; 3]));

                padded_image
                    .copy_from(rgb8, 0, 0)
                    .expect("Failed to copy image");

                *rgb8 = padded_image;
            }
            DynamicImage::ImageRgba8(rgba8) => {
                let mut padded_image = ImageBuffer::from_pixel(max_width, max_height, Rgba([0; 4]));

                padded_image
                    .copy_from(rgba8, 0, 0)
                    .expect("Failed to copy image");

                *rgba8 = padded_image;
            }
            _ => panic!("rgb8 or rgba8 are the only supported image types"),
        }
    }

    images
}
