use std::ops::Range;

use candle_core::Result;

pub(super) const POSITION_PLANES: usize = 4;

pub(super) struct ImageChunkLayout {
    pub spans: Vec<(usize, usize)>,
    pub images: Range<usize>,
    pub positions: [Vec<i64>; POSITION_PLANES],
}

impl ImageChunkLayout {
    pub fn new(
        spans: &[(usize, usize)],
        grids: &[Vec<u32>],
        query: Range<usize>,
        merge_size: usize,
    ) -> Result<Self> {
        let positions = query.clone().map(|pos| pos as i64).collect::<Vec<_>>();
        let mut layout = Self {
            spans: Vec::new(),
            images: 0..0,
            positions: std::array::from_fn(|_| positions.clone()),
        };
        for (image, &(start, end)) in spans.iter().enumerate() {
            if end <= query.start || start >= query.end {
                continue;
            }
            if start < query.start || end > query.end {
                candle_core::bail!("HunyuanVL prompt chunk splits an image");
            }
            let Some(grid) = grids.get(image).filter(|grid| grid.len() == 3) else {
                candle_core::bail!("HunyuanVL image span is missing its grid");
            };
            let height = grid[1] as usize / merge_size;
            let width = grid[2] as usize / merge_size;
            if end - start != height * (width + 1) + 2 {
                candle_core::bail!("HunyuanVL image span does not match its grid");
            }
            if layout.spans.is_empty() {
                layout.images.start = image;
            }
            layout.images.end = image + 1;
            layout.spans.push((start - query.start, end - query.start));
            // Planes are (absolute position, width, height, image ordinal).
            let mut position = start + 1 - query.start;
            for row in 0..height {
                for col in 0..=width {
                    layout.positions[1][position] = col as i64;
                    layout.positions[2][position] = row as i64;
                    layout.positions[3][position] = image as i64;
                    position += 1;
                }
            }
        }
        Ok(layout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn later_image_chunk_matches_full_prompt_positions() -> Result<()> {
        let spans = [(2, 10), (16, 21)];
        let grids = [vec![1, 4, 4], vec![1, 2, 4]];
        let full = ImageChunkLayout::new(&spans, &grids, 0..25, 2)?;
        let chunk = ImageChunkLayout::new(&spans, &grids, 14..23, 2)?;
        assert_eq!(chunk.images, 1..2);
        assert_eq!(chunk.spans, vec![(2, 7)]);
        for plane in 0..POSITION_PLANES {
            assert_eq!(chunk.positions[plane], full.positions[plane][14..23]);
        }
        assert_eq!(chunk.positions[1], vec![14, 15, 16, 0, 1, 2, 20, 21, 22]);
        Ok(())
    }

    #[test]
    fn text_chunk_after_images_keeps_absolute_positions() -> Result<()> {
        let chunk = ImageChunkLayout::new(&[(2, 10)], &[vec![1, 4, 4]], 10..13, 2)?;
        assert!(chunk.spans.is_empty());
        assert!(chunk.images.is_empty());
        assert_eq!(chunk.positions, std::array::from_fn(|_| vec![10, 11, 12]));
        Ok(())
    }

    #[test]
    fn image_ordinal_is_the_image_position_in_the_sequence() -> Result<()> {
        let spans = [(2, 10), (16, 21)];
        let grids = [vec![1, 4, 4], vec![1, 2, 4]];
        let full = ImageChunkLayout::new(&spans, &grids, 0..25, 2)?;
        // The image_start and image_end tokens keep their 1D positions.
        assert_eq!(full.positions[3][2], 2);
        assert_eq!(&full.positions[3][3..9], &[0i64; 6]);
        assert_eq!(full.positions[3][9], 9);
        assert_eq!(&full.positions[3][17..20], &[1i64; 3]);

        let chunk = ImageChunkLayout::new(&spans, &grids, 14..23, 2)?;
        assert_eq!(chunk.positions[3], vec![14, 15, 16, 1, 1, 1, 20, 21, 22]);
        Ok(())
    }

    #[test]
    fn later_chunk_keeps_the_global_image_ordinal() -> Result<()> {
        let spans = [(2, 10), (16, 21), (30, 38)];
        let grids = [vec![1, 4, 4], vec![1, 2, 4], vec![1, 4, 4]];
        let chunk = ImageChunkLayout::new(&spans, &grids, 24..40, 2)?;
        assert_eq!(chunk.images, 2..3);
        assert_eq!(chunk.spans, vec![(6, 14)]);
        assert_eq!(&chunk.positions[3][7..13], &[2i64; 6]);
        Ok(())
    }

    #[test]
    fn image_chunks_require_complete_media_items() {
        assert!(ImageChunkLayout::new(&[(2, 10)], &[vec![1, 4, 4]], 5..12, 2).is_err());
    }
}
