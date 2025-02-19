use crate::models::chunkr::output::BoundingBox;
use image::*;
use tempfile::NamedTempFile;

pub fn get_image_dimensions(
    image: &NamedTempFile,
) -> Result<(u32, u32), Box<dyn std::error::Error + Send + Sync>> {
    let img = ImageReader::open(image.path())?
        .with_guessed_format()?
        .decode()?;
    Ok((img.width() as u32, img.height() as u32))
}

pub fn crop_image(
    image: &NamedTempFile,
    bbox: &BoundingBox,
) -> Result<NamedTempFile, Box<dyn std::error::Error>> {
    const MIN_DIMENSION: u32 = 16;

    let img = ImageReader::open(image.path())?
        .with_guessed_format()?
        .decode()?;
    let mut image = img.to_rgb8();

    // Calculate padded dimensions
    let target_width = bbox.width.max(MIN_DIMENSION as f32) as u32;
    let target_height = bbox.height.max(MIN_DIMENSION as f32) as u32;
    let padding_width = target_width.saturating_sub(bbox.width as u32);
    let padding_height = target_height.saturating_sub(bbox.height as u32);

    // Adjust position to center the box with padding
    let left = bbox.left as i32 - (padding_width / 2) as i32;
    let top = bbox.top as i32 - (padding_height / 2) as i32;

    // Ensure we don't crop outside image bounds
    let left = left.max(0) as u32;
    let top = top.max(0) as u32;
    let width = target_width.min(image.width() - left);
    let height = target_height.min(image.height() - top);

    let sub_img = imageops::crop(&mut image, left, top, width, height);
    let output_file = NamedTempFile::new()?;
    sub_img
        .to_image()
        .save_with_format(output_file.path(), ImageFormat::Jpeg)?;

    Ok(output_file)
}
