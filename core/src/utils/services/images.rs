use crate::models::chunkr::output::BoundingBox;
use image::*;
use tempfile::NamedTempFile;

pub fn get_image_dimensions(
    image: &NamedTempFile,
) -> Result<(u32, u32), Box<dyn std::error::Error>> {
    let img = ImageReader::open(image.path())?
        .with_guessed_format()?
        .decode()?;
    Ok((img.width() as u32, img.height() as u32))
}

pub fn crop_image(
    image: &NamedTempFile,
    bbox: &BoundingBox,
) -> Result<NamedTempFile, Box<dyn std::error::Error>> {
    let img = ImageReader::open(image.path())?
        .with_guessed_format()?
        .decode()?;
    let mut image = img.to_rgb8();
    let sub_img = imageops::crop(
        &mut image,
        bbox.left as u32,
        bbox.top as u32,
        bbox.width as u32,
        bbox.height as u32,
    );
    let output_file = NamedTempFile::new()?;
    sub_img
        .to_image()
        .save_with_format(output_file.path(), ImageFormat::Jpeg)?;

    Ok(output_file)
}
