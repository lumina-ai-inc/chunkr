use crate::models::server::segment::Segment;
use image::*;
use std::path::PathBuf;

pub fn crop_image(
    image_path: &PathBuf,
    segment: &Segment,
    output_dir: &PathBuf
) -> Result<(), Box<dyn std::error::Error>> {
    let img = ImageReader::open(image_path)?.with_guessed_format()?.decode()?;
    let mut image = img.to_rgb8();
    let sub_img = imageops::crop(
        &mut image,
        segment.bbox.left as u32,
        segment.bbox.top as u32,
        segment.bbox.width as u32,
        segment.bbox.height as u32
    );
    let output_path = output_dir.join(format!("{}.jpg", segment.segment_id));
    sub_img.to_image().save_with_format(output_path, ImageFormat::Jpeg).unwrap();
    Ok(())
}
