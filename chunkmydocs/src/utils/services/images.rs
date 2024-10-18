use crate::models::server::segment::Segment;
use image::*;
use imageproc::{
    contrast::adaptive_threshold,
    drawing::draw_hollow_rect_mut,
    filter::median_filter,
    rect::Rect,
};
use std::{ error::Error, fs::{ self, File }, io::BufWriter, path::{ Path, PathBuf } };
use tempfile::NamedTempFile;
use crate::models::workers::table_ocr::TableStructure;

pub fn annotate_image(
    input_image_path: &Path,
    table_structures: &[TableStructure],
    output_folder: &Path
) -> Result<(), Box<dyn Error>> {
    let mut img = image::open(input_image_path)?.to_rgba8();

    let color = Rgba([255, 0, 0, 255]);

    for table in table_structures {
        for cell in &table.cells {
            let bbox = &cell.cell;
            let rect = Rect::at(bbox.left as i32, bbox.top as i32).of_size(
                bbox.width as u32,
                bbox.height as u32
            );

            draw_hollow_rect_mut(&mut img, rect, color);
        }
    }

    fs::create_dir_all(output_folder)?;

    let input_file_name = input_image_path.file_name().ok_or("Invalid input file name")?;
    let output_image_path = output_folder.join(input_file_name);

    let rgb_img = DynamicImage::ImageRgba8(img).into_rgb8();

    rgb_img.save_with_format(&output_image_path, image::ImageFormat::Jpeg)?;

    Ok(())
}

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

pub async fn preprocess_image(
    input_path: &std::path::Path
) -> Result<NamedTempFile, Box<dyn std::error::Error>> {
    let img = ImageReader::open(input_path)?.with_guessed_format()?.decode()?;
    let gray_img = img.to_luma8();
    let enhanced_img = imageops::contrast(&gray_img, 1.1);
    let denoised_img = median_filter(&enhanced_img, 3, 3);

    let binary_img = adaptive_threshold(
        &ImageBuffer::from_raw(
            denoised_img.width(),
            denoised_img.height(),
            denoised_img.clone().into_raw()
        ).unwrap(),
        61
    );

    let temp_file = NamedTempFile::new()?;
    let file = File::create(temp_file.path())?;
    let mut w = BufWriter::new(file);
    binary_img.write_to(&mut w, image::ImageFormat::Png)?;
    Ok(temp_file)
}
