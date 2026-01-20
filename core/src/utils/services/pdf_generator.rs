/// PDF Generation for Viewing Purposes Only
/// 
/// This module handles PDF conversion ONLY for viewing in the frontend.
/// OCR processing uses original files (DOCX, XLSX, images) directly with Docling.
/// 
/// This separation provides:
/// - Faster OCR (no waiting for PDF conversion)
/// - Better quality extraction (Docling sees original formatting)
/// - Viewer compatibility (users can still see PDFs)

use crate::utils::services::file_operations::{check_file_type, convert_to_pdf};
use crate::utils::storage::services::{download_to_tempfile, upload_to_s3};
use std::error::Error;

/// Generate PDF for viewing purposes (async from OCR processing)
/// 
/// This function:
/// 1. Downloads original file from S3
/// 2. Converts to PDF if needed (DOCX/XLSX/images → PDF via LibreOffice/ImageMagick)
/// 3. Uploads PDF to pdf_location for viewer
/// 4. Returns presigned URL
/// 
/// Note: This runs AFTER OCR has already processed the original file with Docling
pub async fn generate_viewer_pdf(
    input_location: &str,
    pdf_location: &str,
    mime_type: &str,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    println!("📄 Generating PDF for viewer: {}", input_location);
    
    // Download original file
    let temp_file = download_to_tempfile(input_location, None, mime_type)
        .await
        .map_err(|e| -> Box<dyn Error + Send + Sync> { format!("Download failed: {}", e).into() })?;
    
    // Check if already PDF
    let (detected_mime, _extension) = check_file_type(&temp_file, None)
        .map_err(|e| -> Box<dyn Error + Send + Sync> { format!("File type check failed: {}", e).into() })?;
    
    if detected_mime == "application/pdf" {
        println!("✅ File is already PDF, uploading to pdf_location");
        // Just upload the original file as the PDF
        upload_to_s3(pdf_location, temp_file.path())
            .await
            .map_err(|e| -> Box<dyn Error + Send + Sync> { format!("S3 upload failed: {}", e).into() })?;
    } else {
        println!("🔄 Converting {} to PDF for viewer", detected_mime);
        // Convert to PDF for viewing
        let pdf_file = convert_to_pdf(&temp_file, None)
            .map_err(|e| -> Box<dyn Error + Send + Sync> { format!("PDF conversion failed: {}", e).into() })?;
        upload_to_s3(pdf_location, pdf_file.path())
            .await
            .map_err(|e| -> Box<dyn Error + Send + Sync> { format!("S3 upload failed: {}", e).into() })?;
        println!("✅ PDF generated and uploaded for viewing");
    }
    
    // Generate presigned URL (with correct parameters)
    // Signature: (location, external, expires_in, base64_urls, mime_type)
    let pdf_url = crate::utils::storage::services::generate_presigned_url(pdf_location, true, None, false, "application/pdf")
        .await
        .map_err(|e| -> Box<dyn Error + Send + Sync> { format!("Presigned URL failed: {}", e).into() })?;
    
    Ok(pdf_url)
}

/// Check if PDF needs to be generated for a given file
pub fn needs_pdf_conversion(mime_type: &str) -> bool {
    mime_type != "application/pdf"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_needs_pdf_conversion() {
        assert!(!needs_pdf_conversion("application/pdf"));
        assert!(needs_pdf_conversion("application/vnd.openxmlformats-officedocument.wordprocessingml.document"));
        assert!(needs_pdf_conversion("image/png"));
        assert!(needs_pdf_conversion("application/vnd.ms-excel"));
    }
}
