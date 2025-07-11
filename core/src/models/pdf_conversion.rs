use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
/// Request body for PDF conversion
pub struct PdfConversionRequest {
    /// URL or base64 encoded PDF file to convert
    pub file: String,
    /// Scaling factor for the images
    #[serde(default = "default_scaling_factor")]
    pub scaling_factor: f32,
    /// Whether to return base64 encoded images or presigned URLs
    pub base64_urls: bool,
}

fn default_scaling_factor() -> f32 {
    2.0
}

#[derive(Debug, Serialize, Deserialize)]
/// Response body for PDF conversion
pub struct PdfConversionResponse {
    /// List of presigned URLs or base64 encoded images for the converted images
    pub images: Vec<String>,
}
