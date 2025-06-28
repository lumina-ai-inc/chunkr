use std::sync::Arc;
use tempfile::NamedTempFile;

fn generate_uuid() -> String {
    uuid::Uuid::new_v4().to_string()
}

#[derive(Debug, Clone)]
pub struct ImageConversionResult {
    pub image_id: String,
    pub image_file: Arc<NamedTempFile>,
    pub html_reference: String,
    pub range: Option<String>,
}

impl ImageConversionResult {
    pub fn new(image_file: Arc<NamedTempFile>, html_reference: String) -> Self {
        Self {
            image_id: generate_uuid(),
            image_file,
            html_reference,
            range: None,
        }
    }

    pub fn set_range(&mut self, range: String) {
        self.range = Some(range);
    }
}

#[derive(Debug, Clone)]
pub struct HtmlConversionResult {
    pub html_id: String,
    pub html_file: Arc<NamedTempFile>,
    pub embedded_images: Vec<ImageConversionResult>,
}

impl HtmlConversionResult {
    pub fn new(html_file: Arc<NamedTempFile>, embedded_images: Vec<ImageConversionResult>) -> Self {
        Self {
            html_id: generate_uuid(),
            html_file,
            embedded_images,
        }
    }
}
