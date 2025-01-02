use std::error::Error;
use std::sync::Arc;
use tempfile::NamedTempFile;

use crate::models::chunkr::output::OutputResponse;
use crate::models::chunkr::task::TaskPayload;
use crate::utils::services::file_operations::{check_file_type, convert_to_pdf};

#[derive(Debug, Clone)]
pub struct Pipeline {
    pub input_file: Arc<NamedTempFile>,
    pub mime_type: String,
    pub output: Option<Vec<OutputResponse>>,
    pub page_count: Option<u32>,
    pub pages: Option<Vec<Arc<NamedTempFile>>>,
    pub pdf_file: Arc<NamedTempFile>,
    pub task_payload: TaskPayload,
}

impl Pipeline {
    pub fn new(file: NamedTempFile, task_payload: TaskPayload) -> Result<Self, Box<dyn Error>> {
        let input_file = Arc::new(file);
        let mime_type = check_file_type(&input_file)?;
        let pdf_file = match mime_type.as_str() {
            "application/pdf" => input_file.clone(),
            _ => Arc::new(convert_to_pdf(&input_file)?),
        };

        Ok(Self {
            input_file,
            mime_type,
            output: None,
            page_count: None,
            pages: None,
            pdf_file,
            task_payload,
        })
    }
}
