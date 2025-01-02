use std::error::Error;
use std::sync::Arc;
use tempfile::NamedTempFile;

use crate::models::chunkr::task::Configuration;
use crate::models::chunkr::output::OutputResponse;
use crate::utils::services::file_operations::{check_file_type, convert_to_pdf};

#[derive(Debug, Clone)]
pub struct Pipeline {
    pub current_configuration: Configuration,
    pub input_file: Arc<NamedTempFile>,
    pub mime_type: String,
    pub output: Option<Vec<OutputResponse>>,
    pub page_count: Option<u32>,
    pub pages: Option<Vec<Arc<NamedTempFile>>>,
    pub pdf_file: Arc<NamedTempFile>,
    pub previous_configuration: Option<Configuration>,
    pub task_id: String,
}

impl Pipeline {
    pub fn new(task_id: String, file: NamedTempFile, current_configuration: Configuration, previous_configuration: Option<Configuration>) -> Result<Self, Box<dyn Error>> {
        let input_file = Arc::new(file);
        let mime_type = check_file_type(&input_file)?;
        let pdf_file = match mime_type.as_str() {
            "application/pdf" => input_file.clone(),
            _ => Arc::new(convert_to_pdf(&input_file)?),
        };

        Ok(Self {
            current_configuration,
            input_file,
            mime_type,
            output: None,
            page_count: None,
            pages: None,
            pdf_file,
            previous_configuration,
            task_id,
        })
    }
}
