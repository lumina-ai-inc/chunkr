use std::error::Error;
use std::sync::Arc;
use tempfile::{NamedTempFile, TempDir};

use crate::models::chunkr::task::Configuration;
use crate::utils::services::file_operations::{check_file_type, convert_to_pdf};

#[derive(Debug, Clone)]
pub struct Pipeline {
    pub current_configuration: Configuration,
    pub input_file: Arc<NamedTempFile>,
    pub mime_type: String,
    pub page_count: Option<u32>,
    pub pages: Option<Vec<Arc<NamedTempFile>>>,
    pub pdf_file: Arc<NamedTempFile>,
    pub previous_configuration: Option<Configuration>,
    pub task_id: String,
    pub tempdir: Arc<TempDir>,
}

impl Pipeline {
    pub fn new(task_id: String, file: NamedTempFile, current_configuration: Configuration, previous_configuration: Option<Configuration>) -> Result<Self, Box<dyn Error>> {
        let tempdir = Arc::new(TempDir::new().unwrap());
        let input_file = NamedTempFile::new_in(tempdir.path())?;
        std::fs::copy(file.path(), input_file.path())?;

        let input_file = Arc::new(input_file);
        let mime_type = check_file_type(&input_file)?;
        let pdf_file = match mime_type.as_str() {
            "application/pdf" => input_file.clone(),
            _ => Arc::new(convert_to_pdf(&input_file)?),
        };

        Ok(Self {
            task_id,
            tempdir,
            input_file,
            pdf_file,
            pages: None,
            page_count: None,
            mime_type,
            current_configuration,
            previous_configuration,
        })
    }
}
