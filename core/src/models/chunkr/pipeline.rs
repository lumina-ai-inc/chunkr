use std::error::Error;
use std::sync::Arc;
use tempfile::{NamedTempFile, TempDir};

use crate::utils::services::file_operations::{check_file_type, convert_to_pdf};

#[derive(Debug, Clone)]
pub struct Pipeline {
    pub task_id: String,
    pub tempdir: Arc<TempDir>,
    pub input_file: Option<Arc<NamedTempFile>>,
    pub pdf_file: Option<Arc<NamedTempFile>>,
    pub pages: Option<Vec<Arc<NamedTempFile>>>,
}

impl Pipeline {
    pub fn new(task_id: String) -> Self {
        Self {
            task_id,
            tempdir: Arc::new(TempDir::new().unwrap()),
            input_file: None,
            pdf_file: None,
            pages: None,
        }
    }

    pub fn set_input_file(&mut self, file: NamedTempFile) -> Result<(), Box<dyn Error>> {
        // Ensure the file is in our tempdir
        let input_file = NamedTempFile::new_in(self.tempdir.path())?;
        std::fs::copy(file.path(), input_file.path())?;

        let input_file = Arc::new(input_file);
        let mime_type = check_file_type(&input_file)?;

        self.pdf_file = match mime_type.as_str() {
            "application/pdf" => Some(input_file.clone()),
            _ => Some(Arc::new(convert_to_pdf(&input_file)?)),
        };

        self.input_file = Some(input_file);
        Ok(())
    }
}
