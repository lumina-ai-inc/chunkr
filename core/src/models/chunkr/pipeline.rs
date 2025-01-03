use crate::models::chunkr::output::Chunk;
use crate::models::chunkr::structured_extraction::ExtractedJson;
use crate::models::chunkr::task::{Status, TaskPayload};
use crate::utils::services::file_operations::{check_file_type, convert_to_pdf};
use crate::utils::services::log::log_task;
use std::error::Error;
use std::sync::Arc;
use tempfile::NamedTempFile;

#[derive(Debug, Clone)]
pub struct Pipeline {
    pub chunks: Option<Vec<Chunk>>,
    pub extracted_json: Option<ExtractedJson>,
    pub input_file: Option<Arc<NamedTempFile>>,
    pub mime_type: Option<String>,
    pub page_count: Option<u32>,
    pub pages: Option<Vec<Arc<NamedTempFile>>>,
    pub pdf_file: Option<Arc<NamedTempFile>>,
    pub status: Option<Status>,
    pub task_payload: Option<TaskPayload>,
}

impl Pipeline {
    pub fn new() -> Self {
        Self {
            chunks: None,
            extracted_json: None,
            input_file: None,
            mime_type: None,
            page_count: None,
            pages: None,
            pdf_file: None,
            status: None,
            task_payload: None,
        }
    }

    pub async fn init(
        &mut self,
        file: NamedTempFile,
        task_payload: TaskPayload,
    ) -> Result<(), Box<dyn Error>> {
        self.input_file = Some(Arc::new(file));
        self.task_payload = Some(task_payload);
        self.mime_type = Some(match check_file_type(&self.input_file.as_ref().unwrap()) {
            Ok(mime_type) => mime_type,
            Err(e) => {
                if e.to_string().contains("Unsupported file type") {
                    self.update_status(Status::Failed, Some(e.to_string()))
                        .await?;
                    return Ok(());
                }
                return Err(e.to_string().into());
            }
        });
        self.pdf_file = match self.mime_type.as_ref().unwrap().as_str() {
            "application/pdf" => Some(self.input_file.clone().unwrap()),
            _ => Some(Arc::new(convert_to_pdf(
                &self.input_file.as_ref().unwrap(),
            )?)),
        };
        self.update_status(Status::Processing, Some("Task started".to_string()))
            .await?;
        Ok(())
    }

    pub async fn update_status(
        &mut self,
        status: Status,
        message: Option<String>,
    ) -> Result<(), Box<dyn Error>> {
        log_task(
            &self.task_payload.as_ref().unwrap().task_id,
            status.clone(),
            Some(&message.unwrap_or_default()),
            None,
        )
        .await?;
        self.status = Some(status);
        Ok(())
    }
}
