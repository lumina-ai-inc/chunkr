use crate::models::chunkr::output::OutputResponse;
use crate::models::chunkr::task::{Status, TaskPayload};
use crate::utils::services::file_operations::{check_file_type, convert_to_pdf};
use crate::utils::services::task::{get_status, update_status};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use std::error::Error;
use std::sync::Arc;
use tempfile::NamedTempFile;

#[derive(Debug, Clone)]
pub struct Pipeline {
    pub output: OutputResponse,
    pub input_file: Option<Arc<NamedTempFile>>,
    pub mime_type: Option<String>,
    pub page_count: Option<u32>,
    pub page_images: Option<Vec<Arc<NamedTempFile>>>,
    pub pdf_file: Option<Arc<NamedTempFile>>,
    pub segment_images: DashMap<String, Arc<NamedTempFile>>,
    pub status: Option<Status>,
    pub task_payload: Option<TaskPayload>,
}

impl Pipeline {
    pub fn new() -> Self {
        Self {
            output: OutputResponse::default(),
            input_file: None,
            mime_type: None,
            page_count: None,
            page_images: None,
            pdf_file: None,
            segment_images: DashMap::new(),
            status: None,
            task_payload: None,
        }
    }

    pub async fn init(
        &mut self,
        file: NamedTempFile,
        task_payload: TaskPayload,
    ) -> Result<(), Box<dyn Error>> {
        self.task_payload = Some(task_payload);
        if self.get_remote_status().await? == Status::Cancelled {
            return Ok(());
        }
        self.input_file = Some(Arc::new(file));
        self.mime_type = Some(match check_file_type(&self.input_file.as_ref().unwrap()) {
            Ok(mime_type) => mime_type,
            Err(e) => {
                if e.to_string().contains("Unsupported file type") {
                    self.update_remote_status(Status::Failed, Some(e.to_string()))
                        .await?;
                    return Ok(());
                }
                println!("Error checking file type: {:?}", e);
                return Err(e.to_string().into());
            }
        });
        self.pdf_file = match self.mime_type.as_ref().unwrap().as_str() {
            "application/pdf" => Some(self.input_file.clone().unwrap()),
            _ => Some(Arc::new(convert_to_pdf(
                &self.input_file.as_ref().unwrap(),
            )?)),
        };
        update_status(
            &self.get_task_id(),
            &self.get_user_id(),
            Status::Processing,
            Some("Task started"),
            Some(Utc::now()),
            None,
            None,
            Some(self.mime_type.as_ref().unwrap()),
        )
        .await?;
        self.status = Some(Status::Processing);
        Ok(())
    }

    fn get_task_id(&self) -> String {
        self.task_payload
            .as_ref()
            .ok_or("Task payload is not initialized")
            .unwrap()
            .task_id
            .clone()
    }

    fn get_user_id(&self) -> String {
        self.task_payload
            .as_ref()
            .ok_or("Task payload is not initialized")
            .unwrap()
            .user_id
            .clone()
    }

    async fn get_remote_status(&mut self) -> Result<Status, Box<dyn std::error::Error>> {
        let status = get_status(&self.get_task_id(), &self.get_user_id()).await?;
        self.status = Some(status.clone());
        Ok(status)
    }

    pub async fn update_remote_status(
        &mut self,
        status: Status,
        message: Option<String>,
    ) -> Result<(), Box<dyn Error>> {
        update_status(
            &self.get_task_id(),
            &self.get_user_id(),
            status.clone(),
            Some(&message.unwrap_or_default()),
            None,
            None,
            None,
            None,
        )
        .await?;
        self.status = Some(status);
        Ok(())
    }

    pub async fn finish_and_update_remote_status(
        &mut self,
        status: Status,
        message: Option<String>,
    ) -> Result<(), Box<dyn Error>> {
        let finished_at = Utc::now();
        let expires_at: Option<DateTime<Utc>> = self
            .task_payload
            .as_ref()
            .unwrap()
            .current_configuration
            .expires_in
            .map(|seconds| finished_at + chrono::Duration::seconds(seconds as i64));
        update_status(
            &self.get_task_id(),
            &self.get_user_id(),
            status.clone(),
            Some(&message.unwrap_or_default()),
            None,
            Some(finished_at),
            expires_at,
            None,
        )
        .await?;
        self.status = Some(status);
        Ok(())
    }
}
