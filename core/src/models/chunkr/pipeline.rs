use crate::models::chunkr::output::OutputResponse;
use crate::models::chunkr::task::{Status, Task, TaskPayload};
use crate::utils::services::file_operations::convert_to_pdf;
use crate::utils::storage::services::download_to_tempfile;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use std::error::Error;
use std::sync::Arc;
use tempfile::NamedTempFile;

#[derive(Debug, Clone)]
pub struct Pipeline {
    pub output: OutputResponse,
    pub input_file: Option<Arc<NamedTempFile>>,
    pub page_images: Option<Vec<Arc<NamedTempFile>>>,
    pub pdf_file: Option<Arc<NamedTempFile>>,
    pub segment_images: DashMap<String, Arc<NamedTempFile>>,
    pub task: Option<Task>,
    pub task_payload: Option<TaskPayload>,
}

impl Pipeline {
    pub fn new() -> Self {
        Self {
            output: OutputResponse::default(),
            input_file: None,
            page_images: None,
            pdf_file: None,
            segment_images: DashMap::new(),
            task: None,
            task_payload: None,
        }
    }

    pub async fn init(&mut self, task_payload: TaskPayload) -> Result<(), Box<dyn Error>> {
        let mut task = Task::get(&task_payload.task_id, &task_payload.user_id).await?;
        if task.status == Status::Cancelled {
            if task_payload.previous_status.is_some() && task_payload.previous_message.is_some() {
                task.update(
                    task_payload.previous_status,
                    task_payload.previous_message.as_deref(),
                    None,
                    None,
                    None,
                    None,
                )
                .await?;
            }
            return Ok(());
        }
        self.input_file = Some(Arc::new(
            download_to_tempfile(&task.input_location, None).await?,
        ));
        self.pdf_file = match task.mime_type.as_ref().unwrap().as_str() {
            "application/pdf" => Some(self.input_file.clone().unwrap()),
            _ => Some(Arc::new(convert_to_pdf(
                &self.input_file.as_ref().unwrap(),
            )?)),
        };
        task.update(
            Some(Status::Processing),
            Some("Task started"),
            None,
            Some(Utc::now()),
            None,
            None,
        )
        .await?;
        self.task_payload = Some(task_payload);
        self.task = Some(task);
        Ok(())
    }

    pub fn get_task(&self) -> Task {
        self.task
            .as_ref()
            .ok_or("Task is not initialized")
            .unwrap()
            .clone()
    }

    pub fn update_task(&mut self, task: Task) {
        self.task = Some(task);
    }

    pub fn get_task_payload(&self) -> TaskPayload {
        self.task_payload
            .as_ref()
            .ok_or("Task payload is not initialized")
            .unwrap()
            .clone()
    }

    pub async fn finish_and_update_task(
        &mut self,
        status: Status,
        message: Option<String>,
    ) -> Result<(), Box<dyn Error>> {
        let finished_at = Utc::now();
        let expires_at: Option<DateTime<Utc>> = self
            .get_task()
            .configuration
            .expires_in
            .map(|seconds| finished_at + chrono::Duration::seconds(seconds as i64));
        self.get_task()
            .update(
                Some(status),
                Some(&message.unwrap_or_default()),
                None,
                Some(finished_at),
                expires_at,
                None,
            )
            .await?;
        self.get_task()
            .upload_artifacts(
                self.page_images.clone().unwrap(),
                &self.segment_images,
                &self.output,
                &self.pdf_file.clone().unwrap(),
            )
            .await?;
        Ok(())
    }
}
