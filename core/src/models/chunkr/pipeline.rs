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
    pub input_file: Option<Arc<NamedTempFile>>,
    pub output: OutputResponse,
    pub page_images: Option<Vec<Arc<NamedTempFile>>>,
    pub pdf_file: Option<Arc<NamedTempFile>>,
    pub segment_images: DashMap<String, Arc<NamedTempFile>>,
    pub task: Option<Task>,
    pub task_payload: Option<TaskPayload>,
}

impl Pipeline {
    pub fn new() -> Self {
        Self {
            input_file: None,
            output: OutputResponse::default(),
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
            if task_payload.previous_configuration.is_some() {
                task.update(
                    task_payload.previous_status,
                    task_payload.previous_message,
                    task_payload.previous_configuration,
                    None,
                    None,
                    None,
                )
                .await?;
            }
            return Ok(());
        }
        if task_payload.previous_configuration.is_some() {
            let (input_file, pdf_file, page_images, segment_images, output) =
                task.get_artifacts().await?;
            self.input_file = Some(Arc::new(input_file));
            self.pdf_file = Some(Arc::new(pdf_file));
            self.page_images = Some(page_images.into_iter().map(|file| Arc::new(file)).collect());
            self.segment_images = segment_images
                .into_iter()
                .map(|(k, v)| (k, Arc::new(v)))
                .collect();
            self.output = output;
            println!("Task initialized with artifacts");
        } else {
            self.input_file = Some(Arc::new(
                download_to_tempfile(&task.input_location, None).await?,
            ));
            self.pdf_file = match task.mime_type.as_ref().unwrap().as_str() {
                "application/pdf" => Some(self.input_file.clone().unwrap()),
                _ => Some(Arc::new(convert_to_pdf(
                    &self.input_file.as_ref().unwrap(),
                )?)),
            };
            println!("Task initialized with input file");
        }
        task.update(
            Some(Status::Processing),
            Some("Task started".to_string()),
            None,
            Some(Utc::now()),
            None,
            None,
        )
        .await?;
        self.task_payload = Some(task_payload.clone());
        self.task = Some(task.clone());
        Ok(())
    }

    pub fn get_task(&self) -> Result<Task, Box<dyn Error>> {
        self.task
            .as_ref()
            .ok_or_else(|| "Task is not initialized".into())
            .map(|task| task.clone())
    }

    pub fn get_task_payload(&self) -> Result<TaskPayload, Box<dyn Error>> {
        self.task_payload
            .as_ref()
            .ok_or_else(|| "Task payload is not initialized".into())
            .map(|payload| payload.clone())
    }

    pub async fn complete(
        &mut self,
        status: Status,
        message: Option<String>,
    ) -> Result<(), Box<dyn Error>> {
        let mut task = self.get_task()?;
        let task_payload = self.get_task_payload()?;
        let finished_at = Utc::now();
        let expires_at = task
            .configuration
            .expires_in
            .map(|seconds| finished_at + chrono::Duration::seconds(seconds as i64));

        async fn revert_to_previous(
            task: &mut Task,
            payload: &TaskPayload,
        ) -> Result<(), Box<dyn Error>> {
            if payload.previous_configuration.is_some() {
                task.update(
                    payload.previous_status.clone(),
                    payload.previous_message.clone(),
                    payload.previous_configuration.clone(),
                    None,
                    None,
                    None,
                )
                .await?;
            }
            Ok(())
        }

        async fn update_success(
            task: &mut Task,
            status: Status,
            message: Option<String>,
            page_images: Vec<Arc<NamedTempFile>>,
            segment_images: &DashMap<String, Arc<NamedTempFile>>,
            output: &OutputResponse,
            pdf_file: Arc<NamedTempFile>,
            finished_at: DateTime<Utc>,
            expires_at: Option<DateTime<Utc>>,
        ) -> Result<(), Box<dyn Error>> {
            task.upload_artifacts(page_images, segment_images, output, &pdf_file)
                .await?;
            task.update(
                Some(status),
                message,
                None,
                Some(finished_at),
                expires_at,
                None,
            )
            .await?;
            Ok(())
        }

        if status == Status::Failed {
            if task_payload.previous_configuration.is_none() {
                task.update(
                    Some(status),
                    message,
                    None,
                    None,
                    Some(finished_at),
                    expires_at,
                )
                .await?;
                return Ok(());
            } else {
                return revert_to_previous(&mut task, &task_payload).await;
            }
        } else {
            match update_success(
                &mut task,
                status,
                message,
                self.page_images.clone().unwrap(),
                &self.segment_images,
                &self.output,
                self.pdf_file.clone().unwrap(),
                finished_at,
                expires_at,
            )
            .await
            {
                Ok(_) => Ok(()),
                Err(e) => {
                    println!("Error in completing task: {:?}", e);
                    revert_to_previous(&mut task, &task_payload).await?;
                    Err(e)
                }
            }
        }
    }
}
