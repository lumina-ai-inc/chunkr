use crate::configs::worker_config;
use crate::models::output::Chunk;
use crate::models::task::{Status, Task, TaskPayload, TimeoutError};
use crate::utils::services::file_operations::convert_to_pdf;
use crate::utils::services::pdf::count_pages;
use crate::utils::storage::services::download_to_tempfile;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use std::error::Error;
use std::sync::Arc;
use strum_macros::{Display, EnumString};
use tempfile::NamedTempFile;

#[cfg(feature = "memory_profiling")]
use memtrack::track_mem;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Display, EnumString)]
pub enum PipelineStep {
    #[cfg(feature = "azure")]
    #[strum(serialize = "azure_analysis")]
    AzureAnalysis,
    #[strum(serialize = "chunking")]
    Chunking,
    #[strum(serialize = "chunkr_analysis")]
    ChunkrAnalysis,
    #[strum(serialize = "convert_to_images")]
    ConvertToImages,
    #[strum(serialize = "crop")]
    Crop,
    #[strum(serialize = "segment_processing")]
    SegmentProcessing,
}

pub trait PipelineStepMessages {
    fn start_message(&self) -> String;
    fn error_message(&self) -> String;
}

impl PipelineStepMessages for PipelineStep {
    fn start_message(&self) -> String {
        match self {
            #[cfg(feature = "azure")]
            PipelineStep::AzureAnalysis => "Running Azure analysis".to_string(),
            PipelineStep::Chunking => "Chunking".to_string(),
            PipelineStep::ChunkrAnalysis => "Running Chunkr analysis".to_string(),
            PipelineStep::ConvertToImages => "Converting pages to images".to_string(),
            PipelineStep::Crop => "Cropping segments".to_string(),
            PipelineStep::SegmentProcessing => "Processing segments".to_string(),
        }
    }

    fn error_message(&self) -> String {
        match self {
            #[cfg(feature = "azure")]
            PipelineStep::AzureAnalysis => "Failed to run Azure analysis".to_string(),
            PipelineStep::Chunking => "Failed to chunk".to_string(),
            PipelineStep::ChunkrAnalysis => "Failed to run Chunkr analysis".to_string(),
            PipelineStep::ConvertToImages => "Failed to convert pages to images".to_string(),
            PipelineStep::Crop => "Failed to crop segments".to_string(),
            PipelineStep::SegmentProcessing => {
                "Failed to process segments - LLM processing error".to_string()
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Pipeline {
    pub input_file: Option<Arc<NamedTempFile>>,
    pub chunks: Vec<Chunk>,
    pub page_images: Option<Vec<Arc<NamedTempFile>>>,
    pub pdf_file: Option<Arc<NamedTempFile>>,
    pub segment_images: DashMap<String, Arc<NamedTempFile>>,
    pub task: Option<Task>,
    pub task_payload: Option<TaskPayload>,
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl Pipeline {
    pub fn new() -> Self {
        Self {
            input_file: None,
            chunks: Vec::new(),
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
            self.page_images = Some(page_images.into_iter().map(Arc::new).collect());
            self.segment_images = segment_images
                .into_iter()
                .map(|(k, v)| (k, Arc::new(v)))
                .collect();
            self.chunks = output.chunks;
            println!("Task initialized with artifacts");
        } else {
            self.input_file = Some(Arc::new(
                download_to_tempfile(&task.input_location, None, task.mime_type.as_ref().unwrap())
                    .await?,
            ));
            self.pdf_file = match task.mime_type.as_ref().unwrap().as_str() {
                "application/pdf" => Some(self.input_file.clone().unwrap()),
                _ => Some(Arc::new(convert_to_pdf(
                    self.input_file.as_ref().unwrap(),
                    None,
                )?)),
            };
            println!("Task initialized with input file");
        }
        let page_count = count_pages(self.pdf_file.as_ref().unwrap())?;
        task.update(
            Some(Status::Processing),
            Some("Task started".to_string()),
            None,
            Some(page_count),
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
            .cloned()
    }

    pub fn get_task_payload(&self) -> Result<TaskPayload, Box<dyn Error>> {
        self.task_payload
            .as_ref()
            .ok_or_else(|| "Task payload is not initialized".into())
            .cloned()
    }

    pub fn get_mime_type(&self) -> Result<String, Box<dyn Error>> {
        Ok(self.get_task()?.mime_type.as_ref().unwrap().clone())
    }

    pub fn get_scaling_factor(&self) -> Result<f32, Box<dyn Error>> {
        if self.get_mime_type()?.starts_with("image/") {
            Ok(1.0)
        } else {
            let worker_config = worker_config::Config::from_env()?;
            if self.get_task()?.configuration.high_resolution {
                Ok(worker_config.high_res_scaling_factor)
            } else {
                Ok(1.0)
            }
        }
    }

    pub fn get_file(&self) -> Result<Arc<NamedTempFile>, Box<dyn Error>> {
        if self.get_mime_type()?.starts_with("image/") {
            Ok(self.input_file.as_ref().unwrap().clone())
        } else {
            Ok(self.pdf_file.as_ref().unwrap().clone())
        }
    }

    /// Execute a step in the pipeline
    #[cfg_attr(feature = "memory_profiling", track_mem)]
    pub async fn execute_step(
        &mut self,
        step: PipelineStep,
        max_retries: u32,
    ) -> Result<(), Box<dyn Error>> {
        let start = std::time::Instant::now();

        let mut task = self.get_task()?;

        let mut retries = 0;
        while retries < max_retries {
            // Update task status to processing and message to step start message
            let message = match retries > 0 {
                true => format!(
                    "{} | retry {}/{}",
                    step.start_message(),
                    retries + 1,
                    max_retries
                ),
                false => step.start_message(),
            };
            println!("Executing step: {}", message);
            match task
                .update(
                    Some(Status::Processing),
                    Some(message),
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .await
            {
                Ok(_) => (),
                Err(e) => {
                    if e.is::<TimeoutError>() {
                        println!("Task timed out");
                    } else {
                        println!("Error in updating task: {:?}", e);
                    }
                    return Err(e);
                }
            };

            // Execute step
            let result = match step {
                #[cfg(feature = "azure")]
                PipelineStep::AzureAnalysis => crate::pipeline::azure_analysis::process(self).await,
                PipelineStep::Chunking => crate::pipeline::chunking::process(self).await,
                PipelineStep::ConvertToImages => {
                    crate::pipeline::convert_to_images::process(self).await
                }
                PipelineStep::Crop => crate::pipeline::crop::process(self).await,
                PipelineStep::ChunkrAnalysis => {
                    crate::pipeline::chunkr_analysis::process(self).await
                }
                PipelineStep::SegmentProcessing => {
                    crate::pipeline::segment_processing::process(self).await
                }
            };

            let duration = start.elapsed();

            // Check if step succeeded or failed
            match result {
                Ok(_) => {
                    println!(
                        "Step {} took {:?} with page count {:?}",
                        step,
                        duration,
                        self.get_task()?.page_count.unwrap_or(0)
                    );
                    return Ok(());
                }
                Err(e) => {
                    println!("Error {} in step {}", e, step);
                    retries += 1;
                    if retries < max_retries {
                        task.update(
                            Some(Status::Processing),
                            Some(step.error_message()),
                            None,
                            None,
                            None,
                            None,
                            None,
                        )
                        .await?;
                        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                    }
                }
            }
        }
        // If step failed after max_retries, complete task with failed status and error message
        self.complete(Status::Failed, Some(step.error_message()))
            .await?;
        Err("Maximum retries exceeded".into())
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
            chunks: Vec<Chunk>,
            pdf_file: Arc<NamedTempFile>,
            finished_at: DateTime<Utc>,
            expires_at: Option<DateTime<Utc>>,
        ) -> Result<(), Box<dyn Error>> {
            task.upload_artifacts(page_images, segment_images, chunks, &pdf_file)
                .await?;
            task.update(
                Some(status),
                message,
                None,
                None,
                None,
                Some(finished_at),
                expires_at,
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
                    None,
                    Some(finished_at),
                    expires_at,
                )
                .await?;
                Ok(())
            } else {
                revert_to_previous(&mut task, &task_payload).await
            }
        } else {
            match update_success(
                &mut task,
                status,
                message,
                self.page_images.clone().unwrap(),
                &self.segment_images,
                self.chunks.clone(),
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
