use crate::configs::worker_config;
use crate::models::chunkr::chunk_processing::ChunkProcessing;
use crate::models::chunkr::output::{OutputResponse, Segment, SegmentType};
use crate::models::chunkr::segment_processing::SegmentProcessing;
use crate::models::chunkr::structured_extraction::JsonSchema;
use crate::models::chunkr::upload::{OcrStrategy, SegmentationStrategy};
use crate::utils::clients::get_pg_client;
use crate::utils::services::file_operations::check_file_type;
use crate::utils::storage::services::delete_folder;
use crate::utils::storage::services::{download_to_tempfile, generate_presigned_url, upload_to_s3};
use actix_multipart::form::tempfile::TempFile;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use futures::future::try_join_all;
use postgres_types::{FromSql, ToSql};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use strum_macros::{Display, EnumString};
use tempfile::NamedTempFile;
use utoipa::ToSchema;
use uuid::Uuid;

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
pub struct Task {
    pub api_key: Option<String>,
    pub configuration: Configuration,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub file_name: Option<String>,
    pub file_size: i64,
    pub finished_at: Option<DateTime<Utc>>,
    pub image_folder_location: String,
    pub input_location: String,
    pub message: Option<String>,
    pub mime_type: Option<String>,
    pub output_location: String,
    pub page_count: Option<u32>,
    pub pdf_location: String,
    pub status: Status,
    pub started_at: Option<DateTime<Utc>>,
    pub task_id: String,
    pub task_url: Option<String>,
    pub user_id: String,
    pub version: Option<String>,
}

impl Task {
    pub async fn new(
        user_id: &str,
        api_key: Option<String>,
        configuration: &Configuration,
        file: &TempFile,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let temp_file = &file.file;
        let mime_type = check_file_type(temp_file)?;

        let client = get_pg_client().await?;
        let worker_config = worker_config::Config::from_env().unwrap();

        let task_id = Uuid::new_v4().to_string();
        let file_name = file.file_name.as_deref().unwrap_or("unknown.pdf");
        let file_size = file.size;
        let status = Status::Starting;
        let base_url  = worker_config.server_url;
        let task_url = format!("{}/api/v1/task/{}", base_url, task_id);
        let (input_location, pdf_location, output_location, image_folder_location) =
        Self::generate_s3_paths(user_id, &task_id, file_name);
        let message = "Task queued".to_string();
        let created_at = Utc::now();
        let version = worker_config.version;

        let file_path  = PathBuf::from(file.file.path());
        upload_to_s3(&input_location, &file_path).await?;
        
        let configuration_json = serde_json::to_string(&configuration)?;
        client
            .execute(
                "INSERT INTO TASKS (
                    api_key,
                    configuration,
                    created_at,
                    file_name,
                    file_size,
                    image_folder_location,
                    input_location,
                    message,
                    mime_type,
                    output_location,
                    page_count,
                    pdf_location,
                    status,
                    task_id,
                    task_url,
                    user_id,
                    version,
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17
                ) ON CONFLICT (task_id) DO NOTHING",
                &[
                    &api_key,
                    &configuration_json,
                    &created_at,
                    &file_name,
                    &(file_size as i64),
                    &image_folder_location,
                    &input_location,
                    &message,
                    &mime_type,
                    &output_location,
                    &(0 as i32),
                    &pdf_location,
                    &status.to_string(),
                    &task_id,
                    &task_url,
                    &user_id,
                    &version,
                ],
            )
            .await?;

        Ok(Self {
            api_key,
            configuration: configuration.clone(),
            created_at,
            expires_at: None,
            file_name: Some(file_name.to_string()),
            file_size: file_size as i64,
            finished_at: None,
            image_folder_location,
            input_location,
            message: Some(message),
            mime_type: Some(mime_type),
            output_location,
            page_count: Some(0),
            pdf_location,
            status,
            started_at: None,
            task_id: task_id.clone(),
            task_url: Some(task_url),
            user_id: user_id.to_string(),
            version: Some(version),  
        })
    }

    pub async fn get(
        task_id: &str,
        user_id: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let client = get_pg_client().await?;
        let row = client
            .query_one(
                "SELECT 
                    api_key,
                    configuration,
                    created_at,
                    expires_at,
                    file_name,
                    file_size,
                    finished_at,
                    image_folder_location,
                    input_location,
                    message,
                    mime_type,
                    output_location,
                    page_count,
                    pdf_location,
                    status,
                    started_at,
                    task_id,
                    task_url,
                    user_id,
                    version
                FROM tasks 
                WHERE task_id = $1 
                AND user_id = $2",
                &[&task_id, &user_id],
            )
            .await?;

        let file_name = row.get("file_name");
        let (input_location, pdf_location, output_location, image_folder_location) =
            Self::generate_s3_paths(user_id, task_id, file_name);
        let page_count: Option<i32> = row.get("page_count");
        let page_count = page_count.map(|count| count as u32);
        Ok(Self {
            api_key: row.get("api_key"),
            configuration: serde_json::from_str(&row.get::<_, String>("configuration"))?,
            created_at: row.get("created_at"),
            expires_at: row.get("expires_at"),
            file_size: row.get("file_size"),
            file_name: row.get("file_name"),
            finished_at: row.get("finished_at"),
            image_folder_location: row
                .get::<_, Option<String>>("image_folder_location")
                .unwrap_or(image_folder_location),
            input_location: row
                .get::<_, Option<String>>("input_location")
                .unwrap_or(input_location),
            message: row.get("message"),
            mime_type: row.get("mime_type"),
            output_location: row
                .get::<_, Option<String>>("output_location")
                .unwrap_or(output_location),
            page_count,
            pdf_location: row
                .get::<_, Option<String>>("pdf_location")
                .unwrap_or(pdf_location),
            status: Status::from_str(&row.get::<_, String>("status"))?,
            started_at: row.get("started_at"),
            task_id: row.get("task_id"),
            task_url: row.get("task_url"),
            user_id: row.get("user_id"),
            version: row.get("version"),
        })
    }

    async fn create_output(&self) -> Result<OutputResponse, Box<dyn std::error::Error>> {
        let temp_file = download_to_tempfile(&self.output_location, None).await?;
        let json_content: String = tokio::fs::read_to_string(temp_file.path()).await?;
        let mut output_response: OutputResponse = serde_json::from_str(&json_content)?;
        async fn process(segment: &mut Segment) -> Result<String, Box<dyn std::error::Error>> {
            let url = generate_presigned_url(segment.image.as_ref().unwrap(), true, None)
                .await
                .ok();
            if segment.segment_type == SegmentType::Picture {
                segment.html = Some(format!(
                    "<img src=\"{}\" />",
                    url.clone().unwrap_or_default()
                ));
                segment.markdown = Some(format!("![Image]({})", url.clone().unwrap_or_default()));
            }
            Ok(url.clone().unwrap_or_default())
        }
        let futures = output_response
            .chunks
            .iter_mut()
            .flat_map(|chunk| chunk.segments.iter_mut())
            .filter(|segment| segment.image.is_some())
            .map(|segment| process(segment));

        try_join_all(futures).await?;
        Ok(output_response)
    }

    pub async fn update(
        &mut self,
        status: Option<Status>,
        message: Option<&str>,
        configuration: Option<Configuration>,
        started_at: Option<DateTime<Utc>>,
        finished_at: Option<DateTime<Utc>>,
        expires_at: Option<DateTime<Utc>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let client = get_pg_client().await?;
        let mut update_parts = vec![];

        if let Some(status) = status {
            update_parts.push(format!("status = '{:?}'", status));
            self.status = status;
        }
        if let Some(msg) = message {
            update_parts.push(format!("message = '{}'", msg));
            self.message = Some(msg.to_string());
        }
        if let Some(dt) = started_at {
            update_parts.push(format!("started_at = '{}'", dt));
            self.started_at = Some(dt);
        }
        if let Some(dt) = finished_at {
            update_parts.push(format!("finished_at = '{}'", dt));
            self.finished_at = Some(dt);
        }
        if let Some(dt) = expires_at {
            update_parts.push(format!("expires_at = '{}'", dt));
            self.expires_at = Some(dt);
        }

        if let Some(configuration) = configuration {
            update_parts.push(format!(
                "configuration = '{}'",
                serde_json::to_string(&configuration)?
            ));
            self.configuration = configuration;
        }

        let query = format!(
            "UPDATE tasks SET {} WHERE task_id = '{}' AND user_id = '{}'",
            update_parts.join(", "),
            self.task_id,
            self.user_id
        );

        client.execute(&query, &[]).await?;

        Ok(())
    }

    pub async fn delete(&self) -> Result<(), Box<dyn std::error::Error>> {
        if self.status != Status::Succeeded && self.status != Status::Failed {
            return Err(format!("Task cannot be deleted: status is {}", self.status).into());
        }
        let client = get_pg_client().await?;
        let worker_config = worker_config::Config::from_env().unwrap();
        let bucket_name = worker_config.s3_bucket;
        let folder_location = format!("s3://{}/{}/{}", bucket_name, self.user_id, self.task_id);
        delete_folder(&folder_location).await?;
        client
            .execute(
                "DELETE FROM tasks WHERE task_id = $1 AND user_id = $2",
                &[&self.task_id, &self.user_id],
            )
            .await?;
        Ok(())
    }

    pub async fn cancel(&self) -> Result<(), Box<dyn std::error::Error>> {
        let client = get_pg_client().await?;
        client
            .execute(
                "UPDATE TASKS 
             SET status = 'Cancelled', message = 'Task cancelled'
             WHERE task_id = $1 
             AND status = 'Starting'
             AND user_id = $2",
                &[&self.task_id, &self.user_id],
            )
            .await
            .map_err(|_| "Error updating task status".to_string())?;

        Ok(())
    }

    pub async fn upload_artifacts(
        &mut self,
        page_images: Vec<Arc<NamedTempFile>>,
        segment_images: &DashMap<String, Arc<NamedTempFile>>,
        output: &OutputResponse,
        pdf_file: &NamedTempFile,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.update(
            Some(Status::Processing),
            Some("Finishing up"),
            None,
            None,
            Some(Utc::now()),
            None,
        )
        .await?;

        let mut output_response = output.clone();

        for (idx, page) in page_images.iter().enumerate() {
            let s3_key = format!(
                "{}/{}/page_{}.jpg",
                self.image_folder_location, "pages", idx
            );
            upload_to_s3(&s3_key, page.path()).await?;
        }

        for pair in segment_images.iter() {
            let segment_id = pair.key();
            let temp_file = pair.value();
            let s3_key = format!("{}/{}.jpg", self.image_folder_location, segment_id);
            upload_to_s3(&s3_key, temp_file.path()).await?;
        }

        output_response.chunks.iter_mut().for_each(|chunk| {
            chunk.segments.iter_mut().for_each(|segment| {
                if segment_images.contains_key(&segment.segment_id) {
                    segment.image = Some(format!(
                        "{}/{}.jpg",
                        self.image_folder_location, segment.segment_id
                    ));
                }
            });
        });

        let mut output_temp_file = NamedTempFile::new()?;
        output_temp_file.write(serde_json::to_string(&output_response)?.as_bytes())?;
        upload_to_s3(&self.output_location, output_temp_file.path()).await?;
        upload_to_s3(&self.pdf_location, pdf_file.path()).await?;

        Ok(())
    }

    fn generate_s3_paths(
        user_id: &str,
        task_id: &str,
        file_name: &str,
    ) -> (String, String, String, String) {
        let worker_config = worker_config::Config::from_env().unwrap();
        let bucket_name = worker_config.s3_bucket;
        let base_path = format!("s3://{}/{}/{}", bucket_name, user_id, task_id);
        let path = Path::new(file_name);
        let file_stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");

        (
            format!("{}/{}", base_path, file_name),      // input_location
            format!("{}/{}.pdf", base_path, file_stem),  // pdf_location
            format!("{}/{}.json", base_path, file_stem), // output_location
            format!("{}/images", base_path),             // image_folder_location
        )
    }

    pub async fn to_task_response(&self) -> Result<TaskResponse, Box<dyn std::error::Error>> {
        let input_file_url = generate_presigned_url(&self.input_location, true, None)
            .await
            .map_err(|_| "Error getting input file url")?;
        let mut pdf_url = None;
        let mut output = None;
        if self.status == Status::Succeeded {
            pdf_url = Some(
                generate_presigned_url(&self.pdf_location, true, None)
                    .await
                    .map_err(|_| "Error getting pdf url")?,
            );
            output = Some(self.create_output().await?);
        }
        Ok(TaskResponse {
            task_id: self.task_id.clone(),
            status: self.status.clone(),
            created_at: self.created_at,
            finished_at: self.finished_at,
            expires_at: self.expires_at,
            output,
            input_file_url: Some(input_file_url),
            task_url: self.task_url.clone(),
            message: self.message.clone().unwrap_or_default(),
            configuration: self.configuration.clone(),
            file_name: self.file_name.clone(),
            page_count: self.page_count,
            pdf_url,
            started_at: self.started_at,
        })
    }

    pub fn to_task_payload(
        &self,
        previous_configuration: Option<Configuration>,
        previous_status: Option<Status>,
        previous_message: Option<String>,
        previous_version: Option<String>,
    ) -> TaskPayload {
        TaskPayload {
            previous_configuration,
            previous_status,
            previous_message,
            previous_version,
            task_id: self.task_id.clone(),
            user_id: self.user_id.clone(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema)]
pub struct TaskResponse {
    pub configuration: Configuration,
    /// The date and time when the task was created and queued.
    pub created_at: DateTime<Utc>,
    /// The date and time when the task will expire.
    pub expires_at: Option<DateTime<Utc>>,
    /// The name of the file.
    pub file_name: Option<String>,
    /// The date and time when the task was finished.
    pub finished_at: Option<DateTime<Utc>>,
    /// The presigned URL of the input file.
    pub input_file_url: Option<String>,
    /// A message describing the task's status or any errors that occurred.
    pub message: String,
    pub output: Option<OutputResponse>,
    /// The number of pages in the file.
    pub page_count: Option<u32>,
    /// The presigned URL of the PDF file.
    pub pdf_url: Option<String>,
    /// The date and time when the task was started.
    pub started_at: Option<DateTime<Utc>>,
    pub status: Status,
    /// The unique identifier for the task.
    pub task_id: String,
    /// The presigned URL of the task.
    pub task_url: Option<String>,
}

#[derive(
    Debug,
    Clone,
    Serialize,
    Deserialize,
    ToSql,
    FromSql,
    PartialEq,
    Eq,
    EnumString,
    Display,
    ToSchema,
)]
/// The status of the task.

pub enum Status {
    Starting,
    Processing,
    Succeeded,
    Failed,
    Cancelled,
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSql, FromSql, ToSchema)]
/// The configuration used for the task.
pub struct Configuration {
    pub chunk_processing: ChunkProcessing,
    #[serde(alias = "expires_at")]
    /// The number of seconds until task is deleted.
    /// Expried tasks can **not** be updated, polled or accessed via web interface.
    pub expires_in: Option<i32>,
    /// Whether to use high-resolution images for cropping and post-processing.
    pub high_resolution: bool,
    pub json_schema: Option<JsonSchema>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[deprecated]
    pub model: Option<Model>,
    pub ocr_strategy: OcrStrategy,
    pub segment_processing: SegmentProcessing,
    pub segmentation_strategy: SegmentationStrategy,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[deprecated]
    /// The target number of words in each chunk. If 0, each chunk will contain a single segment.
    pub target_chunk_length: Option<i32>,
}

// TODO: Move to output

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema, ToSql, FromSql)]
#[deprecated]
pub enum Model {
    Fast,
    HighQuality,
}

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
pub struct TaskPayload {
    pub previous_configuration: Option<Configuration>,
    pub previous_message: Option<String>,
    pub previous_status: Option<Status>,
    pub previous_version: Option<String>,
    pub task_id: String,
    pub user_id: String,
}
