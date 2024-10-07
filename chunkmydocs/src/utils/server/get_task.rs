use crate::models::server::extract::Configuration;
use crate::models::server::segment::{Chunk, SegmentType};
use crate::models::server::task::{Status, TaskResponse};
use crate::utils::db::deadpool_postgres::{Client, Pool};
use crate::utils::storage::services::{download_to_tempfile, generate_presigned_url};
use aws_sdk_s3::Client as S3Client;
use chrono::{DateTime, Utc};
use reqwest;
use serde_json;

pub async fn get_task(
    pool: &Pool,
    s3_client: &S3Client,
    task_id: String,
    user_id: String,
) -> Result<TaskResponse, Box<dyn std::error::Error>> {
    let client: Client = pool.get().await?;
    let task = client
        .query_one(
            "SELECT * FROM TASKS WHERE task_id = $1 AND user_id = $2",
            &[&task_id, &user_id],
        )
        .await?;

    let expires_at: Option<DateTime<Utc>> = task.get("expires_at");
    if expires_at.is_some() && expires_at.unwrap() < Utc::now() {
        return Err("Task expired".into());
    }

    create_task_from_row(&task, s3_client).await
}

pub async fn create_task_from_row(
    row: &tokio_postgres::Row,
    s3_client: &S3Client,
) -> Result<TaskResponse, Box<dyn std::error::Error>> {
    let task_id: String = row.get("task_id");
    let status: Status = row
        .get::<_, Option<String>>("status")
        .and_then(|m| m.parse().ok())
        .ok_or("Invalid status")?;
    let created_at: DateTime<Utc> = row.get("created_at");
    let finished_at: Option<DateTime<Utc>> = row.get("finished_at");
    let expires_at: Option<DateTime<Utc>> = row.get("expires_at");
    let message = row.get::<_, Option<String>>("message").unwrap_or_default();
    let file_name = row.get::<_, Option<String>>("file_name");
    let page_count = row.get::<_, Option<i32>>("page_count");
    let s3_pdf_location: Option<String> = row.get("pdf_location");
    let pdf_location = match s3_pdf_location {
        Some(location) => generate_presigned_url(s3_client, &location, None)
            .await
            .ok(),
        None => None,
    };
    let input_location: String = row.get("input_location");
    let input_file_url = generate_presigned_url(s3_client, &input_location, None)
        .await
        .map_err(|_| "Error getting input file url")?;

    let output_location: String = row.get("output_location");
    let output = if status == Status::Succeeded {
        Some(process_output(s3_client, &output_location).await?)
    } else {
        None
    };

    let task_url: Option<String> = row.get("task_url");
    let configuration: Configuration = row
        .get::<_, Option<String>>("configuration")
        .and_then(|c| serde_json::from_str(&c).ok())
        .ok_or("Invalid configuration")?;

    Ok(TaskResponse {
        task_id,
        status,
        created_at,
        finished_at,
        expires_at,
        message,
        output,
        input_file_url: Some(input_file_url),
        task_url,
        configuration,
        file_name,
        page_count,
        pdf_url: pdf_location.map(|s| s.to_string()),
    })
}

async fn process_output(
    s3_client: &S3Client,
    output_location: &str,
) -> Result<Vec<Chunk>, Box<dyn std::error::Error>> {
    let temp_file =
        download_to_tempfile(s3_client, &reqwest::Client::new(), output_location, None).await?;
    let json_content: String = tokio::fs::read_to_string(temp_file.path()).await?;
    let mut chunks: Vec<Chunk> = serde_json::from_str(&json_content)?;

    for chunk in &mut chunks {
        for segment in &mut chunk.segments {
            if let Some(image) = segment.image.as_ref() {
                let url = generate_presigned_url(s3_client, image, None).await.ok();
                segment.image = url.clone();
                if segment.segment_type == SegmentType::Picture {
                    segment.html = Some(format!(
                        "<img src=\"{}\" />",
                        url.clone().unwrap_or_default()
                    ));
                    segment.markdown =
                        Some(format!("![Image]({})", url.clone().unwrap_or_default()));
                }
            }
        }
    }

    Ok(chunks)
}
