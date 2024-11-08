use super::get_task::create_task_from_row;
use crate::models::server::task::TaskResponse;
use crate::utils::db::deadpool_postgres::{Client, Pool};
use aws_sdk_s3::Client as S3Client;

pub async fn get_tasks(
    pool: &Pool,
    s3_client: &S3Client,
    external_s3_client: &S3Client,
    user_id: String,
    page: i64,
    limit: i64,
) -> Result<Vec<TaskResponse>, Box<dyn std::error::Error>> {
    let client: Client = pool.get().await?;
    let offset = (page - 1) * limit;
    let tasks = client.query(
        "SELECT * FROM TASKS WHERE user_id = $1 AND  (expires_at > NOW() OR expires_at IS NULL) ORDER BY created_at DESC OFFSET $2 LIMIT $3",
        &[&user_id, &offset, &limit]
    ).await?;

    let mut task_responses = Vec::new();

    for row in tasks {
        match create_task_from_row(&row, s3_client, external_s3_client).await {
            Ok(task_response) => task_responses.push(task_response),
            Err(e) => eprintln!("Error processing task row: {}", e),
        }
    }

    Ok(task_responses)
}
