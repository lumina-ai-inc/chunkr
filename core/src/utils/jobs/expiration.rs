use crate::utils::configs::worker_config::Config as WorkerConfig;
use crate::utils::db::deadpool_postgres::Pool;
use crate::utils::storage::services::delete_folder;
use aws_sdk_s3::Client as S3Client;
use futures::future::try_join_all;

pub async fn expire(
    pg_pool: &Pool,
    s3_client: &S3Client,
) -> Result<(), Box<dyn std::error::Error>> {
    let client = pg_pool.get().await?;
    let worker_config = WorkerConfig::from_env()?;
    let bucket_name = worker_config.s3_bucket;

    let expired_tasks = client
        .query(
            "SELECT user_id, task_id FROM tasks WHERE expires_at < CURRENT_TIMESTAMP and finished_at > CURRENT_TIMESTAMP",
            &[],
        )
        .await?;

    let deletion_futures = expired_tasks.iter().map(|row| {
        let user_id: &str = row.get("user_id");
        let task_id: &str = row.get("task_id");
        let folder_location = format!("s3://{}/{}/{}", bucket_name, user_id, task_id);
        let s3_client = s3_client.clone();

        async move {
            if let Err(e) = delete_folder(&s3_client, &folder_location).await {
                println!("Error deleting S3 folder {}: {:?}", folder_location, e);
            }
            Ok::<_, Box<dyn std::error::Error>>(())
        }
    });

    try_join_all(deletion_futures).await?;

    let rows_affected = client
        .execute(
            "DELETE FROM tasks WHERE expires_at < CURRENT_TIMESTAMP",
            &[],
        )
        .await?;

    println!("Deleted {} expired tasks", rows_affected);
    Ok(())
}
