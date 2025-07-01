use crate::configs::worker_config::Config as WorkerConfig;
use crate::utils::clients;
use crate::utils::storage::services::delete_folder;
use futures::future::try_join_all;

pub async fn expire() -> Result<(), Box<dyn std::error::Error>> {
    let client = clients::get_pg_client().await?;
    let worker_config = WorkerConfig::from_env()?;
    let bucket_name = worker_config.s3_bucket;

    let expired_tasks = client
        .query(
            "SELECT user_id, task_id 
            FROM tasks 
            WHERE expires_at < CURRENT_TIMESTAMP 
            AND finished_at < CURRENT_TIMESTAMP 
            AND status in ('Succeeded', 'Failed', 'Cancelled')",
            &[],
        )
        .await?;

    let deletion_futures = expired_tasks.iter().map(|row| {
        let user_id: &str = row.get("user_id");
        let task_id: &str = row.get("task_id");
        let folder_location = format!("s3://{bucket_name}/{user_id}/{task_id}");
        async move {
            if let Err(e) = delete_folder(&folder_location).await {
                println!("Error deleting S3 folder {folder_location}: {e:?}");
            }
            Ok::<_, Box<dyn std::error::Error>>(())
        }
    });

    try_join_all(deletion_futures).await?;

    let rows_affected = client
        .execute(
            "DELETE FROM tasks 
            WHERE expires_at < CURRENT_TIMESTAMP 
            AND finished_at < CURRENT_TIMESTAMP 
            AND status in ('Succeeded', 'Failed', 'Cancelled')",
            &[],
        )
        .await?;

    println!("Deleted {rows_affected} expired tasks");
    Ok(())
}
