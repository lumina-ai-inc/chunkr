use crate::configs::postgres_config::Client;
use crate::configs::worker_config::Config as WorkerConfig;
use crate::utils::clients::get_pg_client;
use crate::utils::storage::services::delete_folder;

pub async fn delete_task(
    task_id: String,
    user_id: String,
) -> Result<(), Box<dyn std::error::Error>> {
    let worker_config = WorkerConfig::from_env()?;
    let bucket_name = worker_config.s3_bucket;
    let client: Client = get_pg_client().await?;
    let folder_location = format!("s3://{}/{}/{}", bucket_name, user_id, task_id);
    delete_folder(&folder_location).await?;
    let result = client
        .execute(
            "DELETE FROM TASKS WHERE task_id = $1 AND user_id = $2",
            &[&task_id, &user_id],
        )
        .await
        .map_err(|_| "Error deleting task".to_string())?;

    if result == 0 {
        return Err("Task not found".into());
    }

    Ok(())
}
