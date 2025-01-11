use crate::configs::postgres_config::Client;
use crate::models::chunkr::task::Status;
use crate::utils::clients::get_pg_client;
use crate::utils::services::status::get_task;

pub async fn cancel_task(task_id: String) -> Result<(), Box<dyn std::error::Error>> {
    let client: Client = get_pg_client().await?;

    let status = get_task(&task_id).await?;
    match Some(status) {
        None => return Err("Task not found".into()),
        Some(status) if status != Status::Starting => {
            return Err(format!("Task cannot be cancelled: status is {}", status).into())
        }
        _ => {}
    }

    client
        .execute(
            "UPDATE TASKS 
             SET status = 'Cancelled', message = 'Task cancelled'
             WHERE task_id = $1 
             AND status = 'Starting'",
            &[&task_id],
        )
        .await
        .map_err(|_| "Error updating task status".to_string())?;

    Ok(())
}
