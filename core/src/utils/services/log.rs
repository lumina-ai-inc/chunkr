use crate::models::chunkr::task::Status;
use crate::utils::db::deadpool_postgres::{Client, Pool};
use chrono::{DateTime, Utc};

pub async fn log_task(
    task_id: &str,
    status: Status,
    message: Option<&str>,
    finished_at: Option<DateTime<Utc>>,
    pool: &Pool,
) -> Result<(), Box<dyn std::error::Error>> {
    let client: Client = pool.get().await?;

    let task_query = format!(
        "UPDATE tasks SET status = '{:?}', message = '{}', finished_at = '{:?}' WHERE task_id = '{}'",
        status,
        message.unwrap_or_default(),
        finished_at.unwrap_or_default(),
        task_id
    );

    client.execute(&task_query, &[]).await?;

    Ok(())
}
