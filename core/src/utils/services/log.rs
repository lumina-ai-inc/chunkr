use crate::configs::postgres_config::Client;
use crate::models::chunkr::task::Status;
use crate::utils::clients::get_pg_client;
use chrono::{DateTime, Utc};

pub async fn log_task(
    task_id: &str,
    status: Status,
    message: Option<&str>,
    finished_at: Option<DateTime<Utc>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let client: Client = get_pg_client().await?;

    let finished_at_str = finished_at.map_or("NULL".to_string(), |dt| format!("'{}'", dt));

    let task_query = format!(
        "UPDATE tasks SET status = '{:?}', message = '{}', finished_at = {} WHERE task_id = '{}'",
        status,
        message.unwrap_or_default(),
        finished_at_str,
        task_id
    );

    client.execute(&task_query, &[]).await?;

    Ok(())
}
