use crate::configs::postgres_config::Client;
use crate::models::chunkr::task::Status;
use crate::utils::clients::get_pg_client;
use chrono::{DateTime, Utc};
use std::str::FromStr;

pub async fn update_status(
    task_id: &str,
    user_id: &str,
    status: Status,
    message: Option<&str>,
    started_at: Option<DateTime<Utc>>,
    finished_at: Option<DateTime<Utc>>,
    expires_at: Option<DateTime<Utc>>,
    mime_type: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let client: Client = get_pg_client().await?;
    let mut update_parts = vec![format!("status = '{:?}'", status)];
    if let Some(msg) = message {
        update_parts.push(format!("message = '{}'", msg));
    }
    if let Some(dt) = started_at {
        update_parts.push(format!("started_at = '{}'", dt));
    }
    if let Some(dt) = finished_at {
        update_parts.push(format!("finished_at = '{}'", dt));
    }
    if let Some(dt) = expires_at {
        update_parts.push(format!("expires_at = '{}'", dt));
    }
    if let Some(mime_type) = mime_type {
        update_parts.push(format!("mime_type = '{}'", mime_type));
    }
    let task_query = format!(
        "UPDATE tasks SET {} WHERE task_id = '{}' AND user_id = '{}'",
        update_parts.join(", "),
        task_id,
        user_id
    );
    client.execute(&task_query, &[]).await?;

    Ok(())
}

pub async fn get_status(
    task_id: &str,
    user_id: &str,
) -> Result<Status, Box<dyn std::error::Error>> {
    let client: Client = get_pg_client().await?;
    let status = Status::from_str(
        &client
            .query_one(
                "SELECT status FROM tasks WHERE task_id = $1 AND user_id = $2",
                &[&task_id, &user_id],
            )
            .await
            .map_err(|e| format!("Error fetching task: {}", e))?
            .get::<_, String>(0),
    )
    .map_err(|e| format!("Failed to parse status: {}", e))?;
    Ok(status)
}
