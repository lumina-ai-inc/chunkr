use super::get_task::create_task_from_row;
use crate::configs::postgres_config::Client;
use crate::models::chunkr::task::TaskResponse;
use crate::utils::clients::get_pg_client;
use futures::future::join_all;

pub async fn get_tasks(
    user_id: String,
    page: i64,
    limit: i64,
) -> Result<Vec<TaskResponse>, Box<dyn std::error::Error>> {
    let client: Client = get_pg_client().await?;
    let offset = (page - 1) * limit;
    let tasks = client.query(
        "SELECT * FROM TASKS WHERE user_id = $1 AND  (expires_at > NOW() OR expires_at IS NULL) ORDER BY created_at DESC OFFSET $2 LIMIT $3",
        &[&user_id, &offset, &limit]
    ).await?;

    let futures = tasks.iter().map(|row| create_task_from_row(row));

    let results = join_all(futures).await;
    let task_responses: Vec<TaskResponse> = results
        .into_iter()
        .filter_map(|result| match result {
            Ok(task_response) => Some(task_response),
            Err(e) => {
                eprintln!("Error processing task row: {}", e);
                None
            }
        })
        .collect();

    Ok(task_responses)
}
