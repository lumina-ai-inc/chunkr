use crate::models::chunkr::task::{Task, TaskResponse};
use crate::models::chunkr::tasks::TasksQuery;
use crate::utils::clients::get_pg_client;
use futures::future::try_join_all;

pub async fn get_tasks(
    user_id: String,
    task_query: TasksQuery,
) -> Result<Vec<TaskResponse>, Box<dyn std::error::Error>> {
    let client = get_pg_client().await?;

    let mut conditions = vec![format!("user_id = '{}'", user_id)];
    conditions.push("(expires_at > NOW() OR expires_at IS NULL)".to_string());

    if let Some(start) = task_query.start {
        conditions.push(format!("created_at >= '{}'", start));
    }

    if let Some(end) = task_query.end {
        conditions.push(format!("created_at <= '{}'", end));
    }

    let pagination = match (task_query.page, task_query.limit) {
        (Some(p), Some(l)) => Ok(format!("OFFSET {} LIMIT {}", (p - 1) * l, l)),
        (None, Some(l)) => Ok(format!("LIMIT {}", l)),
        (Some(_), None) => Err("Limit is required when page is provided".to_string()),
        _ => Ok("".to_string()),
    };

    let query = format!(
        "SELECT task_id FROM TASKS WHERE {} ORDER BY created_at DESC {}",
        conditions.join(" AND "),
        pagination?
    );

    let task_ids = client
        .query(&query, &[])
        .await?
        .into_iter()
        .map(|row| row.get::<_, String>("task_id"))
        .collect::<Vec<String>>();

    let futures = task_ids.iter().map(|task_id| {
        let user_id = user_id.clone();
        let task_id = task_id.clone();
        async move {
            match Task::get(&task_id, &user_id).await {
                Ok(task) => match task
                    .to_task_response(
                        task_query.include_chunks.unwrap_or(false),
                        task_query.base64_urls.unwrap_or(false),
                    )
                    .await
                {
                    Ok(response) => {
                        Ok::<Option<TaskResponse>, Box<dyn std::error::Error>>(Some(response))
                    }
                    Err(e) => {
                        println!("Error converting task {}: {}", task_id, e);
                        Ok(None)
                    }
                },
                Err(e) => {
                    println!("Error fetching task {}: {}", task_id, e);
                    Ok(None)
                }
            }
        }
    });

    let task_responses = try_join_all(futures)
        .await?
        .into_iter()
        .flatten()
        .collect();
    Ok(task_responses)
}
