use crate::models::chunkr::task::{Task, TaskResponse};
use crate::utils::clients::get_pg_client;
use futures::future::try_join_all;

pub async fn get_tasks(
    user_id: String,
    page: i64,
    limit: i64,
) -> Result<Vec<TaskResponse>, Box<dyn std::error::Error>> {
    let client = get_pg_client().await?;
    let offset = (page - 1) * limit;
    let task_ids = client
        .query(
            "SELECT task_id FROM TASKS WHERE user_id = $1 AND (expires_at > NOW() OR expires_at IS NULL) ORDER BY created_at DESC OFFSET $2 LIMIT $3",
            &[&user_id, &offset, &limit]
        )
        .await?
        .into_iter()
        .map(|row| row.get::<_, String>("task_id"))
        .collect::<Vec<String>>();

    let futures = task_ids.iter().map(|task_id| {
        let user_id = user_id.clone();
        let task_id = task_id.clone();
        async move {
            match Task::get(&task_id, &user_id).await {
                Ok(task) => match task.to_task_response(false).await {
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
        .filter_map(|x| x)
        .collect();
    Ok(task_responses)
}
