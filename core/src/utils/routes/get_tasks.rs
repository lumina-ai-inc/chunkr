use crate::models::chunkr::task::{Task, TaskDetails, TaskResponse};
use crate::utils::clients::get_pg_client;
use futures::future::try_join_all;
pub async fn get_tasks(
    user_id: String,
    page: i64,
    limit: i64,
    include_output: bool,
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
                Ok(task) => match task.to_task_response(include_output).await {
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

pub async fn get_task_details(
    start: chrono::DateTime<chrono::Utc>,
    end: chrono::DateTime<chrono::Utc>,
    email: Option<&str>,
) -> Result<Vec<TaskDetails>, Box<dyn std::error::Error>> {
    let client = get_pg_client().await?;
    let query = if let Some(_email) = email {
        "SELECT t.task_id, t.user_id, u.email, 
                CONCAT(u.first_name, ' ', u.last_name) as name,
                t.page_count, t.created_at, t.finished_at as completed_at,
                t.status
         FROM tasks t
         LEFT JOIN users u ON t.user_id = u.user_id
         WHERE t.created_at >= $1 AND t.created_at <= $2 AND u.email = $3
         ORDER BY t.created_at DESC
         LIMIT 100;"
    } else {
        "SELECT t.task_id, t.user_id, u.email, 
                CONCAT(u.first_name, ' ', u.last_name) as name,
                t.page_count, t.created_at, t.finished_at as completed_at,
                t.status
         FROM tasks t
         LEFT JOIN users u ON t.user_id = u.user_id
         WHERE t.created_at >= $1 AND t.created_at <= $2
         ORDER BY t.created_at DESC
         LIMIT 100;"
    };
    let rows = if let Some(_email) = email {
        client.query(query, &[&start, &end, &_email]).await?
    } else {
        client.query(query, &[&start, &end]).await?
    };
    Ok(rows
        .iter()
        .map(|row| TaskDetails {
            task_id: row.get("task_id"),
            user_id: row.get("user_id"),
            email: row.get("email"),
            name: row.get("name"),
            page_count: row.get("page_count"),
            created_at: row.get("created_at"),
            completed_at: row.get("completed_at"),
            status: row.get("status"),
        })
        .collect())
}
