use crate::configs::worker_config::Config as WorkerConfig;
use crate::models::task::TaskPayload;
use crate::utils::clients::get_redis_pool;
use deadpool_redis::redis::cmd;

pub async fn queue_task_payload(
    task_payload: TaskPayload,
) -> Result<(), Box<dyn std::error::Error>> {
    let pool = get_redis_pool();
    let mut conn = pool.get().await.unwrap();
    let worker_config = WorkerConfig::from_env().expect("Failed to load worker config");
    match cmd("RPUSH")
        .arg(&worker_config.queue_task)
        .arg(serde_json::to_string(&task_payload).expect("Failed to serialize task payload"))
        .query_async::<i64>(&mut conn)
        .await
    {
        Ok(_) => Ok(()),
        Err(e) => Err(e.to_string().into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::auth::UserInfo;
    use crate::utils::clients::initialize;

    #[tokio::test]
    async fn test_queue_task_payload() {
        initialize().await;
        let task_payload = TaskPayload {
            previous_configuration: None,
            previous_message: None,
            previous_status: None,
            previous_version: None,
            task_id: "test_task_id".to_string(),
            user_info: UserInfo {
                user_id: "test_user_id".to_string(),
                email: None,
                first_name: None,
                last_name: None,
                api_key: None,
            },
            trace_context: None,
        };
        queue_task_payload(task_payload).await.unwrap();
    }
}
