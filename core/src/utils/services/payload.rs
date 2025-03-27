use crate::configs::redis_config::{Pipeline, RedisResult};
use crate::configs::worker_config::Config as WorkerConfig;
use crate::models::task::TaskPayload;
use crate::utils::clients::get_redis_pool;

pub async fn queue_task_payload(task_payload: TaskPayload) -> RedisResult<()> {
    let pool = get_redis_pool();
    let mut conn = pool.get().await.unwrap();
    let worker_config = WorkerConfig::from_env().expect("Failed to load worker config");
    let mut pipe = Pipeline::new();
    pipe.rpush(
        worker_config.queue_task,
        serde_json::to_string(&task_payload).expect("Failed to serialize task payload"),
    );
    pipe.atomic().query_async(&mut conn).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
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
            user_id: "test_user_id".to_string(),
        };
        queue_task_payload(task_payload).await.unwrap();
    }
}
