use crate::models::auth::UserInfo;
use crate::models::task::{Configuration, Task, TaskResponse};
use crate::utils::services::payload::queue_task_payload;
use std::error::Error;
use tempfile::NamedTempFile;

pub async fn create_task(
    file: &NamedTempFile,
    file_name: Option<String>,
    user_info: &UserInfo,
    configuration: &Configuration,
) -> Result<TaskResponse, Box<dyn Error>> {
    // #region agent log
    let _ = std::fs::OpenOptions::new().create(true).append(true).open("/Users/harishmaiya/Documents/GitHub/data-extract/.cursor/debug.log").and_then(|mut f| std::io::Write::write_all(&mut f, format!("{{\"location\":\"create_task.rs:7\",\"message\":\"create_task called\",\"data\":{{\"userId\":\"{}\",\"fileName\":\"{:?}\"}},\"timestamp\":{},\"sessionId\":\"debug-session\",\"runId\":\"file-upload\",\"hypothesisId\":\"H1\"}}\n", user_info.user_id, file_name, std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis()).as_bytes()));
    // #endregion
    
    let task = match Task::new(
        user_info.user_id.as_str(),
        user_info.clone().api_key,
        configuration,
        file,
        file_name,
    )
    .await {
        Ok(t) => {
            // #region agent log
            let _ = std::fs::OpenOptions::new().create(true).append(true).open("/Users/harishmaiya/Documents/GitHub/data-extract/.cursor/debug.log").and_then(|mut f| std::io::Write::write_all(&mut f, format!("{{\"location\":\"create_task.rs:13\",\"message\":\"Task::new succeeded\",\"data\":{{\"taskId\":\"{}\"}},\"timestamp\":{},\"sessionId\":\"debug-session\",\"runId\":\"file-upload\",\"hypothesisId\":\"H1-H4\"}}\n", t.task_id, std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis()).as_bytes()));
            // #endregion
            t
        },
        Err(e) => {
            // #region agent log
            let error_str = format!("{}", e).replace("\"", "\\\"");
            let _ = std::fs::OpenOptions::new().create(true).append(true).open("/Users/harishmaiya/Documents/GitHub/data-extract/.cursor/debug.log").and_then(|mut f| std::io::Write::write_all(&mut f, format!("{{\"location\":\"create_task.rs:18\",\"message\":\"Task::new FAILED\",\"data\":{{\"error\":\"{}\"}},\"timestamp\":{},\"sessionId\":\"debug-session\",\"runId\":\"file-upload\",\"hypothesisId\":\"H1-H4\"}}\n", error_str, std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis()).as_bytes()));
            // #endregion
            return Err(e);
        }
    };
    
    let extraction_payload = task.to_task_payload(None, None, None, None, user_info);
    
    // #region agent log
    let _ = std::fs::OpenOptions::new().create(true).append(true).open("/Users/harishmaiya/Documents/GitHub/data-extract/.cursor/debug.log").and_then(|mut f| std::io::Write::write_all(&mut f, format!("{{\"location\":\"create_task.rs:24\",\"message\":\"Queueing task to BOTH queues\",\"data\":{{\"taskId\":\"{}\"}},\"timestamp\":{},\"sessionId\":\"debug-session\",\"runId\":\"file-upload\",\"hypothesisId\":\"H5\"}}\n", task.task_id, std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis()).as_bytes()));
    // #endregion
    
    // Queue to generic task queue (legacy)
    if let Err(e) = queue_task_payload(extraction_payload).await {
        // #region agent log
        let error_str = format!("{}", e).replace("\"", "\\\"");
        let _ = std::fs::OpenOptions::new().create(true).append(true).open("/Users/harishmaiya/Documents/GitHub/data-extract/.cursor/debug.log").and_then(|mut f| std::io::Write::write_all(&mut f, format!("{{\"location\":\"create_task.rs:29\",\"message\":\"queue_task_payload FAILED\",\"data\":{{\"error\":\"{}\"}},\"timestamp\":{},\"sessionId\":\"debug-session\",\"runId\":\"file-upload\",\"hypothesisId\":\"H5\"}}\n", error_str, std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis()).as_bytes()));
        // #endregion
        return Err(e);
    }
    
    // ALSO queue to deal_documents queue for the deal document worker
    // This ensures documents get processed with the optimized Docling pipeline
    let pool = crate::utils::clients::get_redis_pool();
    let mut conn = pool.get().await?;
    let deal_doc_message = serde_json::json!({
        "document_id": task.task_id.clone(),
        "deal_id": "mock_deal",  // Will be overridden for real deals
        "user_id": user_info.user_id.clone(),
        "s3_location": task.input_location.clone(),  // Full S3 URI is correct format
    });
    
    // #region agent log
    let _ = std::fs::OpenOptions::new().create(true).append(true).open("/Users/harishmaiya/Documents/GitHub/data-extract/.cursor/debug.log").and_then(|mut f| std::io::Write::write_all(&mut f, format!("{{\"location\":\"create_task.rs:47\",\"message\":\"Queueing to deal_documents\",\"data\":{{\"taskId\":\"{}\",\"s3_location\":\"{}\"}},\"timestamp\":{},\"sessionId\":\"debug-session\",\"runId\":\"file-upload\",\"hypothesisId\":\"H5\"}}\n", task.task_id, task.input_location, std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis()).as_bytes()));
    // #endregion
    
    deadpool_redis::redis::cmd("RPUSH")
        .arg("deal_documents")
        .arg(serde_json::to_string(&deal_doc_message)?)
        .query_async::<i64>(&mut conn)
        .await
        .map_err(|e| -> Box<dyn Error> { format!("Failed to queue to deal_documents: {}", e).into() })?;
    
    // #region agent log
    let _ = std::fs::OpenOptions::new().create(true).append(true).open("/Users/harishmaiya/Documents/GitHub/data-extract/.cursor/debug.log").and_then(|mut f| std::io::Write::write_all(&mut f, format!("{{\"location\":\"create_task.rs:34\",\"message\":\"Task queued successfully to BOTH queues\",\"data\":{{\"taskId\":\"{}\"}},\"timestamp\":{},\"sessionId\":\"debug-session\",\"runId\":\"file-upload\",\"hypothesisId\":\"H1-H5\"}}\n", task.task_id, std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis()).as_bytes()));
    // #endregion
    
    let task_response: TaskResponse = task.to_task_response(false, false).await?;
    Ok(task_response)
}
