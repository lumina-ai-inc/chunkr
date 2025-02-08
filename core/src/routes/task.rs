use crate::models::chunkr::auth::UserInfo;
use crate::models::chunkr::task::{Task, TaskQuery, TaskResponse};
use crate::models::chunkr::upload;
use crate::models::chunkr::upload_multipart;
use crate::utils::routes::cancel_task::cancel_task;
use crate::utils::routes::create_task;
use crate::utils::routes::delete_task::delete_task;
use crate::utils::routes::get_task::get_task;
use crate::utils::routes::update_task::update_task;
use actix_multipart::form::MultipartForm;
use actix_web::{web, Error, HttpResponse};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use tempfile;

/// Get Task
///
/// Retrieves detailed information about a task by its ID, including:
/// - Processing status
/// - Task configuration
/// - Output data (if processing is complete)
/// - File metadata (name, page count)
/// - Timestamps (created, started, finished)
/// - Presigned URLs for accessing files
///
/// This endpoint can be used to:
/// 1. Poll the task status during processing
/// 2. Retrieve the final output once processing is complete
/// 3. Access task metadata and configuration
#[utoipa::path(
    get,
    path = "/task/{task_id}",
    context_path = "/api/v1",
    tag = "Task",
    params(
        ("task_id" = Option<String>, Path, description = "Id of the task to retrieve"),
        ("base64_urls" = Option<bool>, Query, description = "Whether to return base64 encoded URLs. If false, the URLs will be returned as presigned URLs."),
        ("include_chunks" = Option<bool>, Query, description = "Whether to include chunks in the output response"),
    ),
    responses(
        (status = 200, description = "Detailed information describing the task", body = TaskResponse),
        (status = 500, description = "Internal server error related to getting the task", body = String),
    ),
    security(
        ("api_key" = []),
    )
)]
pub async fn get_task_route(
    task_id: web::Path<String>,
    task_query: web::Query<TaskQuery>,
    user_info: web::ReqData<UserInfo>,
) -> Result<HttpResponse, Error> {
    let task_id = task_id.into_inner();
    let user_id = user_info.user_id.clone();

    match get_task(task_id, user_id, task_query.into_inner()).await {
        Ok(task_response) => Ok(HttpResponse::Ok().json(task_response)),
        Err(e) => {
            eprintln!("Error getting task: {:?}", e);
            if e.to_string().contains("expired") || e.to_string().contains("not found") {
                Ok(HttpResponse::NotFound().body("Task not found"))
            } else {
                Ok(HttpResponse::InternalServerError().body(e.to_string()))
            }
        }
    }
}

/// Create Task
///
/// Queues a document for processing and returns a TaskResponse containing:
/// - Task ID for status polling
/// - Initial configuration
/// - File metadata
/// - Processing status
/// - Creation timestamp
/// - Presigned URLs for file access
///
/// The returned task will typically be in a `Starting` or `Processing` state.
/// Use the `GET /task/{task_id}` endpoint to poll for completion.
#[utoipa::path(
    post,
    path = "/task/parse",
    context_path = "/api/v1",
    tag = "Task",
    request_body(content = upload::CreateForm, description = "JSON request to create a task", content_type = "application/json"),
    responses(
        (status = 200, description = "Detailed information describing the task, its status and processed outputs", body = TaskResponse),
        (status = 500, description = "Internal server error related to creating the task", body = String),
    ),
    security(
        ("api_key" = []),
    )
)]
pub async fn create_task_route(
    payload: web::Json<upload::CreateForm>,
    user_info: web::ReqData<UserInfo>,
) -> Result<HttpResponse, Error> {
    let configuration = payload.to_configuration();

    // Decode base64 with proper error handling
    let file = match STANDARD.decode(&payload.file) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Error decoding base64 data: {:?}", e);
            return Ok(HttpResponse::BadRequest().body("Invalid base64 data"));
        }
    };

    // Create a temporary file
    let mut temp_file = match tempfile::NamedTempFile::new() {
        Ok(file) => file,
        Err(e) => {
            eprintln!("Error creating temporary file: {:?}", e);
            return Ok(HttpResponse::InternalServerError().body("Failed to process file"));
        }
    };

    // Write the decoded data to the temporary file
    if let Err(e) = std::io::Write::write_all(&mut temp_file, &file) {
        eprintln!("Error writing to temporary file: {:?}", e);
        return Ok(HttpResponse::InternalServerError().body("Failed to process file"));
    }

    let result = create_task::create_task(&temp_file, &user_info, &configuration).await;

    // Clean up the temporary file
    if let Err(e) = temp_file.close() {
        eprintln!("Error cleaning up temporary file: {:?}", e);
    }

    match result {
        Ok(task_response) => Ok(HttpResponse::Ok().json(task_response)),
        Err(e) => {
            let error_message = e.to_string();
            if error_message
                .to_lowercase()
                .contains("usage limit exceeded")
            {
                Ok(HttpResponse::TooManyRequests().body("Usage limit exceeded"))
            } else if error_message
                .to_lowercase()
                .contains("unsupported file type")
            {
                Ok(HttpResponse::BadRequest().body("Unsupported file type"))
            } else {
                eprintln!("Error creating task: {:?}", e);
                Ok(HttpResponse::InternalServerError().body("Failed to create task"))
            }
        }
    }
}

/// Update Task
///
/// Updates an existing task's configuration and reprocesses the document.
///
/// Requirements:
/// - Task must have status `Succeeded` or `Failed`
/// - New configuration must be different from the current one
///
/// The returned task will typically be in a `Starting` or `Processing` state.
/// Use the `GET /task/{task_id}` endpoint to poll for completion.
#[utoipa::path(
    patch,
    path = "/task/{task_id}/parse",
    context_path = "/api/v1",
    tag = "Task",
    request_body(content = upload::UpdateForm, description = "JSON request to update an task", content_type = "application/json"),
    responses(
        (status = 200, description = "Detailed information describing the task, its status and processed outputs", body = TaskResponse),
        (status = 500, description = "Internal server error related to updating the task", body = String),
    ),
    security(
        ("api_key" = []),
    )
)]
pub async fn update_task_route(
    payload: web::Json<upload::UpdateForm>,
    task_id: web::Path<String>,
    user_info: web::ReqData<UserInfo>,
) -> Result<HttpResponse, Error> {
    let task_id = task_id.into_inner();
    let user_id = user_info.user_id.clone();
    let previous_task = match Task::get(&task_id, &user_id).await {
        Ok(task) => task,
        Err(_) => return Err(actix_web::error::ErrorNotFound("Task not found")),
    };
    let configuration = payload.to_configuration(&previous_task.configuration);
    let result = update_task(&previous_task, &configuration).await;
    match result {
        Ok(task_response) => Ok(HttpResponse::Ok().json(task_response)),
        Err(e) => {
            let error_message = e.to_string();
            if error_message.contains("Usage limit exceeded") {
                Ok(HttpResponse::TooManyRequests().body("Usage limit exceeded"))
            } else if error_message.contains("Task cannot be updated") {
                Ok(HttpResponse::BadRequest().body(error_message))
            } else {
                eprintln!("Error creating task: {:?}", e);
                Ok(HttpResponse::InternalServerError().body("Failed to create task"))
            }
        }
    }
}

/// Delete Task
///
/// Delete a task by its ID.
///
/// Requirements:
/// - Task must have status `Succeeded` or `Failed`
#[utoipa::path(
    delete,
    path = "/task/{task_id}",
    context_path = "/api/v1",
    tag = "Task",
    params(
        ("task_id" = Option<String>, Path, description = "Id of the task to delete")
    ),
    responses(
        (status = 200, description = "Task deleted successfully"),
        (status = 500, description = "Internal server error related to deleting the task", body = String),
    ),
    security(
        ("api_key" = []),
    )
)]
pub async fn delete_task_route(
    task_id: web::Path<String>,
    user_info: web::ReqData<UserInfo>,
) -> Result<HttpResponse, Error> {
    let task_id = task_id.into_inner();
    let user_id = user_info.user_id.clone();
    match delete_task(task_id, user_id).await {
        Ok(_) => Ok(HttpResponse::Ok().body("Task deleted")),
        Err(e) => {
            eprintln!("Error deleting task: {:?}", e);
            if e.to_string().contains("expired") || e.to_string().contains("not found") {
                Ok(HttpResponse::NotFound().body("Task not found"))
            } else {
                Ok(HttpResponse::InternalServerError().body(e.to_string()))
            }
        }
    }
}

/// Cancel Task
///
/// Cancel a task that hasn't started processing yet:
/// - For new tasks: Status will be updated to `Cancelled`
/// - For updating tasks: Task will revert to the previous state
///
/// Requirements:
/// - Task must have status `Starting`
#[utoipa::path(
    get,
    path = "/task/{task_id}/cancel",
    context_path = "/api/v1",
    tag = "Task",
    params(
        ("task_id" = Option<String>, Path, description = "Id of the task to cancel")
    ),
    responses(
        (status = 200, description = "Task cancelled successfully"),
        (status = 500, description = "Internal server error related to cancelling the task", body = String),
    ),
    security(
        ("api_key" = []),
    )
)]
pub async fn cancel_task_route(
    task_id: web::Path<String>,
    user_info: web::ReqData<UserInfo>,
) -> Result<HttpResponse, Error> {
    let task_id = task_id.into_inner();
    let user_id = user_info.user_id.clone();
    match cancel_task(&task_id, &user_id).await {
        Ok(_) => Ok(HttpResponse::Ok().body("Task cancelled")),
        Err(e) => {
            eprintln!("Error cancelling task: {:?}", e);
            if e.to_string().contains("not found") {
                Ok(HttpResponse::NotFound().body("Task not found"))
            } else if e.to_string().contains("cannot be cancelled") {
                Ok(HttpResponse::BadRequest().body(e.to_string()))
            } else {
                Ok(HttpResponse::InternalServerError().body(e.to_string()))
            }
        }
    }
}

/// Create Task Multipart
///
/// Queues a document for processing and returns a TaskResponse containing:
/// - Task ID for status polling
/// - Initial configuration
/// - File metadata
/// - Processing status
/// - Creation timestamp
/// - Presigned URLs for file access
///
/// The returned task will typically be in a `Starting` or `Processing` state.
/// Use the `GET /task/{task_id}` endpoint to poll for completion.
#[utoipa::path(
    post,
    path = "/task",
    context_path = "/api/v1",
    tag = "Task",
    request_body(content = upload_multipart::CreateFormMultipart, description = "Multipart form request to create an task", content_type = "multipart/form-data"),
    responses(
        (status = 200, description = "Detailed information describing the task, its status and processed outputs", body = TaskResponse),
        (status = 500, description = "Internal server error related to creating the task", body = String),
    ),
    security(
        ("api_key" = []),
    )
)]
#[deprecated]
pub async fn create_task_route_multipart(
    form: MultipartForm<upload_multipart::CreateFormMultipart>,
    user_info: web::ReqData<UserInfo>,
) -> Result<HttpResponse, Error> {
    let form = form.into_inner();
    let file_data = &form.file;
    let configuration = form.to_configuration();
    let result = create_task::create_task_multipart(file_data, &user_info, &configuration).await;
    if let Ok(metadata) = std::fs::metadata(file_data.file.path()) {
        if metadata.is_file() {
            if let Err(e) = std::fs::remove_file(file_data.file.path()) {
                eprintln!("Error deleting temporary file: {:?}", e);
            }
        }
    }
    match result {
        Ok(task_response) => Ok(HttpResponse::Ok().json(task_response)),
        Err(e) => {
            let error_message = e.to_string();
            if error_message
                .to_lowercase()
                .contains("usage limit exceeded")
            {
                Ok(HttpResponse::TooManyRequests().body("Usage limit exceeded"))
            } else if error_message
                .to_lowercase()
                .contains("unsupported file type")
            {
                Ok(HttpResponse::BadRequest().body("Unsupported file type"))
            } else {
                eprintln!("Error creating task: {:?}", e);
                Ok(HttpResponse::InternalServerError().body("Failed to create task"))
            }
        }
    }
}

/// Update Task Multipart
///
/// Updates an existing task's configuration and reprocesses the document.
///
/// Requirements:
/// - Task must have status `Succeeded` or `Failed`
/// - New configuration must be different from the current one
///
/// The returned task will typically be in a `Starting` or `Processing` state.
/// Use the `GET /task/{task_id}` endpoint to poll for completion.
#[utoipa::path(
    patch,
    path = "/task/{task_id}",
    context_path = "/api/v1",
    tag = "Task",
    request_body(content = upload_multipart::UpdateFormMultipart, description = "Multipart form request to update an task", content_type = "multipart/form-data"),
    responses(
        (status = 200, description = "Detailed information describing the task, its status and processed outputs", body = TaskResponse),
        (status = 500, description = "Internal server error related to updating the task", body = String),
    ),
    security(
        ("api_key" = []),
    )
)]
#[deprecated]
pub async fn update_task_route_multipart(
    form: MultipartForm<upload_multipart::UpdateFormMultipart>,
    task_id: web::Path<String>,
    user_info: web::ReqData<UserInfo>,
) -> Result<HttpResponse, Error> {
    let task_id = task_id.into_inner();
    let user_id = user_info.user_id.clone();
    let form = form.into_inner();
    let previous_task = match Task::get(&task_id, &user_id).await {
        Ok(task) => task,
        Err(_) => return Err(actix_web::error::ErrorNotFound("Task not found")),
    };
    let configuration = form.to_configuration(&previous_task.configuration);
    let result = update_task(&previous_task, &configuration).await;
    match result {
        Ok(task_response) => Ok(HttpResponse::Ok().json(task_response)),
        Err(e) => {
            let error_message = e.to_string();
            if error_message.contains("Usage limit exceeded") {
                Ok(HttpResponse::TooManyRequests().body("Usage limit exceeded"))
            } else if error_message.contains("Task cannot be updated") {
                Ok(HttpResponse::BadRequest().body(error_message))
            } else {
                eprintln!("Error creating task: {:?}", e);
                Ok(HttpResponse::InternalServerError().body("Failed to create task"))
            }
        }
    }
}
