use crate::models::chunkr::auth::UserInfo;
use crate::models::chunkr::task::Configuration;
use crate::models::chunkr::task::TaskResponse;
use crate::models::chunkr::upload::UploadForm;
use crate::utils::routes::create_task::create_task;
use crate::utils::routes::get_task::get_task;
use actix_multipart::form::MultipartForm;
use actix_web::{web, Error, HttpRequest, HttpResponse};
use uuid::Uuid;

/// Get Task
///
/// Keep track of the progress of an extraction task by polling this route with the task ID.
#[utoipa::path(
    get,
    path = "/task/{task_id}",
    context_path = "/api/v1",
    tag = "Task",
    params(
        ("task_id" = Option<String>, Path, description = "Id of the task to retrieve")
    ),
    responses(
        (status = 200, description = "Detailed information describing the task", body = TaskResponse),
        (status = 500, description = "Internal server error related to getting the task", body = String),
    ),
    security(
        ("api_key" = []),
    )
)]
pub async fn get_task_status(
    task_id: web::Path<String>,
    user_info: web::ReqData<UserInfo>,
    _req: HttpRequest,
) -> Result<HttpResponse, Error> {
    let task_id = task_id.into_inner();

    // Validate task_id as UUID
    if Uuid::parse_str(&task_id).is_err() {
        return Ok(HttpResponse::BadRequest().body("Invalid task ID format"));
    }

    let user_id = user_info.user_id.clone();

    match get_task(task_id, user_id).await {
        Ok(task_response) => Ok(HttpResponse::Ok().json(task_response)),
        Err(e) => {
            eprintln!("Error getting task status: {:?}", e);
            Ok(HttpResponse::InternalServerError().body(e.to_string()))
        }
    }
}

/// Create Task
///
/// Queue a document for extraction and get a task ID back to poll for status
#[utoipa::path(
    post,
    path = "/task",
    context_path = "/api/v1",
    tag = "Task",
    request_body(content = UploadForm, description = "Multipart form request to create an task", content_type = "multipart/form-data"),
    responses(
        (status = 200, description = "Detailed information describing the task such that it's status can be polled for", body = TaskResponse),
        (status = 500, description = "Internal server error related to creating the task", body = String),
    ),
    security(
        ("api_key" = []),
    )
)]
pub async fn create_extraction_task(
    form: MultipartForm<UploadForm>,
    user_info: web::ReqData<UserInfo>,
) -> Result<HttpResponse, Error> {
    let form = form.into_inner();
    let file_data = &form.file;
    let configuration = Configuration {
        chunk_processing: form.get_chunk_processing().unwrap_or_default(),
        expires_in: form.expires_in.map(|e| e.into_inner()),
        high_resolution: form
            .high_resolution
            .map(|e| e.into_inner())
            .unwrap_or(false),
        json_schema: form.json_schema.map(|js| js.into_inner()),
        model: None,
        ocr_strategy: form
            .ocr_strategy
            .map(|e| e.into_inner())
            .unwrap_or_default(),
        segment_processing: form
            .segment_processing
            .map(|e| e.into_inner())
            .unwrap_or_default(),
        segmentation_strategy: form
            .segmentation_strategy
            .map(|e| e.into_inner())
            .unwrap_or_default(),
        target_chunk_length: None,
    };

    let result = create_task(file_data, &user_info, &configuration).await;

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
            if error_message.contains("Usage limit exceeded") {
                Ok(HttpResponse::TooManyRequests().body("Usage limit exceeded"))
            } else {
                eprintln!("Error creating task: {:?}", e);
                Ok(HttpResponse::InternalServerError().body("Failed to create task"))
            }
        }
    }
}
