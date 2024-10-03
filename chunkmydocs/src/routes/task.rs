use crate::models::auth::auth::UserInfo;
use crate::models::server::extract::{Configuration, UploadForm, OcrStrategy};
use crate::utils::db::deadpool_postgres::Pool;
use crate::utils::server::create_task::create_task;
use crate::utils::server::get_task::get_task;
use actix_multipart::form::MultipartForm;
use actix_web::{web, Error, HttpRequest, HttpResponse};
use aws_sdk_s3::Client as S3Client;
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
    pool: web::Data<Pool>,
    s3_client: web::Data<S3Client>,
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

    match get_task(&pool, &s3_client, task_id, user_id).await {
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
    req: HttpRequest,
    form: MultipartForm<UploadForm>,
    user_info: web::ReqData<UserInfo>,
) -> Result<HttpResponse, Error> {
    let form = form.into_inner();
    let pool = req.app_data::<web::Data<Pool>>().unwrap();
    let s3_client = req.app_data::<web::Data<S3Client>>().unwrap();
    let file_data = &form.file;
    let configuration = Configuration {
        model: form.model.into_inner(),
        target_chunk_length: form.target_chunk_length.map(|t| t.into_inner()),
        ocr_strategy: form.ocr_strategy.map(|t| t.into_inner()).unwrap_or(OcrStrategy::default()),
    };

    let result = create_task(pool, s3_client, file_data, &user_info, &configuration).await;

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
