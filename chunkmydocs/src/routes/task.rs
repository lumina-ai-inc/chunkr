use crate::models::server::extract::{Configuration, UploadForm};
use crate::models::auth::auth::UserInfo;
use crate::utils::server::create_task::create_task;
use crate::utils::server::get_task::get_task;
use crate::utils::db::deadpool_postgres::Pool;
use actix_multipart::form::MultipartForm;
use actix_web::{web, Error, HttpRequest, HttpResponse};
use aws_sdk_s3::Client as S3Client;
use uuid::Uuid;

/// Get Extraction Task Status
///
/// Keep track of the progress of an extraction task by polling this route with the task ID.
#[utoipa::path(
    get,
    path = "/task/{task_id}",
    context_path = "/api",
    tag = "task",
    params(
        ("task_id" = Option<String>, Path, description = "Id of the task to get."),
    ),
    responses(
        (status = 200, description = "Detailed information describing the extraction task", body = TaskResponse),
        (status = 502, description = "Internal server error related to getting the extraction task", body = String),
    ),
)]
pub async fn get_task_status(
    pool: web::Data<Pool>,
    s3_client: web::Data<S3Client>,
    task_id: web::Path<String>,
    _req: HttpRequest,
) -> Result<HttpResponse, Error> {
    let task_id = task_id.into_inner();

    // Validate task_id as UUID
    if Uuid::parse_str(&task_id).is_err() {
        return Ok(HttpResponse::BadRequest().body("Invalid task ID format"));
    }

    match get_task(&pool, &s3_client, task_id).await {
        Ok(task_response) => Ok(HttpResponse::Ok().json(task_response)),
        Err(e) => {
            eprintln!("Error getting task status: {:?}", e);
            Ok(HttpResponse::InternalServerError().body(e.to_string()))
        }
    }
}

/// Create Extraction Task
///
/// Queue a document for extraction and get a task ID back to poll for status
#[utoipa::path(
    post,
    path = "/task",
    context_path = "/api",
    tag = "task",
    request_body(content = UploadForm, description = "Multipart form encoded data to create an extraction task", content_type = "multipart/form-data"),
    responses(
        (status = 200, description = "Detailed information describing the extraction task such that it's status can be poll'ed for", body = TaskResponse),
        (status = 502, description = "Internal server error related to creating the extraction task", body = String),
    ),
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
    };

    let result = create_task(
        pool,
        s3_client,
        file_data,
        &user_info,
        &configuration,
    )
    .await;

    // Delete temporary files after create_task has finished
    if let Err(e) = std::fs::remove_file(file_data.file.path()) {
        eprintln!("Error deleting temporary file: {:?}", e);
    }

    match result {
        Ok(task_response) => Ok(HttpResponse::Ok().json(task_response)),
        Err(e) => {
            eprintln!("Error creating task: {:?}", e);
            Ok(HttpResponse::InternalServerError().body("Failed to create task"))
        }
    }
}
