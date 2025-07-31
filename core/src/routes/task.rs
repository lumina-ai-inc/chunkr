use crate::configs::otel_config;
use crate::models::auth::UserInfo;
use crate::models::task::{Task, TaskQuery, TaskResponse};
use crate::models::upload;
use crate::models::upload_multipart;
use crate::utils::routes::cancel_task::cancel_task;
use crate::utils::routes::create_task;
use crate::utils::routes::delete_task::delete_task;
use crate::utils::routes::get_task::get_task;
use crate::utils::routes::update_task::update_task;
use crate::utils::services::file_operations::get_base64;
use actix_multipart::form::MultipartForm;
use actix_web::{web, Error, HttpResponse};
use opentelemetry::{
    trace::{Span, Tracer},
    Context, KeyValue,
};
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
    context_path = "",
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
    let otel_config = otel_config::Config::from_env().unwrap();
    let tracer = otel_config.get_tracer(otel_config::ServiceName::Server);
    let mut span = tracer.start_with_context(
        otel_config::SpanName::GetTask.to_string(),
        &Context::current(),
    );

    let task_id = task_id.into_inner();
    let user_id = user_info.user_id.clone();

    span.set_attribute(KeyValue::new("task_id", task_id.clone()));

    match get_task(task_id, user_id, task_query.into_inner()).await {
        Ok(task_response) => {
            span.end();
            Ok(HttpResponse::Ok().json(task_response))
        }
        Err(e) => {
            eprintln!("Error getting task: {e:?}");
            span.end();
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
    context_path = "",
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
    let otel_config = otel_config::Config::from_env().unwrap();
    let tracer = otel_config.get_tracer(otel_config::ServiceName::Server);
    let mut span = tracer.start_with_context(
        otel_config::SpanName::CreateTask.to_string(),
        &Context::current(),
    );

    let configuration = match payload.to_configuration() {
        Ok(config) => {
            span.set_attribute(KeyValue::new(
                "configuration",
                serde_json::to_string(&config).unwrap(),
            ));
            config
        }
        Err(e) => {
            span.end();
            return Ok(HttpResponse::BadRequest().body(e));
        }
    };

    let (base64_data, filename) = match get_base64(payload.file.clone()).await {
        Ok(file) => file,
        Err(e) => match e.to_string().contains("Invalid base64 data") {
            true => {
                span.end();
                return Ok(HttpResponse::BadRequest().body("Invalid base64 data"));
            }
            false => {
                span.end();
                return Ok(HttpResponse::InternalServerError().body("Failed to process file"));
            }
        },
    };

    let mut temp_file = match tempfile::NamedTempFile::new() {
        Ok(file) => file,
        Err(_) => {
            span.end();
            return Ok(HttpResponse::InternalServerError().body("Failed to process file"));
        }
    };

    if std::io::Write::write_all(&mut temp_file, &base64_data).is_err() {
        span.end();
        return Ok(HttpResponse::InternalServerError().body("Failed to process file"));
    };

    let result = create_task::create_task(
        &temp_file,
        filename.or(payload.file_name.clone()),
        &user_info,
        &configuration,
    )
    .await;

    match result {
        Ok(task_response) => {
            span.set_attribute(KeyValue::new("task_id", task_response.task_id.clone()));
            span.end();
            Ok(HttpResponse::Ok().json(task_response))
        }
        Err(e) => {
            let error_message = e.to_string();
            span.end();

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
                eprintln!("Error creating task: {e:?}");
                Ok(HttpResponse::InternalServerError().body("Failed to create task"))
            }
        }
    }
}

/// Update Task
///
/// Updates an existing task's configuration and reprocesses the document.
/// The original configuration will be used for all values that are not provided in the update.
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
    context_path = "",
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
    let otel_config = otel_config::Config::from_env().unwrap();
    let tracer = otel_config.get_tracer(otel_config::ServiceName::Server);
    let mut span = tracer.start_with_context(
        otel_config::SpanName::UpdateTask.to_string(),
        &Context::current(),
    );

    let task_id = task_id.into_inner();
    let user_id = user_info.user_id.clone();

    span.set_attribute(KeyValue::new("task_id", task_id.clone()));

    let previous_task = match Task::get(&task_id, &user_id).await {
        Ok(task) => task,
        Err(_) => {
            span.end();
            return Err(actix_web::error::ErrorNotFound("Task not found"));
        }
    };
    let configuration = match payload.to_configuration(&previous_task.configuration) {
        Ok(config) => {
            span.set_attribute(KeyValue::new(
                "configuration",
                serde_json::to_string(&config).unwrap(),
            ));
            config
        }
        Err(e) => {
            span.end();
            return Ok(HttpResponse::BadRequest().body(e));
        }
    };
    let result = update_task(&previous_task, &configuration, &user_info).await;
    match result {
        Ok(task_response) => {
            span.end();
            Ok(HttpResponse::Ok().json(task_response))
        }
        Err(e) => {
            let error_message = e.to_string();
            span.end();
            if error_message.contains("Usage limit exceeded") {
                Ok(HttpResponse::TooManyRequests().body("Usage limit exceeded"))
            } else if error_message.contains("Task cannot be updated") {
                Ok(HttpResponse::BadRequest().body(error_message))
            } else {
                eprintln!("Error creating task: {e:?}");
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
    context_path = "",
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
    let otel_config = otel_config::Config::from_env().unwrap();
    let tracer = otel_config.get_tracer(otel_config::ServiceName::Server);
    let mut span = tracer.start_with_context(
        otel_config::SpanName::DeleteTask.to_string(),
        &Context::current(),
    );

    let task_id = task_id.into_inner();
    let user_id = user_info.user_id.clone();

    span.set_attribute(KeyValue::new("task_id", task_id.clone()));

    match delete_task(task_id, user_id).await {
        Ok(_) => {
            span.end();
            Ok(HttpResponse::Ok().body("Task deleted"))
        }
        Err(e) => {
            eprintln!("Error deleting task: {e:?}");
            span.end();
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
    context_path = "",
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
    let otel_config = otel_config::Config::from_env().unwrap();
    let tracer = otel_config.get_tracer(otel_config::ServiceName::Server);
    let mut span = tracer.start_with_context(
        otel_config::SpanName::CancelTask.to_string(),
        &Context::current(),
    );

    let task_id = task_id.into_inner();
    let user_id = user_info.user_id.clone();

    span.set_attribute(KeyValue::new("task_id", task_id.clone()));

    match cancel_task(&task_id, &user_id).await {
        Ok(_) => {
            span.end();
            Ok(HttpResponse::Ok().body("Task cancelled"))
        }
        Err(e) => {
            eprintln!("Error cancelling task: {e:?}");
            span.end();
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
/// **This endpoint is deprecated**
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
    context_path = "",
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
    let form = &form.into_inner();
    let configuration = form.to_configuration().clone();
    let result = create_task::create_task(
        &form.file.file,
        form.file.file_name.clone(),
        &user_info,
        &configuration,
    )
    .await;
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
            } else if error_message.contains("must have a filename") {
                Ok(HttpResponse::BadRequest().body("File must have a filename"))
            } else {
                eprintln!("Error creating task: {e:?}");
                Ok(HttpResponse::InternalServerError().body("Failed to create task"))
            }
        }
    }
}

/// Update Task Multipart
///
/// **This endpoint is deprecated**
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
    context_path = "",
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
    let result = update_task(&previous_task, &configuration, &user_info).await;
    match result {
        Ok(task_response) => Ok(HttpResponse::Ok().json(task_response)),
        Err(e) => {
            let error_message = e.to_string();
            if error_message.contains("Usage limit exceeded") {
                Ok(HttpResponse::TooManyRequests().body("Usage limit exceeded"))
            } else if error_message.contains("Task cannot be updated") {
                Ok(HttpResponse::BadRequest().body(error_message))
            } else {
                eprintln!("Error creating task: {e:?}");
                Ok(HttpResponse::InternalServerError().body("Failed to create task"))
            }
        }
    }
}
