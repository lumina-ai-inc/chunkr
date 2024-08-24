use crate::models::extraction::api::ApiInfo;
use crate::models::extraction::extraction::UploadForm;
use crate::utils::db::deadpool_postgres::Pool;
use crate::utils::server::create_task::create_task;
use actix_multipart::form::MultipartForm;
use actix_web::{web, Error, HttpRequest, HttpResponse};
use uuid::Uuid;

pub async fn extract_files(
    req: HttpRequest,
    form: MultipartForm<UploadForm>,
    api_info: web::ReqData<ApiInfo>,
) -> Result<HttpResponse, Error> {
    let pool = req.app_data::<web::Data<Pool>>().unwrap();
    let api_key = api_info.api_key.clone();
    let user_id = api_info.user_id.clone();
    let task_id = Uuid::new_v4().to_string();

    // Process files
    let file_data = &form.file;

    // Call create_task function
    let model = form.model.to_internal();
    let result = create_task(
        &pool,
        file_data,
        task_id,
        user_id,
        &api_key.to_string(),
        model,
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
