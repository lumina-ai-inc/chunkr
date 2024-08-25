use crate::utils::db::deadpool_postgres::Pool;
use crate::utils::server::get_task::get_task;
use actix_web::{web, Error, HttpRequest, HttpResponse};
use uuid::Uuid;

pub async fn get_task_status(
    pool: web::Data<Pool>,
    task_id: web::Path<String>,
    _req: HttpRequest,
) -> Result<HttpResponse, Error> {
    let task_id = task_id.into_inner();

    // Validate task_id as UUID
    if Uuid::parse_str(&task_id).is_err() {
        return Ok(HttpResponse::BadRequest().body("Invalid task ID format"));
    }

    match get_task(&pool, task_id).await {
        Ok(task_response) => Ok(HttpResponse::Ok().json(task_response)),
        Err(e) => {
            eprintln!("Error getting task status: {:?}", e);
            Ok(HttpResponse::InternalServerError().body(e.to_string()))
        }
    }
}
