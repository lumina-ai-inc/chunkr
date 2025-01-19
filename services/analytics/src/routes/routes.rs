use actix_web::{web, HttpResponse};
use chrono::{DateTime, Utc};
use crate::queries;
use deadpool_postgres::Pool;
use log;

#[derive(serde::Deserialize)]
pub struct Payload {
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    email: Option<String>,
}

pub async fn lifetime_pages(pool: web::Data<Pool>) -> HttpResponse {
    match queries::queries::get_lifetime_pages(&pool).await {
        Ok(data) => HttpResponse::Ok().json(data),
        Err(e) => {
            log::error!("Lifetime pages error: {:?}", e);
            HttpResponse::InternalServerError().finish()
        }
    }
}

pub async fn pages_per_day(
    pool: web::Data<Pool>,
    params: web::Query<Payload>,
) -> HttpResponse {
    match queries::queries::get_pages_per_day(&pool, params.start, params.end, params.email.as_deref()).await {
        Ok(data) => HttpResponse::Ok().json(data),
        Err(_) => HttpResponse::InternalServerError().finish(),
    }
}

pub async fn top_users(
    pool: web::Data<Pool>,
    params: web::Json<Payload>,
) -> HttpResponse {
    match queries::queries::get_top_users(&pool, params.start, params.end, 5).await {
        Ok(data) => HttpResponse::Ok().json(data),
        Err(_) => HttpResponse::InternalServerError().finish(),
    }
}

pub async fn status_breakdown(
    pool: web::Data<Pool>,
    params: web::Query<Payload>,
) -> HttpResponse {
    match queries::queries::get_status_breakdown(&pool, params.start, params.end, params.email.as_deref()).await {
        Ok(data) => HttpResponse::Ok().json(data),
        Err(_) => HttpResponse::InternalServerError().finish(),
    }
}

pub async fn user_info(
    pool: web::Data<Pool>,
    params: web::Json<Payload>,
) -> HttpResponse {
    if let Some(email) = &params.email {
        match queries::queries::get_user_info(&pool, email, params.start, params.end).await {
            Ok(data) => HttpResponse::Ok().json(data),
            Err(_) => HttpResponse::InternalServerError().finish(),
        }
    } else {
        HttpResponse::BadRequest().finish()
    }
}

pub async fn task_details(
    pool: web::Data<Pool>,
    params: web::Query<Payload>,
) -> HttpResponse {
    match queries::queries::get_task_details(&pool, params.start, params.end, params.email.as_deref()).await {
        Ok(data) => HttpResponse::Ok().json(data),
        Err(_) => HttpResponse::InternalServerError().finish(),
    }
}

pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(web::resource("/").route(web::get().to(|| async { "Hello World!" })))
        .service(web::resource("/lifetime-pages").route(web::get().to(lifetime_pages)))
        .service(web::resource("/pages-per-day").route(web::get().to(pages_per_day)))
        .service(web::resource("/top-users").route(web::post().to(top_users)))
        .service(web::resource("/status-breakdown").route(web::get().to(status_breakdown)))
        .service(web::resource("/user-info").route(web::post().to(user_info)))
        .service(web::resource("/task-details").route(web::get().to(task_details)));
}