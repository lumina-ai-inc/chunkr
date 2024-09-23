use crate::utils::configs::extraction_config::Config;
use actix_web::HttpResponse;

/// Health Check
///
/// Confirmation that the service can respond to requests
#[utoipa::path(
    get,
    path = "/health",
    context_path = "",
    tag = "Health",
    responses(
        (status = 200, description = "Confirmation that the service can respond to requests and the version of the service", body = String),
    ),
)]
pub async fn health_check() -> HttpResponse {
    let config = Config::from_env().unwrap();
    let version = config.version;
    let message = format!("OK - Version {}", version);
    HttpResponse::Ok().body(message)
}
