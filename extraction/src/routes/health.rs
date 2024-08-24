use crate::utils::extraction_config::extraction_config::Config;
use actix_web::HttpResponse;
pub async fn health_check() -> HttpResponse {
    let config = Config::from_env().unwrap();
    let version = config.version;
    let message = format!("OK - Version {}", version);
    HttpResponse::Ok().body(message)
}
