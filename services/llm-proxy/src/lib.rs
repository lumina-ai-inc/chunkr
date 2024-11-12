use actix_cors::Cors;
use actix_web::{web, App, HttpServer, HttpResponse, HttpRequest};
use actix_web::middleware::Logger;
use awc::Client;
use clap::Parser;
use env_logger::Env;
use serde_json;
use std::process::Command;

async fn health() -> HttpResponse {
    HttpResponse::Ok().json(serde_json::json!({
        "status": "healthy"
    }))
}

async fn proxy(req: HttpRequest, body: web::Bytes, args: web::Data<Args>) -> HttpResponse {
    let client = Client::default();
    
    let mut forwarded_req = client.request(req.method().clone(), &forward_url).timeout(std::time::Duration::from_secs(timeout));
    
    for (header_name, header_value) in req.headers() {
        forwarded_req = forwarded_req.insert_header((header_name.clone(), header_value.clone()));
    }
    
    match forwarded_req.send_body(body).await {
        Ok(mut client_resp) => {
            match client_resp.body().await {
                Ok(body) => {
                    match serde_json::from_slice::<serde_json::Value>(&body) {
                        Ok(json_body) => {
                            HttpResponse::Ok()
                                .content_type("application/json")
                                .json(json_body)
                        },
                        Err(_) => {
                            HttpResponse::BadGateway()
                                .json(serde_json::json!({
                                    "error": "Invalid JSON response from upstream server"
                                }))
                        }
                    }
                },
                Err(e) => {
                    HttpResponse::InternalServerError()
                        .json(serde_json::json!({
                            "error": format!("Error reading response body: {}", e)
                        }))
                }
            }
        },
        Err(e) => {
            HttpResponse::InternalServerError()
                .json(serde_json::json!({
                    "error": format!("Error forwarding request: {}", e)
                }))
        }
    }
}

pub fn main() -> std::io::Result<()> {
    actix_web::rt::System::new().block_on(async move {
        env_logger::init_from_env(Env::default().default_filter_or("info"));

        HttpServer::new(move || {
            App::new()
                .wrap(Cors::permissive())
                .wrap(Logger::default())
                .wrap(Logger::new("%a %{User-Agent}i"))
                .app_data(web::PayloadConfig::new(10 * 1024 * 1024 * 1024))
                .route("/", web::get().to(health))
                .default_service(web::to(proxy))
        })
        .bind("0.0.0.0:8000")?
        .run()
        .await
    })
}
