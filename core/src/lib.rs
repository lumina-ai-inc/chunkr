#![allow(deprecated)]

use actix_cors::Cors;
use actix_multipart::form::MultipartFormConfig;
use actix_multipart::MultipartError;
use actix_web::get;
use actix_web::middleware::Logger;
use actix_web::Error;
use actix_web::HttpRequest;
use actix_web::{web, App, HttpServer};
use diesel_migrations::{embed_migrations, EmbeddedMigrations, MigrationHarness};
use env_logger::Env;
use utoipa::{
    openapi::security::{ApiKey, ApiKeyValue, SecurityScheme},
    Modify, OpenApi,
};
use utoipa_redoc::{Redoc, Servable};
use utoipa_swagger_ui::SwaggerUi;

pub mod configs;
pub mod jobs;
pub mod middleware;
pub mod models;
pub mod pipeline;
pub mod routes;
pub mod utils;

use jobs::init::init_jobs;
use middleware::auth::AuthMiddlewareFactory;
use routes::github::get_github_repo_info;
use routes::health::health_check;
use routes::stripe::{
    create_setup_intent, create_stripe_session, get_invoice_detail, get_monthly_usage,
    get_user_invoices, stripe_webhook,
};
use routes::structured_extraction::handle_structured_extraction_route;
use routes::task::{
    cancel_task_route, create_task_route, delete_task_route, get_task_route, update_task_route,
};
use routes::tasks::get_tasks_route;
use routes::usage::get_usage;
use routes::user::get_or_create_user;
use utils::clients::initialize;
use utils::routes::admin_user::get_or_create_admin_user;

pub const MIGRATIONS: EmbeddedMigrations = embed_migrations!("./migrations");

fn run_migrations(url: &str) {
    use diesel::prelude::*;

    let mut conn = diesel::pg::PgConnection::establish(url).expect("Failed to connect to database");
    conn.run_pending_migrations(MIGRATIONS)
        .expect("Failed to run migrations");

    println!("Migrations ran successfully");
}

#[derive(OpenApi)]
#[openapi(
    info(
        title = "Chunkr API",
        description = "API service for document layout analysis and chunking to convert document into RAG/LLM-ready data.",
        contact(name = "Chunkr", url = "https://chunkr.ai", email = "ishaan@lumina.sh"),
        version = "1.0.0"
    ),
    servers((url = "https://api.chunkr.ai", description = "Production server")),
    paths(
        routes::health::health_check,
        routes::task::create_task_route,
        routes::task::get_task_route,
        routes::task::delete_task_route,
        routes::task::cancel_task_route,
        routes::task::update_task_route,
    ),
    components(
        schemas(
            models::chunkr::chunk_processing::ChunkProcessing,
            models::chunkr::cropping::CroppingStrategy,
            models::chunkr::output::BoundingBox,
            models::chunkr::output::Chunk,
            models::chunkr::output::OCRResult,
            models::chunkr::output::OutputResponse,
            models::chunkr::output::Segment,
            models::chunkr::output::SegmentType,
            models::chunkr::segment_processing::AutoGenerationConfig,
            models::chunkr::segment_processing::GenerationStrategy,
            models::chunkr::segment_processing::LlmGenerationConfig,
            models::chunkr::segment_processing::SegmentProcessing,
            models::chunkr::structured_extraction::ExtractedField,
            models::chunkr::structured_extraction::ExtractedJson,
            models::chunkr::structured_extraction::JsonSchema,
            models::chunkr::structured_extraction::Property,
            models::chunkr::structured_extraction::ExtractionRequest,
            models::chunkr::structured_extraction::ExtractionResponse,
            models::chunkr::task::Configuration,
            models::chunkr::task::Model,
            models::chunkr::task::Status,
            models::chunkr::task::TaskResponse,
            models::chunkr::upload::OcrStrategy,
            models::chunkr::upload::SegmentationStrategy,
            models::chunkr::upload::CreateForm,
            models::chunkr::upload::UpdateForm,
        )
    ),
    modifiers(&SecurityAddon),
    tags(
        (name = "Health", description = "Endpoint for checking the health of the service."),
        (name = "Task", description = "Endpoints for creating and getting task status")
    )
)]
pub struct ApiDoc;

struct SecurityAddon;

impl Modify for SecurityAddon {
    fn modify(&self, openapi: &mut utoipa::openapi::OpenApi) {
        let components = openapi
            .components
            .as_mut()
            .expect("Safe to expect since the component was already registered");
        components.add_security_scheme(
            "api_key",
            SecurityScheme::ApiKey(ApiKey::Header(ApiKeyValue::new("Authorization"))),
        );
    }
}

#[get("/openapi.json")]
pub async fn get_openapi_spec_handler() -> impl actix_web::Responder {
    web::Json(ApiDoc::openapi())
}

pub fn main() -> std::io::Result<()> {
    actix_web::rt::System::new().block_on(async move {
        env_logger::init_from_env(Env::default().default_filter_or("info"));
        initialize().await;
        run_migrations(&std::env::var("PG__URL").expect("PG__URL must be set in .env file"));
        get_or_create_admin_user()
            .await
            .expect("Failed to create admin user");
        init_jobs();
        fn handle_multipart_error(err: MultipartError, _: &HttpRequest) -> Error {
            println!("Multipart error: {}", err);
            Error::from(err)
        }

        let max_size: usize = std::env::var("MAX_TOTAL_LIMIT")
            .unwrap_or_else(|_| "1073741824".to_string())
            .parse()
            .expect("MAX_TOTAL_LIMIT must be a valid usize");
        let max_memory_size: usize = std::env::var("MAX_MEMORY_LIMIT")
            .unwrap_or_else(|_| "1073741824".to_string())
            .parse()
            .expect("MAX_MEMORY_LIMIT must be a valid usize");
        let timeout: usize = std::env::var("TIMEOUT")
            .unwrap_or_else(|_| "600".to_string())
            .parse()
            .expect("TIMEOUT must be a valid usize");
        let timeout = std::time::Duration::from_secs(timeout.try_into().unwrap());
        HttpServer::new(move || {
            let mut app = App::new()
                .wrap(Cors::permissive())
                .wrap(Logger::default())
                .wrap(Logger::new("%a %{User-Agent}i"))
                .app_data(
                    MultipartFormConfig::default()
                        .total_limit(max_size)
                        .memory_limit(max_memory_size)
                        .error_handler(handle_multipart_error),
                )
                .service(Redoc::with_url("/redoc", ApiDoc::openapi()))
                .route("/", web::get().to(health_check))
                .route("/health", web::get().to(health_check))
                .route("/github", web::get().to(get_github_repo_info))
                .service(
                    SwaggerUi::new("/swagger-ui/{_:.*}")
                        .url("/docs/openapi.json", ApiDoc::openapi()),
                );

            let api_scope = web::scope("/api/v1")
                .wrap(AuthMiddlewareFactory)
                .route("/user", web::get().to(get_or_create_user))
                .route(
                    "/structured_extract",
                    web::post().to(handle_structured_extraction_route),
                )
                .service(
                    web::scope("/task")
                        .route("", web::post().to(create_task_route))
                        .route("/{task_id}", web::get().to(get_task_route))
                        .route("/{task_id}", web::delete().to(delete_task_route))
                        .route("/{task_id}", web::patch().to(update_task_route))
                        .route("/{task_id}/cancel", web::get().to(cancel_task_route)),
                )
                .route("/tasks", web::get().to(get_tasks_route))
                .route("/usage", web::get().to(get_usage))
                .route("/usage/monthly", web::get().to(get_monthly_usage));

            if std::env::var("STRIPE__API_KEY").is_ok() {
                app = app.route("/stripe/webhook", web::post().to(stripe_webhook));

                let stripe_scope = web::scope("/stripe")
                    .wrap(AuthMiddlewareFactory)
                    .route("/create-setup-intent", web::get().to(create_setup_intent))
                    .route("/create-session", web::get().to(create_stripe_session))
                    .route("/invoices", web::get().to(get_user_invoices))
                    .route("/invoice/{invoice_id}", web::get().to(get_invoice_detail));

                app = app.service(stripe_scope);
            }

            app.service(api_scope)
        })
        .bind("0.0.0.0:8000")?
        .keep_alive(timeout)
        .run()
        .await
    })
}
