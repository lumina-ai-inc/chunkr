#![allow(deprecated)]

use actix_cors::Cors;
use actix_multipart::form::MultipartFormConfig;
use actix_multipart::MultipartError;
use actix_web::get;
use actix_web::middleware::Logger;
use actix_web::Error;
use actix_web::HttpRequest;
use actix_web::{web, App, HttpServer};
use actix_web_opentelemetry::RequestTracing;
use diesel_migrations::{embed_migrations, EmbeddedMigrations, MigrationHarness};
use env_logger::Env;
use utoipa::{
    openapi::security::{ApiKey, ApiKeyValue, SecurityScheme},
    Modify, OpenApi,
};
use utoipa_redoc::{Redoc, Servable};
use utoipa_swagger_ui::SwaggerUi;

// pub mod agents; // Temporarily disabled - needs Diesel to tokio-postgres conversion
pub mod configs;
pub mod data;
pub mod events;
pub mod jobs;
pub mod middleware;
pub mod models;
pub mod pipeline;
pub mod routes;
pub mod services;
pub mod utils;

use jobs::init::init_jobs;
use middleware::auth::AuthMiddlewareFactory;
// Deal routes temporarily disabled - need Diesel to tokio-postgres conversion
// use routes::deal::{
//     approve_facts_route, calculate_underwriting_route, create_deal_route, get_deal_documents,
//     get_deal_facts, get_deal_route, get_deals_route, reset_facts_route, update_fact_route,
//     upload_deal_documents,
// };
use routes::github::get_github_repo_info;
use routes::health::health_check;
use routes::llm::get_models_ids;
use routes::stripe::{
    create_checkout_session, create_setup_intent, create_stripe_session,
    get_billing_portal_session, get_checkout_session, get_invoice_detail, get_monthly_usage,
    get_user_invoices, stripe_webhook,
};
use routes::task::{
    cancel_task_route, create_task_route, create_task_route_multipart, delete_task_route,
    get_task_route, update_task_route, update_task_route_multipart,
};
use routes::tasks::get_tasks_route;
use routes::user::get_or_create_user;
use utils::clients::initialize;
use utils::routes::admin_user::get_or_create_admin_user;

pub const MIGRATIONS: EmbeddedMigrations = embed_migrations!("./migrations");

const ONE_GB: usize = 1024 * 1024 * 1024; // 1 GB in bytes

fn run_migrations(url: &str) {
    use diesel::prelude::*;
    
    // Retry logic for production environments
    let max_retries = 5;
    let mut attempt = 0;
    
    loop {
        attempt += 1;
        println!("Migration attempt {}/{}", attempt, max_retries);
        
        match diesel::pg::PgConnection::establish(url) {
            Ok(mut conn) => {
                match conn.run_pending_migrations(MIGRATIONS) {
                    Ok(_) => {
                        println!("✅ Migrations ran successfully");
                        
                        // Log migration audit trail
                        if let Ok(env) = std::env::var("ENV") {
                            if env == "production" {
                                println!("📝 Migration audit: Environment={}, Timestamp={}", 
                                    env, chrono::Utc::now().to_rfc3339());
                            }
                        }
                        return;
                    }
                    Err(e) => {
                        eprintln!("❌ Migration error: {}", e);
                        if attempt >= max_retries {
                            panic!("Failed to run migrations after {} attempts", max_retries);
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("❌ Database connection error: {}", e);
                if attempt >= max_retries {
                    panic!("Failed to connect to database after {} attempts", max_retries);
                }
            }
        }
        
        // Wait before retrying
        std::thread::sleep(std::time::Duration::from_secs(5));
    }
}

#[derive(OpenApi)]
#[openapi(
    info(
        title = "Orin API",
        description = "AI-powered real estate deal analysis platform for fund managers.",
        contact(name = "Orin", url = "https://orin.ai", email = "support@orin.ai"),
        version = "1.0.0"
    ),
    servers((url = "https://api.orin.ai", description = "Production server")),
    paths(
        routes::health::health_check,
        routes::task::create_task_route,
        routes::task::get_task_route,
        routes::task::delete_task_route,
        routes::task::cancel_task_route,
        routes::task::update_task_route,
        routes::tasks::get_tasks_route,
    ),
    components(
        schemas(
            models::chunk_processing::ChunkProcessing,
            models::cropping::CroppingStrategy,
            models::output::BoundingBox,
            models::output::Chunk,
            models::output::OCRResult,
            models::output::OutputResponse,
            models::output::Segment,
            models::output::SegmentType,
            models::segment_processing::AutoGenerationConfig,
            models::segment_processing::GenerationStrategy,
            models::segment_processing::LlmGenerationConfig,
            models::segment_processing::SegmentProcessing,
            models::task::Configuration,
            models::task::Model,
            models::task::Status,
            models::task::TaskResponse,
            models::upload::OcrStrategy,
            models::upload::SegmentationStrategy,
            models::upload::CreateForm,
            models::upload::UpdateForm,
            models::upload_multipart::CreateFormMultipart,
            models::upload_multipart::UpdateFormMultipart,
        )
    ),
    modifiers(&SecurityAddon),
    tags(
        (name = "Health", description = "Endpoint for checking the health of the service."),
        (name = "Task", description = "Endpoints for managing individual tasks - create, read, update, delete and cancel operations"),
        (name = "Tasks", description = "Endpoints for listing multiple tasks")
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
        if let Err(e) = configs::otel_config::Config::from_env()
            .map(|config| config.init_tracer(configs::otel_config::ServiceName::Server))
            .map_err(|e| e.to_string())
        {
            eprintln!("Failed to initialize OpenTelemetry tracer: {}", e);
        }

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
            .unwrap_or_else(|_| ONE_GB.to_string())
            .parse()
            .expect("MAX_TOTAL_LIMIT must be a valid usize");
        let max_memory_size: usize = std::env::var("MAX_MEMORY_LIMIT")
            .unwrap_or_else(|_| ONE_GB.to_string())
            .parse()
            .expect("MAX_MEMORY_LIMIT must be a valid usize");
        let timeout: usize = std::env::var("TIMEOUT")
            .unwrap_or_else(|_| "600".to_string())
            .parse()
            .expect("TIMEOUT must be a valid usize");
        let timeout = std::time::Duration::from_secs(timeout.try_into().unwrap());
        HttpServer::new(move || {
            let mut app = App::new()
                .wrap(RequestTracing::new())
                .wrap(
                    Cors::default()
                        .allow_any_origin()
                        .allow_any_method()
                        .allow_any_header()
                        .supports_credentials()
                )
                .wrap(Logger::default())
                .wrap(Logger::new("%a %{User-Agent}i"))
                .app_data(
                    MultipartFormConfig::default()
                        .total_limit(max_size)
                        .memory_limit(max_memory_size)
                        .error_handler(handle_multipart_error),
                )
                .app_data(web::JsonConfig::default().limit(max_size))
                .service(Redoc::with_url("/redoc", ApiDoc::openapi()))
                .route("/", web::get().to(health_check))
                .route("/health", web::get().to(health_check))
                .route("/github", web::get().to(get_github_repo_info))
                .route("/llm/models", web::get().to(get_models_ids))
                // Public conversation route (no auth)
                .route("/api/v1/conversations/public/message", web::post().to(routes::conversation::send_public_message))
                .service(
                    SwaggerUi::new("/swagger-ui/{_:.*}")
                        .url("/docs/openapi.json", ApiDoc::openapi()),
                );

            let api_scope = web::scope("/api/v1")
                .wrap(AuthMiddlewareFactory)
                .route("/user", web::get().to(get_or_create_user))
                // Deal routes temporarily disabled - need Diesel to tokio-postgres conversion
                // .service(
                //     web::scope("/deals")
                //         .route("", web::post().to(create_deal_route))
                //         .route("", web::get().to(get_deals_route))
                //         .route("/{deal_id}", web::get().to(get_deal_route))
                //         .route("/{deal_id}/documents", web::post().to(upload_deal_documents))
                //         .route("/{deal_id}/documents", web::get().to(get_deal_documents))
                //         .route("/{deal_id}/facts", web::get().to(get_deal_facts))
                //         .route("/{deal_id}/facts/{fact_id}", web::patch().to(update_fact_route))
                //         .route("/{deal_id}/facts/approve", web::post().to(approve_facts_route))
                //         .route("/{deal_id}/facts/reset", web::post().to(reset_facts_route))
                //         .route("/{deal_id}/underwrite", web::post().to(calculate_underwriting_route)),
                // )
                .service(
                    web::scope("/conversations")
                        .route("", web::post().to(routes::conversation::create_conversation))
                        .route("", web::get().to(routes::conversation::list_conversations))
                        .route("/{conversation_id}", web::get().to(routes::conversation::get_conversation))
                        .route("/{conversation_id}/messages", web::post().to(routes::conversation::send_message))
                        .route("/{conversation_id}", web::delete().to(routes::conversation::delete_conversation)),
                )
                .service(
                    web::scope("/task")
                        .route("", web::post().to(create_task_route_multipart))
                        .route("/parse", web::post().to(create_task_route))
                        .route("/{task_id}", web::get().to(get_task_route))
                        .route("/{task_id}", web::delete().to(delete_task_route))
                        .route("/{task_id}", web::patch().to(update_task_route_multipart))
                        .route("/{task_id}/parse", web::patch().to(update_task_route))
                        .route("/{task_id}/cancel", web::get().to(cancel_task_route)),
                )
                .route("/tasks", web::get().to(get_tasks_route))
                .route("/usage/monthly", web::get().to(get_monthly_usage));

            // Stripe integration - only enabled if API key is configured
            if std::env::var("STRIPE__API_KEY").is_ok() && 
               std::env::var("FEATURES__STRIPE_ENABLED").unwrap_or_else(|_| "false".to_string()) == "true" {
                println!("💳 Stripe integration enabled");
                app = app.route("/stripe/webhook", web::post().to(stripe_webhook));

                let stripe_scope = web::scope("/stripe")
                    .wrap(AuthMiddlewareFactory)
                    .route("/create-setup-intent", web::get().to(create_setup_intent))
                    .route("/create-session", web::get().to(create_stripe_session))
                    .route("/invoices", web::get().to(get_user_invoices))
                    .route("/invoice/{invoice_id}", web::get().to(get_invoice_detail))
                    .route("/checkout", web::post().to(create_checkout_session))
                    .route(
                        "/checkout/{session_id}",
                        web::get().to(get_checkout_session),
                    )
                    .route("/billing-portal", web::get().to(get_billing_portal_session));

                app = app.service(stripe_scope);
            } else {
                println!("ℹ️  Stripe integration disabled");
            }

            app.service(api_scope)
        })
        .bind("0.0.0.0:8000")?
        .keep_alive(timeout)
        .run()
        .await
    })
}
