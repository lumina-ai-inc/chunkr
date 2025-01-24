use actix_cors::Cors;
use actix_web::{App, HttpServer, middleware::Logger, web};
use deadpool_postgres::{Manager, Pool, tokio_postgres::NoTls};
use std::env;

pub mod models;
pub mod queries;
pub mod routes;
mod middleware;
use middleware::auth::ApiKeyMiddlewareFactory;


#[actix_web::main]
pub async fn main() -> std::io::Result<()> {
    dotenv::dotenv().ok();
    env_logger::init();

    let db_url = env::var("DATABASE_URL").expect("DATABASE_URL not set");
    let config = db_url.parse().expect("Invalid DB URL");
    let manager = Manager::new(config, NoTls);
    let pool = Pool::builder(manager)
        .max_size(10)
        .build()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    HttpServer::new(move || {
        App::new()
            .wrap(Cors::permissive())
            .wrap(Logger::default())
            .app_data(web::Data::new(pool.clone()))
            .service(
                web::scope("")
                    .wrap(ApiKeyMiddlewareFactory)
                    .configure(crate::routes::routes::configure_routes)
            )
    })
    .bind(("0.0.0.0", 8000))?
    .run()
    .await
}