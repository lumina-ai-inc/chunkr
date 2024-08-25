use crate::models::extraction::api::ApiInfo;
use crate::utils::db::deadpool_postgres::{Client, Pool};
use actix_web::{
    dev::{forward_ready, Service, ServiceRequest, ServiceResponse, Transform},
    web, Error, HttpMessage,
};
use futures_util::future::LocalBoxFuture;
use std::future::{ready, Ready};
use std::rc::Rc;

pub struct ApiKeyMiddlewareFactory;

impl<S, B> Transform<S, ServiceRequest> for ApiKeyMiddlewareFactory
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error> + 'static,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type InitError = ();
    type Transform = ApiKeyMiddleware<S>;
    type Future = Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        ready(Ok(ApiKeyMiddleware {
            service: Rc::new(service),
        }))
    }
}

pub struct ApiKeyMiddleware<S> {
    service: Rc<S>,
}

impl<S, B> Service<ServiceRequest> for ApiKeyMiddleware<S>
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error> + 'static,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type Future = LocalBoxFuture<'static, Result<Self::Response, Self::Error>>;

    forward_ready!(service);

    fn call(&self, req: ServiceRequest) -> Self::Future {
        let srv = self.service.clone();

        Box::pin(async move {
            let api_key = req
                .headers()
                .get("x-api-key")
                .and_then(|value| value.to_str().ok());

            if api_key.is_none() {
                actix_web::error::ErrorUnauthorized("API key is missing");
            }

            let pool = match req.app_data::<web::Data<Pool>>() {
                Some(pool) => pool.clone(),
                None => {
                    println!("Database pool not found");
                    return Err(actix_web::error::ErrorInternalServerError(
                        "Database pool not found",
                    ));
                }
            };
            let client = match pool.get().await {
                Ok(client) => client,
                Err(e) => {
                    eprintln!("Error getting Postgres client from pool: {:?}", e);
                    return Err(actix_web::error::ErrorInternalServerError(
                        "Failed to get client",
                    ));
                }
            };

            match validator(api_key.unwrap_or_default(), client).await {
                Ok(api_info) => {
                    req.extensions_mut().insert(api_info);
                    let res = srv.call(req).await?;
                    Ok(res)
                }
                Err(e) => Err(e),
            }
        })
    }
}

async fn validator(api_key: &str, client: Client) -> Result<ApiInfo, Error> {
    println!("Api key: {}", api_key);

    if api_key.is_empty() {
        return Err(actix_web::error::ErrorUnauthorized("API key is missing"));
    }

    // Query the database for the API key
    let row = match client
        .query_opt(
            "SELECT user_id FROM API_KEYS WHERE key = $1 AND active = true AND deleted = false",
            &[&api_key],
        )
        .await
    {
        Ok(row) => row,
        Err(e) => {
            eprintln!("Error querying API key: {:?}", e);
            return Err(actix_web::error::ErrorInternalServerError(
                "Failed to find API key",
            ));
        }
    };

    // Check if the API key is valid and retrieve the user_id
    let user_id = match row {
        Some(row) => row.get::<_, String>("user_id"),
        None => {
            eprintln!("Error: Invalid or inactive API key");
            return Err(actix_web::error::ErrorUnauthorized(
                "Invalid or inactive API key",
            ));
        }
    };

    // Attach the user_id to the request
    let api_info = ApiInfo {
        api_key: api_key.to_string(),
        user_id,
    };

    Ok(api_info)
}
