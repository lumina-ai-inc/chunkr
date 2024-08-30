use crate::models::auth::auth::{ UserInfo, Claims };
use crate::utils::configs::auth_config::Config;
use crate::utils::db::deadpool_postgres::Pool;
use actix_web::{
    dev::{ forward_ready, Service, ServiceRequest, ServiceResponse, Transform },
    web,
    Error,
    HttpMessage,
};
use futures_util::future::LocalBoxFuture;
use std::future::{ ready, Ready };
use std::rc::Rc;


pub struct ApiKeyMiddlewareFactory;

impl<S, B> Transform<S, ServiceRequest>
    for ApiKeyMiddlewareFactory
    where
        S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error> + 'static,
        S::Future: 'static,
        B: 'static
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type InitError = ();
    type Transform = ApiKeyMiddleware<S>;
    type Future = Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        ready(
            Ok(ApiKeyMiddleware {
                service: Rc::new(service),
            })
        )
    }
}

pub struct ApiKeyMiddleware<S> {
    service: Rc<S>,
}

impl<S, B> Service<ServiceRequest>
    for ApiKeyMiddleware<S>
    where
        S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error> + 'static,
        S::Future: 'static,
        B: 'static
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type Future = LocalBoxFuture<'static, Result<Self::Response, Self::Error>>;

    forward_ready!(service);

    fn call(&self, req: ServiceRequest) -> Self::Future {
        let srv = self.service.clone();

        Box::pin(async move {
            let authorization = req
                .headers()
                .get("Authorization")
                .and_then(|value| value.to_str().ok());

            if authorization.is_none() {
                return Err(actix_web::error::ErrorUnauthorized("Authorization header is missing"));
            }

            if authorization.unwrap().starts_with("Bearer ") {
                let token = authorization.unwrap().split("Bearer ").nth(1).unwrap();
                match bearer_token_validator(token).await {
                    Ok(user_info) => {
                        req.extensions_mut().insert(user_info);
                        let res = srv.call(req).await?;
                        Ok(res)
                    }
                    Err(e) => Err(e),
                }
            } else {
                
                match api_key_validator(authorization.unwrap_or_default(), &req).await {
                    Ok(user_info) => {
                        req.extensions_mut().insert(user_info);
                        let res = srv.call(req).await?;
                        Ok(res)
                    }
                    Err(e) => Err(e),
                }
            }
        })
    }
}

async fn bearer_token_validator(token: &str) -> Result<UserInfo, Error> {
    let config = Config::from_env().unwrap();
    let user_id = "1234".to_string();
    Ok(UserInfo {
        api_key: None,
        user_id,
    })
}

async fn api_key_validator(api_key: &str, req: &ServiceRequest) -> Result<UserInfo, Error> {
    if api_key.is_empty() {
        return Err(actix_web::error::ErrorUnauthorized("API key is missing"));
    }

    let pool = match req.app_data::<web::Data<Pool>>() {
        Some(pool) => pool.clone(),
        None => {
            println!("Database pool not found");
            return Err(
                actix_web::error::ErrorInternalServerError("Database pool not found")
            );
        }
    };
    let client = match pool.get().await {
        Ok(client) => client,
        Err(e) => {
            eprintln!("Error getting Postgres client from pool: {:?}", e);
            return Err(
                actix_web::error::ErrorInternalServerError("Failed to get client")
            );
        }
    };

    // Query the database for the API key
    let row = match
        client.query_opt(
            "SELECT user_id FROM API_KEYS WHERE key = $1 AND active = true AND deleted = false",
            &[&api_key]
        ).await
    {
        Ok(row) => row,
        Err(e) => {
            eprintln!("Error querying API key: {:?}", e);
            return Err(actix_web::error::ErrorInternalServerError("Failed to find API key"));
        }
    };

    // Check if the API key is valid and retrieve the user_id
    let user_id = match row {
        Some(row) => row.get::<_, String>("user_id"),
        None => {
            eprintln!("Error: Invalid or inactive API key");
            return Err(actix_web::error::ErrorUnauthorized("Invalid or inactive API key"));
        }
    };

    // Attach the user_id to the request
    let user_info = UserInfo {
        api_key: Some(api_key.to_string()),
        user_id,
    };

    Ok(user_info)
}
