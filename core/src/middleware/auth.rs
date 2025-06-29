use crate::configs::auth_config::Config;
use crate::models::auth::UserInfo;
use crate::utils::clients::get_pg_client;
use actix_web::{
    dev::{forward_ready, Service, ServiceRequest, ServiceResponse, Transform},
    Error, HttpMessage,
};
use futures_util::future::LocalBoxFuture;
use jsonwebtoken::{decode, Algorithm, DecodingKey, Validation};
use lazy_static::lazy_static;
use reqwest::Client;
use serde_json::Value;
use std::future::{ready, Ready};
use std::rc::Rc;
use std::sync::Arc;
use tokio::sync::OnceCell;

lazy_static! {
    static ref DECODING_KEY: Arc<OnceCell<DecodingKey>> = Arc::new(OnceCell::new());
}

async fn get_decoding_key() -> &'static DecodingKey {
    DECODING_KEY
        .get_or_init(|| async {
            let config = Config::from_env().expect("Failed to load auth config");
            let client = Client::new();
            let url = format!(
                "{}/realms/{}/protocol/openid-connect/certs",
                config.keycloak_url, config.keycloak_realm
            );

            let response = client
                .get(url)
                .send()
                .await
                .expect("Failed to fetch JWKS")
                .json::<Value>()
                .await
                .expect("Failed to parse JWKS response");

            let rs256_key = response["keys"]
                .as_array()
                .expect("JWKS keys should be an array")
                .iter()
                .find(|key| key["alg"] == "RS256")
                .expect("No RS256 key found in JWKS");

            DecodingKey::from_rsa_components(
                rs256_key["n"]
                    .as_str()
                    .expect("Missing 'n' component in RSA key"),
                rs256_key["e"]
                    .as_str()
                    .expect("Missing 'e' component in RSA key"),
            )
            .expect("Invalid RSA public key")
        })
        .await
}

pub struct AuthMiddlewareFactory;

impl<S, B> Transform<S, ServiceRequest> for AuthMiddlewareFactory
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error> + 'static,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type InitError = ();
    type Transform = AuthMiddleware<S>;
    type Future = Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        ready(Ok(AuthMiddleware {
            service: Rc::new(service),
        }))
    }
}

pub struct AuthMiddleware<S> {
    service: Rc<S>,
}

impl<S, B> Service<ServiceRequest> for AuthMiddleware<S>
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
            // don't validate CORS pre-flight requests
            if req.method() == "OPTIONS" {
                let res = srv.call(req).await?;
                return Ok(res);
            }
            let authorization = req
                .headers()
                .get("Authorization")
                .and_then(|value| value.to_str().ok());

            if authorization.is_none() {
                return Err(actix_web::error::ErrorUnauthorized(
                    "Authorization header is missing",
                ));
            }

            let user_info = if authorization.unwrap().starts_with("Bearer ") {
                let token = authorization.unwrap().split("Bearer ").nth(1).unwrap();
                bearer_token_validator(token).await
            } else {
                api_key_validator(authorization.unwrap_or_default()).await
            }?;

            user_info.add_attributes_to_ctx();
            req.extensions_mut().insert(user_info);
            srv.call(req).await
        })
    }
}

async fn bearer_token_validator(token: &str) -> Result<UserInfo, Error> {
    let mut validation = Validation::new(Algorithm::RS256);
    validation.validate_aud = false;

    let decoding_key = get_decoding_key().await;

    match decode::<Value>(token, decoding_key, &validation) {
        Ok(data) => {
            let user_info = UserInfo {
                api_key: None,
                user_id: data.claims["sub"].as_str().unwrap_or_default().to_string(),
                email: data.claims["email"].as_str().map(|s| s.to_string()),
                first_name: data.claims["given_name"].as_str().map(|s| s.to_string()),
                last_name: data.claims["family_name"].as_str().map(|s| s.to_string()),
            };
            Ok(user_info)
        }
        Err(err) => {
            eprintln!("Token validation error: {err:?}");
            Err(actix_web::error::ErrorUnauthorized("Invalid token payload"))
        }
    }
}

async fn api_key_validator(api_key: &str) -> Result<UserInfo, Error> {
    if api_key.is_empty() {
        return Err(actix_web::error::ErrorUnauthorized("API key is missing"));
    }

    let client = match get_pg_client().await {
        Ok(client) => client,
        Err(e) => {
            eprintln!("Error getting Postgres client from pool: {e:?}");
            return Err(actix_web::error::ErrorInternalServerError(
                "Failed to get client",
            ));
        }
    };

    // Query the database for the API key
    let row = match
        client.query_opt(
            "SELECT u.user_id, u.email, u.first_name, u.last_name FROM API_KEYS AS ak LEFT JOIN USERS AS u ON ak.user_id = u.user_id WHERE ak.key = $1 AND ak.active = true AND ak.deleted = false",
            &[&api_key]
        ).await
    {
        Ok(row) => row,
        Err(e) => {
            eprintln!("Failed to find API key: {e:?}");
            return Err(actix_web::error::ErrorInternalServerError("Failed to find API key"));
        }
    };

    // Check if the API key is valid and retrieve the user_id
    let row = match row {
        Some(row) => row,
        None => {
            eprintln!("Error: Invalid or inactive API key");
            return Err(actix_web::error::ErrorUnauthorized(
                "Invalid or inactive API key",
            ));
        }
    };

    // Attach the user_id to the request
    let user_info = UserInfo {
        api_key: Some(api_key.to_string()),
        user_id: row.get::<_, String>("user_id"),
        email: Some(row.get::<_, String>("email")),
        first_name: Some(row.get::<_, String>("first_name")),
        last_name: Some(row.get::<_, String>("last_name")),
    };

    Ok(user_info)
}
