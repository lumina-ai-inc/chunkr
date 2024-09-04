use crate::utils::configs::stripe_config::Config as StripeConfig;
use crate::utils::db::deadpool_postgres::Pool;
use crate::utils::stripe::stripe::{create_stripe_customer, create_stripe_subscription};
use actix_web::{web, Error, HttpResponse};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub struct SubscriptionRequest {
    user_id: String,
    email: String,
}

#[derive(Serialize)]
pub struct SubscriptionResponse {
    customer_id: String,
    subscription: serde_json::Value,
}

pub async fn create_subscription(
    pool: web::Data<Pool>,
    stripe_config: web::Data<StripeConfig>,
    req: web::Json<SubscriptionRequest>,
) -> Result<HttpResponse, Error> {
    let client = pool.get().await.map_err(|e| {
        eprintln!("Error connecting to database: {:?}", e);
        actix_web::error::ErrorInternalServerError("Database connection error")
    })?;

    // Check if customer_id exists for the user
    let row = client
        .query_opt(
            "SELECT customer_id FROM users WHERE user_id = $1",
            &[&req.user_id],
        )
        .await
        .map_err(|e| {
            eprintln!("Database error: {:?}", e);
            actix_web::error::ErrorInternalServerError("Database error")
        })?;

    let customer_id = if let Some(row) = row {
        row.get::<_, Option<String>>("customer_id")
    } else {
        None
    };

    let customer_id = match customer_id {
        Some(id) if !id.is_empty() => id,
        _ => {
            // Create new Stripe customer
            let new_customer_id = create_stripe_customer(&req.email).await.map_err(|e| {
                eprintln!("Error creating Stripe customer: {:?}", e);
                actix_web::error::ErrorInternalServerError("Error creating Stripe customer")
            })?;

            // Update user with new customer_id
            client
                .execute(
                    "UPDATE users SET customer_id = $1 WHERE user_id = $2",
                    &[&new_customer_id, &req.user_id],
                )
                .await
                .map_err(|e| {
                    eprintln!("Error updating user with customer_id: {:?}", e);
                    actix_web::error::ErrorInternalServerError("Database error")
                })?;

            new_customer_id
        }
    };

    // Create Stripe subscription
    let subscription = create_stripe_subscription(&customer_id, &stripe_config)
        .await
        .map_err(|e| {
            eprintln!("Error creating Stripe subscription: {:?}", e);
            actix_web::error::ErrorInternalServerError("Error creating Stripe subscription")
        })?;

    let response = SubscriptionResponse {
        customer_id,
        subscription,
    };

    Ok(HttpResponse::Ok().json(response))
}
