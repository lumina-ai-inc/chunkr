use crate::models::auth::auth::UserInfo;
use crate::utils::configs::stripe_config::Config as StripeConfig;
use crate::utils::db::deadpool_postgres::Pool;
use crate::utils::stripe::stripe::{create_stripe_customer, create_stripe_setup_intent};
use actix_web::{web, Error, HttpRequest, HttpResponse};
use serde::Serialize;
use stripe::CustomerId;
#[derive(Serialize)]
pub struct SetupIntentResponse {
    customer_id: String,
    setup_intent: serde_json::Value,
}

pub async fn create_setup_intent(
    pool: web::Data<Pool>,
    user_info: web::ReqData<UserInfo>,
) -> Result<HttpResponse, Error> {
    let client = pool.get().await.map_err(|e| {
        eprintln!("Error connecting to database: {:?}", e);
        actix_web::error::ErrorInternalServerError("Database connection error")
    })?;
    let stripe_config = StripeConfig::from_env().map_err(|e| {
        eprintln!("Error loading Stripe configuration: {:?}", e);
        actix_web::error::ErrorInternalServerError("Configuration error")
    })?;
    // Check if customer_id exists for the user
    let row = client
        .query_opt(
            "SELECT customer_id FROM users WHERE user_id = $1",
            &[&user_info.user_id],
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
            let email = user_info
                .email
                .as_ref()
                .ok_or_else(|| actix_web::error::ErrorBadRequest("User email is required"))?;
            let new_customer_id = create_stripe_customer(email).await.map_err(|e| {
                eprintln!("Error creating Stripe customer: {:?}", e);
                actix_web::error::ErrorInternalServerError("Error creating Stripe customer")
            })?;

            // Update user with new customer_id
            client
                .execute(
                    "UPDATE users SET customer_id = $1 WHERE user_id = $2",
                    &[&new_customer_id, &user_info.user_id],
                )
                .await
                .map_err(|e| {
                    eprintln!("Error updating user with customer_id: {:?}", e);
                    actix_web::error::ErrorInternalServerError("Database error")
                })?;

            new_customer_id
        }
    };

    // Create Stripe setup intent
    let setup_intent = create_stripe_setup_intent(&customer_id, &stripe_config)
        .await
        .map_err(|e| {
            eprintln!("Error creating Stripe setup intent: {:?}", e);
            actix_web::error::ErrorInternalServerError("Error creating Stripe setup intent")
        })?;

    let response = SetupIntentResponse {
        customer_id,
        setup_intent,
    };

    Ok(HttpResponse::Ok().json(response))
}

pub async fn stripe_webhook(
    pool: web::Data<Pool>,
    req: HttpRequest,
    payload: web::Bytes,
) -> Result<HttpResponse, Error> {
    println!("stripewebhook");
    let stripe_config = StripeConfig::from_env().map_err(|e| {
        eprintln!("Error loading Stripe configuration: {:?}", e);
        actix_web::error::ErrorInternalServerError("Configuration error")
    })?;

    let payload = payload.to_vec();
    let sig_header = req
        .headers()
        .get("Stripe-Signature")
        .ok_or_else(|| actix_web::error::ErrorBadRequest("Missing Stripe-Signature header"))?
        .to_str()
        .map_err(|_| actix_web::error::ErrorBadRequest("Invalid Stripe-Signature header"))?;

    let endpoint_secret = stripe_config.webhook_secret;

    let payload_str = std::str::from_utf8(&payload)
        .map_err(|_| actix_web::error::ErrorBadRequest("Invalid payload"))?;

    let event = stripe::Webhook::construct_event(payload_str, &sig_header, &endpoint_secret)
        .map_err(|_| actix_web::error::ErrorBadRequest("Invalid webhook signature"))?;
    println!("Received event: {:?}", event);

    match event.type_ {
        stripe::EventType::SetupIntentCanceled => {
            if let stripe::EventObject::SetupIntent(setup_intent) = event.data.object {
                println!("Setup Intent Canceled: {:?}", setup_intent);
            }
        }
        stripe::EventType::SetupIntentCreated => {
            if let stripe::EventObject::SetupIntent(setup_intent) = event.data.object {
                println!("Setup Intent Created: {:?}", setup_intent);
            }
        }
        stripe::EventType::SetupIntentRequiresAction => {
            if let stripe::EventObject::SetupIntent(setup_intent) = event.data.object {
                println!("Setup Intent Requires Action: {:?}", setup_intent);
            }
        }
        stripe::EventType::SetupIntentSetupFailed => {
            if let stripe::EventObject::SetupIntent(setup_intent) = event.data.object {
                println!("Setup Intent Setup Failed: {:?}", setup_intent);
            }
        }
        stripe::EventType::SetupIntentSucceeded => {
            if let stripe::EventObject::SetupIntent(setup_intent) = event.data.object {
                println!("Setup Intent Succeeded: {:?}", setup_intent);
                let customer_id = match &setup_intent.customer {
                    Some(stripe::Expandable::Id(id)) => id.clone(),
                    Some(stripe::Expandable::Object(customer)) => customer.id.clone(),
                    None => {
                        return Err(actix_web::error::ErrorBadRequest("No customer ID found").into())
                    }
                };
                match upgrade_user(customer_id.to_string(), pool).await {
                    Ok(_) => println!("User upgrade successful"),
                    Err(e) => eprintln!("Error upgrading user: {:?}", e),
                }
            }
        }
        _ => {
            println!("Unhandled event type: {:?}", event.type_);
        }
    }

    Ok(HttpResponse::Ok().finish())
}
async fn upgrade_user(customer_id: String, pool: web::Data<Pool>) -> Result<HttpResponse, Error> {
    let client = pool.get().await.map_err(|e| {
        eprintln!("Error connecting to database: {:?}", e);
        actix_web::error::ErrorInternalServerError("Database connection error")
    })?;

    // Calculate remaining pages for each usage type and insert into discounts table
    let remaining_pages_query = "
        INSERT INTO discounts (user_id, usage_type, remaining_pages)
        SELECT user_id, usage_type, usage_limit - usage AS remaining_pages
        FROM USAGE
        WHERE user_id = $1
    ";
    client
        .execute(remaining_pages_query, &[&customer_id])
        .await
        .map_err(|e| {
            eprintln!("Error inserting into discounts table: {:?}", e);
            actix_web::error::ErrorInternalServerError("Error processing discount")
        })?;

    // Update USAGE table
    let update_usage_query = "
        UPDATE USAGE
        SET usage_limit = 100000, usage = 0
        WHERE user_id = $1
    ";
    client
        .execute(update_usage_query, &[&customer_id])
        .await
        .map_err(|e| {
            eprintln!("Error updating usage table: {:?}", e);
            actix_web::error::ErrorInternalServerError("Error updating usage")
        })?;

    // Update users table to change tier to 'Paid'
    let update_user_tier_query = "
        UPDATE users
        SET tier = 'Paid'
        WHERE customer_id = $1
    ";
    client
        .execute(update_user_tier_query, &[&customer_id])
        .await
        .map_err(|e| {
            eprintln!("Error updating users table: {:?}", e);
            actix_web::error::ErrorInternalServerError("Error updating user tier")
        })?;

    // Return a valid HttpResponse
    Ok(HttpResponse::Ok().finish())
}
