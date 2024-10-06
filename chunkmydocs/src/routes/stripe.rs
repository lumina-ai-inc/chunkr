use crate::models::auth::auth::UserInfo;
use crate::models::server::user::{InvoiceStatus, Tier, UsageType};
use crate::utils::configs::stripe_config::Config as StripeConfig;
use crate::utils::db::deadpool_postgres::Pool;
use crate::utils::server::get_user::get_monthly_usage_count;
use crate::utils::server::get_user::{get_invoice_information, get_invoices};
use crate::utils::stripe::stripe::{
    create_customer_session, create_stripe_customer, create_stripe_setup_intent,
    set_default_payment_method, update_invoice_status,
};
use actix_web::{web, Error, HttpRequest, HttpResponse};
use serde::Serialize;
use stripe::InvoiceStatus as StripeInvoiceStatus;
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

pub async fn create_stripe_session(
    pool: web::Data<Pool>,
    user_info: web::ReqData<UserInfo>,
) -> Result<HttpResponse, Error> {
    let stripe_config = StripeConfig::from_env().map_err(|e| {
        eprintln!("Error loading Stripe configuration: {:?}", e);
        actix_web::error::ErrorInternalServerError("Configuration error")
    })?;
    let client = pool.get().await.map_err(|e| {
        eprintln!("Error connecting to database: {:?}", e);
        actix_web::error::ErrorInternalServerError("Database connection error")
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
    let session = create_customer_session(&customer_id, &stripe_config)
        .await
        .map_err(|e| {
            eprintln!("Error creating Stripe session: {:?}", e);
            actix_web::error::ErrorInternalServerError("Error creating Stripe session")
        })?;

    Ok(HttpResponse::Ok().json(session))
}

pub async fn stripe_webhook(
    pool: web::Data<Pool>,
    req: HttpRequest,
    payload: web::Bytes,
) -> Result<HttpResponse, Error> {
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

    let endpoint_secret = stripe_config.clone().webhook_secret;

    let payload_str = std::str::from_utf8(&payload)
        .map_err(|_| actix_web::error::ErrorBadRequest("Invalid payload"))?;

    let event = stripe::Webhook::construct_event(payload_str, &sig_header, &endpoint_secret)
        .map_err(|_| actix_web::error::ErrorBadRequest("Invalid webhook signature"))?;

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
                match set_default_payment_method(&customer_id.to_string(), &stripe_config).await {
                    Ok(_) => println!("Default payment method set successfully"),
                    Err(e) => eprintln!("Error setting default payment method: {:?}", e),
                }
            }
        }

        stripe::EventType::InvoicePaid => {
            if let stripe::EventObject::Invoice(invoice) = event.data.object {
                println!("Invoice Paid: {:?}", invoice);
                let invoice_id = invoice.id.clone();
                let status: Option<StripeInvoiceStatus> = invoice.status.clone();
                let status_str = match status {
                    Some(StripeInvoiceStatus::Paid) => InvoiceStatus::Paid.to_string(),
                    Some(StripeInvoiceStatus::Open) => InvoiceStatus::Ongoing.to_string(),
                    Some(StripeInvoiceStatus::Void) => InvoiceStatus::Canceled.to_string(),
                    Some(StripeInvoiceStatus::Uncollectible) => {
                        InvoiceStatus::NoInvoice.to_string()
                    }
                    _ => {
                        return Err(
                            actix_web::error::ErrorBadRequest("Invalid invoice status").into()
                        )
                    }
                };
                update_invoice_status(&invoice_id.to_string(), &status_str, &pool).await?;
            }
        }
        stripe::EventType::InvoicePaymentActionRequired => {
            if let stripe::EventObject::Invoice(invoice) = event.data.object {
                println!("Invoice Payment Action Required: {:?}", invoice);
                let invoice_id = invoice.id.clone();
                let status: Option<StripeInvoiceStatus> = invoice.status.clone();
                let status_str = match status {
                    Some(StripeInvoiceStatus::Open) => InvoiceStatus::NeedsAction.to_string(),
                    Some(StripeInvoiceStatus::Uncollectible) => {
                        InvoiceStatus::NeedsAction.to_string()
                    }
                    _ => {
                        return Err(
                            actix_web::error::ErrorBadRequest("Invalid invoice status").into()
                        )
                    }
                };
                update_invoice_status(&invoice_id.to_string(), &status_str, &pool).await?;
            }
        }
        stripe::EventType::InvoicePaymentFailed => {
            if let stripe::EventObject::Invoice(invoice) = event.data.object {
                println!("Invoice Payment Failed: {:?}", invoice);
                let invoice_id = invoice.id.clone();
                let status_str = InvoiceStatus::PastDue.to_string();
                update_invoice_status(&invoice_id.to_string(), &status_str, &pool).await?;
            }
        }
        stripe::EventType::InvoicePaymentSucceeded => {
            if let stripe::EventObject::Invoice(invoice) = event.data.object {
                println!("Invoice Payment Succeeded: {:?}", invoice);
                let invoice_id = invoice.id.clone();
                let status: Option<StripeInvoiceStatus> = invoice.status.clone();
                let status_str = match status {
                    Some(StripeInvoiceStatus::Paid) => InvoiceStatus::Paid.to_string(),
                    Some(StripeInvoiceStatus::Open) => InvoiceStatus::Ongoing.to_string(),
                    Some(StripeInvoiceStatus::Void) => InvoiceStatus::Canceled.to_string(),
                    Some(StripeInvoiceStatus::Uncollectible) => {
                        InvoiceStatus::NoInvoice.to_string()
                    }
                    _ => {
                        return Err(
                            actix_web::error::ErrorBadRequest("Invalid invoice status").into()
                        )
                    }
                };
                update_invoice_status(&invoice_id.to_string(), &status_str, &pool).await?;
            }
        } //update USERS and INVOICES table
        _ => {
            println!("Unhandled event type: {:?}", event.type_);
        } //do these... test invoice...
    }

    Ok(HttpResponse::Ok().finish())
}
async fn upgrade_user(customer_id: String, pool: web::Data<Pool>) -> Result<HttpResponse, Error> {
    let client = pool.get().await.map_err(|e| {
        eprintln!("Error connecting to database: {:?}", e);
        actix_web::error::ErrorInternalServerError("Database connection error")
    })?;
    let tier_query = "
        SELECT tier
        FROM users
        WHERE customer_id = $1
    ";
    let row = client
        .query_one(tier_query, &[&customer_id])
        .await
        .map_err(|e| {
            eprintln!("Error querying tier from users table: {:?}", e);
            actix_web::error::ErrorInternalServerError("Error querying user tier")
        })?;
    let user_tier: String = row.get("tier");

    if user_tier == "PayAsYouGo" || user_tier == "SelfHosted" {
        return Ok(HttpResponse::Ok().body("User does not need to be upgraded"));
    }

    let user_id_query = "
        SELECT user_id
        FROM users
        WHERE customer_id = $1
    ";
    let row = client
        .query_one(user_id_query, &[&customer_id])
        .await
        .map_err(|e| {
            eprintln!("Error querying user_id from users table: {:?}", e);
            actix_web::error::ErrorInternalServerError("Error querying user_id")
        })?;
    let user_id: String = row.get("user_id");
    // Calculate remaining pages for each usage type and insert into discounts table

    let remaining_pages_query = "
    INSERT INTO discounts (user_id, usage_type, amount)
    SELECT user_id, usage_type, usage_limit AS amount
    FROM USAGE
    WHERE user_id = $1;
    ";
    client
        .execute(remaining_pages_query, &[&user_id])
        .await
        .map_err(|e| {
            eprintln!("Error inserting into discounts table: {:?}", e);
            actix_web::error::ErrorInternalServerError("Error processing discount")
        })?;

    // Update USAGE table with new usage limits for PayAsYouGo tier
    let update_fast_usage_query = "
        UPDATE USAGE
        SET usage_limit = $1::integer
        WHERE user_id = $2 AND usage_type = 'Fast';
    ";
    let update_high_quality_usage_query = "
        UPDATE USAGE
        SET usage_limit = $1::integer
        WHERE user_id = $2 AND usage_type = 'HighQuality';
    ";
    let update_segment_usage_query = "
        UPDATE USAGE
        SET usage_limit = $1::integer
        WHERE user_id = $2 AND usage_type = 'Segment';
    ";

    let fast_limit = UsageType::Fast.get_usage_limit(&Tier::PayAsYouGo) as i32;
    let high_quality_limit = UsageType::HighQuality.get_usage_limit(&Tier::PayAsYouGo) as i32;
    let segment_limit = UsageType::Segment.get_usage_limit(&Tier::PayAsYouGo) as i32;

    println!("Updating fast usage for customer_id: {}", customer_id);
    client
        .execute(update_fast_usage_query, &[&fast_limit, &user_id])
        .await
        .map_err(|e| {
            eprintln!("Error updating fast usage table: {:?}", e);
            actix_web::error::ErrorInternalServerError("Error updating fast usage")
        })?;
    println!(
        "Successfully updated fast usage for customer_id: {}",
        user_id
    );

    println!("Updating high quality usage for customer_id: {}", user_id);
    client
        .execute(
            update_high_quality_usage_query,
            &[&high_quality_limit, &user_id],
        )
        .await
        .map_err(|e| {
            eprintln!("Error updating high quality usage table: {:?}", e);
            actix_web::error::ErrorInternalServerError("Error updating high quality usage")
        })?;
    println!(
        "Successfully updated high quality usage for customer_id: {}",
        user_id
    );

    println!("Updating segment usage for customer_id: {}", user_id);
    client
        .execute(update_segment_usage_query, &[&segment_limit, &user_id])
        .await
        .map_err(|e| {
            eprintln!("Error updating segment usage table: {:?}", e);
            actix_web::error::ErrorInternalServerError("Error updating segment usage")
        })?;
    println!(
        "Successfully updated segment usage for customer_id: {}",
        user_id
    );

    // Update users table to change tier to 'PayAsYouGo'
    let update_user_tier_query = "
        UPDATE users
        SET tier = 'PayAsYouGo'
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
// Define a route to get invoices for a user
pub async fn get_user_invoices(
    user_info: web::ReqData<UserInfo>,
    pool: web::Data<Pool>,
) -> Result<HttpResponse, actix_web::Error> {
    let user_id = user_info.user_id.clone();
    let invoices = get_invoices(user_id, &pool).await?;
    Ok(HttpResponse::Ok().json(invoices))
}

// Define a route to get detailed information for a specific invoice
pub async fn get_invoice_detail(
    invoice_id: web::Path<String>,
    pool: web::Data<Pool>,
) -> Result<HttpResponse, actix_web::Error> {
    let invoice_detail = get_invoice_information(invoice_id.into_inner(), &pool).await?;
    Ok(HttpResponse::Ok().json(invoice_detail))
}
pub async fn get_monthly_usage(
    user_info: web::ReqData<UserInfo>,
    pool: web::Data<Pool>,
) -> Result<HttpResponse, actix_web::Error> {
    let user_id = user_info.user_id.clone();
    let monthly_usage = get_monthly_usage_count(user_id, &pool).await?;
    Ok(HttpResponse::Ok().json(monthly_usage))
}
