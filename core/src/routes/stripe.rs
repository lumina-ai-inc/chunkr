use crate::configs::postgres_config::Pool;
use crate::configs::stripe_config::Config;
use crate::models::chunkr::auth::UserInfo;
use crate::models::chunkr::user::{InvoiceStatus, Tier, UsageType};
use crate::utils::routes::get_user::get_monthly_usage_count;
use crate::utils::routes::get_user::{get_invoice_information, get_invoices};
use crate::utils::stripe::stripe_utils::{
    cancel_stripe_subscription, create_customer_session, create_stripe_customer,
    create_stripe_invoice_for_overage, create_stripe_setup_intent, create_stripe_subscription,
    set_default_payment_method, update_invoice_status,
};
use actix_web::{web, Error, HttpRequest, HttpResponse};
use postgres_types::ToSql;
use serde::{Deserialize, Serialize};
use stripe::EventType;
use stripe::InvoiceStatus as StripeInvoiceStatus;
use utoipa::ToSchema;
use uuid::Uuid;

#[derive(Serialize)]
pub struct SetupIntentResponse {
    customer_id: String,
    setup_intent: serde_json::Value,
}

#[derive(Deserialize, ToSchema, ToSql, Debug)]
pub struct SubscriptionRequest {
    pub tier: String,            // e.g. "Starter", "Dev", etc.
    pub stripe_price_id: String, // The corresponding Stripe Price ID for the tier
}

pub async fn create_setup_intent(
    pool: web::Data<Pool>,
    user_info: web::ReqData<UserInfo>,
) -> Result<HttpResponse, Error> {
    let client = pool.get().await.map_err(|e| {
        eprintln!("Error connecting to database: {:?}", e);
        actix_web::error::ErrorInternalServerError("Database connection error")
    })?;
    let stripe_config = Config::from_env().map_err(|e| {
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
    let stripe_config = Config::from_env().map_err(|e| {
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
    let stripe_config = Config::from_env().map_err(|e| {
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

    let client = pool.get().await.map_err(|e| {
        eprintln!("DB connect error: {:?}", e);
        actix_web::error::ErrorInternalServerError("DB error")
    })?;
    // TODO: update and downgrade subscription
    match event.type_ {
        EventType::CustomerSubscriptionCreated => {
            if let stripe::EventObject::Subscription(sub) = event.data.object {
                let user_id_opt = find_user_id_by_customer(&sub.customer, &client).await;
                if let Some(user_id) = user_id_opt {
                    // For demonstration, we set last_paid_status = 'True' if it's active
                    let stripe_sub_id = sub.id.to_string();
                    let status_str = sub.status.clone();
                    let last_paid_status = match status_str.as_str() {
                        "active" | "trialing" => "True",
                        _ => "False",
                    };
                    // Update DB
                    let _ = client
                        .execute(
                            "WITH sub_update AS (
                                UPDATE subscriptions
                                SET stripe_subscription_id = $1,
                                    last_paid_status = $2, 
                                    updated_at = NOW()
                                WHERE user_id = $3
                                RETURNING tier
                            )
                            UPDATE users u
                            SET tier = s.tier
                            FROM sub_update s
                            WHERE u.user_id = $3;

                            INSERT INTO monthly_usage (
                                user_id, usage_type, usage, overage_usage, 
                                year, month, tier, usage_limit
                            )
                            SELECT 
                                $3, 'Page', 0, 0,
                                EXTRACT(YEAR FROM CURRENT_TIMESTAMP),
                                EXTRACT(MONTH FROM CURRENT_TIMESTAMP),
                                t.tier,
                                t.usage_limit
                            FROM tiers t
                            JOIN sub_update s ON s.tier = t.tier
                            ON CONFLICT (user_id, usage_type, year, month) 
                            DO UPDATE SET
                                tier = EXCLUDED.tier,
                                usage_limit = EXCLUDED.usage_limit",
                            &[&stripe_sub_id, &last_paid_status, &user_id],
                        )
                        .await;
                }
            }
        }
        EventType::CustomerSubscriptionUpdated => {
            if let stripe::EventObject::Subscription(sub) = event.data.object {
                let user_id_opt = find_user_id_by_customer(&sub.customer, &client).await;
                if let Some(user_id) = user_id_opt {
                    // Extract the single subscription item's price ID
                    // (assuming you only have one active item).
                    let items = sub.items.data; // Access data directly since items is not Option
                    if let Some(item) = items.get(0) {
                        if let Some(price) = &item.price {
                            let price_id = price.id.clone();
                            // match price_id to local tier ...
                            let new_tier = match price_id.as_str() {
                                x if x == std::env::var("STARTER_PRICE_ID").unwrap_or_default() => {
                                    "Starter"
                                }
                                x if x == std::env::var("DEV_PRICE_ID").unwrap_or_default() => {
                                    "Dev"
                                }
                                x if x == std::env::var("TEAM_PRICE_ID").unwrap_or_default() => {
                                    "Team"
                                }
                                _ => "Free", // or "Unknown"
                            };

                            // Then set last_paid_status if sub is active/trialing
                            let status_str = sub.status.clone();
                            let last_paid_status = match status_str.as_str() {
                                "active" | "trialing" => "True",
                                _ => "False",
                            };

                            // Update DB
                            let _ = client
                                .execute(
                                    "WITH sub_update AS (
                                    UPDATE subscriptions
                                    SET stripe_subscription_id = $1,
                                        last_paid_status = $2,
                                        tier = $3,
                                        updated_at = NOW()
                                    WHERE user_id = $4
                                    RETURNING tier
                                )
                                UPDATE users
                                SET tier = s.tier
                                FROM sub_update s
                                WHERE users.user_id = $4;",
                                    &[&sub.id.to_string(), &last_paid_status, &new_tier, &user_id],
                                )
                                .await;

                            // Optionally also update monthly_usage if needed.
                        }
                    }
                }
            }
        }
        EventType::CustomerSubscriptionDeleted => {
            if let stripe::EventObject::Subscription(sub) = event.data.object {
                let user_id_opt = find_user_id_by_customer(&sub.customer, &client).await;
                if let Some(user_id) = user_id_opt {
                    // Mark subscription row canceled or remove entirely
                    let _ = client
                        .execute(
                            "UPDATE subscriptions
                         SET last_paid_status = 'Canceled',
                             updated_at = NOW()
                         WHERE user_id = $1",
                            &[&user_id],
                        )
                        .await;
                }
            }
        }
        EventType::InvoicePaid => {
            if let stripe::EventObject::Invoice(invoice) = event.data.object {
                let stripe_invoice_id = invoice.id;
                let status = invoice.status.clone();
                let invoice_status = match status {
                    Some(StripeInvoiceStatus::Paid) => InvoiceStatus::Paid.to_string(),
                    Some(StripeInvoiceStatus::Open) => InvoiceStatus::Ongoing.to_string(),
                    Some(StripeInvoiceStatus::Void) => InvoiceStatus::Canceled.to_string(),
                    Some(StripeInvoiceStatus::Uncollectible) => {
                        InvoiceStatus::NoInvoice.to_string()
                    }
                    _ => "Unknown".to_string(),
                };
                // Update local invoice row
                let _ = update_invoice_status(&stripe_invoice_id, &invoice_status, &pool).await;
            }
        }
        EventType::InvoicePaymentFailed => {
            if let stripe::EventObject::Invoice(invoice) = event.data.object {
                let stripe_invoice_id = invoice.id;
                let _ = update_invoice_status(
                    &stripe_invoice_id,
                    &InvoiceStatus::PastDue.to_string(),
                    &pool,
                )
                .await;
            }
        }
        EventType::InvoicePaymentSucceeded => {
            if let stripe::EventObject::Invoice(invoice) = event.data.object {
                let stripe_invoice_id = invoice.id;
                let status_str = match invoice.status.clone() {
                    Some(StripeInvoiceStatus::Paid) => InvoiceStatus::Paid.to_string(),
                    Some(StripeInvoiceStatus::Open) => InvoiceStatus::Ongoing.to_string(),
                    Some(StripeInvoiceStatus::Void) => InvoiceStatus::Canceled.to_string(),
                    Some(StripeInvoiceStatus::Uncollectible) => {
                        InvoiceStatus::NoInvoice.to_string()
                    }
                    _ => "Unknown".to_string(),
                };
                // update local DB
                let _ = update_invoice_status(&stripe_invoice_id, &status_str, &pool).await;
            }
        }
        _ => {
            println!("Unhandled event type in webhook: {:?}", event.type_);
        }
    }

    // Return 200 OK so Stripe knows we processed the webhook
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
) -> Result<HttpResponse, actix_web::Error> {
    let user_id = user_info.user_id.clone();
    let invoices = get_invoices(user_id).await?;
    Ok(HttpResponse::Ok().json(invoices))
}

// Define a route to get detailed information for a specific invoice
pub async fn get_invoice_detail(
    invoice_id: web::Path<String>,
) -> Result<HttpResponse, actix_web::Error> {
    let invoice_detail = get_invoice_information(invoice_id.into_inner()).await?;
    Ok(HttpResponse::Ok().json(invoice_detail))
}

pub async fn get_monthly_usage(
    user_info: web::ReqData<UserInfo>,
) -> Result<HttpResponse, actix_web::Error> {
    let user_id = user_info.user_id.clone();
    let monthly_usage = get_monthly_usage_count(user_id).await?;
    Ok(HttpResponse::Ok().json(monthly_usage))
}

/// POST /subscribe
/// Creates (or updates) a user subscription in Stripe and stores it in the local subscriptions table.
pub async fn subscribe_user(
    pool: web::Data<Pool>,
    user_info: web::ReqData<UserInfo>,
    form: web::Json<SubscriptionRequest>,
) -> Result<HttpResponse, Error> {
    // 1) Ensure we have a valid DB client
    let mut client = pool.get().await.map_err(|e| {
        eprintln!("Error connecting to DB: {:?}", e);
        actix_web::error::ErrorInternalServerError("DB Connection Error")
    })?;

    // 2) Load or create Stripe Customer
    let stripe_config = Config::from_env().map_err(|e| {
        eprintln!("Stripe configuration error: {:?}", e);
        actix_web::error::ErrorInternalServerError("Stripe Config Error")
    })?;
    let user_id = &user_info.user_id;

    let row = client
        .query_opt(
            "SELECT customer_id FROM users WHERE user_id = $1",
            &[user_id],
        )
        .await
        .map_err(|_| actix_web::error::ErrorInternalServerError("Failed to query user"))?;

    let customer_id = match row {
        Some(r) => {
            let cid: Option<String> = r.get("customer_id");
            if let Some(c) = cid {
                c
            } else {
                // Create a new Stripe customer
                let email = user_info.email.clone().unwrap_or_default();
                let ncid = create_stripe_customer(&email).await.map_err(|e| {
                    eprintln!("Stripe create customer error: {:?}", e);
                    actix_web::error::ErrorInternalServerError("Could not create stripe customer")
                })?;
                client
                    .execute(
                        "UPDATE users SET customer_id = $1 WHERE user_id = $2",
                        &[&ncid, user_id],
                    )
                    .await
                    .map_err(|e| {
                        eprintln!("Failed to update user with new customer_id: {:?}", e);
                        actix_web::error::ErrorInternalServerError("DB error updating user")
                    })?;
                ncid
            }
        }
        None => {
            // No record found, create new customer
            let email = user_info.email.clone().unwrap_or_default();
            let ncid = create_stripe_customer(&email).await.map_err(|e| {
                eprintln!("Stripe create customer error: {:?}", e);
                actix_web::error::ErrorInternalServerError("Could not create stripe customer")
            })?;
            client
                .execute(
                    "UPDATE users SET customer_id = $1 WHERE user_id = $2",
                    &[&ncid, user_id],
                )
                .await
                .map_err(|e| {
                    eprintln!("Failed to update user with new customer_id: {:?}", e);
                    actix_web::error::ErrorInternalServerError("DB error updating user")
                })?;
            ncid
        }
    };

    // 3) Create or Update the subscription in Stripe
    let subscription_json = create_stripe_subscription(
        &customer_id,
        &form.stripe_price_id, // e.g. "price_1234"
        &stripe_config,
    )
    .await
    .map_err(|e| {
        eprintln!("Error creating subscription in stripe: {:?}", e);
        actix_web::error::ErrorInternalServerError("Could not create subscription")
    })?;

    let stripe_subscription_id = subscription_json["id"]
        .as_str()
        .unwrap_or_default()
        .to_string();

    // 4) Insert or update your local `subscriptions` table
    //    Example: We store a row keyed by subscription_id = user_id (or random).
    //    If you prefer each user can have multiple subscriptions, store them separately with unique IDs.
    let local_subscription_id = Uuid::new_v4().to_string();
    client
        .execute(
            r#"
            INSERT INTO subscriptions (subscription_id, stripe_subscription_id, user_id, tier, last_paid_date, last_paid_status)
            VALUES ($1, $2, $3, $4, NOW(), 'True')
            ON CONFLICT (user_id) DO UPDATE
              SET stripe_subscription_id = EXCLUDED.stripe_subscription_id,
                  tier = EXCLUDED.tier,
                  updated_at = NOW()
            "#,
            &[
                &local_subscription_id,
                &stripe_subscription_id,
                user_id,
                &form.tier,
            ],
        )
        .await
        .map_err(|e| {
            eprintln!("Failed inserting subscription locally: {:?}", e);
            actix_web::error::ErrorInternalServerError("DB error inserting subscription")
        })?;

    // 5) Optionally set the new default payment method from user's setup if desired
    //    e.g., if the user just added a card via setup_intent, etc.
    if let Err(e) = set_default_payment_method(&customer_id, &stripe_config).await {
        eprintln!("Could not set default payment method: {:?}", e);
    }

    // 6) Return subscription object from Stripe
    Ok(HttpResponse::Ok().json(subscription_json))
}

/// DELETE /subscribe
/// Cancel the user's subscription in Stripe and mark it canceled in your local DB.
pub async fn cancel_subscription(
    pool: web::Data<Pool>,
    user_info: web::ReqData<UserInfo>,
) -> Result<HttpResponse, Error> {
    let mut client = pool.get().await.map_err(|e| {
        eprintln!("DB connection error: {:?}", e);
        actix_web::error::ErrorInternalServerError("DB connection error")
    })?;

    let stripe_config = Config::from_env().map_err(|e| {
        eprintln!("Stripe config error: {:?}", e);
        actix_web::error::ErrorInternalServerError("Stripe Config Error")
    })?;

    // Get the user's subscription
    let subscription_row = client
        .query_opt(
            "SELECT stripe_subscription_id FROM subscriptions WHERE user_id = $1",
            &[&user_info.user_id],
        )
        .await
        .map_err(|e| {
            eprintln!("Query error: {:?}", e);
            actix_web::error::ErrorInternalServerError("Failed to query subscription")
        })?;

    if let Some(row) = subscription_row {
        let stripe_sub_id: String = row.get("stripe_subscription_id");
        // Cancel in Stripe
        let canceled_sub = cancel_stripe_subscription(&stripe_sub_id, false, false, &stripe_config)
            .await
            .map_err(|e| {
                eprintln!("Failed to cancel subscription in Stripe: {:?}", e);
                actix_web::error::ErrorInternalServerError("Stripe cancel subscription failed")
            })?;

        // Mark local subscription as canceled
        client
            .execute(
                "UPDATE subscriptions SET last_paid_status = 'Canceled', updated_at = NOW() WHERE user_id = $1",
                &[&user_info.user_id],
            )
            .await
            .map_err(|e| {
                eprintln!("Failed to update local subscription: {:?}", e);
                actix_web::error::ErrorInternalServerError("DB update failed")
            })?;

        return Ok(HttpResponse::Ok().json(canceled_sub));
    }

    Ok(HttpResponse::NotFound().json("No active subscription found"))
}

/// Helper to find user_id from a Stripe "Expandable<Customer>".
/// Returns None if no match found.
async fn find_user_id_by_customer(
    customer_expandable: &stripe::Expandable<stripe::Customer>,
    client: &tokio_postgres::Client,
) -> Option<String> {
    if let stripe::Expandable::Id(c_id) = customer_expandable {
        if let Ok(row_opt) = client
            .query_opt(
                "SELECT user_id FROM users WHERE customer_id = $1",
                &[&c_id.to_string()],
            )
            .await
        {
            if let Some(row) = row_opt {
                return Some(row.get("user_id"));
            }
        }
    }
    None
}
