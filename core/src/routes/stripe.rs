use crate::utils::clients::get_pg_client;
use crate::configs::stripe_config::Config;
use crate::models::chunkr::auth::UserInfo;
use crate::models::chunkr::user::{InvoiceStatus, Tier, UsageType};
use crate::utils::routes::get_user::get_monthly_usage_count;
use crate::utils::routes::get_user::{get_invoice_information, get_invoices};
use crate::utils::stripe::stripe_utils::{
    cancel_stripe_subscription, create_customer_session, create_stripe_customer,
create_stripe_setup_intent, create_stripe_subscription,
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
    pub tier: String, // e.g. "Starter", "Dev", etc.
}

pub async fn create_setup_intent(
    user_info: web::ReqData<UserInfo>,
) -> Result<HttpResponse, Error> {
    let client = get_pg_client().await.map_err(|e| {
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
    user_info: web::ReqData<UserInfo>,
) -> Result<HttpResponse, Error> {
    let stripe_config = Config::from_env().map_err(|e| {
        eprintln!("Error loading Stripe configuration: {:?}", e);
        actix_web::error::ErrorInternalServerError("Configuration error")
    })?;
    let client = get_pg_client().await.map_err(|e| {
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

    let client = get_pg_client().await.map_err(|e| {
        eprintln!("DB connect error: {:?}", e);
        actix_web::error::ErrorInternalServerError("DB error")
    })?;


    match event.type_ {
        EventType::CustomerSubscriptionCreated => {
            if let stripe::EventObject::Subscription(sub) = event.data.object {
                let user_id_opt: Option<String> = find_user_id_by_customer(&sub.customer, &client).await;
                if let Some(user_id) = user_id_opt {
                    let stripe_sub_id = sub.id.to_string();
                    let last_paid_status = match sub.status.as_str() {
                        "active" | "trialing" => "True",
                        _ => "False",
                    };
                    let tier = if let Some(item) = sub.items.data.get(0) {
                        if let Some(price) = &item.price {
                            match price.id.as_str() {
                                x if x == stripe_config.starter_price_id => "Starter",
                                x if x == stripe_config.dev_price_id => "Dev",
                                x if x == stripe_config.team_price_id => "Team",
                                _ => "Free"
                            }
                        } else {
                            "Free"
                        }
                    } else {
                        "Free"
                    };
                    let res_one = client
                        .execute(
                            "WITH sub_upsert AS (
                                INSERT INTO subscriptions (
                                    stripe_subscription_id,
                                    user_id,
                                    tier,
                                    last_paid_status,
                                    last_paid_date,
                                    created_at,
                                    updated_at
                                )
                                VALUES ($1, $3, $4, $2, NOW(), NOW(), NOW())
                                ON CONFLICT (user_id) DO UPDATE
                                SET stripe_subscription_id = $1,
                                    last_paid_status = $2,
                                    tier = $4,
                                    last_paid_date = NOW(),
                                    updated_at = NOW()
                                RETURNING tier
                            )
                            UPDATE users u
                            SET tier = s.tier,
                                invoice_status = $2
                            FROM sub_upsert s
                            WHERE u.user_id = $3",
                            &[&stripe_sub_id, &last_paid_status, &user_id, &tier],
                        )
                        .await;

                    if let Err(e) = res_one {
                        eprintln!("Error syncing subscriptions/users: {:?}", e);
                    }

                    let res_two = client
                        .execute(
                            "WITH new_data AS (
                                SELECT
                                    $1 as user_id, 
                                    'Page' as usage_type, 
                                    0 as usage, 
                                    0 as overage_usage,
                                    EXTRACT(YEAR FROM CURRENT_TIMESTAMP) as year,
                                    EXTRACT(MONTH FROM CURRENT_TIMESTAMP) as month,
                                    $2 as tier,
                                    t.usage_limit
                                FROM tiers t
                                WHERE t.tier = $2
                            )
                            INSERT INTO monthly_usage (
                                user_id, usage_type, usage, overage_usage,
                                year, month, tier, usage_limit
                            )
                            SELECT * FROM new_data
                            ON CONFLICT (user_id, usage_type, year, month)
                            DO UPDATE
                            SET tier = EXCLUDED.tier,
                                usage_limit = EXCLUDED.usage_limit",
                            &[&user_id, &tier],
                        )
                        .await;

                    if let Err(e) = res_two {
                        eprintln!("Error syncing monthly_usage: {:?}", e);
                    }
                }
            }
        }
        EventType::CustomerSubscriptionUpdated => {
            if let stripe::EventObject::Subscription(sub) = event.data.object {
                let user_id_opt = find_user_id_by_customer(&sub.customer, &client).await;
                if let Some(user_id) = user_id_opt {
                    if let Some(item) = sub.items.data.get(0) {
                        if let Some(price) = &item.price {
                            let new_tier = match price.id.as_str() {
                                x if x == stripe_config.starter_price_id => "Starter",
                                x if x == stripe_config.dev_price_id => "Dev", 
                                x if x == stripe_config.team_price_id => "Team",
                                _ => "Free"
                            };
                            let last_paid_status = match sub.status.as_str() {
                                "active" | "trialing" => "True",
                                _ => "False",
                            };

                            let res_one = client
                                .execute(
                                    "WITH sub_upsert AS (
                                        INSERT INTO subscriptions (
                                            stripe_subscription_id,
                                            user_id,
                                            tier,
                                            last_paid_status,
                                            last_paid_date,
                                            created_at,
                                            updated_at
                                        )
                                        VALUES ($1, $4, $3, $2, NOW(), NOW(), NOW())
                                        ON CONFLICT (user_id) DO UPDATE
                                        SET stripe_subscription_id = $1,
                                            last_paid_status = $2,
                                            tier = $3,
                                            last_paid_date = NOW(),
                                            updated_at = NOW()
                                        RETURNING tier
                                    )
                                    UPDATE users u
                                    SET tier = s.tier,
                                        invoice_status = $2
                                    FROM sub_upsert s
                                    WHERE u.user_id = $4",
                                    &[
                                        &sub.id.to_string(),
                                        &last_paid_status,
                                        &new_tier,
                                        &user_id,
                                    ],
                                )
                                .await;

                            if let Err(e) = res_one {
                                eprintln!("Error syncing subscriptions/users: {:?}", e);
                            }

                            let res_two = client
                                .execute(
                                    "WITH new_data AS (
                                        SELECT
                                            $1 as user_id, 
                                            'Page' as usage_type, 
                                            0 as usage, 
                                            0 as overage_usage,
                                            EXTRACT(YEAR FROM CURRENT_TIMESTAMP) as year,
                                            EXTRACT(MONTH FROM CURRENT_TIMESTAMP) as month,
                                            $2 as tier,
                                            t.usage_limit
                                        FROM tiers t
                                        WHERE t.tier = $2
                                    )
                                    INSERT INTO monthly_usage (
                                        user_id, usage_type, usage, overage_usage,
                                        year, month, tier, usage_limit
                                    )
                                    SELECT * FROM new_data
                                    ON CONFLICT (user_id, usage_type, year, month)
                                    DO UPDATE
                                    SET tier = EXCLUDED.tier,
                                        usage_limit = EXCLUDED.usage_limit",
                                    &[&user_id, &new_tier],
                                )
                                .await;

                            if let Err(e) = res_two {
                                eprintln!("Error syncing monthly_usage: {:?}", e);
                            }
                        }
                    }
                }
            }
        }
        EventType::CustomerSubscriptionDeleted => {
            if let stripe::EventObject::Subscription(sub) = event.data.object {
                let user_id_opt = find_user_id_by_customer(&sub.customer, &client).await;
                if let Some(user_id) = user_id_opt {
                    let res_one = client
                        .execute(
                            "UPDATE subscriptions 
                             SET last_paid_status = 'Canceled',
                                 updated_at = NOW()
                             WHERE user_id = $1",
                            &[&user_id],
                        )
                        .await;

                    if let Err(e) = res_one {
                        eprintln!("Error updating subscription: {:?}", e);
                    }

                    let res_two = client
                        .execute(
                            "UPDATE users 
                             SET invoice_status = 'Canceled'
                             WHERE user_id = $1",
                            &[&user_id],
                        )
                        .await;

                    if let Err(e) = res_two {
                        eprintln!("Error updating user: {:?}", e);
                    }

                    let res_three = client
                        .execute(
                            "UPDATE monthly_usage
                             SET tier = 'Free',
                                 usage_limit = (SELECT usage_limit FROM tiers WHERE tier = 'Free')
                             WHERE user_id = $1 
                             AND year = EXTRACT(YEAR FROM CURRENT_TIMESTAMP)
                             AND month = EXTRACT(MONTH FROM CURRENT_TIMESTAMP)",
                            &[&user_id],
                        )
                        .await;

                    if let Err(e) = res_three {
                        eprintln!("Error updating monthly usage: {:?}", e);
                    }
                }
            }
        }
        EventType::InvoicePaid => {
            if let stripe::EventObject::Invoice(invoice) = event.data.object {
                let user_id_opt = if let Some(customer) = &invoice.customer {
                    find_user_id_by_customer(customer, &client).await
                } else {
                    None
                };
                if let Some(user_id) = user_id_opt {
                    let stripe_invoice_id = invoice.id;
                    let invoice_status = match invoice.status {
                        Some(StripeInvoiceStatus::Paid) => InvoiceStatus::Paid.to_string(),
                        Some(StripeInvoiceStatus::Open) => InvoiceStatus::Ongoing.to_string(),
                        Some(StripeInvoiceStatus::Void) => InvoiceStatus::Canceled.to_string(),
                        Some(StripeInvoiceStatus::Uncollectible) => InvoiceStatus::NoInvoice.to_string(),
                        _ => "Unknown".to_string(),
                    };
                    let _ = update_invoice_status(&stripe_invoice_id, &user_id, &invoice_status).await;
                }
            }
        }
        EventType::InvoicePaymentFailed => {
            if let stripe::EventObject::Invoice(invoice) = event.data.object {
                let user_id_opt = if let Some(customer) = &invoice.customer {
                    find_user_id_by_customer(customer, &client).await
                } else {
                    None
                };
                if let Some(user_id) = user_id_opt {
                    let stripe_invoice_id = invoice.id;
                    let _ = update_invoice_status(
                        &stripe_invoice_id,
                        &user_id,
                        &InvoiceStatus::PastDue.to_string(),
                    )
                    .await;
                }
            }
        }
        EventType::InvoicePaymentSucceeded => {
            if let stripe::EventObject::Invoice(invoice) = event.data.object {
                let user_id_opt = if let Some(customer) = &invoice.customer {
                    find_user_id_by_customer(customer, &client).await
                } else {
                    None
                };
                if let Some(user_id) = user_id_opt {
                    let stripe_invoice_id = invoice.id;
                    let status_str = match invoice.status {
                        Some(StripeInvoiceStatus::Paid) => InvoiceStatus::Paid.to_string(),
                        Some(StripeInvoiceStatus::Open) => InvoiceStatus::Ongoing.to_string(),
                        Some(StripeInvoiceStatus::Void) => InvoiceStatus::Canceled.to_string(),
                        Some(StripeInvoiceStatus::Uncollectible) => InvoiceStatus::NoInvoice.to_string(),
                        _ => "Unknown".to_string(),
                    };
                    let _ = update_invoice_status(&stripe_invoice_id, &user_id, &status_str).await;
                }
            }
        }
        _ => {
            println!("Unhandled event type in webhook: {:?}", event.type_);
        }
    }

    println!("Webhook processed successfully for event type: {:?}", event.type_);
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
    user_info: web::ReqData<UserInfo>,
    form: web::Json<SubscriptionRequest>,
) -> Result<HttpResponse, Error> {
    // 1) Ensure we have a valid DB client
    let client = get_pg_client().await.map_err(|e| {
        eprintln!("Error connecting to DB: {:?}", e);
        actix_web::error::ErrorInternalServerError("DB Connection Error")
    })?;

    // 2) Load or create Stripe Customer
    let stripe_config = Config::from_env().map_err(|e| {
        eprintln!("Stripe configuration error: {:?}", e);
        actix_web::error::ErrorInternalServerError("Stripe Config Error")
    })?;

    let tier = form.tier.clone();
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
        &tier, // e.g. "price_1234"
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
    client
        .execute(
            r#"
            INSERT INTO subscriptions (user_id, stripe_subscription_id, tier, last_paid_date, last_paid_status)
            VALUES ($1, $2, $3, NOW(), 'True')
            ON CONFLICT (user_id) DO UPDATE
              SET stripe_subscription_id = EXCLUDED.stripe_subscription_id,
                  tier = EXCLUDED.tier,
                  updated_at = NOW()
            "#,
            &[user_id, &stripe_subscription_id, &form.tier],
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
    user_info: web::ReqData<UserInfo>,
) -> Result<HttpResponse, Error> {
    let mut client = get_pg_client().await.map_err(|e| {
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
    let customer_id = match customer_expandable {
        stripe::Expandable::Id(id) => id.to_string(),
        stripe::Expandable::Object(customer) => customer.id.to_string(),
    };

    client
        .query_opt(
            "SELECT user_id FROM users WHERE customer_id = $1",
            &[&customer_id],
        )
        .await
        .ok()
        .flatten()
        .map(|row| row.get("user_id"))
}
