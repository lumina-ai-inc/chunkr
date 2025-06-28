use crate::configs::stripe_config::Config as StripeConfig;
use crate::utils::clients::get_pg_client;
use log::info;
use reqwest::Client as ReqwestClient;

pub async fn create_stripe_customer(email: &str) -> Result<String, Box<dyn std::error::Error>> {
    let stripe_config = StripeConfig::from_env()?;
    let client = ReqwestClient::new();
    let stripe_response = client
        .post("https://api.stripe.com/v1/customers")
        .header("Authorization", format!("Bearer {}", stripe_config.api_key))
        .form(&[("name", email)])
        .send()
        .await?;

    if !stripe_response.status().is_success() {
        return Err(Box::new(std::io::Error::other(
            "Failed to create Stripe customer",
        )));
    }

    let stripe_customer: serde_json::Value = stripe_response.json().await?;
    let stripe_customer_id = stripe_customer["id"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    Ok(stripe_customer_id)
}
pub async fn update_invoice_status(
    stripe_invoice_id: &str,
    user_id: &str,
    status: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let client = get_pg_client().await.map_err(|e| {
        eprintln!("Error connecting to database: {e:?}");
        actix_web::error::ErrorInternalServerError("Database connection error")
    })?;
    let _ = client
        .execute(
            "UPDATE invoices SET invoice_status = $1 WHERE stripe_invoice_id = $2",
            &[&status, &stripe_invoice_id],
        )
        .await?;

    let is_paid = matches!(status, "Paid" | "Open");

    client
        .execute(
            "UPDATE subscriptions SET last_paid_status = $1 WHERE user_id = $2",
            &[&is_paid, &user_id],
        )
        .await?;
    client
        .execute(
            "UPDATE users SET invoice_status = $1 WHERE user_id = $2",
            &[&is_paid, &user_id],
        )
        .await?;
    Ok(())
}
pub async fn create_stripe_setup_intent(
    customer_id: &str,
    stripe_config: &StripeConfig,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let client = ReqwestClient::new();

    let form_data = vec![
        ("customer", customer_id),
        ("payment_method_types[]", "card"),
        ("usage", "off_session"),
    ];

    let stripe_response = match client
        .post("https://api.stripe.com/v1/setup_intents")
        .header("Authorization", format!("Bearer {}", stripe_config.api_key))
        .form(&form_data)
        .send()
        .await
    {
        Ok(response) => response,
        Err(e) => return Err(Box::new(e)),
    };

    if !stripe_response.status().is_success() {
        let error_message = match stripe_response.text().await {
            Ok(text) => format!("Failed to create Stripe SetupIntent: {text}"),
            Err(_) => "Failed to create Stripe SetupIntent".to_string(),
        };
        return Err(Box::new(std::io::Error::other(error_message)));
    }

    let setup_intent: serde_json::Value = match stripe_response.json().await {
        Ok(json) => json,
        Err(e) => return Err(Box::new(e)),
    };

    Ok(setup_intent)
}
pub async fn create_customer_session(
    customer_id: &str,
    stripe_config: &StripeConfig,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let client = ReqwestClient::new();

    let form_data = vec![
        ("customer", customer_id),
        ("components[payment_element][enabled]", "true"),
        (
            "components[payment_element][features][payment_method_redisplay]",
            "enabled",
        ),
        (
            "components[payment_element][features][payment_method_allow_redisplay_filters][]",
            "always",
        ),
        (
            "components[payment_element][features][payment_method_allow_redisplay_filters][]",
            "limited",
        ),
        (
            "components[payment_element][features][payment_method_allow_redisplay_filters][]",
            "unspecified",
        ),
        (
            "components[payment_element][features][payment_method_remove]",
            "enabled",
        ),
        (
            "components[payment_element][features][payment_method_save]",
            "disabled",
        ),
        (
            "components[payment_element][features][payment_method_redisplay_limit]",
            "1",
        ),
        // (
        //     "components[payment_element][features][payment_method_save_usage]",
        //     "off_session",
        // ),
    ];

    let stripe_response = client
        .post("https://api.stripe.com/v1/customer_sessions")
        .header("Authorization", format!("Bearer {}", stripe_config.api_key))
        .form(&form_data)
        .send()
        .await?;

    if !stripe_response.status().is_success() {
        let error_message = stripe_response.text().await?;
        return Err(Box::new(std::io::Error::other(format!(
            "Failed to create Stripe Customer Session: {error_message}"
        ))));
    }

    let session: serde_json::Value = stripe_response.json().await?;

    Ok(session)
}

pub async fn create_stripe_checkout_session(
    customer_id: &str,
    tier: &str,
    stripe_config: &StripeConfig,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let price_id = match tier {
        "Starter" => stripe_config.starter_price_id.clone(),
        "Dev" => stripe_config.dev_price_id.clone(),
        "Growth" => stripe_config.growth_price_id.clone(),
        _ => return Err("Unsupported tier".into()),
    };

    let client = ReqwestClient::new();

    let return_url = format!(
        "{}/checkout/return?session_id={{CHECKOUT_SESSION_ID}}",
        stripe_config.return_url.trim_end_matches('/')
    );

    let form_data = vec![
        ("mode", "subscription"),
        ("customer", customer_id),
        ("line_items[0][price]", &price_id),
        ("line_items[0][quantity]", "1"),
        ("ui_mode", "embedded"),
        ("allow_promotion_codes", "true"),
    ];

    let stripe_response = client
        .post("https://api.stripe.com/v1/checkout/sessions")
        .header("Authorization", format!("Bearer {}", stripe_config.api_key))
        .form(&form_data)
        .query(&[("return_url", return_url)])
        .send()
        .await?;

    if !stripe_response.status().is_success() {
        let err_body = stripe_response.text().await.unwrap_or_default();
        return Err(format!("Failed to create checkout session: {err_body}").into());
    }

    let checkout_session: serde_json::Value = stripe_response.json().await?;
    Ok(checkout_session)
}

pub async fn get_stripe_checkout_session(
    session_id: &str,
    stripe_config: &StripeConfig,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let client = ReqwestClient::new();

    let url = format!("https://api.stripe.com/v1/checkout/sessions/{session_id}");
    let auth = format!("Bearer {}", stripe_config.api_key);
    let stripe_response = client
        .get(&url)
        .header("Authorization", auth)
        .send()
        .await?;

    if !stripe_response.status().is_success() {
        let err_body = stripe_response.text().await.unwrap_or_default();
        return Err(format!("Failed to get checkout session line items: {err_body}").into());
    }

    let line_items: serde_json::Value = stripe_response.json().await?;
    Ok(line_items)
}
// pub async fn create_stripe_portal_configuration(
//     stripe_config: &StripeConfig,
// ) -> Result<String, Box<dyn std::error::Error>> {
//     let client = ReqwestClient::new();

//     let form_data = vec![
//         ("features[subscription_update][enabled]", "true"),
//         ("features[payment_method_update][enabled]", "true"),
//         (
//             "features[subscription_update][default_allowed_updates][]",
//             "price",
//         ),
//         ("features[subscription_pause][enabled]", "false"),
//         ("features[subscription_cancel][enabled]", "true"),
//         ("features[subscription_cancel][mode]", "immediately"),
//         (
//             "features[subscription_update][proration_behavior]",
//             "always_invoice",
//         ),
//         (
//             "features[subscription_update][products][0][product]",
//             &stripe_config.starter_product_id,
//         ),
//         (
//             "features[subscription_update][products][0][prices][]",
//             &stripe_config.starter_price_id,
//         ),
//         (
//             "features[subscription_update][products][1][product]",
//             &stripe_config.dev_product_id,
//         ),
//         (
//             "features[subscription_update][products][1][prices][]",
//             &stripe_config.dev_price_id,
//         ),
//         (
//             "features[subscription_update][products][2][product]",
//             &stripe_config.growth_product_id,
//         ),
//         (
//             "features[subscription_update][products][2][prices][]",
//             &stripe_config.growth_price_id,
//         ),
//     ];

//     info!(
//         "Creating Stripe portal configuration with form data: {:?}",
//         form_data
//     );
//     let stripe_response = client
//         .post("https://api.stripe.com/v1/billing_portal/configurations")
//         .header("Authorization", format!("Bearer {}", stripe_config.api_key))
//         .form(&form_data)
//         .send()
//         .await?;

//     if !stripe_response.status().is_success() {
//         let err_body = stripe_response.text().await.unwrap_or_default();
//         return Err(format!("Failed to create portal configuration: {}", err_body).into());
//     }

//     let config: serde_json::Value = stripe_response.json().await?;
//     info!("Created portal configuration with ID: {}", config["id"]);
//     Ok(config["id"].as_str().unwrap_or_default().to_string())
// }

pub async fn create_stripe_billing_portal_session(
    customer_id: &str,
    stripe_config: &StripeConfig,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let client = ReqwestClient::new();

    // First update the subscription billing cycle

    // Create or retrieve the portal configuration ID
    // let config_id = create_stripe_portal_configuration(stripe_config).await?;

    let return_url = format!(
        "{}/dashboard",
        stripe_config.return_url.trim_end_matches('/')
    );

    let form_data = vec![
        ("customer", customer_id),
        ("return_url", &return_url),
        // ("configuration", &config_id),
    ];

    info!("Creating billing portal session for customer {customer_id}");
    let stripe_response = client
        .post("https://api.stripe.com/v1/billing_portal/sessions")
        .header("Authorization", format!("Bearer {}", stripe_config.api_key))
        .form(&form_data)
        .send()
        .await?;

    if !stripe_response.status().is_success() {
        let err_body = stripe_response.text().await.unwrap_or_default();
        info!("Failed to create billing portal session: {err_body}");
        return Err(format!("Failed to create billing portal session: {err_body}").into());
    }

    let portal_session: serde_json::Value = stripe_response.json().await?;
    info!(
        "Created billing portal session with URL: {}",
        portal_session["url"]
    );

    Ok(portal_session)
}

pub async fn update_subscription_billing_cycle(
    subscription_id: &str,
    stripe_config: &StripeConfig,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let client = ReqwestClient::new();

    let url = format!("https://api.stripe.com/v1/subscriptions/{subscription_id}");

    let form_data = vec![("billing_cycle_anchor", "now")];

    let stripe_response = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", stripe_config.api_key))
        .form(&form_data)
        .send()
        .await?;

    if !stripe_response.status().is_success() {
        let err_body = stripe_response.text().await.unwrap_or_default();
        return Err(format!("Failed to update subscription: {err_body}").into());
    }

    let updated_subscription: serde_json::Value = stripe_response.json().await?;
    Ok(updated_subscription)
}

/// Gets the customer's payment methods and sets the most recent one as default for invoices
pub async fn set_customer_default_payment_method(
    customer_id: &str,
    config: &StripeConfig,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let client = ReqwestClient::new();

    // Step 1: List customer's payment methods
    let url = format!("https://api.stripe.com/v1/customers/{customer_id}/payment_methods");

    let response = client
        .get(&url)
        .header("Authorization", format!("Bearer {}", config.api_key))
        .query(&[("limit", "5"), ("type", "card")])
        .send()
        .await?
        .json::<serde_json::Value>()
        .await?;

    // Find the most recent payment method
    if let Some(data) = response.get("data").and_then(|d| d.as_array()) {
        if let Some(payment_method) = data.first() {
            if let Some(payment_method_id) = payment_method.get("id").and_then(|id| id.as_str()) {
                // Step 2: Update customer to set default payment method
                let update_url = format!("https://api.stripe.com/v1/customers/{customer_id}");

                let params = [(
                    "invoice_settings[default_payment_method]",
                    payment_method_id,
                )];

                client
                    .post(&update_url)
                    .header("Authorization", format!("Bearer {}", config.api_key))
                    .form(&params)
                    .send()
                    .await?;

                return Ok(());
            }
        }
    }

    Err("No payment methods found for customer".into())
}
