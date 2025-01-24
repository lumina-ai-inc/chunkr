use crate::configs::stripe_config::Config as StripeConfig;
use crate::utils::clients::get_pg_client;
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
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
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
        eprintln!("Error connecting to database: {:?}", e);
        actix_web::error::ErrorInternalServerError("Database connection error")
    })?;
    let _ = client
        .execute(
            "UPDATE invoices SET invoice_status = $1 WHERE stripe_invoice_id = $2",
            &[&status, &stripe_invoice_id],
        )
        .await?;

    let is_paid = match status {
        "Paid" | "Open" => true,
        _ => false,
    };

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
            Ok(text) => format!("Failed to create Stripe SetupIntent: {}", text),
            Err(_) => "Failed to create Stripe SetupIntent".to_string(),
        };
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            error_message,
        )));
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
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!(
                "Failed to create Stripe Customer Session: {}",
                error_message
            ),
        )));
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
        "Team" => stripe_config.team_price_id.clone(),
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
        ("return_url", &return_url),
    ];

    let stripe_response = client
        .post("https://api.stripe.com/v1/checkout/sessions")
        .header("Authorization", format!("Bearer {}", stripe_config.api_key))
        .form(&form_data)
        .send()
        .await?;

    if !stripe_response.status().is_success() {
        let err_body = stripe_response.text().await.unwrap_or_default();
        return Err(format!("Failed to create checkout session: {}", err_body).into());
    }

    let checkout_session: serde_json::Value = stripe_response.json().await?;
    Ok(checkout_session)
}

pub async fn get_stripe_checkout_session(
    session_id: &str,
    stripe_config: &StripeConfig,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let client = ReqwestClient::new();

    let url = format!("https://api.stripe.com/v1/checkout/sessions/{}", session_id);
    let auth = format!("Bearer {}", stripe_config.api_key);
    println!("curl -X GET '{}' -H 'Authorization: {}'", url, auth);
    let stripe_response = client
        .get(&url)
        .header("Authorization", auth)
        .send()
        .await?;

    if !stripe_response.status().is_success() {
        let err_body = stripe_response.text().await.unwrap_or_default();
        return Err(format!("Failed to get checkout session line items: {}", err_body).into());
    }

    let line_items: serde_json::Value = stripe_response.json().await?;
    Ok(line_items)
}

pub async fn create_stripe_billing_portal_session(
    customer_id: &str,
    stripe_config: &StripeConfig,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let client = ReqwestClient::new();

    let form_data = vec![
        ("customer", customer_id),
        ("features[invoice_history][enabled]", "true"),
        ("return_url", &stripe_config.return_url),
    ];

    let stripe_response = client
        .post("https://api.stripe.com/v1/billing_portal/sessions")
        .header("Authorization", format!("Bearer {}", stripe_config.api_key))
        .form(&form_data)
        .send()
        .await?;

    if !stripe_response.status().is_success() {
        let err_body = stripe_response.text().await.unwrap_or_default();
        return Err(format!("Failed to create billing portal session: {}", err_body).into());
    }

    let portal_session: serde_json::Value = stripe_response.json().await?;
    Ok(portal_session)
}
