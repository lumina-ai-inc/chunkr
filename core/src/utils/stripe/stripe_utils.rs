use crate::utils::clients::get_pg_client;
use crate::configs::stripe_config::Config as StripeConfig;
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

    client
        .execute(
            "UPDATE subscriptions SET last_paid_status = $1 WHERE user_id = $2",
            &[&status, &user_id],
        )
        .await?;
    client
        .execute(
            "UPDATE users SET invoice_status = $1 WHERE user_id = $2",
            &[&status, &user_id],
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
pub async fn list_payment_methods(
    customer_id: &str,
    stripe_config: &StripeConfig,
) -> Result<Vec<serde_json::Value>, Box<dyn std::error::Error>> {
    let client = ReqwestClient::new();

    let url = format!(
        "https://api.stripe.com/v1/customers/{}/payment_methods",
        customer_id
    );

    let stripe_response = client
        .get(&url)
        .header("Authorization", format!("Bearer {}", stripe_config.api_key))
        .send()
        .await?;

    if !stripe_response.status().is_success() {
        let error_message = stripe_response.text().await?;
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Failed to list payment methods: {}", error_message),
        )));
    }

    let response: serde_json::Value = stripe_response.json().await?;
    let payment_methods = response["data"]
        .as_array()
        .ok_or("No payment methods found")?
        .to_vec();

    Ok(payment_methods)
}

pub async fn delete_payment_method(
    payment_method_id: &str,
    stripe_config: &StripeConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let client = ReqwestClient::new();

    let url = format!(
        "https://api.stripe.com/v1/payment_methods/{}/detach",
        payment_method_id
    );

    let stripe_response = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", stripe_config.api_key))
        .send()
        .await?;

    if !stripe_response.status().is_success() {
        let error_message = stripe_response.text().await?;
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Failed to delete payment method: {}", error_message),
        )));
    }

    Ok(())
}

pub async fn update_payment_method(
    payment_method_id: &str,
    update_data: serde_json::Value,
    stripe_config: &StripeConfig,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let client = ReqwestClient::new();

    let url = format!(
        "https://api.stripe.com/v1/payment_methods/{}",
        payment_method_id
    );

    let stripe_response = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", stripe_config.api_key))
        .json(&update_data)
        .send()
        .await?;

    if !stripe_response.status().is_success() {
        let error_message = stripe_response.text().await?;
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Failed to update payment method: {}", error_message),
        )));
    }

    let updated_payment_method: serde_json::Value = stripe_response.json().await?;

    Ok(updated_payment_method)
}
pub async fn set_default_payment_method(
    customer_id: &str,
    stripe_config: &StripeConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let client = ReqwestClient::new();

    // First, get all payment methods for the customer
    let list_url = format!(
        "https://api.stripe.com/v1/payment_methods?customer={}&type=card",
        customer_id
    );

    let list_response = client
        .get(&list_url)
        .header("Authorization", format!("Bearer {}", stripe_config.api_key))
        .send()
        .await?;

    if !list_response.status().is_success() {
        let error_message = list_response.text().await?;
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Failed to list payment methods: {}", error_message),
        )));
    }

    let payment_methods: serde_json::Value = list_response.json().await?;

    // Sort payment methods by created date (descending) and get the most recent one
    let most_recent_payment_method = payment_methods["data"]
        .as_array()
        .and_then(|methods| {
            methods
                .iter()
                .max_by_key(|method| method["created"].as_i64())
        })
        .ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::Other, "No payment methods found")
        })?;

    let payment_method_id = most_recent_payment_method["id"].as_str().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::Other, "Invalid payment method ID")
    })?;

    // Set the most recent payment method as default
    let update_url = format!("https://api.stripe.com/v1/customers/{}", customer_id);

    let update_response = client
        .post(&update_url)
        .header("Authorization", format!("Bearer {}", stripe_config.api_key))
        .form(&[(
            "invoice_settings[default_payment_method]",
            payment_method_id,
        )])
        .send()
        .await?;

    if !update_response.status().is_success() {
        let error_message = update_response.text().await?;
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Failed to set default payment method: {}", error_message),
        )));
    }

    Ok(())
}

/// Create a subscription in Stripe, specifying the customer_id and a price_id (from your Stripe Dashboard).
/// Returns the Stripe subscription object in JSON form if successful.
pub async fn create_stripe_subscription(
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
    let form_data = vec![
        ("customer", customer_id.to_string()),
        ("items[0][price]", price_id),
        ("collection_method", "charge_automatically".to_string()),
        ("payment_behavior", "default_incomplete".to_string()),
        ("payment_settings[save_default_payment_method]", "on_subscription".to_string()),
        ("expand[]", "latest_invoice.payment_intent".to_string()),
    ];
    let resp = client
        .post("https://api.stripe.com/v1/subscriptions")
        .header("Authorization", format!("Bearer {}", stripe_config.api_key))
        .form(&form_data)
        .send()
        .await?;
    if !resp.status().is_success() {
        let err_body = resp.text().await.unwrap_or_default();
        return Err(format!("Failed to create subscription in Stripe: {}", err_body).into());
    }
    let subscription_json: serde_json::Value = resp.json().await?;
    Ok(subscription_json)
}

/// Cancel a Stripe subscription immediately (or at period's end if desired).
/// Returns the canceled subscription in JSON form if successful.
pub async fn cancel_stripe_subscription(
    stripe_subscription_id: &str,
    invoice_now: bool, // if you want to invoice all pending usage
    prorate: bool,     // whether to apply proration
    stripe_config: &StripeConfig,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let client = ReqwestClient::new();

    let cancel_subscription_url = format!(
        "https://api.stripe.com/v1/subscriptions/{}?{}{}",
        stripe_subscription_id,
        if invoice_now { "invoice_now=true&" } else { "" },
        if prorate {
            "prorate=true"
        } else {
            "prorate=false"
        },
    );

    let stripe_response = client
        .delete(&cancel_subscription_url)
        .header("Authorization", format!("Bearer {}", stripe_config.api_key))
        .send()
        .await?;

    if !stripe_response.status().is_success() {
        let err_body = stripe_response.text().await.unwrap_or_default();
        return Err(format!("Failed to cancel subscription in Stripe: {}", err_body).into());
    }

    let canceled_subscription: serde_json::Value = stripe_response.json().await?;
    Ok(canceled_subscription)
}

/// (Optional) You may want a helper that creates a standalone overage invoice in Stripe,
/// if you want the charge to appear in the user's Stripe billing.
/// Then you can store the returned invoice ID in your local "invoices" table.
/// Adjust line_items or invoice items as needed.
pub async fn create_stripe_invoice_for_overage(
    customer_id: &str,
    amount_in_cents: i64,
    stripe_config: &StripeConfig,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let client = ReqwestClient::new();

    // 1) Create an InvoiceItem (line item) for the overage
    let invoice_item_url = "https://api.stripe.com/v1/invoiceitems";
    let invoice_item_form = vec![
        ("customer", customer_id.to_string()),
        ("amount", amount_in_cents.to_string()), // e.g. 500 = $5.00
        ("currency", "usd".to_string()),
        ("description", "Overage Charges".to_string()),
    ];

    let ii_response = client
        .post(invoice_item_url)
        .header("Authorization", format!("Bearer {}", stripe_config.api_key))
        .form(&invoice_item_form)
        .send()
        .await?;

    if !ii_response.status().is_success() {
        let err_body = ii_response.text().await.unwrap_or_default();
        return Err(format!("Failed to create invoice item for overage: {}", err_body).into());
    }

    // 2) Create the invoice
    let create_invoice_url = "https://api.stripe.com/v1/invoices";
    let invoice_form = vec![
        ("customer", customer_id.to_string()),
        ("auto_advance", "true".to_string()), // automatically try to finalize & pay
        ("collection_method", "charge_automatically".to_string()),
    ];

    let invoice_response = client
        .post(create_invoice_url)
        .header("Authorization", format!("Bearer {}", stripe_config.api_key))
        .form(&invoice_form)
        .send()
        .await?;

    if !invoice_response.status().is_success() {
        let err_body = invoice_response.text().await.unwrap_or_default();
        return Err(format!("Failed to create invoice for overage: {}", err_body).into());
    }

    let created_invoice: serde_json::Value = invoice_response.json().await?;
    Ok(created_invoice)
}

/// Update an existing subscription to a new tier
pub async fn update_stripe_subscription(
    stripe_subscription_id: &str,
    new_tier: &str,
    prorate: bool,
    stripe_config: &StripeConfig,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let price_id = match new_tier {
        "Starter" => stripe_config.starter_price_id.clone(),
        "Dev" => stripe_config.dev_price_id.clone(),
        "Team" => stripe_config.team_price_id.clone(),
        _ => return Err("Unsupported tier".into()),
    };

    let client = ReqwestClient::new();
    let update_url = format!(
        "https://api.stripe.com/v1/subscriptions/{}",
        stripe_subscription_id
    );

    // Subscriptions with a single "items[0]".
    // If you had multiple items, you'd manipulate them individually.
    let form_data = vec![
        ("items[0][price]", price_id),
        // "always_invoice" or "none"
        // This decides how Stripe handles prorations.
        (
            "proration_behavior",
            if prorate { "create_prorations" } else { "none" }.to_string(),
        ),
    ];

    let stripe_response = client
        .post(&update_url)
        .header("Authorization", format!("Bearer {}", stripe_config.api_key))
        .form(&form_data)
        .send()
        .await?;

    if !stripe_response.status().is_success() {
        let err_body = stripe_response.text().await.unwrap_or_default();
        return Err(format!("Failed to update subscription: {}", err_body).into());
    }

    let subscription_json: serde_json::Value = stripe_response.json().await?;
    Ok(subscription_json)
}
