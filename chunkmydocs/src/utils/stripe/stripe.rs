use crate::utils::configs::stripe_config::Config as StripeConfig;
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

pub async fn create_stripe_subscription(
    customer_id: &str,
    stripe_config: &StripeConfig,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let client = ReqwestClient::new();
    let mut items = vec![
        ("items[0][price]", stripe_config.page_fast_price_id.as_str()),
        (
            "items[1][price]",
            stripe_config.page_high_quality_price_id.as_str(),
        ),
        ("items[2][price]", stripe_config.segment_price_id.as_str()),
    ];

    let mut form_data = vec![
        ("automatic_tax[enabled]", "true"),
        ("currency", "usd"),
        ("customer", customer_id),
        ("off_session", "true"),
        ("payment_behavior", "error_if_incomplete"),
        ("proration_behavior", "none"),
    ];
    form_data.append(&mut items);

    let stripe_response = client
        .post("https://api.stripe.com/v1/subscriptions")
        .header("Authorization", format!("Bearer {}", stripe_config.api_key))
        .form(&form_data)
        .send()
        .await?;

    if !stripe_response.status().is_success() {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Failed to create Stripe subscription",
        )));
    }

    let stripe_subscription: serde_json::Value = stripe_response.json().await?;
    Ok(stripe_subscription)
}
// collect payment information
pub async fn create_setup_intent(
    stripe_config: &StripeConfig,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let client = ReqwestClient::new();

    let form_data = vec![("usage", "off_session"), ("payment_method_types[]", "card")];

    let stripe_response = client
        .post("https://api.stripe.com/v1/setup_intents")
        .header("Authorization", format!("Bearer {}", stripe_config.api_key))
        .form(&form_data)
        .send()
        .await?;

    if !stripe_response.status().is_success() {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Failed to create Stripe setup intent",
        )));
    }

    let setup_intent: serde_json::Value = stripe_response.json().await?;
    Ok(setup_intent)
}

pub fn get_client_secret(setup_intent: &serde_json::Value) -> Option<String> {
    setup_intent["client_secret"].as_str().map(String::from)
}
pub async fn record_usage(
    stripe_config: &StripeConfig,
    customer_id: &str,
    event_name: &str,
    value: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let client = ReqwestClient::new();
    let value_str = value.to_string(); // Create a longer-lived value

    let form_data = vec![
        ("event_name", event_name),
        ("payload[value]", &value_str),
        ("payload[stripe_customer_id]", customer_id),
    ];

    let stripe_response = client
        .post("https://api.stripe.com/v1/billing/meter_events")
        .header("Authorization", format!("Bearer {}", stripe_config.api_key))
        .form(&form_data)
        .send()
        .await?;

    if !stripe_response.status().is_success() {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Failed to record usage for event: {}", event_name),
        )));
    }

    Ok(())
}

pub async fn record_high_quality_pages_usage(
    stripe_config: &StripeConfig,
    customer_id: &str,
    pages: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    record_usage(stripe_config, customer_id, "high_quality_pages", pages).await
}

pub async fn record_fast_pages_usage(
    stripe_config: &StripeConfig,
    customer_id: &str,
    pages: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    record_usage(stripe_config, customer_id, "fast_pages", pages).await
}

pub async fn record_segments_usage(
    stripe_config: &StripeConfig,
    customer_id: &str,
    segments: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    record_usage(stripe_config, customer_id, "segments", segments).await
}
