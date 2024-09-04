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
