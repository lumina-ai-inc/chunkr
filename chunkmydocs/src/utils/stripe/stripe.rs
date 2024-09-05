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
