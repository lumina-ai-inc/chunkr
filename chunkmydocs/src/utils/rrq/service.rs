use config::{ Config as ConfigTrait, ConfigError };
use dotenvy::dotenv;
use lazy_static::lazy_static;
use reqwest::Client;
use serde::Deserialize;
use std::{ sync::Once, time::{ Duration, Instant } };

use crate::models::rrq::{
    consume::{ ConsumePayload, ConsumeResponse },
    produce::ProducePayload,
    status::{ StatusPayload, StatusResult },
};

lazy_static! {
    static ref CLIENT: Client = Client::new();
}

static INIT: Once = Once::new();

#[derive(Debug, Deserialize)]
struct Config {
    url: String,
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        INIT.call_once(|| {
            dotenv().ok();
        });

        ConfigTrait::builder()
            .add_source(config::Environment::default().prefix("RRQ").separator("__"))
            .build()?
            .try_deserialize()
    }
}

pub async fn health() -> Result<String, Box<dyn std::error::Error>> {
    let cfg = Config::from_env()?;
    let response = CLIENT.get(&cfg.url)
        .timeout(Duration::from_secs(60))
        .send().await
        .map_err(|e| e.to_string())?
        .error_for_status()
        .map_err(|e| {
            let status = e.status().unwrap_or_default();
            let error_body = e.to_string();
            println!("Error sending request: {} {}", status, error_body);
            format!("HTTP error in health {}: {}", status, error_body)
        })?;
    let body = response.text().await?;
    Ok(body)
}

pub async fn produce(items: Vec<ProducePayload>) -> Result<String, Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    let cfg = Config::from_env()?;
    let mut responses = Vec::new();
    for chunk in items.chunks(120) {
        println!("Time taken to start produce: {:?}", start_time.elapsed());
        let response = CLIENT.post(&format!("{}/produce", cfg.url))
            .timeout(Duration::from_secs(10))
            .json(chunk)
            .send().await
            .map_err(|e| e.to_string())?
            .error_for_status()
            .map_err(|e| {
                let status = e.status().unwrap_or_default();
                let error_body = e.to_string();
                println!("Error sending request: {} {}", status, error_body);
                format!("HTTP error produce {}: {}", status, error_body)
            })?;
        let body = response.text().await?;
        println!("Time taken to produce: {:?}", start_time.elapsed());
        responses.push(body);
    }

    println!("Total time taken to produce: {:?}", start_time.elapsed());

    Ok(responses.join("\n"))
}

pub async fn consume(
    payload: ConsumePayload
) -> Result<Vec<ConsumeResponse>, Box<dyn std::error::Error>> {
    let cfg = Config::from_env()?;
    let response = CLIENT.post(&format!("{}/consume", cfg.url))
        .timeout(Duration::from_secs(5))
        .json(&payload)
        .send().await
        .map_err(|e| e.to_string())?
        .error_for_status()
        .map_err(|e| {
            let status = e.status().unwrap_or_default();
            let error_body = e.to_string();
            println!("Error sending request: {} {}", status, error_body);
            format!("HTTP error in consume {}: {}", status, error_body)
        })?;
    let consume_payload: Vec<ConsumeResponse> = response.json().await?;
    Ok(consume_payload)
}

pub async fn complete(payloads: Vec<StatusPayload>) -> Result<String, Box<dyn std::error::Error>> {
    let cfg = Config::from_env()?;
    let mut success_count = 0;
    let mut failure_count = 0;
    for payload in &payloads {
        match payload.result {
            StatusResult::Success => {
                success_count += 1;
            }
            StatusResult::Failure => {
                failure_count += 1;
            }
        }
    }
    println!(
        "Processing {} items: {} successes, {} failures",
        payloads.len(),
        success_count,
        failure_count
    );

    let mut responses = Vec::new();

    let response = CLIENT.post(&format!("{}/complete", cfg.url))
        .timeout(Duration::from_secs(60))
        .json(&payloads)
        .send().await
        .map_err(|e| e.to_string())?
        .error_for_status()
        .map_err(|e| {
            let status = e.status().unwrap_or_default();
            let error_body = e.to_string();
            println!("Error sending request: {} {}", status, error_body);
            format!("HTTP error in success {}: {}", status, error_body)
        })?;
    let body = response.text().await?;
    responses.push(body);

    Ok(responses.join("\n"))
}
