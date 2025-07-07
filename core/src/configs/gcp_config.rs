use chrono::{DateTime, Duration, Utc};
use config::{Config as ConfigTrait, ConfigError};
use dotenvy::dotenv_override;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::process::Command;
use std::sync::Mutex;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GcpError {
    #[error("Failed to execute gcloud command: {0}")]
    CommandError(String),
    #[error("Failed to parse gcloud output: {0}")]
    ParseError(String),
    #[error("Token expired or invalid")]
    TokenExpired,
}

#[derive(Debug, Clone)]
struct CachedToken {
    token: String,
    expires_at: DateTime<Utc>,
}

impl CachedToken {
    fn new(token: String, ttl_seconds: i64) -> Self {
        Self {
            token,
            expires_at: Utc::now() + Duration::seconds(ttl_seconds),
        }
    }

    fn is_expired(&self) -> bool {
        Utc::now() > self.expires_at
    }
}

// Global cache for tokens with mutex for thread safety
static TOKEN_CACHE: Lazy<Mutex<HashMap<String, CachedToken>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    #[serde(default = "default_cache_ttl_seconds")]
    pub cache_ttl_seconds: i64,
}

fn default_cache_ttl_seconds() -> i64 {
    3300 // 55 minutes (tokens typically expire in 1 hour)
}

impl Default for Config {
    fn default() -> Self {
        Self {
            cache_ttl_seconds: default_cache_ttl_seconds(),
        }
    }
}

impl Config {
    pub fn new(cache_ttl_seconds: i64) -> Self {
        Self { cache_ttl_seconds }
    }

    pub fn from_env() -> Result<Self, ConfigError> {
        dotenv_override().ok();

        ConfigTrait::builder()
            .add_source(config::Environment::default().prefix("GCP").separator("__"))
            .build()?
            .try_deserialize()
    }

    /// Get the GCP access token using gcloud CLI with caching
    pub fn get_access_token(&self) -> Result<String, GcpError> {
        let cache_key = "gcp_access_token";

        // Check cache first
        {
            let cache = TOKEN_CACHE.lock().unwrap();
            if let Some(cached_token) = cache.get(cache_key) {
                if !cached_token.is_expired() {
                    return Ok(cached_token.token.clone());
                }
            }
        }

        // Cache miss or expired, fetch new token
        let token = self.fetch_access_token_from_gcloud()?;

        // Update cache
        {
            let mut cache = TOKEN_CACHE.lock().unwrap();
            cache.insert(
                cache_key.to_string(),
                CachedToken::new(token.clone(), self.cache_ttl_seconds),
            );
        }

        Ok(token)
    }

    /// Fetch access token directly from gcloud CLI
    fn fetch_access_token_from_gcloud(&self) -> Result<String, GcpError> {
        let output = Command::new("gcloud")
            .args(["auth", "application-default", "print-access-token"])
            .output()
            .map_err(|e| {
                GcpError::CommandError(format!("Failed to execute gcloud command: {e}"))
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(GcpError::CommandError(format!(
                "gcloud command failed with exit code {}: {}",
                output.status.code().unwrap_or(-1),
                stderr
            )));
        }

        let token = String::from_utf8(output.stdout)
            .map_err(|e| GcpError::ParseError(format!("Invalid UTF-8 in gcloud output: {e}")))?
            .trim()
            .to_string();

        if token.is_empty() {
            return Err(GcpError::ParseError(
                "Empty token received from gcloud".to_string(),
            ));
        }

        Ok(token)
    }

    /// Clear the token cache (useful for testing or forcing refresh)
    pub fn clear_cache(&self) {
        let mut cache = TOKEN_CACHE.lock().unwrap();
        cache.clear();
    }

    /// Get cached token info for debugging
    pub fn get_cache_info(&self) -> Option<(String, DateTime<Utc>, bool)> {
        let cache = TOKEN_CACHE.lock().unwrap();
        cache.get("gcp_access_token").map(|cached_token| {
            (
                cached_token.token.clone(),
                cached_token.expires_at,
                cached_token.is_expired(),
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cached_token_expiry() {
        let token = CachedToken::new("test_token".to_string(), -1); // Already expired
        assert!(token.is_expired());

        let token = CachedToken::new("test_token".to_string(), 3600); // 1 hour
        assert!(!token.is_expired());
    }

    #[test]
    fn test_gcp_config_creation() {
        let config = Config::default();
        assert_eq!(config.cache_ttl_seconds, 3300);

        let config = Config::new(1800);
        assert_eq!(config.cache_ttl_seconds, 1800);
    }

    #[test]
    fn test_gcp_config_from_env() {
        // Test default values when no environment variables are set
        let config = Config::from_env().unwrap_or_default();
        assert_eq!(config.cache_ttl_seconds, 3300);

        // Test with environment variable set
        std::env::set_var("GCP__CACHE_TTL_SECONDS", "1800");
        let config = Config::from_env().unwrap_or_default();
        assert_eq!(config.cache_ttl_seconds, 1800);

        // Clean up
        std::env::remove_var("GCP__CACHE_TTL_SECONDS");
    }

    #[test]
    fn test_cache_operations() {
        let config = Config::default();

        // Clear cache first
        config.clear_cache();
        assert!(config.get_cache_info().is_none());

        // Note: We can't test the actual gcloud command in unit tests
        // as it requires gcloud CLI to be installed and authenticated
    }

    #[test]
    fn test_get_access_token() {
        let config = Config::default();
        let token = config.get_access_token().unwrap();
        println!("Token: {token}");
    }
}
