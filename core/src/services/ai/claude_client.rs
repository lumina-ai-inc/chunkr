use reqwest::{Client, header};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::error::Error;
use std::sync::Arc;
use tokio::sync::OnceCell;

static CLAUDE_CLIENT: OnceCell<Arc<ClaudeHttpClient>> = OnceCell::const_new();

/// Initialize Claude HTTP client
async fn get_claude_http_client() -> Result<Arc<ClaudeHttpClient>, Box<dyn Error + Send + Sync>> {
    CLAUDE_CLIENT
        .get_or_try_init(|| async {
            let api_key = std::env::var("ANTHROPIC_API_KEY")
                .map_err(|_| "ANTHROPIC_API_KEY environment variable not set")?;
            
            let mut headers = header::HeaderMap::new();
            headers.insert(
                "x-api-key",
                header::HeaderValue::from_str(&api_key)
                    .map_err(|e| format!("Invalid API key: {}", e))?,
            );
            headers.insert(
                "anthropic-version",
                header::HeaderValue::from_static("2023-06-01"),
            );
            headers.insert(
                header::CONTENT_TYPE,
                header::HeaderValue::from_static("application/json"),
            );

            let client = Client::builder()
                .default_headers(headers)
                .build()
                .map_err(|e| format!("Failed to build HTTP client: {}", e))?;

            Ok(Arc::new(ClaudeHttpClient { client, api_key }))
        })
        .await
        .map(|client| client.clone())
}

struct ClaudeHttpClient {
    client: Client,
    api_key: String,
}

pub struct ClaudeClient {
    http_client: Arc<ClaudeHttpClient>,
    model: String,
}

impl ClaudeClient {
    pub async fn new() -> Result<Self, Box<dyn Error + Send + Sync>> {
        let http_client = get_claude_http_client().await?;
        let model = std::env::var("CLAUDE_MODEL")
            .unwrap_or_else(|_| "claude-3-5-sonnet-20241022".to_string());
        
        Ok(Self { http_client, model })
    }

    /// Analyze underwriting with qualitative insights
    pub async fn analyze_underwriting(
        &self,
        facts: &str,
        context: &str,
        system_prompt: &str,
    ) -> Result<UnderwritingAnalysis, Box<dyn Error + Send + Sync>> {
        let user_message = format!(
            "Context:\n{}\n\nFacts:\n{}\n\nPlease provide a comprehensive underwriting analysis.",
            context, facts
        );

        let response = self.chat_completion(
            system_prompt,
            &user_message,
            4096,
            0.3,
        ).await?;

        // Parse the response into structured analysis
        // For now, return as-is, but in production you'd want structured output
        Ok(UnderwritingAnalysis {
            summary: response.content.clone(),
            insights: vec![],
            recommendations: vec![],
            risks: vec![],
            tokens_used: response.tokens_used,
        })
    }

    /// Generate investor memo
    pub async fn generate_investor_memo(
        &self,
        deal_data: &str,
        template: &str,
        system_prompt: &str,
    ) -> Result<InvestorMemo, Box<dyn Error + Send + Sync>> {
        let user_message = format!(
            "Template Requirements:\n{}\n\nDeal Data:\n{}\n\nPlease generate a professional investor memo.",
            template, deal_data
        );

        let response = self.chat_completion(
            system_prompt,
            &user_message,
            8192,
            0.5,
        ).await?;

        Ok(InvestorMemo {
            content: response.content,
            sections: vec![], // Would be parsed from structured output
            tokens_used: response.tokens_used,
        })
    }

    /// Chat completion with Claude
    pub async fn chat_completion(
        &self,
        system_prompt: &str,
        user_message: &str,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<ClaudeResponse, Box<dyn Error + Send + Sync>> {
        let request_body = json!({
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        });

        let response = self.http_client.client
            .post("https://api.anthropic.com/v1/messages")
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(format!("Claude API error {}: {}", status, error_text).into());
        }

        let api_response: ClaudeApiResponse = response.json().await?;

        // Extract text content from the response
        let content = api_response.content
            .iter()
            .filter_map(|c| {
                if c.content_type == "text" {
                    c.text.clone()
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n");

        let tokens_used = api_response.usage.input_tokens + api_response.usage.output_tokens;

        Ok(ClaudeResponse {
            content,
            tokens_used: Some(tokens_used),
            model: api_response.model,
            stop_reason: api_response.stop_reason,
        })
    }

    /// Multi-turn conversation
    pub async fn chat_conversation(
        &self,
        system_prompt: &str,
        messages: Vec<ConversationMessage>,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<ClaudeResponse, Box<dyn Error + Send + Sync>> {
        let messages_json: Vec<serde_json::Value> = messages
            .into_iter()
            .map(|msg| {
                json!({
                    "role": msg.role,
                    "content": msg.content
                })
            })
            .collect();

        let request_body = json!({
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": messages_json
        });

        let response = self.http_client.client
            .post("https://api.anthropic.com/v1/messages")
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(format!("Claude API error {}: {}", status, error_text).into());
        }

        let api_response: ClaudeApiResponse = response.json().await?;

        let content = api_response.content
            .iter()
            .filter_map(|c| {
                if c.content_type == "text" {
                    c.text.clone()
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n");

        let tokens_used = api_response.usage.input_tokens + api_response.usage.output_tokens;

        Ok(ClaudeResponse {
            content,
            tokens_used: Some(tokens_used),
            model: api_response.model,
            stop_reason: api_response.stop_reason,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ClaudeApiResponse {
    id: String,
    #[serde(rename = "type")]
    response_type: String,
    role: String,
    content: Vec<ContentBlock>,
    model: String,
    stop_reason: Option<String>,
    stop_sequence: Option<String>,
    usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    content_type: String,
    text: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Usage {
    input_tokens: i32,
    output_tokens: i32,
}

#[derive(Debug, Clone)]
pub struct ClaudeResponse {
    pub content: String,
    pub tokens_used: Option<i32>,
    pub model: String,
    pub stop_reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ConversationMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnderwritingAnalysis {
    pub summary: String,
    pub insights: Vec<String>,
    pub recommendations: Vec<String>,
    pub risks: Vec<String>,
    pub tokens_used: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvestorMemo {
    pub content: String,
    pub sections: Vec<MemoSection>,
    pub tokens_used: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoSection {
    pub title: String,
    pub content: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires Claude API key
    async fn test_chat_completion() {
        let client = ClaudeClient::new().await.unwrap();
        let response = client
            .chat_completion(
                "You are a helpful assistant.",
                "Say hello in one sentence.",
                100,
                0.7,
            )
            .await
            .unwrap();
        
        assert!(!response.content.is_empty());
        assert!(response.tokens_used.is_some());
    }

    #[tokio::test]
    #[ignore] // Requires Claude API key
    async fn test_analyze_underwriting() {
        let client = ClaudeClient::new().await.unwrap();
        let analysis = client
            .analyze_underwriting(
                "NOI: $100,000, DSCR: 1.5",
                "10-unit multifamily property",
                "You are a real estate underwriting analyst.",
            )
            .await
            .unwrap();
        
        assert!(!analysis.summary.is_empty());
    }
}

