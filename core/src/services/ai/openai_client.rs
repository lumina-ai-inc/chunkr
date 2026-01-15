use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
        ChatCompletionRequestUserMessageArgs,
        CreateChatCompletionRequestArgs, CreateEmbeddingRequestArgs,
        EmbeddingInput,
    },
    Client,
};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::sync::Arc;
use tokio::sync::OnceCell;

static OPENAI_CLIENT: OnceCell<Arc<Client<OpenAIConfig>>> = OnceCell::const_new();

/// Initialize OpenAI client
pub async fn get_openai_client() -> Result<Arc<Client<OpenAIConfig>>, Box<dyn Error + Send + Sync>> {
    OPENAI_CLIENT
        .get_or_try_init(|| async {
            let api_key = std::env::var("OPENAI_API_KEY")
                .map_err(|_| "OPENAI_API_KEY environment variable not set")?;
            
            let config = OpenAIConfig::new().with_api_key(api_key);
            let client = Client::with_config(config);
            Ok(Arc::new(client))
        })
        .await
        .map(|client| client.clone())
}

pub struct OpenAIClient {
    client: Arc<Client<OpenAIConfig>>,
    model: String,
}

impl OpenAIClient {
    pub async fn new() -> Result<Self, Box<dyn Error + Send + Sync>> {
        let client = get_openai_client().await?;
        let model = std::env::var("OPENAI_MODEL")
            .unwrap_or_else(|_| "gpt-4-turbo-preview".to_string());
        
        Ok(Self { client, model })
    }

    /// Extract structured facts from OCR text using GPT-4 with JSON mode
    pub async fn extract_facts_from_ocr(
        &self,
        document_type: &str,
        ocr_text: &str,
        system_prompt: &str,
    ) -> Result<StructuredFacts, Box<dyn Error + Send + Sync>> {
        let user_message = format!(
            "Document Type: {}\n\nDocument Text:\n{}",
            document_type, ocr_text
        );

        let messages = vec![
            ChatCompletionRequestMessage::System(
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(system_prompt)
                    .build()?,
            ),
            ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessageArgs::default()
                    .content(user_message)
                    .build()?,
            ),
        ];

        use async_openai::types::ResponseFormat;
        
        let request = CreateChatCompletionRequestArgs::default()
            .model(&self.model)
            .messages(messages)
            .response_format(ResponseFormat::JsonObject)
            .temperature(0.1)
            .build()?;

        let response = self.client.chat().create(request).await?;

        let content = response
            .choices
            .first()
            .and_then(|choice| choice.message.content.as_ref())
            .ok_or("No response from OpenAI")?;

        let facts: StructuredFacts = serde_json::from_str(content)?;
        Ok(facts)
    }

    /// Chat completion with optional tools/functions
    pub async fn chat_completion(
        &self,
        messages: Vec<ChatCompletionRequestMessage>,
        temperature: Option<f32>,
    ) -> Result<ChatResponse, Box<dyn Error + Send + Sync>> {
        let mut request_builder = CreateChatCompletionRequestArgs::default();
        request_builder.model(&self.model).messages(messages);

        if let Some(temp) = temperature {
            request_builder.temperature(temp);
        }

        let request = request_builder.build()?;
        let response = self.client.chat().create(request).await?;

        let content = response
            .choices
            .first()
            .and_then(|choice| choice.message.content.as_ref())
            .cloned()
            .unwrap_or_default();

        let tokens_used = response.usage.map(|u| u.total_tokens as i32);

        Ok(ChatResponse {
            content,
            tokens_used,
            model: response.model,
        })
    }

    /// Create embeddings for text
    pub async fn create_embedding(
        &self,
        text: &str,
    ) -> Result<Vec<f32>, Box<dyn Error + Send + Sync>> {
        let request = CreateEmbeddingRequestArgs::default()
            .model("text-embedding-3-small")
            .input(EmbeddingInput::String(text.to_string()))
            .build()?;

        let response = self.client.embeddings().create(request).await?;

        let embedding = response
            .data
            .first()
            .ok_or("No embedding returned")?
            .embedding
            .clone();

        Ok(embedding)
    }

    /// Create embeddings for multiple texts
    pub async fn create_embeddings_batch(
        &self,
        texts: Vec<String>,
    ) -> Result<Vec<Vec<f32>>, Box<dyn Error + Send + Sync>> {
        let request = CreateEmbeddingRequestArgs::default()
            .model("text-embedding-3-small")
            .input(EmbeddingInput::StringArray(texts))
            .build()?;

        let response = self.client.embeddings().create(request).await?;

        let embeddings: Vec<Vec<f32>> = response
            .data
            .into_iter()
            .map(|data| data.embedding)
            .collect();

        Ok(embeddings)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredFacts {
    pub facts: Vec<ExtractedFact>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedFact {
    pub fact_type: String,
    pub label: String,
    pub value: String,
    pub unit: Option<String>,
    pub confidence: f64,
    pub source_page: Option<i32>,
    pub source_text: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ChatResponse {
    pub content: String,
    pub tokens_used: Option<i32>,
    pub model: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires OpenAI API key
    async fn test_create_embedding() {
        let client = OpenAIClient::new().await.unwrap();
        let embedding = client.create_embedding("test text").await.unwrap();
        assert_eq!(embedding.len(), 1536);
    }

    #[tokio::test]
    #[ignore] // Requires OpenAI API key
    async fn test_chat_completion() {
        let client = OpenAIClient::new().await.unwrap();
        let messages = vec![ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessageArgs::default()
                .content("Say hello")
                .build()
                .unwrap(),
        )];
        
        let response = client.chat_completion(messages, Some(0.7)).await.unwrap();
        assert!(!response.content.is_empty());
    }
}

