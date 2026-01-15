use crate::agents::common::{AgentContext, AgentError, AgentResult, log_execution};
use crate::agents::document_parser::DocumentParserAgent;
use crate::models::output::OCRResult;
use crate::services::ai::{ClaudeClient, OpenAIClient, VectorDB};
use crate::services::ai::prompts::{
    get_chat_system_prompt, CHAT_ORCHESTRATOR_SYSTEM, INTENT_CLASSIFIER_PROMPT,
};
use crate::utils::clients::get_pg_client;
use chrono::Utc;
// Removed diesel - using tokio-postgres instead
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as JsonValue};
use std::error::Error;
use uuid::Uuid;

pub struct ChatOrchestratorAgent {
    openai_client: OpenAIClient,
    claude_client: ClaudeClient,
    vector_db: VectorDB,
}

impl ChatOrchestratorAgent {
    pub async fn new() -> Result<Self, Box<dyn Error + Send + Sync>> {
        let openai_client = OpenAIClient::new().await?;
        let claude_client = ClaudeClient::new().await?;
        let vector_db = VectorDB::new().await?;
        
        Ok(Self {
            openai_client,
            claude_client,
            vector_db,
        })
    }

    /// Process a chat message with full context
    pub async fn process_message(
        &self,
        conversation_id: &str,
        user_message: &str,
        user_id: &str,
        deal_id: Option<String>,
        is_authenticated: bool,
    ) -> Result<ChatResult, Box<dyn Error + Send + Sync>> {
        let start_time = std::time::Instant::now();

        // Step 1: Retrieve conversation history
        let history = self.get_conversation_history(conversation_id).await?;

        // Step 2: Retrieve relevant context using RAG
        let context = self.retrieve_relevant_context(user_message, user_id, deal_id.as_deref()).await?;

        // Step 3: Determine user intent
        let intent = self.determine_intent(user_message, &history).await?;

        // Step 4: Build enhanced prompt with context
        let system_prompt = get_chat_system_prompt(is_authenticated);
        let enhanced_message = self.build_contextual_message(
            user_message,
            &context,
            deal_id.as_deref(),
        );

        // Step 5: Generate response (use Claude for better long-form responses)
        let mut messages = vec![];
        
        // Add conversation history
        for msg in &history {
            messages.push(crate::services::ai::claude_client::ConversationMessage {
                role: msg.role.clone(),
                content: msg.content.clone(),
            });
        }
        
        // Add current message
        messages.push(crate::services::ai::claude_client::ConversationMessage {
            role: "user".to_string(),
            content: enhanced_message,
        });

        let response = self.claude_client
            .chat_conversation(system_prompt, messages, 2048, 0.7)
            .await?;

        // Step 6: Store message and response
        self.store_message(conversation_id, "user", user_message).await?;
        self.store_message(conversation_id, "assistant", &response.content).await?;

        // Step 7: Check if action is needed
        let action = self.detect_action(&response.content, &intent);

        let execution_time = start_time.elapsed().as_millis() as i32;

        Ok(ChatResult {
            response: response.content,
            intent: intent.intent_type,
            action,
            context_used: context.len(),
            tokens_used: response.tokens_used,
            execution_time_ms: execution_time,
        })
    }

    /// Retrieve relevant context using vector search (RAG)
    async fn retrieve_relevant_context(
        &self,
        query: &str,
        user_id: &str,
        deal_id: Option<&str>,
    ) -> Result<Vec<ContextItem>, Box<dyn Error + Send + Sync>> {
        // Create query embedding
        let query_embedding = self.openai_client.create_embedding(query).await?;

        // Build filter for user's data
        let mut filter_conditions = vec![];
        
        // Search in documents collection
        let doc_filter = if let Some(deal) = deal_id {
            Some(qdrant_client::qdrant::Filter {
                must: vec![
                    qdrant_client::qdrant::Condition {
                        condition_one_of: Some(
                            qdrant_client::qdrant::condition::ConditionOneOf::Field(
                                qdrant_client::qdrant::FieldCondition {
                                    key: "deal_id".to_string(),
                                    r#match: Some(qdrant_client::qdrant::Match {
                                        match_value: Some(
                                            qdrant_client::qdrant::r#match::MatchValue::Keyword(
                                                deal.to_string()
                                            )
                                        ),
                                    }),
                                    ..Default::default()
                                }
                            )
                        ),
                    }
                ],
                ..Default::default()
            })
        } else {
            None
        };

        let doc_results = self.vector_db
            .search_similar("documents", query_embedding.clone(), 3, doc_filter)
            .await?;

        // Search in facts collection
        let fact_filter = if let Some(deal) = deal_id {
            Some(qdrant_client::qdrant::Filter {
                must: vec![
                    qdrant_client::qdrant::Condition {
                        condition_one_of: Some(
                            qdrant_client::qdrant::condition::ConditionOneOf::Field(
                                qdrant_client::qdrant::FieldCondition {
                                    key: "deal_id".to_string(),
                                    r#match: Some(qdrant_client::qdrant::Match {
                                        match_value: Some(
                                            qdrant_client::qdrant::r#match::MatchValue::Keyword(
                                                deal.to_string()
                                            )
                                        ),
                                    }),
                                    ..Default::default()
                                }
                            )
                        ),
                    }
                ],
                ..Default::default()
            })
        } else {
            None
        };

        let fact_results = self.vector_db
            .search_similar("facts", query_embedding, 5, fact_filter)
            .await?;

        // Combine and format results
        let mut context = Vec::new();

        for result in doc_results {
            if let (Some(text), Some(page)) = (
                result.payload.get("text"),
                result.payload.get("page_number"),
            ) {
                context.push(ContextItem {
                    content: text.as_str().unwrap_or("").to_string(),
                    source: format!("Document page {}", page),
                    relevance_score: result.score,
                });
            }
        }

        for result in fact_results {
            if let (Some(label), Some(value)) = (
                result.payload.get("label"),
                result.payload.get("value"),
            ) {
                context.push(ContextItem {
                    content: format!("{}: {}", 
                        label.as_str().unwrap_or(""), 
                        value.as_str().unwrap_or("")
                    ),
                    source: "Extracted Facts".to_string(),
                    relevance_score: result.score,
                });
            }
        }

        // Sort by relevance
        context.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());

        Ok(context)
    }

    /// Determine user intent
    async fn determine_intent(
        &self,
        message: &str,
        history: &[ConversationMessage],
    ) -> Result<Intent, Box<dyn Error + Send + Sync>> {
        let context = if history.is_empty() {
            message.to_string()
        } else {
            let recent: Vec<String> = history
                .iter()
                .rev()
                .take(3)
                .map(|m| format!("{}: {}", m.role, m.content))
                .collect();
            format!("Recent conversation:\n{}\n\nCurrent message: {}", 
                recent.join("\n"), message)
        };

        let response = self.openai_client
            .chat_completion(
                vec![
                    async_openai::types::ChatCompletionRequestMessage::System(
                        async_openai::types::ChatCompletionRequestSystemMessageArgs::default()
                            .content(INTENT_CLASSIFIER_PROMPT)
                            .build()?
                    ),
                    async_openai::types::ChatCompletionRequestMessage::User(
                        async_openai::types::ChatCompletionRequestUserMessageArgs::default()
                            .content(context)
                            .build()?
                    ),
                ],
                Some(0.1),
            )
            .await?;

        let intent_type = if response.content.contains("UPLOAD_DOCUMENT") {
            "UPLOAD_DOCUMENT"
        } else if response.content.contains("ANALYZE_DEAL") {
            "ANALYZE_DEAL"
        } else if response.content.contains("GENERATE_MEMO") {
            "GENERATE_MEMO"
        } else if response.content.contains("VIEW_DATA") {
            "VIEW_DATA"
        } else if response.content.contains("GET_HELP") {
            "GET_HELP"
        } else {
            "ASK_QUESTION"
        };

        Ok(Intent {
            intent_type: intent_type.to_string(),
            confidence: 0.8, // Could extract from response
        })
    }

    /// Build contextual message with RAG results
    fn build_contextual_message(
        &self,
        user_message: &str,
        context: &[ContextItem],
        deal_id: Option<&str>,
    ) -> String {
        if context.is_empty() {
            return user_message.to_string();
        }

        let context_section: String = context
            .iter()
            .take(5) // Limit to top 5 most relevant
            .map(|item| format!("- {} (from {})", item.content, item.source))
            .collect::<Vec<_>>()
            .join("\n");

        let deal_info = if let Some(deal) = deal_id {
            format!("\n\nCurrent Deal ID: {}", deal)
        } else {
            String::new()
        };

        format!(
            "User's question: {}\n\nRelevant context from their data:\n{}{}\n\nPlease provide a helpful response using this context.",
            user_message, context_section, deal_info
        )
    }

    /// Detect if action is needed based on response
    fn detect_action(&self, response: &str, intent: &Intent) -> Option<Action> {
        // Simple action detection - could be enhanced with structured output
        if intent.intent_type == "UPLOAD_DOCUMENT" {
            Some(Action {
                action_type: "trigger_upload".to_string(),
                parameters: json!({}),
            })
        } else if intent.intent_type == "ANALYZE_DEAL" {
            Some(Action {
                action_type: "run_underwriting".to_string(),
                parameters: json!({}),
            })
        } else if intent.intent_type == "GENERATE_MEMO" {
            Some(Action {
                action_type: "generate_memo".to_string(),
                parameters: json!({"template": "lp"}),
            })
        } else {
            None
        }
    }

    /// Get conversation history from database
    async fn get_conversation_history(
        &self,
        conversation_id: &str,
    ) -> Result<Vec<ConversationMessage>, Box<dyn Error + Send + Sync>> {
        let client = get_pg_client().await?;

        let rows = client.query(
            "SELECT role, content FROM messages WHERE conversation_id = $1 ORDER BY created_at ASC LIMIT 20",
            &[&conversation_id]
        ).await?;

        Ok(rows
            .into_iter()
            .map(|row| ConversationMessage {
                role: row.get("role"),
                content: row.get("content"),
            })
            .collect())
    }

    /// Store message in database
    async fn store_message(
        &self,
        conversation_id: &str,
        role: &str,
        content: &str,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let client = get_pg_client().await?;
        let message_id = Uuid::new_v4().to_string();
        let now = Utc::now().naive_utc();

        client.execute(
            "INSERT INTO messages (message_id, conversation_id, role, content, metadata, embedding_id, created_at) VALUES ($1, $2, $3, $4, $5, $6, $7)",
            &[&message_id, &conversation_id, &role, &content, &None::<serde_json::Value>, &None::<String>, &now]
        ).await?;

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ChatResult {
    pub response: String,
    pub intent: String,
    pub action: Option<Action>,
    pub context_used: usize,
    pub tokens_used: Option<i32>,
    pub execution_time_ms: i32,
}

#[derive(Debug, Clone)]
struct ContextItem {
    content: String,
    source: String,
    relevance_score: f32,
}

#[derive(Debug, Clone)]
struct Intent {
    intent_type: String,
    confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    pub action_type: String,
    pub parameters: JsonValue,
}

#[derive(Debug, Clone)]
struct ConversationMessage {
    role: String,
    content: String,
}

// Removed diesel models - using raw SQL queries instead

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires database and API keys
    async fn test_chat_orchestrator() {
        let agent = ChatOrchestratorAgent::new().await.unwrap();
        // Add test implementation
    }
}

