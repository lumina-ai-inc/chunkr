use serde::{Deserialize, Serialize};
use std::error::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event_type", rename_all = "snake_case")]
pub enum DocumentEvent {
    Uploaded {
        document_id: String,
        deal_id: String,
        user_id: String,
        s3_location: String,
        file_name: String,
        document_type: String,
    },
    OCRStarted {
        document_id: String,
        deal_id: String,
        user_id: String,
    },
    OCRCompleted {
        document_id: String,
        deal_id: String,
        user_id: String,
        page_count: i32,
    },
    FactsExtracted {
        document_id: String,
        deal_id: String,
        user_id: String,
        fact_count: i32,
    },
    ProcessingFailed {
        document_id: String,
        deal_id: String,
        user_id: String,
        error: String,
    },
}

impl DocumentEvent {
    pub fn document_id(&self) -> &str {
        match self {
            DocumentEvent::Uploaded { document_id, .. } => document_id,
            DocumentEvent::OCRStarted { document_id, .. } => document_id,
            DocumentEvent::OCRCompleted { document_id, .. } => document_id,
            DocumentEvent::FactsExtracted { document_id, .. } => document_id,
            DocumentEvent::ProcessingFailed { document_id, .. } => document_id,
        }
    }
    
    pub fn deal_id(&self) -> &str {
        match self {
            DocumentEvent::Uploaded { deal_id, .. } => deal_id,
            DocumentEvent::OCRStarted { deal_id, .. } => deal_id,
            DocumentEvent::OCRCompleted { deal_id, .. } => deal_id,
            DocumentEvent::FactsExtracted { deal_id, .. } => deal_id,
            DocumentEvent::ProcessingFailed { deal_id, .. } => deal_id,
        }
    }
    
    pub fn user_id(&self) -> &str {
        match self {
            DocumentEvent::Uploaded { user_id, .. } => user_id,
            DocumentEvent::OCRStarted { user_id, .. } => user_id,
            DocumentEvent::OCRCompleted { user_id, .. } => user_id,
            DocumentEvent::FactsExtracted { user_id, .. } => user_id,
            DocumentEvent::ProcessingFailed { user_id, .. } => user_id,
        }
    }
}

/// Publish an event to the event bus (Redis)
pub async fn publish_event(event: DocumentEvent) -> Result<(), Box<dyn Error + Send + Sync>> {
    let pool = crate::utils::clients::get_redis_pool();
    let mut conn = pool.get().await?;
    
    let event_json = serde_json::to_string(&event)?;
    let channel = format!("events:documents:{}", event.deal_id());
    
    deadpool_redis::redis::cmd("PUBLISH")
        .arg(&channel)
        .arg(&event_json)
        .query_async::<i64>(&mut conn)
        .await?;
    
    println!("Published event to channel {}: {:?}", channel, event);
    Ok(())
}

/// Subscribe to document events for a specific deal
/// Note: This requires a direct Redis connection, not through deadpool
/// For now, clients should poll the Redis list or use a separate subscription service
pub async fn subscribe_to_deal_events(
    _deal_id: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    // TODO: Implement proper pubsub subscription
    // This requires a direct redis connection, not through deadpool
    // For MVP, we'll use polling instead of pubsub
    Ok(())
}

