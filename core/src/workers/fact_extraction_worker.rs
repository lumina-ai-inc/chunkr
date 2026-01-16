use core::utils::clients::get_redis_pool;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
struct FactExtractionMessage {
    document_id: String,
    deal_id: String,
    user_id: String,
    ocr_results: serde_json::Value,
}

/// Worker for extracting facts from processed documents using AI
/// NOTE: AI agent integration temporarily disabled pending Diesel to tokio-postgres conversion
pub async fn start_fact_extraction_worker() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting fact extraction worker (AI agent temporarily disabled)");
    let queue_name = "fact_extraction";
    
    println!("Listening for fact extraction jobs on queue: {}", queue_name);
    println!("NOTE: Fact extraction will be queued but not processed until agents are re-enabled");

    loop {
        let mut conn = get_redis_pool().get().await?;
        let result: Option<(String, String)> = deadpool_redis::redis::cmd("BRPOP")
            .arg(queue_name)
            .arg(0) // 0 = block indefinitely
            .query_async(&mut conn)
            .await?;

        if let Some((_, message_json)) = result {
            println!("Received fact extraction message: {}", message_json);
            
            match process_fact_extraction(&message_json).await {
                Ok(_) => println!("Fact extraction job received (processing pending agent re-enable)"),
                Err(e) => eprintln!("Error processing message: {:?}", e),
            }
        }
    }
}

async fn process_fact_extraction(
    message_json: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let message: FactExtractionMessage = serde_json::from_str(message_json)?;
    
    println!(
        "Fact extraction queued for document {} in deal {} (AI processing pending)",
        message.document_id, message.deal_id
    );

    // TODO: Re-enable once agents module is converted to use tokio-postgres
    // For now, just acknowledge the message
    
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Fact Extraction Worker starting...");
    core::utils::clients::initialize().await;
    start_fact_extraction_worker().await
}
