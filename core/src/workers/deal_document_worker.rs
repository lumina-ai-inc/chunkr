use core::configs::worker_config::Config as WorkerConfig;
use core::pipeline::deal_processing::process_document_from_queue;
use core::utils::clients::get_redis_pool;

/// Worker for processing deal documents
/// This is the new simplified worker that replaces the heavy Chunkr pipeline
pub async fn start_deal_document_worker() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting deal document worker");
    let config = WorkerConfig::from_env()?;
    let queue_name = "deal_documents";
    
    println!("Listening for deal documents on queue: {}", queue_name);

    loop {
        let mut conn = get_redis_pool().get().await?;
        let result: Option<(String, String)> = deadpool_redis::redis::cmd("BRPOP")
            .arg(queue_name)
            .arg(0) // 0 = block indefinitely
            .query_async(&mut conn)
            .await?;

        if let Some((_, message_json)) = result {
            println!("Received document processing message: {}", message_json);
            
            match process_document_from_queue(&message_json).await {
                Ok(_) => println!("Document processed successfully"),
                Err(e) => eprintln!("Error processing document: {:?}", e),
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Deal Document Worker starting...");
    core::utils::clients::initialize().await;
    start_deal_document_worker().await
}

