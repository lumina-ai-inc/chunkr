use chunkmydocs::utils::configs::worker_config;
use chunkmydocs::utils::rrq::consumer::consumer;
use chunkmydocs::utils::workers::structured_extraction::process;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = worker_config::Config::from_env()?;
    println!("Starting structured extraction task processor");
    consumer(process, config.queue_structured_extraction, 1, 1200).await?;
    Ok(())
}
