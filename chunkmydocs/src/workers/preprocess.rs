use chunkmydocs::utils::configs::extraction_config;
use chunkmydocs::utils::rrq::consumer::consumer;
use chunkmydocs::utils::workers::segmentation::process;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = extraction_config::Config::from_env()?;
    println!("Starting task processor");
    consumer(process, config.queue_preprocess.unwrap(), 1, 600).await?;
    Ok(())
}
