use chunkmydocs::utils::configs::extraction_config;
use chunkmydocs::utils::rrq::consumer::consumer;
use chunkmydocs::utils::workers::structured_extract::process;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = extraction_config::Config::from_env()?;
    println!("Starting structured extraction task processor");
    consumer(process, config.queue_structured_extract, 1, 600).await?;
    Ok(())
}
