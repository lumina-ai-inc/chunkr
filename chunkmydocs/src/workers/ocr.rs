use chunkmydocs::utils::configs::worker_config;
use chunkmydocs::utils::rrq::consumer::consumer;
use chunkmydocs::utils::workers::ocr::process;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = worker_config::Config::from_env()?;
    println!("Starting OCR processor");
    consumer(process, config.queue_ocr, 1, 1200).await?;
    Ok(())
}
