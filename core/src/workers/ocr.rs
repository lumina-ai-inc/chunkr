use core::utils::configs::worker_config;
use core::utils::rrq::consumer::consumer;
use core::utils::workers::ocr::process;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = worker_config::Config::from_env()?;
    println!("Starting OCR processor");
    consumer(process, config.queue_ocr, 1, 1200).await?;
    Ok(())
}
