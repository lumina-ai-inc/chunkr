use chunkmydocs::utils::configs::extraction_config;
use chunkmydocs::utils::rrq::consumer::consumer;
use chunkmydocs::utils::workers::segmentation::process;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = extraction_config::Config::from_env()?;
    println!("Starting OCR processor");
    consumer(process, config.extraction_queue_ocr.unwrap(), 1, 600).await?;
    Ok(())
}
