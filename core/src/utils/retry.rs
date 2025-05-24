use crate::configs::worker_config::Config as WorkerConfig;
use std::time::Duration;
use tokio::time::sleep;

/// Retry a function with exponential backoff.
///
/// If the function returns an error, it will retry the function up to `max_retries` times.
/// The delay between retries is exponential, starting at 1 seconds and capped at 10 seconds.
pub async fn retry_with_backoff<T, E, Fut, F>(f: F) -> Result<T, E>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<T, E>>,
    E: std::fmt::Debug,
{
    let worker_config = WorkerConfig::from_env().unwrap();
    let max_retries = worker_config.max_retries;
    let mut retries = 1;
    loop {
        match f().await {
            Ok(result) => return Ok(result),
            Err(e) => {
                if retries >= max_retries {
                    return Err(e);
                }
                let delay = Duration::from_millis((2u64.pow(retries) * 250).min(10000));
                retries += 1;
                println!(
                    "Error: {:?}. Retrying ({}/{}) in {:?}...",
                    e, retries, max_retries, delay
                );
                sleep(delay).await;
            }
        }
    }
}
