use crate::configs::redis_config::{create_pool as create_redis_pool, Pool};
use crate::models::chunkr::output::{OCRResult, Segment};
use crate::models::chunkr::segmentation::ObjectDetectionResponse;
use crate::utils::rate_limit::{create_segmentation_rate_limiter, RateLimiter};
use crate::utils::retry::retry_with_backoff;
use once_cell::sync::OnceCell;
use std::error::Error;
use std::sync::Arc;
use tempfile::NamedTempFile;

static SEGMENTATION_RATE_LIMITER: OnceCell<RateLimiter> = OnceCell::new();
static POOL: OnceCell<Pool> = OnceCell::new();
static SEGMENTATION_TIMEOUT: OnceCell<u64> = OnceCell::new();
static TOKEN_TIMEOUT: OnceCell<u64> = OnceCell::new();

fn init_throttle() {
    POOL.get_or_init(|| create_redis_pool());
    SEGMENTATION_RATE_LIMITER.get_or_init(|| {
        create_segmentation_rate_limiter(POOL.get().unwrap().clone(), "segmentation")
    });
    SEGMENTATION_TIMEOUT.get_or_init(|| 120);
    TOKEN_TIMEOUT.get_or_init(|| 10000);
}

async fn vgt_segmentation(
    temp_file: Arc<NamedTempFile>,
    ocr_results: Vec<OCRResult>,
) -> Result<ObjectDetectionResponse, Box<dyn Error + Send + Sync>> {
    todo!()
}

async fn perform_segmentation(
    temp_file: Arc<NamedTempFile>,
    ocr_results: Vec<OCRResult>,
) -> Result<ObjectDetectionResponse, Box<dyn Error + Send + Sync>> {
    init_throttle();
    let rate_limiter = SEGMENTATION_RATE_LIMITER.get().unwrap();
    Ok(retry_with_backoff(|| async {
        rate_limiter
            .acquire_token_with_timeout(std::time::Duration::from_secs(
                *TOKEN_TIMEOUT.get().unwrap(),
            ))
            .await?;
        vgt_segmentation(Arc::clone(&temp_file), ocr_results.clone()).await
    })
    .await?)
}

pub async fn create_segments(
    temp_file: Arc<NamedTempFile>,
    ocr_results: Vec<OCRResult>,
    page_number: u32,
) -> Result<Vec<Segment>, Box<dyn Error + Send + Sync>> {
    let response = perform_segmentation(temp_file, ocr_results.clone()).await?;
    let segments = response.instances.to_segments(page_number, ocr_results);
    Ok(segments)
}
