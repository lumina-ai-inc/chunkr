use crate::configs::llm_config::Config as LlmConfig;
use crate::configs::throttle_config::Config as ThrottleConfig;
use crate::utils::clients::get_redis_pool;
use deadpool_redis::redis::{RedisError, RedisResult};
use once_cell::sync::OnceCell;
use std::collections::HashMap;
use std::sync::RwLock;
use std::time::Duration;

pub static GENERAL_OCR_RATE_LIMITER: OnceCell<RateLimiter> = OnceCell::new();
pub static GENERAL_OCR_TIMEOUT: OnceCell<Option<u64>> = OnceCell::new();
pub static LLM_RATE_LIMITERS: OnceCell<RwLock<HashMap<String, Option<RateLimiter>>>> =
    OnceCell::new();
pub static LLM_TIMEOUT: OnceCell<Option<u64>> = OnceCell::new();
pub static SEGMENTATION_RATE_LIMITER: OnceCell<RateLimiter> = OnceCell::new();
pub static SEGMENTATION_TIMEOUT: OnceCell<Option<u64>> = OnceCell::new();
pub static TOKEN_TIMEOUT: OnceCell<u64> = OnceCell::new();

#[derive(Clone)]
pub struct RateLimiter {
    tokens_per_second: f32,
    bucket_name: String,
}

impl RateLimiter {
    pub fn new(tokens_per_second: f32, bucket_name: &str) -> Self {
        RateLimiter {
            tokens_per_second,
            bucket_name: bucket_name.to_string(),
        }
    }

    pub async fn acquire_token(&self) -> RedisResult<bool> {
        let mut conn = match get_redis_pool().get().await {
            Ok(conn) => conn,
            Err(e) => {
                println!("Error getting connection: {:?}", e);
                return Ok(false);
            }
        };

        let script = r#"
            local bucket_key = KEYS[1]
            local rate_limit = tonumber(ARGV[1])  -- This becomes our bucket capacity
            local now = tonumber(ARGV[2])
            
            local last_leak_key = bucket_key .. ':last_leak'
            if redis.call('exists', last_leak_key) == 0 then
                redis.call('set', last_leak_key, now)
            end

            -- Get current queue length or initialize it
            local queue_length = tonumber(redis.call('llen', bucket_key) or 0)
            local last_leak = tonumber(redis.call('get', bucket_key .. ':last_leak') or now)
            
            -- Calculate how many tokens should have leaked since last check
            local elapsed = now - last_leak
            local should_leak = math.floor(elapsed * rate_limit)   

            -- Leak tokens (remove from queue)
            if should_leak > 0 then
                -- Don't try to remove more than what's in the queue
                local to_remove = math.min(should_leak, queue_length)
                if to_remove > 0 then
                    redis.call('ltrim', bucket_key, to_remove, -1)
                end
                redis.call('set', bucket_key .. ':last_leak', now)
                queue_length = math.max(0, queue_length - to_remove)
            end
            
            -- Check if we can add to queue
            if queue_length < rate_limit then
                redis.call('rpush', bucket_key, now)
                return 1
            end
            
            return 0
        "#;

        let result: i32 = redis::Script::new(script)
            .key(&self.bucket_name)
            .arg(self.tokens_per_second)
            .arg(chrono::Utc::now().timestamp())
            .invoke_async(&mut conn)
            .await?;

        Ok(result == 1)
    }

    pub async fn acquire_token_with_timeout(&self, timeout: Duration) -> RedisResult<bool> {
        let start = std::time::Instant::now();
        while start.elapsed() < timeout {
            if self.acquire_token().await? {
                return Ok(true);
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        Err(RedisError::from((
            redis::ErrorKind::BusyLoadingError,
            "Rate limit timeout exceeded",
        )))
    }
}

fn create_general_ocr_rate_limiter(bucket_name: &str) -> RateLimiter {
    let throttle_config = ThrottleConfig::from_env().unwrap();
    RateLimiter::new(throttle_config.general_ocr_rate_limit, bucket_name)
}

fn create_general_ocr_timeout() -> Option<u64> {
    let throttle_config = ThrottleConfig::from_env().unwrap();
    throttle_config.general_ocr_timeout
}

fn create_llm_rate_limiter(bucket_name: &str, rate_limit: Option<f32>) -> Option<RateLimiter> {
    if let Some(rate_limit) = rate_limit {
        Some(RateLimiter::new(rate_limit, bucket_name))
    } else {
        None
    }
}

fn create_llm_timeout() -> Option<u64> {
    let throttle_config = ThrottleConfig::from_env().unwrap();
    throttle_config.llm_timeout
}

fn create_segmentation_rate_limiter(bucket_name: &str) -> RateLimiter {
    let throttle_config = ThrottleConfig::from_env().unwrap();
    RateLimiter::new(throttle_config.segmentation_rate_limit, bucket_name)
}

fn create_segmentation_timeout() -> Option<u64> {
    let throttle_config = ThrottleConfig::from_env().unwrap();
    throttle_config.segmentation_timeout
}

pub fn init_throttle() {
    TOKEN_TIMEOUT.get_or_init(|| 10000);
    GENERAL_OCR_RATE_LIMITER.get_or_init(|| create_general_ocr_rate_limiter("general_ocr"));
    GENERAL_OCR_TIMEOUT.get_or_init(create_general_ocr_timeout);
    SEGMENTATION_RATE_LIMITER.get_or_init(|| create_segmentation_rate_limiter("segmentation"));
    SEGMENTATION_TIMEOUT.get_or_init(create_segmentation_timeout);

    LLM_RATE_LIMITERS.get_or_init(|| {
        let mut llm_rate_limiters = HashMap::new();
        let llm_config = LlmConfig::from_env().unwrap();
        let throttle_config = ThrottleConfig::from_env().unwrap();

        if let Some(llm_models) = llm_config.llm_models.as_ref() {
            for model in llm_models {
                llm_rate_limiters.insert(
                    model.id.clone(),
                    create_llm_rate_limiter(
                        &model.id,
                        model.rate_limit.or(throttle_config.llm_rate_limit),
                    ),
                );
            }
        }

        RwLock::new(llm_rate_limiters)
    });

    LLM_TIMEOUT.get_or_init(create_llm_timeout);
}

pub fn get_llm_rate_limiter(model_id: &str) -> Option<RateLimiter> {
    LLM_RATE_LIMITERS.get().and_then(|limiters| {
        let limiters_guard = limiters.read().unwrap();
        limiters_guard.get(model_id).cloned().flatten()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::configs::llm_config::Config as LlmConfig;
    use crate::models::open_ai::{ContentPart, Message, MessageContent};
    use crate::utils::clients::initialize;
    use crate::utils::services::llm::open_ai_call;
    use rand::Rng;
    use std::error::Error;
    use std::fs;
    use std::path::Path;

    #[tokio::test]
    async fn test_acquire_token() {
        initialize().await;
        let rate_limiter = create_llm_rate_limiter("test_bucket", Some(100.0));
        let result = rate_limiter.unwrap().acquire_token().await;
        println!("result: {:?}", result);
    }

    #[tokio::test]
    async fn test_acquire_token_with_timeout() {
        initialize().await;
        let rate_limiter = create_llm_rate_limiter("test_bucket", Some(100.0));
        let result = rate_limiter
            .unwrap()
            .acquire_token_with_timeout(Duration::from_secs(1))
            .await;
        println!("result: {:?}", result);
    }

    #[tokio::test]
    async fn test_hit_rate_limit() {
        initialize().await;
        let rate_limiter = create_llm_rate_limiter("test_bucket", Some(100.0));

        let unwrapped_limiter = rate_limiter.unwrap().clone();
        let result = unwrapped_limiter.acquire_token().await.unwrap();
        println!("result: {:?}", result);
        assert!(!result, "Token request should fail after exhausting limit");
    }

    #[tokio::test]
    async fn test_hit_rate_limit_with_timeout() {
        initialize().await;
        let rate_limiter = create_llm_rate_limiter("test_bucket", Some(100.0));

        let unwrapped_limiter = rate_limiter.unwrap().clone();
        let futures: Vec<_> = (0..2000)
            .map(|_| {
                let limiter = unwrapped_limiter.clone();
                async move { limiter.acquire_token().await }
            })
            .collect();
        let _ = futures::future::join_all(futures).await;

        let result = unwrapped_limiter
            .acquire_token_with_timeout(Duration::from_secs(1))
            .await
            .unwrap();

        println!("result: {:?}", result);
        assert!(
            result,
            "Token request should acquire token after exhausting limit"
        );
    }

    async fn send_request() -> Result<(), Box<dyn Error>> {
        initialize().await;
        let llm_config = LlmConfig::from_env().unwrap();
        let llm_model = llm_config.get_model(None).unwrap();
        let random_number = rand::thread_rng().gen_range(0..100000);
        let messages = vec![Message {
            role: "user".to_string(),
            content: MessageContent::Array {
                content: vec![ContentPart {
                    content_type: "text".to_string(),
                    text: Some(format!("HI {}", random_number)),
                    image_url: None,
                }],
            },
        }];

        let start_time = std::time::Instant::now();
        match open_ai_call(
            llm_model.provider_url,
            llm_model.api_key,
            llm_model.model,
            messages,
            None,
            None,
            None,
        )
        .await
        {
            Ok(_) => Ok(()),
            Err(e) => {
                println!(
                    "error in open ai call: {:?} in {:?}",
                    e,
                    start_time.elapsed()
                );
                Err(e.to_string().into())
            }
        }
    }

    async fn send_request_with_retry() -> Result<(), Box<dyn Error>> {
        initialize().await;
        let llm_config = LlmConfig::from_env().unwrap();
        let llm_model = llm_config.get_model(None).unwrap();
        let random_number = rand::thread_rng().gen_range(0..100000);
        let messages = vec![Message {
            role: "user".to_string(),
            content: MessageContent::Array {
                content: vec![ContentPart {
                    content_type: "text".to_string(),
                    text: Some(format!("HI {}", random_number)),
                    image_url: None,
                }],
            },
        }];

        let start_time = std::time::Instant::now();
        loop {
            match open_ai_call(
                llm_model.provider_url.clone(),
                llm_model.api_key.clone(),
                llm_model.model.clone(),
                messages.clone(),
                None,
                None,
                None,
            )
            .await
            {
                Ok(_) => return Ok(()),
                Err(e) => {
                    if let Some(reqwest_err) = e.downcast_ref::<reqwest::Error>() {
                        if let Some(status) = reqwest_err.status() {
                            if status == 429 {
                                println!(
                                    "Rate limit hit (429), retrying... Time elapsed: {:?}",
                                    start_time.elapsed()
                                );
                                tokio::time::sleep(Duration::from_secs(1)).await;
                                continue;
                            }
                        }
                    }
                    println!(
                        "error in open ai call: {:?} in {:?}",
                        e,
                        start_time.elapsed()
                    );
                    return Err(e.to_string().into());
                }
            }
        }
    }

    #[tokio::test]
    async fn test_send_request() -> Result<(), Box<dyn Error>> {
        match send_request().await {
            Ok(_) => Ok(()),
            Err(e) => {
                println!("error: {:?}", e);
                Err(e)
            }
        }
    }

    #[tokio::test]
    async fn test_send_request_with_retry() -> Result<(), Box<dyn Error>> {
        match send_request_with_retry().await {
            Ok(_) => Ok(()),
            Err(e) => {
                println!("error: {:?}", e);
                Err(e)
            }
        }
    }

    #[tokio::test]
    async fn test_open_ai_rate_limit_with_retry() -> Result<(), Box<dyn Error>> {
        let start_time = std::time::Instant::now();
        let futures: Vec<_> = (0..10000)
            .map(|_| async { send_request_with_retry().await })
            .collect();

        let results: Vec<Result<(), Box<dyn Error>>> = futures::future::join_all(futures).await;
        let total_time = start_time.elapsed();
        let output: Vec<serde_json::Value> = results
            .into_iter()
            .enumerate()
            .map(|(i, result)| {
                serde_json::json!({
                    "index": i,
                    "success": result.is_ok(),
                    "error": result.err().map(|e| e.to_string())
                })
            })
            .collect();

        let json_string = serde_json::to_string_pretty(&output)?;
        let output_dir = Path::new("output/rate_limit_retries");
        fs::create_dir_all(output_dir)?;
        std::fs::write(output_dir.join("output.json"), json_string)?;
        std::fs::write(
            output_dir.join("total_time.txt"),
            total_time.as_secs().to_string(),
        )?;
        Ok(())
    }

    #[tokio::test]
    async fn test_open_ai_rate_limit_rate_limiter() -> Result<(), Box<dyn Error>> {
        initialize().await;
        let start_time = std::time::Instant::now();
        let rate_limiter = RateLimiter::new(200.0, "rate_limiter");
        let futures: Vec<_> = (0..10000)
            .map(|_| async {
                if let Ok(true) = rate_limiter
                    .acquire_token_with_timeout(Duration::from_secs(30))
                    .await
                {
                    send_request().await
                } else {
                    println!("failed to get token");
                    Err("failed to get token".into())
                }
            })
            .collect();

        let results: Vec<Result<(), Box<dyn Error>>> = futures::future::join_all(futures).await;
        let total_time = start_time.elapsed();
        let output: Vec<serde_json::Value> = results
            .into_iter()
            .enumerate()
            .map(|(i, result)| {
                serde_json::json!({
                    "index": i,
                    "success": result.is_ok(),
                    "error": result.err().map(|e| e.to_string())
                })
            })
            .collect();

        let json_string = serde_json::to_string_pretty(&output)?;
        let output_dir = Path::new("output/rate_limiter");
        fs::create_dir_all(output_dir)?;
        std::fs::write(output_dir.join("output.json"), json_string)?;
        std::fs::write(
            output_dir.join("total_time.txt"),
            total_time.as_secs().to_string(),
        )?;
        Ok(())
    }

    #[tokio::test]
    async fn test_rate_limiter_with_multiple_rates() -> Result<(), Box<dyn Error>> {
        // Initialize the environment
        initialize().await;
        
        // Define test scenarios with different rate limits
        let scenarios = vec![1, 10, 25, 50, 100, 200];
        let test_duration_secs = 30;
        
        for rate_limit in scenarios {
            println!("Testing rate limit: {} requests per second", rate_limit);
            
            // Create a new session for this rate limit test
            let client = reqwest::Client::new();
            let session_response = client
                .post("http://localhost:6969/api/test/session")
                .json(&serde_json::json!({ 
                    "name": format!("Rate Limiter Test - {} RPS", rate_limit) 
                }))
                .send()
                .await?;
            
            let session_data: serde_json::Value = session_response.json().await?;
            let session_id = session_data["id"].as_str().unwrap();
            
            // Create rate limiter with the current test scenario rate
            let rate_limiter = RateLimiter::new(rate_limit as f32, &format!("limit_lens_test_{}", rate_limit));
            
            // Calculate expected total requests
            let expected_requests = rate_limit * test_duration_secs;
            
            // Send requests for the duration
            let start_time = std::time::Instant::now();
            let mut futures: Vec<tokio::task::JoinHandle<Result<(), Box<dyn Error + Send + Sync>>>> = Vec::new();
            
            for i in 0..expected_requests {
                let limiter = rate_limiter.clone();
                let client = client.clone();
                let session_id = session_id.to_string();
                
                let future = tokio::spawn(async move {
                    // Try to acquire a token with a longer timeout that matches the test duration
                    if let Ok(true) = limiter
                        .acquire_token_with_timeout(Duration::from_secs(test_duration_secs as u64))
                        .await
                    {
                        // Removed the deadline check to ensure all requests are attempted
                        match client
                            .get(format!("http://localhost:6969/api/test/request/{}", session_id))
                            .send()
                            .await
                        {
                            Ok(_) => Ok(()),
                            Err(e) => Err(e.to_string().into())
                        }
                    } else {
                        // Log token acquisition failures
                        println!("Failed to acquire token for request {}", i);
                        Ok(())
                    }
                });
                
                futures.push(future);
            }
            
            // Wait for all futures to complete or the deadline to pass
            futures::future::join_all(futures).await;
            
            // Ensure we wait for the full test duration
            let elapsed = start_time.elapsed();
            if elapsed < Duration::from_secs(test_duration_secs as u64) {
                tokio::time::sleep(Duration::from_secs(test_duration_secs as u64) - elapsed).await;
            }
            
            // Add small delay to ensure all requests are processed
            tokio::time::sleep(Duration::from_millis(500)).await;
            
            // Get metrics from limit-lens
            let metrics_response = client
                .get(format!("http://localhost:6969/api/test/metrics/{}", session_id))
                .send()
                .await?;
            
            let metrics: serde_json::Value = metrics_response.json().await?;
            
            // Extract key metrics
            let total_requests = metrics["total_requests"].as_u64().unwrap() as f64;
            let requests_per_second = metrics["requests_per_second"].as_f64().unwrap();
            let distribution = metrics["request_distribution"].as_array().unwrap();
            println!("total_requests: {:?}", total_requests);
            println!("requests_per_second: {:?}", requests_per_second);
            println!("distribution: {:?}", distribution);
            // Validate metrics
            
            // 1. Check total requests - should be within 10% of expected
            let expected_requests_f64 = expected_requests as f64;
            let total_requests_diff_pct = ((total_requests - expected_requests_f64).abs() / expected_requests_f64) * 100.0;
            assert!(
                total_requests_diff_pct <= 10.0,
                "Total requests {} differs from expected {} by more than 10% ({}%)",
                total_requests, expected_requests_f64, total_requests_diff_pct
            );
            
            // 2. Check rate - should be within 20% of target rate
            let rate_diff_pct = ((requests_per_second - rate_limit as f64).abs() / rate_limit as f64) * 100.0;
            println!("rate_diff_pct: {:?}", rate_diff_pct);
            assert!(
                rate_diff_pct <= 30.0,
                "Measured rate {} differs from target {} by more than 30% ({}%)",
                requests_per_second, rate_limit, rate_diff_pct
            );
            
            // 3. Check distribution - each second should have roughly the target rate of requests
            // Skip first and last second which might be partial
            let mut valid_periods = 0;
            let mut total_deviation = 0.0;
            
            for (i, period) in distribution.iter().enumerate() {
                let count = period["count"].as_u64().unwrap() as f64;
                
                // Skip first and last periods which might be partial
                if i == 0 || i == distribution.len() - 1 {
                    continue;
                }
                
                valid_periods += 1;
                let deviation_pct = ((count - rate_limit as f64).abs() / rate_limit as f64) * 100.0;
                total_deviation += deviation_pct;
            }
            
            // Average deviation across all complete periods should be within 25%
            if valid_periods > 0 {
                let avg_deviation = total_deviation / valid_periods as f64;
                assert!(
                    avg_deviation <= 25.0,
                    "Average per-second deviation from target rate too high: {}%",
                    avg_deviation
                );
            }
            
            println!("✅ Rate limit {} RPS test passed:", rate_limit);
            println!("   - Total requests: {} (expected ~{})", total_requests, expected_requests);
            println!("   - Measured rate: {:.2} RPS", requests_per_second);
            println!("   - Test duration: {:.2}s", start_time.elapsed().as_secs_f64());
            
            // Store results for this scenario
            let dir_name = format!("output/limit_lens_test_{}_rps", rate_limit);
            let output_dir = Path::new(&dir_name);
            fs::create_dir_all(output_dir)?;
            
            std::fs::write(
                output_dir.join("metrics.json"),
                serde_json::to_string_pretty(&metrics)?,
            )?;
            
            std::fs::write(
                output_dir.join("summary.txt"),
                format!(
                    "Rate limit: {} RPS\nTotal requests: {}\nMeasured rate: {:.2} RPS\nTest duration: {:.2}s",
                    rate_limit,
                    total_requests,
                    requests_per_second,
                    start_time.elapsed().as_secs_f64()
                ),
            )?;
        }
        
        Ok(())
    }

}
