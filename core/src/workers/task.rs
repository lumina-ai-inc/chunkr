use core::configs::pdfium_config::Config as PdfiumConfig;
use core::configs::worker_config::Config as WorkerConfig;
use core::models::pipeline::{Pipeline, PipelineStep};
use core::models::task::Status;
use core::models::task::TaskPayload;
use core::utils::clients::get_redis_pool;
use core::utils::clients::initialize;

#[cfg(feature = "memory_profiling")]
use memtrack::track_mem;

/// Orchestrate the task
///
/// This function defines the order of the steps in the pipeline.
fn orchestrate_task(
    _pipeline: &mut Pipeline,
) -> Result<Vec<PipelineStep>, Box<dyn std::error::Error>> {
    let mut steps = vec![PipelineStep::ConvertToImages];

    #[cfg(feature = "azure")]
    {
        match _pipeline.get_task()?.configuration.pipeline.clone() {
            Some(core::models::task::PipelineType::Azure) => {
                steps.push(PipelineStep::AzureAnalysis)
            }
            _ => steps.push(PipelineStep::ChunkrAnalysis),
        }
    }
    #[cfg(not(feature = "azure"))]
    {
        steps.push(PipelineStep::ChunkrAnalysis);
    }

    steps.push(PipelineStep::Crop);
    steps.push(PipelineStep::SegmentProcessing);
    steps.push(PipelineStep::Chunking);
    Ok(steps)
}

#[cfg_attr(feature = "memory_profiling", track_mem)]
pub async fn process(
    task_payload: TaskPayload,
    max_retries: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut pipeline = Pipeline::new();
    pipeline.init(task_payload.clone()).await?;
    if pipeline.get_task()?.status != Status::Processing {
        println!(
            "Skipping task as status is {:?}",
            pipeline.get_task()?.status
        );
        return Ok(());
    }
    let start_time = std::time::Instant::now();
    for step in orchestrate_task(&mut pipeline)? {
        pipeline.execute_step(step, max_retries).await?;
        if pipeline.get_task()?.status != Status::Processing {
            return Ok(());
        }
    }
    let end_time = std::time::Instant::now();
    println!(
        "Task took {:?} to complete with page count {:?}",
        end_time.duration_since(start_time),
        pipeline.get_task()?.page_count.unwrap_or(0)
    );
    pipeline
        .complete(Status::Succeeded, Some("Task succeeded".to_string()))
        .await?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting task processor");
    let config = WorkerConfig::from_env()?;
    PdfiumConfig::from_env()?.ensure_binary().await?;
    initialize().await;
    println!("Listening for tasks on queue: {}", &config.queue_task);

    loop {
        let mut conn = get_redis_pool().get().await.unwrap();
        let result: Option<(String, String)> = redis::cmd("BRPOP")
            .arg(&config.queue_task)
            .arg(0) // 0 = block indefinitely
            .query_async(&mut conn)
            .await?;

        if let Some((_, task_json)) = result {
            {
                match serde_json::from_str::<TaskPayload>(&task_json) {
                    Ok(payload) => match process(payload, config.max_retries).await {
                        Ok(_) => println!("Task processed successfully"),
                        Err(e) => eprintln!("Error processing task: {}", e),
                    },
                    Err(e) => eprintln!("Failed to parse task: {}", e),
                }

                // Force a minor GC via MALLOC_TRIM (requires libc feature)
                #[cfg(target_os = "linux")]
                unsafe {
                    // This tells glibc to release free memory back to the OS
                    libc::malloc_trim(0);
                    #[cfg(feature = "memory_profiling")]
                    println!("Memory trimmed");
                }
            }

            // Create a small delay to give the OS time to reclaim memory
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }
    }
}
