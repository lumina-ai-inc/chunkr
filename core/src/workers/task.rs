use core::configs::pdfium_config::Config as PdfiumConfig;
use core::configs::worker_config::Config as WorkerConfig;
use core::models::pipeline::Pipeline;
use core::models::task::Status;
use core::models::task::TaskPayload;
use core::utils::clients::get_redis_pool;

#[cfg(feature = "azure")]
use core::pipeline::azure;
use core::pipeline::chunking;
use core::pipeline::convert_to_images;
use core::pipeline::crop;
use core::pipeline::segment_processing;
use core::pipeline::segmentation_and_ocr;
// use core::pipeline::structured_extraction;
use core::utils::clients::initialize;

#[cfg(feature = "memory_profiling")]
use memtrack::track_mem;

/// Execute a step in the pipeline
#[cfg_attr(feature = "memory_profiling", track_mem)]
async fn execute_step(
    step: &str,
    pipeline: &mut Pipeline,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Executing step: {}", step);
    let start = std::time::Instant::now();
    match step {
        #[cfg(feature = "azure")]
        "azure" => azure::process(pipeline).await,
        "chunking" => chunking::process(pipeline).await,
        "convert_to_images" => convert_to_images::process(pipeline).await,
        "crop" => crop::process(pipeline).await,
        "segmentation_and_ocr" => segmentation_and_ocr::process(pipeline).await,
        "segment_processing" => segment_processing::process(pipeline).await,
        // "structured_extraction" => structured_extraction::process(pipeline).await,
        _ => Err(format!("Unknown function: {}", step).into()),
    }?;
    let duration = start.elapsed();
    println!(
        "Step {} took {:?} with page count {:?}",
        step,
        duration,
        pipeline.get_task()?.page_count.unwrap_or(0)
    );
    Ok(())
}

/// Orchestrate the task
///
/// This function defines the order of the steps in the pipeline.
fn orchestrate_task(
    pipeline: &mut Pipeline,
) -> Result<Vec<&'static str>, Box<dyn std::error::Error>> {
    let mut steps = vec!["convert_to_images"];
    #[cfg(feature = "azure")]
    {
        match pipeline.get_task()?.configuration.pipeline.clone() {
            Some(core::models::task::PipelineType::Azure) => steps.push("azure"),
            _ => steps.push("segmentation_and_ocr"),
        }
    }
    #[cfg(not(feature = "azure"))]
    {
        steps.push("segmentation_and_ocr");
    }
    let chunk_processing = pipeline.get_task()?.configuration.chunk_processing.clone();
    if chunk_processing.target_length > 1 {
        steps.push("chunking");
    }
    steps.push("crop");
    steps.push("segment_processing");
    // let structured_extraction = pipeline
    //     .get_task()?
    //     .configuration
    //     .structured_extraction
    //     .clone();
    // if structured_extraction.is_some() {
    //     steps.push("structured_extraction");
    // }
    Ok(steps)
}

#[cfg_attr(feature = "memory_profiling", track_mem)]
pub async fn process(task_payload: TaskPayload) -> Result<(), Box<dyn std::error::Error>> {
    let mut pipeline = Pipeline::new();
    let result: Result<(), Box<dyn std::error::Error>> = (async {
        pipeline.init(task_payload).await?;
        if pipeline.get_task()?.status != Status::Processing {
            println!(
                "Skipping task as status is {:?}",
                pipeline.get_task()?.status
            );
            return Ok(());
        }
        let start_time = std::time::Instant::now();
        for step in orchestrate_task(&mut pipeline)? {
            execute_step(step, &mut pipeline).await?;
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
    })
    .await;

    match result {
        Ok(_) => Ok(()),
        Err(e) => {
            let message = "Task failed".to_string();
            pipeline.complete(Status::Failed, Some(message)).await?;
            // match task_payload.attempt >= task_payload.max_attempts {
            //     true => {
            //         let message = "Task failed".to_string();
            //         pipeline.complete(Status::Failed, Some(message)).await?;
            //     }
            //     false => {
            //         let message =
            //             format!("Retrying task {}/{}", payload.attempt, payload.max_attempts);
            //         pipeline
            //             .get_task()?
            //             .update(
            //                 Some(Status::Processing),
            //                 Some(message),
            //                 None,
            //                 None,
            //                 None,
            //                 None,
            //                 None,
            //             )
            //             .await?;
            //     }
            // };
            Err(e)
        }
    }
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
            match serde_json::from_str::<TaskPayload>(&task_json) {
                Ok(payload) => match process(payload).await {
                    Ok(_) => println!("Task processed successfully"),
                    Err(e) => eprintln!("Error processing task: {}", e),
                },
                Err(e) => eprintln!("Failed to parse task: {}", e),
            }
        }
    }
}
