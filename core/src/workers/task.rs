use core::configs::pdfium_config::Config as PdfiumConfig;
use core::configs::worker_config::Config as WorkerConfig;
use core::configs::{job_config, otel_config};
use core::models::pipeline::{Pipeline, PipelineStep};
use core::models::task::TaskPayload;
use core::models::task::{Status, Task};
use core::utils::clients::get_redis_pool;
use core::utils::clients::initialize;
use opentelemetry::trace::{Span, TraceContextExt, Tracer};
use opentelemetry::{global, Context, KeyValue};

#[cfg(feature = "memory_profiling")]
use memtrack::track_mem;

/// Orchestrate the task
///
/// This function defines the order of the steps in the pipeline.
fn orchestrate_task(
    pipeline: &mut Pipeline,
) -> Result<Vec<PipelineStep>, Box<dyn std::error::Error>> {
    let is_spreadsheet = pipeline.get_task()?.is_spreadsheet;
    let mut steps = vec![];
    if is_spreadsheet {
        steps.push(PipelineStep::ConvertExcelToHtml);
        steps.push(PipelineStep::IdentifyTablesInSheet);
    } else {
        steps.push(PipelineStep::ConvertToImages);
        #[cfg(feature = "azure")]
        {
            match pipeline.get_task()?.configuration.pipeline.clone() {
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
    tracer: opentelemetry::global::BoxedTracer,
) -> Result<(), Box<dyn std::error::Error>> {
    let start_time = std::time::Instant::now();
    let timeout_duration =
        std::time::Duration::from_secs(job_config::Config::from_env()?.task_timeout.into());
    println!("Timeout duration: {timeout_duration:?}");
    let mut pipeline = Pipeline::new();

    let process_result = match tokio::time::timeout(timeout_duration, async {
        let mut pipeline_init_span = tracer.start_with_context(
            otel_config::SpanName::PipelineInit.to_string(),
            &Context::current(),
        );
        match pipeline.init(task_payload.clone()).await {
            Ok(_) => {
                if let Some(page_count) = pipeline.get_task()?.page_count {
                    opentelemetry::Context::current()
                        .span()
                        .set_attribute(KeyValue::new("page_count", i64::from(page_count)));
                }
                if let Some(mime_type) = pipeline.get_task()?.mime_type.clone() {
                    opentelemetry::Context::current()
                        .span()
                        .set_attribute(KeyValue::new("mime_type", mime_type));
                }
            }
            Err(e) => {
                let mut task =
                    Task::get(&task_payload.task_id, &task_payload.user_info.user_id).await?;
                let message = match e.to_string().contains("LibreOffice") {
                    true => "Failed to convert file to PDF".to_string(),
                    false => "Failed to initialize task".to_string(),
                };
                if task.status == Status::Processing {
                    task.update(
                        Some(Status::Failed),
                        Some(message),
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                    )
                    .await?;
                }
                return Err(e);
            }
        }
        pipeline_init_span.end();
        let status = pipeline.get_task()?.status;
        if status != Status::Processing {
            println!("Skipping task as status is {status:?}");
            opentelemetry::Context::current().span().add_event(
                "task_skipped",
                vec![opentelemetry::KeyValue::new("status", status.to_string())],
            );
            return Ok(());
        }

        for step in orchestrate_task(&mut pipeline)? {
            pipeline.execute_step(step, max_retries, &tracer).await?;
            if pipeline.get_task()?.status != Status::Processing {
                return Ok::<(), Box<dyn std::error::Error>>(());
            }
        }
        Ok(())
    })
    .await
    {
        Ok(result) => {
            result?;
            pipeline
                .complete(Status::Succeeded, Some("Task succeeded".to_string()))
                .await
        }
        Err(e) => {
            // Update task status to Failed when timeout occurs in the worker
            println!("Task timed out after {timeout_duration:?}");
            match Task::get(&task_payload.task_id, &task_payload.user_info.user_id).await {
                Ok(mut task) => {
                    if let Err(update_err) = task
                        .update(
                            Some(Status::Failed),
                            Some("Task timed out".to_string()),
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                        )
                        .await
                    {
                        println!("Failed to update task status on timeout: {update_err}");
                    } else {
                        println!(
                            "Updated task {} to Failed status due to timeout",
                            task_payload.task_id
                        );
                    }
                }
                Err(get_err) => {
                    println!("Failed to get task for timeout update: {get_err}");
                }
            }
            // NOTE: The timeout CRON job serves as a backup to catch any tasks that might be missed
            Err(Box::new(e) as Box<dyn std::error::Error>)
        }
    };

    let end_time = std::time::Instant::now();
    let pages_per_second = pipeline.get_task()?.page_count.unwrap_or(0) as f64
        / end_time.duration_since(start_time).as_secs() as f64;
    println!(
        "Task took {:?} to complete with page count {:?} and page per second {:?}",
        end_time.duration_since(start_time),
        pipeline.get_task()?.page_count.unwrap_or(0),
        pages_per_second
    );
    opentelemetry::Context::current()
        .span()
        .set_attribute(KeyValue::new("pages_per_second", pages_per_second));
    process_result?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting task processor");
    let config = WorkerConfig::from_env()?;
    PdfiumConfig::from_env()?.ensure_binary().await?;
    initialize().await;
    if let Err(e) = core::configs::otel_config::Config::from_env()
        .map(|config| config.init_tracer(core::configs::otel_config::ServiceName::TaskWorker))
        .map_err(|e| e.to_string())
    {
        eprintln!("Failed to initialize OpenTelemetry tracer: {e}");
    }
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
                    Ok(payload) => {
                        let parent_context = core::configs::otel_config::Config::inject_context(
                            payload.trace_context.clone(),
                        );
                        let tracer =
                            global::tracer(otel_config::ServiceName::TaskWorker.to_string());
                        let mut span = tracer.start_with_context(
                            otel_config::SpanName::ProcessTask.to_string(),
                            &parent_context,
                        );
                        span.set_attribute(KeyValue::new("task_id", payload.task_id.clone()));
                        for attribute in payload.user_info.get_attributes() {
                            span.set_attribute(attribute);
                        }
                        let _guard = parent_context.with_span(span).attach();
                        match process(payload, config.max_retries, tracer).await {
                            Ok(_) => {
                                println!("Task processed successfully");
                            }
                            Err(e) => {
                                eprintln!("Error processing task: {e}");
                                let context = opentelemetry::Context::current();
                                let span = context.span();
                                span.set_status(opentelemetry::trace::Status::error(e.to_string()));
                                span.record_error(e.as_ref());
                                span.set_attribute(KeyValue::new("error", e.to_string()));
                            }
                        }
                    }
                    Err(e) => eprintln!("Failed to parse task: {e}"),
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
