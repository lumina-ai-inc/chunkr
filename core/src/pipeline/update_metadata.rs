use crate::models::chunkr::pipeline::Pipeline;
use crate::models::chunkr::task::Status;
use crate::utils::clients::get_pg_client;
use crate::utils::services::pdf::count_pages;
use std::error::Error;

/// Update the page count for the task
///
/// This function calculates the page count for the task and updates the database with the page count
pub async fn process(pipeline: &mut Pipeline) -> Result<(), Box<dyn Error>> {
    pipeline
        .get_task()
        .update(
            Some(Status::Processing),
            Some("Counting pages".to_string()),
            None,
            None,
            None,
            None,
        )
        .await?;
    let client = get_pg_client().await?;
    let pdf_file = pipeline.pdf_file.as_ref().unwrap();
    let mut task = pipeline.get_task();
    let task_id = task.task_id.clone();
    let page_count = count_pages(pdf_file)?;
    let task_query = format!(
        "UPDATE tasks SET page_count = {} WHERE task_id = '{}'",
        page_count, task_id
    );
    match client.execute(&task_query, &[]).await {
        Ok(_) => {
            task.page_count = Some(page_count);
            pipeline.task = Some(task.clone());
            Ok(())
        }
        Err(e) => {
            if e.to_string().contains("usage limit exceeded") {
                pipeline
                    .get_task()
                    .update(
                        Some(Status::Failed),
                        Some("Page limit exceeded".to_string()),
                        None,
                        None,
                        None,
                        None,
                    )
                    .await?;
                Ok(())
            } else {
                Err(Box::new(e))
            }
        }
    }
}
