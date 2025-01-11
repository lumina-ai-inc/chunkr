use crate::models::chunkr::pipeline::Pipeline;
use crate::models::chunkr::task::Status;
use crate::utils::clients::get_pg_client;
use crate::utils::services::pdf::count_pages;
use std::error::Error;

/// Update the page count and input file mime type for the task
///
/// This function calculates the page count for the task and updates the database with the page count and input file mime type
pub async fn process(pipeline: &mut Pipeline) -> Result<(), Box<dyn Error>> {
    pipeline
        .update_remote_status(Status::Processing, Some("Counting pages".to_string()))
        .await?;
    let client = get_pg_client().await?;
    let pdf_file = pipeline.pdf_file.as_ref().unwrap();
    pipeline.page_count = Some(count_pages(pdf_file)?);
    let task_id = pipeline.task_payload.as_ref().unwrap().task_id.clone();

    let task_query = format!(
        "UPDATE tasks SET page_count = {}, input_file_type = '{}' WHERE task_id = '{}'",
        pipeline.page_count.unwrap(),
        pipeline.mime_type.as_ref().unwrap(),
        task_id
    );

    match client.execute(&task_query, &[]).await {
        Ok(_) => Ok(()),
        Err(e) => {
            if e.to_string().contains("usage limit exceeded") {
                pipeline
                    .update_remote_status(Status::Failed, Some("Page limit exceeded".to_string()))
                    .await?;
                Ok(())
            } else {
                Err(Box::new(e))
            }
        }
    }
}
