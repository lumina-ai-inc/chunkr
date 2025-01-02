use crate::models::chunkr::pipeline::Pipeline;
use crate::models::chunkr::task::Status;
use crate::utils::db::deadpool_postgres::Pool;
use crate::utils::services::pdf::count_pages;
use std::error::Error;

/// Update the page count and input file mime type for the task
///
/// This function calculates the page count for the task and updates the database with the page count and input file mime type
pub async fn process(
    pipeline: &mut Pipeline,
    pool: &Pool,
) -> Result<(Status, Option<String>), Box<dyn Error>> {
    let client = pool.get().await?;
    let pdf_file = pipeline.pdf_file.as_ref();
    pipeline.page_count = Some(count_pages(pdf_file)?);
    let task_id = pipeline.task_payload.task_id.clone();

    let task_query = format!(
        "UPDATE tasks SET page_count = {}, input_file_type = '{}' WHERE task_id = '{}'",
        pipeline.page_count.unwrap(),
        pipeline.mime_type,
        task_id
    );

    match client.execute(&task_query, &[]).await {
        Ok(_) => Ok((
            Status::Processing,
            Some(format!("Page count: {}", pipeline.page_count.unwrap())),
        )),
        Err(e) => {
            if e.to_string().contains("usage limit exceeded") {
                Ok((Status::Failed, Some("Page limit exceeded".to_string())))
            } else {
                Err(Box::new(e))
            }
        }
    }
}
