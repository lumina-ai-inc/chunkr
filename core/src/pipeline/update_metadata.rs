use crate::models::chunkr::pipeline::Pipeline;
use crate::models::chunkr::task::Status;
use crate::utils::services::pdf::count_pages;
use std::error::Error;

/// Update the page count for the task
///
/// This function calculates the page count for the task and updates the database with the page count
pub async fn process(pipeline: &mut Pipeline) -> Result<(), Box<dyn Error>> {
    let mut task = pipeline.get_task()?;
    task.update(
        Some(Status::Processing),
        Some("Counting pages".to_string()),
        None,
        None,
        None,
        None,
        None,
    )
    .await?;
    let pdf_file = pipeline.pdf_file.as_ref().unwrap();
    let page_count = count_pages(pdf_file)?;
    task.update(None, None, None, Some(page_count), None, None, None)
        .await?;
    Ok(())
}
