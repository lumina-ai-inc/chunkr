use crate::models::chunkr::pipeline::Pipeline;
use crate::models::chunkr::task::Status;
use crate::utils::db::deadpool_postgres::{Client, Pool};
use crate::utils::services::pdf::count_pages;
use std::error::Error;

pub async fn process(
    pipeline: &mut Pipeline,
    pool: &Pool,
) -> Result<(Status, Option<String>), Box<dyn Error>> {
    let client: Client = pool.get().await?;
    let pdf_file = pipeline.pdf_file.as_ref().unwrap();
    let page_count = count_pages(pdf_file)?;
    let task_id = pipeline.task_id.clone();

    let task_query = format!(
        "UPDATE tasks SET page_count = {} WHERE task_id = '{}'",
        page_count, task_id
    );

    match client.execute(&task_query, &[]).await {
        Ok(_) => Ok((
            Status::Processing,
            Some(format!("Page count: {}", page_count)),
        )),
        Err(e) => {
            if e.to_string().contains("Page usage limit exceeded") {
                Ok((Status::Failed, Some("Page limit exceeded".to_string())))
            } else {
                Err(Box::new(e))
            }
        }
    }
}
