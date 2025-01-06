use crate::models::chunkr::pipeline::Pipeline;
use crate::models::chunkr::task::Status;
use crate::utils::storage::services::upload_to_s3;
use std::error::Error;
use std::io::Write;
use tempfile::NamedTempFile;

/// Upsert the output of the pipeline into S3
pub async fn process(pipeline: &mut Pipeline) -> Result<(), Box<dyn Error>> {
    pipeline
        .update_status(Status::Processing, Some("Finishing up".to_string()))
        .await?;

    let output_response = pipeline.output.clone();
    let mut output_temp_file = NamedTempFile::new()?;
    output_temp_file.write(serde_json::to_string(&output_response)?.as_bytes())?;
    upload_to_s3(
        &pipeline.task_payload.as_ref().unwrap().output_location,
        output_temp_file.path(),
    )
    .await?;
    Ok(())
}
