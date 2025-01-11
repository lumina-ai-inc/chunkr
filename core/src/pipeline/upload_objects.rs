use crate::models::chunkr::pipeline::Pipeline;
use crate::models::chunkr::task::Status;
use crate::utils::storage::services::upload_to_s3;
use std::error::Error;
use std::io::Write;
use tempfile::NamedTempFile;

fn generate_segment_image_s3_key(pipeline: &Pipeline, segment_id: &str) -> String {
    format!(
        "{}/{}.jpg",
        pipeline
            .task_payload
            .as_ref()
            .unwrap()
            .image_folder_location,
        segment_id
    )
}

/// Upsert the output of the pipeline into S3
///
/// This function will upload the output of the pipeline `task_payload.output_location`
/// and also upload the images for each segment `task_payload.image_folder_location/{segment_id}.jpg`
pub async fn process(pipeline: &mut Pipeline) -> Result<(), Box<dyn Error>> {
    pipeline
        .update_remote_status(Status::Processing, Some("Finishing up".to_string()))
        .await?;

    let mut output_response = pipeline.output.clone();
    for pair in pipeline.segment_images.iter() {
        let s3_key = generate_segment_image_s3_key(pipeline, pair.key());
        upload_to_s3(&s3_key, pair.value().path()).await?;
    }

    output_response.chunks.iter_mut().for_each(|chunk| {
        chunk.segments.iter_mut().for_each(|segment| {
            if pipeline.segment_images.contains_key(&segment.segment_id) {
                segment.image = Some(generate_segment_image_s3_key(pipeline, &segment.segment_id));
            }
        });
    });

    let mut output_temp_file = NamedTempFile::new()?;
    output_temp_file.write(serde_json::to_string(&output_response)?.as_bytes())?;
    upload_to_s3(
        &pipeline.task_payload.as_ref().unwrap().output_location,
        output_temp_file.path(),
    )
    .await?;

    upload_to_s3(
        &pipeline.task_payload.as_ref().unwrap().pdf_location,
        pipeline.pdf_file.as_ref().unwrap().as_ref().path(),
    )
    .await?;

    Ok(())
}
