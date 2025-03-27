use crate::models::pipeline::Pipeline;
use crate::models::search::ChunkContent;
use crate::models::structured_extraction::StructuredExtractionRequest;
use crate::models::task::Status;
use crate::utils::services::structured_extraction::perform_structured_extraction;
use std::error::Error;
pub async fn process(pipeline: &mut Pipeline) -> Result<(), Box<dyn Error>> {
    let mut task = pipeline.get_task()?;
    task.update(
        Some(Status::Processing),
        Some("Structured extraction started".to_string()),
        None,
        None,
        None,
        None,
        None,
    )
    .await?;
    let mut output_response = pipeline.output.clone();
    let structured_extraction_request = StructuredExtractionRequest {
        contents: output_response
            .chunks
            .iter()
            .map(|chunk| ChunkContent::Full(chunk.clone()))
            .collect(),
        structured_extraction: task.configuration.structured_extraction.unwrap(),
    };
    let structured_results =
        match perform_structured_extraction(structured_extraction_request).await {
            Ok(results) => results,
            Err(e) => {
                println!("Error performing structured extraction: {}", e);
                return Err(e.to_string().into());
            }
        };
    output_response.structured_extraction = Some(structured_results);
    pipeline.output = output_response;
    Ok(())
}
