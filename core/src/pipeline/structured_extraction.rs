use crate::models::chunkr::pipeline::Pipeline;
use crate::models::chunkr::task::Status;
use crate::utils::services::structured_extraction::perform_structured_extraction;
use std::error::Error;

pub async fn process(pipeline: &mut Pipeline) -> Result<(), Box<dyn Error>> {
    pipeline
        .get_task()?
        .update(
            Some(Status::Processing),
            Some("Structured extraction started".to_string()),
            None,
            None,
            None,
            None,
            None,
        )
        .await?;

    // TODO:
    // V1. Send all content to structured extraction
    // V2. Search and send relevant content to structured extraction
    //      - Use embeddigns cache

    // let mut output_response = pipeline.output.clone();

    // let texts: Vec<String> = output_response
    //     .chunks
    //     .iter()
    //     .map(|chunk| {
    //         chunk
    //             .segments
    //             .iter()
    //             .map(|segment| segment.content.clone())
    //             .collect::<Vec<String>>()
    //             .join(" ")
    //     })
    //     .collect();

    // let structured_results = match perform_structured_extraction(
    //     json_schema.ok_or("JSON schema is missing")?,
    //     texts,
    //     "markdown".to_string(),
    // )
    // .await
    // {
    //     Ok(results) => results,
    //     Err(e) => {
    //         println!("Error performing structured extraction: {}", e);
    //         return Err(e.to_string().into());
    //     }
    // };
    // output_response.extracted_json = Some(structured_results);
    // pipeline.output = output_response;

    Ok(())
}
