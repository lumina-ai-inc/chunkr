use crate::configs::llm_config::get_prompt;
use crate::models::chunkr::azure::DocumentAnalysisFeature;
use crate::models::chunkr::pipeline::Pipeline;
use crate::models::chunkr::task::Status;
use crate::utils::services::azure::perform_azure_analysis;
use crate::utils::services::llm;
use futures::future::try_join_all;
use rayon::prelude::*;
use std::collections::HashMap;

/// Use Azure document layout analysis to perform segmentation and ocr
pub async fn process(pipeline: &mut Pipeline) -> Result<(), Box<dyn std::error::Error>> {
    let mut task = pipeline.get_task()?;
    task.update(
        Some(Status::Processing),
        Some("Running Azure analysis".to_string()),
        None,
        None,
        None,
        None,
        None,
    )
    .await?;

    let configuration = pipeline.get_task()?.configuration.clone();
    let scaling_factor = pipeline.get_scaling_factor()?;
    let features = if configuration.high_resolution {
        Some(vec![DocumentAnalysisFeature::OcrHighResolution])
    } else {
        None
    };

    let pages: Vec<_> = pipeline
        .page_images
        .as_ref()
        .unwrap()
        .iter()
        .map(|x| x.as_ref())
        .collect();
    let page_futures = pages.iter().enumerate().map(|(idx, &page)| {
        let prompt = get_prompt("agent-segmentation", &HashMap::new()).unwrap();
        async move {
            let run_layout_analysis = !llm::agent_segment(page, prompt, None).await?;

            if run_layout_analysis {
                println!("Running layout analysis for page {}", idx + 1);
                Ok::<Option<usize>, Box<dyn std::error::Error + Send + Sync>>(Some(idx))
            } else {
                println!("Running page segmentation for page {}", idx + 1);
                Ok::<Option<usize>, Box<dyn std::error::Error + Send + Sync>>(None)
            }
        }
    });
    let layout_analysis_indexes: Vec<usize> = try_join_all(page_futures)
        .await
        .map_err(|e| -> Box<dyn std::error::Error> { e.to_string().into() })?
        .into_iter()
        .filter_map(|x| x)
        .collect();

    let file = pipeline.get_file()?;
    let mut chunks = perform_azure_analysis(
        &file,
        features,
        configuration.segmentation_strategy,
        layout_analysis_indexes,
    )
    .await?;
    chunks.par_iter_mut().for_each(|chunk| {
        chunk.segments.par_iter_mut().for_each(|segment| {
            segment.scale(scaling_factor);
        });
    });
    pipeline.chunks = chunks;
    Ok(())
}
