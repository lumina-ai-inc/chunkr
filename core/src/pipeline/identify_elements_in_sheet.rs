use crate::configs::{llm_config::Config as LlmConfig, otel_config};
use crate::models::excel::IdentifiedElements;
use crate::models::llm::{FallbackStrategy, LlmProcessing};
use crate::models::pipeline::{Element, Pipeline};
use crate::models::upload::{ErrorHandlingStrategy, SegmentationStrategy};
use crate::utils::services::file_operations::get_file_url;
use crate::utils::services::html::clean_html_for_llm;
use crate::utils::services::llm::{structured_output_from_template, LLMError};
use futures::FutureExt;
use opentelemetry::trace::{Span, TraceContextExt, Tracer};
use opentelemetry::Context;
use rayon::prelude::*;
use std::collections::HashMap;
use std::error::Error;
use std::fs;

/// Get the LLM processing for the identify tables in sheet task
///
/// This function will get the excel-specific models from the config, falling back to default models if not configured.
/// It will also check if there's a specific excel fallback model configured.
///
/// # Returns
///
/// A `Result` containing the LLM processing for the identify tables in sheet task.
///
fn get_llm_processing() -> Result<LlmProcessing, Box<dyn Error>> {
    // Get excel-specific models from config, falling back to regular models if not configured
    let llm_config =
        LlmConfig::from_env().map_err(|e| format!("Failed to load LLM config: {e}"))?;
    let excel_model = llm_config
        .get_excel_model(None)
        .map_err(|e| format!("Failed to get excel model: {e}"))?;

    // Check if there's a specific excel fallback model configured
    let excel_fallback_strategy = llm_config
        .llm_models
        .as_ref()
        .and_then(|models| models.iter().find(|model| model.excel_fallback))
        .map(|model| FallbackStrategy::Model(model.id.clone()));

    Ok(LlmProcessing::new(
        Some(excel_model.id),
        excel_fallback_strategy,
        None,
        0.0,
    ))
}

/// Identify tables in sheet
///
/// This function will identify individual tables in the sheet.
pub async fn process(
    pipeline: &mut Pipeline,
    tracer: &opentelemetry::global::BoxedTracer,
) -> Result<(), Box<dyn Error>> {
    let task = pipeline.get_task()?;
    let sheets = pipeline.sheets.clone().ok_or("No sheets found")?;

    let llm_processing = get_llm_processing()?;

    let error_handling = task.configuration.error_handling.clone();
    let segmentation_strategy = task.configuration.segmentation_strategy.clone();
    let image_folder_location = task.image_folder_location.clone();

    // Create futures for each sheet
    let identify_elements_span = tracer.start_with_context(
        otel_config::SpanName::IdentifyElements.to_string(),
        &Context::current(),
    );
    let identify_elements_context = Context::current().with_span(identify_elements_span);

    let identification_futures = sheets.par_iter().enumerate().map(|(index, sheet)| {
        println!("Processing sheet {}: '{}'", index, sheet.sheet_info.name);

        if sheet.sheet_info.start_row.is_none() {
            println!(
                "Skipping sheet for identification {:?} due to no start row",
                sheet.sheet_info.name
            );
            let result: Result<Vec<Element>, Box<dyn Error + Send + Sync>> = Ok(vec![]);
            return futures::future::ready(result).boxed();
        }

        if segmentation_strategy == SegmentationStrategy::Page {
            println!(
                "Skipping sheet '{}' due to Page segmentation strategy",
                sheet.sheet_info.name
            );
            let result: Result<Vec<Element>, Box<dyn Error + Send + Sync>> =
                Ok(sheet.sheet_info.clone().into());
            return futures::future::ready(result).boxed();
        }

        let llm_processing = llm_processing.clone();
        let error_handling = error_handling.clone();
        let image_folder_location = image_folder_location.clone();
        let identify_elements_context = identify_elements_context.clone();
        async move {
            let mut span = tracer.start_with_context(
                otel_config::SpanName::IdentifyTableInSheet.to_string(),
                &identify_elements_context,
            );
            span.set_attribute(opentelemetry::KeyValue::new("sheet_index", index as i64));
            span.set_attribute(opentelemetry::KeyValue::new(
                "sheet_name",
                sheet.sheet_info.name.clone(),
            ));

            let sheet_context = identify_elements_context.with_span(span);
            let html_file = sheet.html_file.clone();

            let html_content = match fs::read_to_string(html_file.path()) {
                Ok(content) => content,
                Err(e) => {
                    let error_msg = e.to_string();
                    sheet_context
                        .span()
                        .set_status(opentelemetry::trace::Status::error(error_msg.clone()));
                    sheet_context.span().record_error(&e);
                    sheet_context
                        .span()
                        .set_attribute(opentelemetry::KeyValue::new("error", error_msg.clone()));
                    sheet_context.span().end();
                    return Err(error_msg.into());
                }
            };

            let cleaned_html = match clean_html_for_llm(&html_content) {
                Ok(html) => html,
                Err(e) => {
                    let error_msg = e.to_string();
                    sheet_context
                        .span()
                        .set_status(opentelemetry::trace::Status::error(error_msg.clone()));
                    sheet_context.span().record_error(e.as_ref());
                    sheet_context
                        .span()
                        .set_attribute(opentelemetry::KeyValue::new("error", error_msg.clone()));
                    sheet_context.span().end();
                    return Err(error_msg.into());
                }
            };

            let s3_location = format!(
                "{}/{}_with_headers.png",
                image_folder_location, sheet.sheet_info.name
            );

            let image_url =
                match get_file_url(&sheet.sheet_capture_with_headers.image, &s3_location).await {
                    Ok(url) => url,
                    Err(e) => {
                        let error_msg = e.to_string();
                        sheet_context
                            .span()
                            .set_status(opentelemetry::trace::Status::error(error_msg.clone()));
                        sheet_context.span().record_error(e.as_ref());
                        sheet_context
                            .span()
                            .set_attribute(opentelemetry::KeyValue::new(
                                "error",
                                error_msg.clone(),
                            ));
                        sheet_context.span().end();
                        return Err(error_msg.into());
                    }
                };
            let mut values = HashMap::new();
            values.insert("cleaned_html".to_string(), cleaned_html);
            values.insert("image_url".to_string(), image_url);

            let result = match structured_output_from_template::<IdentifiedElements>(
                "identify_excel_elements",
                Some("Identify individual elements in the sheet".to_string()),
                "identify_excel_elements",
                &values,
                llm_processing,
                tracer,
                &sheet_context,
            )
            .await
            {
                Ok(identified_elements) => {
                    Ok(identified_elements.try_into()?)
                },
                Err(e) => {
                    // Check if this is a non-retryable client error
                    // TODO: Add a way to let the user know that the sheet was skipped
                    if let Some(LLMError::NonRetryable(_error_msg)) = e.downcast_ref::<LLMError>() {
                        println!("Skipping table identification for sheet {index} due to non-retryable client error: {e}");
                        sheet_context
                            .span()
                            .set_attribute(opentelemetry::KeyValue::new(
                                "skipped_reason",
                                "non_retryable_client_error",
                            ));
                        sheet_context.span().end();
                        return Ok(sheet.sheet_info.clone().into());
                    }
                    if error_handling == ErrorHandlingStrategy::Fail {
                        println!("Failed to identify tables for sheet {index}: {e}");
                        let error_msg = e.to_string();
                        sheet_context
                            .span()
                            .set_status(opentelemetry::trace::Status::error(error_msg.clone()));
                        sheet_context.span().record_error(e.as_ref());
                        sheet_context
                            .span()
                            .set_attribute(opentelemetry::KeyValue::new(
                                "error",
                                error_msg.clone(),
                            ));
                        sheet_context.span().end();
                        Err(error_msg.into())
                    } else {
                        println!("Skipping sheet {index} due to error: {e}");
                        sheet_context
                            .span()
                            .set_attribute(opentelemetry::KeyValue::new(
                                "skipped_reason",
                                "error handling strategy",
                            ));
                        sheet_context
                            .span()
                            .set_attribute(opentelemetry::KeyValue::new("error", e.to_string()));
                        sheet_context.span().end();
                        Ok(sheet.sheet_info.clone().into())
                    }
                }
            };

            sheet_context.span().end();
            result
        }
        .boxed()
    });

    let identification_results: Vec<Vec<Element>> =
        futures::future::try_join_all(identification_futures.collect::<Vec<_>>())
            .await
            .map_err(|e| -> Box<dyn Error> { e })
            .inspect_err(|e| {
                identify_elements_context
                    .span()
                    .set_status(opentelemetry::trace::Status::error(e.to_string()));
                identify_elements_context.span().record_error(e.as_ref());
                identify_elements_context
                    .span()
                    .set_attribute(opentelemetry::KeyValue::new("error", e.to_string()));
            })?;
    identify_elements_context.span().end();

    let mut set_tables_span = tracer.start_with_context(
        otel_config::SpanName::SetTablesOnSheets.to_string(),
        &Context::current(),
    );
    pipeline
        .sheets
        .as_mut()
        .unwrap()
        .par_iter_mut()
        .zip(identification_results.par_iter())
        .try_for_each(
            |(sheet, elements)| -> Result<(), Box<dyn Error + Send + Sync>> {
                match sheet.set_elements(elements.clone()) {
                    Ok(_) => Ok(()),
                    Err(e) => {
                        if error_handling == ErrorHandlingStrategy::Fail {
                            println!(
                                "Failed to set tables for sheet {}: {}",
                                sheet.sheet_info.name, e
                            );
                            Err(e)
                        } else {
                            println!(
                                "Skipping sheet {} due to error: {}",
                                sheet.sheet_info.name, e
                            );
                            Ok(())
                        }
                    }
                }
            },
        )
        .map_err(|e| -> Box<dyn Error> { e.to_string().into() })
        .inspect_err(|e| {
            set_tables_span.set_status(opentelemetry::trace::Status::error(e.to_string()));
            set_tables_span.record_error(e.as_ref());
            set_tables_span.set_attribute(opentelemetry::KeyValue::new("error", e.to_string()));
        })?;

    pipeline
        .sheets_to_chunks(tracer, &Context::current())
        .map_err(|e| -> Box<dyn Error> { e.to_string().into() })
        .inspect_err(|e| {
            set_tables_span.set_status(opentelemetry::trace::Status::error(e.to_string()));
            set_tables_span.record_error(e.as_ref());
            set_tables_span.set_attribute(opentelemetry::KeyValue::new("error", e.to_string()));
        })?;

    set_tables_span.end();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::llm::{FallbackStrategy, LlmProcessing};
    use crate::models::task::Configuration;
    use crate::pipeline::convert_excel_to_html;
    use crate::utils::clients::initialize;
    use opentelemetry::global;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::Arc;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_process() {
        initialize().await;
        let mut input_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        input_file_path.push("input/test.xlsx");
        let input_file = Arc::new(NamedTempFile::new().unwrap());
        fs::copy(input_file_path.clone(), input_file.path()).unwrap();

        // Create empty Pipeline with just the input file
        let mut pipeline = Pipeline::new();
        pipeline.input_file = Some(input_file);

        let mut configuration = Configuration::default();

        let llm_processing = LlmProcessing {
            model_id: Some("gemini-pro-2.5".to_string()),
            fallback_strategy: FallbackStrategy::None,
            max_completion_tokens: None,
            temperature: 0.0,
        };

        configuration.llm_processing = llm_processing;
        // Create a sample task with LLM processing configuration for testing
        let sample_task = crate::models::task::Task::create_sample_task(configuration);
        pipeline.task = Some(sample_task);

        let tracer = global::tracer("test");
        convert_excel_to_html::process(&mut pipeline, &tracer)
            .await
            .unwrap();
        let tracer = global::tracer("test");
        process(&mut pipeline, &tracer).await.unwrap();

        // Save the final output
        let output_dir = PathBuf::from("output/excel/html");
        let sheets_dir = output_dir.join("sheets");

        pipeline.sheets.as_ref().unwrap().iter().for_each(|sheet| {
            println!(
                "For sheet {} found {} tables",
                sheet.sheet_info.name,
                sheet
                    .elements
                    .as_ref()
                    .map(|tables| tables.len())
                    .unwrap_or(0)
            );
            let sheet_dir = sheets_dir.join(sheet.sheet_info.name.to_lowercase());
            sheet
                .elements
                .as_ref()
                .unwrap()
                .iter()
                .enumerate()
                .for_each(|(index, table)| {
                    let table_dir = sheet_dir.join(format!("table_{index}"));
                    fs::create_dir_all(&table_dir).unwrap();

                    let table_json = serde_json::json!({
                        "table_range": table.range.clone(),
                        "header_range": table.header_range.clone(),
                    });
                    fs::write(
                        table_dir.join("table.html"),
                        fs::read_to_string(table.html.as_ref().unwrap().path()).unwrap(),
                    )
                    .unwrap();
                    fs::write(
                        table_dir.join("table.json"),
                        serde_json::to_string(&table_json).unwrap(),
                    )
                    .unwrap();
                });
        });
    }
}
