use crate::configs::otel_config;
use crate::models::pipeline::{Sheet, SheetInfo};
use crate::utils::services::excel::get_worksheets_info;
use crate::utils::services::html::{extract_sheets_from_html, get_html_table_bounds};
use crate::{models::pipeline::Pipeline, utils::services::file_operations::convert_to_html};
use opentelemetry::trace::{Span, TraceContextExt, Tracer};
use opentelemetry::Context;
use rayon::prelude::*;
use std::error::Error;
use std::path::Path;

fn get_sheet_infos(file_path: &Path) -> Result<Vec<SheetInfo>, Box<dyn Error>> {
    let worksheets_info = get_worksheets_info(file_path)?;
    let sheets_info = worksheets_info
        .iter()
        .enumerate()
        .map(|(sheet_idx, (name, start_pos, _end_pos))| {
            let (start_row, start_column) = match start_pos {
                Some((row, col)) => (Some(*row), Some(*col)),
                None => (None, None),
            };
            SheetInfo {
                name: name.to_string(),
                start_row,
                start_column,
                end_row: None,
                end_column: None,
                sheet_number: sheet_idx as u32 + 1,
            }
        })
        .collect();
    Ok(sheets_info)
}

/// Create HTML from Excel
///
/// This function will create HTML from Excel with an additional `data-cell-ref` attribute using `Calamine` and `LibreOffice`.
pub async fn process(
    pipeline: &mut Pipeline,
    tracer: &opentelemetry::global::BoxedTracer,
) -> Result<(), Box<dyn Error>> {
    let input_file = pipeline.input_file.as_ref().expect("Input file not found");

    // Convert Excel to HTML
    let mut span = tracer.start_with_context(
        otel_config::SpanName::LibreOfficeConvertToHtml.to_string(),
        &Context::current(),
    );
    let html_conversion_result = convert_to_html(input_file).inspect_err(|e| {
        span.set_status(opentelemetry::trace::Status::error(e.to_string()));
        span.record_error(e.as_ref());
        span.set_attribute(opentelemetry::KeyValue::new("error", e.to_string()));
    })?;
    span.end();

    // Extract sheets from HTML
    let mut span = tracer.start_with_context(
        otel_config::SpanName::ExtractSheetsFromHtml.to_string(),
        &Context::current(),
    );
    let sheet_htmls =
        extract_sheets_from_html(&html_conversion_result.html_file).inspect_err(|e| {
            span.set_status(opentelemetry::trace::Status::error(e.to_string()));
            span.record_error(e.as_ref());
            span.set_attribute(opentelemetry::KeyValue::new("error", e.to_string()));
        })?;
    span.end();

    // Get sheet infos
    let mut span = tracer.start_with_context(
        otel_config::SpanName::GetSheetInfos.to_string(),
        &Context::current(),
    );
    let mut sheet_infos = get_sheet_infos(input_file.path()).inspect_err(|e| {
        span.set_status(opentelemetry::trace::Status::error(e.to_string()));
        span.record_error(e.as_ref());
        span.set_attribute(opentelemetry::KeyValue::new("error", e.to_string()));
    })?;
    span.end();

    // Create sheets
    let mut create_sheets_span = tracer.start_with_context(
        otel_config::SpanName::CreateSheets.to_string(),
        &Context::current(),
    );
    let sheets: Vec<Sheet> = sheet_infos
        .par_iter_mut()
        .zip(sheet_htmls.par_iter())
        .map(|(sheet_info, sheet_html)| {
            // TODO: This ignores hidden rows and columns (libreoffice does not support changing the visibility)
            let process_sheet_span = tracer.start_with_context(
                otel_config::SpanName::ProcessSheet.to_string(),
                &Context::current(),
            );
            let sheet_info_context = Context::current().with_span(process_sheet_span);
            let mut get_bounds_span = tracer.start_with_context(
                otel_config::SpanName::GetHtmlTableBounds.to_string(),
                &sheet_info_context,
            );
            let (end_row, end_col) = get_html_table_bounds(sheet_html).inspect_err(|e| {
                get_bounds_span.set_status(opentelemetry::trace::Status::error(e.to_string()));
                get_bounds_span.record_error(e.as_ref());
                get_bounds_span.set_attribute(opentelemetry::KeyValue::new("error", e.to_string()));
            })?;

            sheet_info.end_row = Some(end_row as u32);
            sheet_info.end_column = Some(end_col as u32);

            get_bounds_span.set_attribute(opentelemetry::KeyValue::new(
                "sheet_info",
                format!("{sheet_info:?}"),
            ));

            get_bounds_span.end();

            let mut create_sheet_span = tracer.start_with_context(
                otel_config::SpanName::CreateSheet.to_string(),
                &sheet_info_context,
            );
            let result = Sheet::new(
                sheet_info.clone(),
                sheet_html.clone(),
                html_conversion_result.embedded_images.clone(),
                tracer,
                &sheet_info_context,
            )
            .inspect_err(|e| {
                create_sheet_span.set_status(opentelemetry::trace::Status::error(e.to_string()));
                create_sheet_span.record_error(e.as_ref());
                create_sheet_span
                    .set_attribute(opentelemetry::KeyValue::new("error", e.to_string()));
            });

            create_sheet_span.end();
            result
        })
        .collect::<Result<Vec<Sheet>, Box<dyn Error + Send + Sync>>>()
        .map_err(|e| -> Box<dyn Error> { e.to_string().into() })
        .inspect_err(|e| {
            create_sheets_span.set_status(opentelemetry::trace::Status::error(e.to_string()));
            create_sheets_span.record_error(e.as_ref());
            create_sheets_span.set_attribute(opentelemetry::KeyValue::new("error", e.to_string()));
        })?;
    create_sheets_span.end();

    // Set spreadsheet assets
    let span = tracer.start_with_context(
        otel_config::SpanName::SetSpreadsheetAssets.to_string(),
        &Context::current(),
    );
    let context = Context::current().with_span(span);
    let result = pipeline
        .set_spreadsheet_assets(sheets, html_conversion_result.html_file, tracer, &context)
        .await;

    if let Err(e) = &result {
        let span_ref = context.span();
        span_ref.set_status(opentelemetry::trace::Status::error(e.to_string()));
        span_ref.record_error(e.as_ref());
        span_ref.set_attribute(opentelemetry::KeyValue::new("error", e.to_string()));
    }

    context.span().end();
    result?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::models::task::Configuration;
    use crate::utils::clients::initialize;

    use super::*;
    use base64::{engine::general_purpose::STANDARD, Engine as _};
    use std::collections::HashMap;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::Arc;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_process() {
        initialize().await;
        let mut input_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        input_file_path.push("input/LANCASTER-BTR-INTERNAL-8020.xlsx");
        let input_file = Arc::new(NamedTempFile::new().unwrap());
        fs::copy(input_file_path.clone(), input_file.path()).unwrap();

        // Create empty Pipeline with just the input file
        let mut pipeline = Pipeline::new();
        let configuration = Configuration::default();
        let sample_task = crate::models::task::Task::create_sample_task(configuration);
        pipeline.input_file = Some(input_file);
        pipeline.task = Some(sample_task);
        let tracer = opentelemetry::global::tracer("test");
        // Call process
        process(&mut pipeline, &tracer).await.unwrap();

        // Save the final output
        let output_dir = PathBuf::from("output/excel/html");
        let sheets_dir = output_dir.join("sheets");
        fs::create_dir_all(&sheets_dir).unwrap();

        let mut embedded_images_map = HashMap::new();

        // Save individual sheets
        if let Some(sheets) = &pipeline.sheets {
            sheets.iter().for_each(|sheet| {
                let sheet_dir = sheets_dir.join(sheet.sheet_info.name.to_lowercase());
                fs::create_dir_all(&sheet_dir).unwrap();
                fs::write(
                    sheet_dir.join("sheet.html"),
                    fs::read_to_string(sheet.html_file.path()).unwrap(),
                )
                .unwrap();
                fs::write(
                    sheet_dir.join("sheet_with_headers.html"),
                    fs::read_to_string(sheet.html_file_with_headers.path()).unwrap(),
                )
                .unwrap();
                sheet.embedded_images.iter().for_each(|img| {
                    let image_path = sheet_dir.join(img.html_reference.to_lowercase());
                    embedded_images_map.insert(img.html_reference.clone(), image_path.clone());
                    fs::copy(img.image_file.path(), image_path.clone()).unwrap();
                });
                fs::copy(
                    sheet.sheet_capture.image.path(),
                    sheet_dir.join("sheet.png"),
                )
                .unwrap();
                fs::copy(
                    sheet.sheet_capture_with_headers.image.path(),
                    sheet_dir.join("sheet_with_headers.png"),
                )
                .unwrap();
                println!(
                    "Sheet: {} saved in directory: \"{}\"",
                    sheet.sheet_info.name,
                    sheet_dir.display()
                );
            });
        }

        // Save the main HTML file
        if let Some(html_file) = &pipeline.html_file {
            let mut html_content = fs::read_to_string(html_file.path()).unwrap();
            let mut html_content_base64 = html_content.clone();

            // Version with file paths
            embedded_images_map.iter().for_each(|(key, value)| {
                html_content = html_content.replace(key, value.display().to_string().as_str());
            });
            fs::write(output_dir.join("input.html"), html_content).unwrap();

            // Version with base64 encoded images
            if let Some(sheets) = &pipeline.sheets {
                for sheet in sheets {
                    for image in &sheet.embedded_images {
                        let image_data = fs::read(image.image_file.path()).unwrap();
                        let base64_data = STANDARD.encode(&image_data);
                        let data_uri = format!("data:image/png;base64,{base64_data}");
                        html_content_base64 =
                            html_content_base64.replace(&image.html_reference, &data_uri);
                    }
                }
            }
            fs::write(output_dir.join("input_base64.html"), html_content_base64).unwrap();
        }

        if let Some(pdf_file) = &pipeline.pdf_file {
            fs::copy(pdf_file.path(), output_dir.join("input.pdf")).unwrap();
        }
    }

    #[test]
    fn get_sheets_info_test() {
        let mut input_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        input_file_path.push("input/test.xlsx");
        let sheets_info = get_sheet_infos(&input_file_path).unwrap();
        println!("{sheets_info:?}");
    }
}
