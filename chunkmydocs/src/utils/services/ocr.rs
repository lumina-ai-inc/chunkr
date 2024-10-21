use crate::models::server::segment::{BoundingBox, OCRResult};
use crate::models::workers::table_ocr::TableStructure;
// use crate::utils::services::images::preprocess_image;
use crate::utils::services::rapid_ocr::call_rapid_ocr_api;
use crate::utils::services::table_ocr::recognize_table;
use crate::utils::storage::services::download_to_tempfile;
use aws_sdk_s3::Client as S3Client;
use reqwest::Client as ReqwestClient;
use std::collections::HashSet;


pub async fn download_and_ocr(
    s3_client: &S3Client,
    reqwest_client: &ReqwestClient,
    image_location: &str
) -> Result<(Vec<OCRResult>, String, String), Box<dyn std::error::Error>> {
    let original_file = download_to_tempfile(s3_client, reqwest_client, image_location, None).await?;
    let original_file_path = original_file.path().to_owned();
    // let preprocessed_file = preprocess_image(&original_file.path()).await?;
    // let preprocessed_file_path = preprocessed_file.path().to_owned();
    let ocr_results = match call_rapid_ocr_api(&original_file_path).await {
        Ok(ocr_results) => ocr_results,
        Err(e) => {
            return Err(e.to_string().into());
        }
    };
    Ok((ocr_results, "".to_string(), "".to_string()))
}

pub async fn download_and_table_ocr(
    s3_client: &S3Client,
    reqwest_client: &ReqwestClient,
    image_location: &str
) -> Result<(Vec<OCRResult>, String, String), Box<dyn std::error::Error>> {
    let original_file = download_to_tempfile(s3_client, reqwest_client, image_location, None).await?;
    let original_file_path = original_file.path().to_owned();
    let original_file_path_clone = original_file_path.clone();
    // let preprocessed_file = preprocess_image(&original_file.path()).await?;
    // let preprocessed_file_path = preprocessed_file.path().to_owned();

    let table_structure_task = tokio::task::spawn(async move {
        recognize_table(&original_file_path).await
    });
    let rapid_ocr_task = tokio::task::spawn(async move {
        call_rapid_ocr_api(&original_file_path_clone).await
    });
    let ocr_results = match rapid_ocr_task.await {
        Ok(ocr_results) => ocr_results.unwrap_or_default(),
        Err(e) => {
            return Err(e.to_string().into());
        }
    };
    let table_structures = match table_structure_task.await {
        Ok(table_structures) => table_structures.unwrap_or_default(),
        Err(e) => {
            return Err(e.to_string().into());
        }
    };

    let table_structures_with_content = add_content_to_table_structure(table_structures, ocr_results);
    let html = get_table_html(table_structures_with_content.clone());
    let markdown = get_table_markdown(table_structures_with_content.clone());
    let table_ocr_results = table_structures_to_ocr_results(table_structures_with_content);
    Ok((table_ocr_results, html, markdown))
}

fn add_content_to_table_structure(
    mut table_structures: Vec<TableStructure>,
    ocr_results: Vec<OCRResult>
) -> Vec<TableStructure> {
    let mut used_ocr_results = HashSet::new();

    for table in &mut table_structures {
        for cell in &mut table.cells {
            let mut cell_content = Vec::new();

            for (index, ocr_result) in ocr_results.iter().enumerate() {
                if used_ocr_results.contains(&index) {
                    continue;
                }

                let (center_x, center_y) = ocr_result.bbox.get_center();
                if is_point_in_bbox(center_x, center_y, &cell.cell) {
                    cell_content.push(ocr_result.text.clone());
                    used_ocr_results.insert(index);
                }
            }

            cell.content = Some(cell_content.join(" "));
        }
    }

    table_structures
}

fn is_point_in_bbox(x: f32, y: f32, bbox: &BoundingBox) -> bool {
    x >= bbox.left && x <= bbox.left + bbox.width && y >= bbox.top && y <= bbox.top + bbox.height
}

fn get_table_html(
    table_structures: Vec<TableStructure>
) -> String {
    let mut html = String::new();
    html.push_str("<table>");
    for row in table_structures {
        html.push_str("<tr>");
        for cell in row.cells {
            html.push_str(&format!("<td colspan='{}' rowspan='{}'>{}</td>", cell.col_span, cell.row_span, cell.content.unwrap_or_default()));
        }
        html.push_str("</tr>");
    }
    html.push_str("</table>");
    return html;
}

fn get_table_markdown(
    table_structures: Vec<TableStructure>
) -> String {
    let mut markdown = String::new();
    markdown.push_str("|");
    for row in table_structures {
        markdown.push_str("|");
        for cell in row.cells {
            markdown.push_str(&format!("{} |", cell.content.unwrap_or_default()));
        }
    }
    markdown.push_str("\n");
    return markdown;
}

fn table_structures_to_ocr_results(
    table_structures: Vec<TableStructure>
) -> Vec<OCRResult> {
    table_structures
        .iter()
        .flat_map(|table| table.cells.iter())
        .map(|cell| OCRResult {
            bbox: cell.cell.clone(),
            text: cell.content.clone().unwrap_or_default(),
            confidence: cell.confidence,
        })
        .collect()
}
