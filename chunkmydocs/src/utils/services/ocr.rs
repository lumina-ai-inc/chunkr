use crate::models::server::segment::{BoundingBox, OCRResult};
use crate::models::workers::table_ocr::TableStructure;
use crate::utils::services::rapid_ocr::call_rapid_ocr_api;
use crate::utils::services::table_struct::recognize_table;
use crate::utils::storage::services::download_to_tempfile;
use aws_sdk_s3::Client as S3Client;
use reqwest::Client as ReqwestClient;
use std::collections::HashSet;


pub async fn download_and_ocr(
    s3_client: &S3Client,
    reqwest_client: &ReqwestClient,
    image_location: &str
) -> Result<Vec<OCRResult>, Box<dyn std::error::Error>> {
    let temp_file = download_to_tempfile(s3_client, reqwest_client, image_location, None).await?;
    let ocr_results = match call_rapid_ocr_api(&temp_file.path()).await {
        Ok(ocr_results) => ocr_results,
        Err(e) => {
            return Err(e.to_string().into());
        }
    };
    Ok(ocr_results)
}

pub async fn download_and_table_ocr(
    s3_client: &S3Client,
    reqwest_client: &ReqwestClient,
    image_location: &str
) -> Result<Vec<OCRResult>, Box<dyn std::error::Error>> {
    let temp_file = download_to_tempfile(s3_client, reqwest_client, image_location, None).await?;
    let temp_file_path = temp_file.path().to_owned();
    let temp_file_path_clone = temp_file_path.clone();

    let rapid_ocr_task = tokio::task::spawn(async move {
        call_rapid_ocr_api(&temp_file_path_clone).await
    });
    let table_structure_task = tokio::task::spawn(async move {
        recognize_table(&temp_file_path).await
    });
    let ocr_results = match rapid_ocr_task.await {
        Ok(ocr_results) => ocr_results.unwrap(),
        Err(e) => {
            return Err(e.to_string().into());
        }
    };
    let table_structures = match table_structure_task.await {
        Ok(table_structures) => table_structures.unwrap(),
        Err(e) => {
            return Err(e.to_string().into());
        }
    };
    Ok(ocr_results)
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
    x >= bbox.x1 && x <= bbox.x2 && y >= bbox.y1 && y <= bbox.y2
}

fn get_table_html(
    ocr_results: Vec<OCRResult>,
    table_structures: Vec<TableStructure>
) -> Vec<OCRResult> {}

fn get_ocr_results_for_table(
    ocr_results: Vec<OCRResult>,
    table_structure: TableStructure
) -> Vec<OCRResult> {}
