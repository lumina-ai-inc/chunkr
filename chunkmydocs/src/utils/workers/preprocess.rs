use crate::models::rrq::queue::QueuePayload;
use crate::models::server::extract::{ ExtractionPayload, SegmentationStrategy };
use crate::models::server::task::Status;
use crate::utils::configs::extraction_config::Config;
use crate::utils::db::deadpool_postgres::{ Client, create_pool };
use crate::utils::services::{
    log::log_task,
    payload::produce_extraction_payloads,
    pdf::{ pdf_2_images, extract_text_pdf },
};
use crate::utils::configs::s3_config::create_client;
use crate::utils::storage::services::{ download_to_given_tempfile, upload_to_s3 };
use chrono::Utc;
use std::{ error::Error, path::{ Path, PathBuf }, process::Command };
use tempfile::{ NamedTempFile, TempDir };

use crate::models::server::segment::{ Segment, SegmentType, BoundingBox };
use crate::utils::storage::services::download_to_tempfile;
use std::io::Write;
use uuid::Uuid;
use lopdf::Object;

fn is_valid_file_type(file_path: &Path) -> Result<(bool, String), Box<dyn Error>> {
    let output = Command::new("file")
        .arg("--mime-type")
        .arg("-b")
        .arg(file_path.to_str().unwrap())
        .output()?;

    let mime_type = String::from_utf8(output.stdout)?.trim().to_string();

    let is_valid = match mime_type.as_str() {
        | "application/pdf"
        | "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        | "application/msword"
        | "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        | "application/vnd.ms-powerpoint"
        | "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        | "application/vnd.ms-excel" => true,
        _ => false,
    };

    Ok((is_valid, mime_type))
}

fn convert_to_pdf(input_file_path: &Path, output_dir: &Path) -> Result<PathBuf, Box<dyn Error>> {
    let output = Command::new("libreoffice")
        .args(
            &[
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                output_dir.to_str().unwrap(),
                input_file_path.to_str().unwrap(),
            ]
        )
        .output()?;

    if !output.status.success() {
        return Err(
            Box::new(
                std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("LibreOffice conversion failed: {:?}", output)
                )
            )
        );
    }

    let pdf_file_name = input_file_path.file_stem().unwrap().to_str().unwrap().to_string() + ".pdf";
    let pdf_file_path = output_dir.join(pdf_file_name);

    if pdf_file_path.exists() {
        Ok(pdf_file_path)
    } else {
        Err(
            Box::new(
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "Converted PDF file not found in output directory"
                )
            )
        )
    }
}

pub async fn process(payload: QueuePayload) -> Result<(), Box<dyn std::error::Error>> {
    println!("Processing task");
    let s3_client: aws_sdk_s3::Client = create_client().await?;
    let reqwest_client = reqwest::Client::new();
    let mut extraction_payload: ExtractionPayload = serde_json::from_value(payload.payload)?;
    let task_id = extraction_payload.task_id.clone();
    let pg_pool = create_pool();
    let client: Client = pg_pool.get().await?;
    let temp_dir = TempDir::new().unwrap();
    let config = Config::from_env()?;

    let result: Result<bool, Box<dyn std::error::Error>> = (async {
        log_task(
            task_id.clone(),
            Status::Processing,
            Some("Task started".to_string()),
            None,
            &pg_pool
        ).await?;

        let mut input_file: NamedTempFile = NamedTempFile::new()?;
        download_to_given_tempfile(
            &mut input_file,
            &s3_client,
            &reqwest_client,
            &extraction_payload.input_location,
            None
        ).await.map_err(|e| {
            eprintln!("Failed to download input file: {:?}", e);
            e
        })?;

        let (is_valid, detected_mime_type) = is_valid_file_type(&input_file.path()).map_err(|e| {
            eprintln!("Failed to check file type: {:?}", e);
            e
        })?;

        if !is_valid {
            log_task(
                task_id.clone(),
                Status::Failed,
                Some(format!("Not a valid file type: {}", detected_mime_type)),
                Some(Utc::now()),
                &pg_pool
            ).await?;
            return Ok(false);
        }

        let pdf_path = match detected_mime_type.as_str() {
            "application/pdf" => input_file.path().to_path_buf(),
            _ => {
                log_task(
                    task_id.clone(),
                    Status::Processing,
                    Some("Converting to PDF".to_string()),
                    None,
                    &pg_pool
                ).await?;

                convert_to_pdf(&input_file.path(), &input_file.path().parent().unwrap())?
            }
        };

        upload_to_s3(&s3_client, &extraction_payload.pdf_location, &pdf_path).await?;

        log_task(
            task_id.clone(),
            Status::Processing,
            Some("Converting file to images".to_string()),
            None,
            &pg_pool
        ).await?;

        let image_paths = pdf_2_images(&pdf_path, &temp_dir.path()).await?;

        let page_count = image_paths.len() as i32;

        println!("Page count: {}", page_count.clone());
        println!("Page limit: {}", config.page_limit.clone());
       
        if page_count > config.page_limit {
            log_task(
                task_id.clone(),
                Status::Failed,
                Some(format!("File must be less than {} pages", config.page_limit)),
                Some(Utc::now()),
                &pg_pool
            ).await?;
            return Ok(false);
        }
        
        extraction_payload.page_count = Some(page_count);

        let update_page_count = client.execute(
            "UPDATE tasks SET page_count = $1, input_file_type = $2 WHERE task_id = $3",
            &[&page_count, &detected_mime_type, &task_id]
        ).await;

        (match update_page_count {
            Ok(_) => Ok::<_, Box<dyn std::error::Error>>(()),
            Err(e) => {
                if e.to_string().to_lowercase().contains("usage limit exceeded") {
                    log_task(
                        task_id.clone(),
                        Status::Failed,
                        Some("Task failed: Usage limit exceeded".to_string()),
                        Some(Utc::now()),
                        &pg_pool
                    ).await?;
                    Ok(())
                } else {
                    Err(e.into())
                }
            }
        })?;

        for image_path in image_paths {
            let image_name = image_path.file_name().unwrap().to_str().unwrap().to_string();
            let image_location = format!(
                "{}/{}",
                extraction_payload.image_folder_location,
                image_name.to_string()
            );
            upload_to_s3(&s3_client, &image_location, &image_path).await?;
        }

        Ok(true)
    }).await;
    
    match result {
        Ok(value) => {
            println!("Task succeeded");
            if value {
                match extraction_payload.model {
                    SegmentationStrategy::PdlaFast => {
                        produce_extraction_payloads(config.queue_fast, extraction_payload).await?;
                        log_task(
                            task_id.clone(),
                            Status::Processing,
                            Some("Segmentation queued".to_string()),
                            None,
                            &pg_pool
                        ).await?;
                    },
                    SegmentationStrategy::Pdla => {
                        produce_extraction_payloads(config.queue_high_quality, extraction_payload).await?;
                        log_task(
                            task_id.clone(),
                            Status::Processing,
                            Some("Segmentation queued".to_string()),
                            None,
                            &pg_pool
                        ).await?;
                    },
                    SegmentationStrategy::Page => {
                        let mut segments = Vec::new();
                        let pdf_file = download_to_tempfile(&s3_client, &reqwest_client, &extraction_payload.pdf_location, None).await?;
                        let page_texts = extract_text_pdf(&pdf_file.path()).await?;
                        let doc = lopdf::Document::load(pdf_file.path())?;

                        for (page_num, obj_id) in doc.get_pages() {
                            if let Ok(page_dict) = doc.get_dictionary(obj_id) {
                                if let Ok(mediabox) = page_dict.get(b"MediaBox").and_then(Object::as_array) {
                                    if mediabox.len() >= 4 {
                                        let x1 = mediabox[0].as_float().unwrap_or(0.0);
                                        let y1 = mediabox[1].as_float().unwrap_or(0.0);
                                        let x2 = mediabox[2].as_float().unwrap_or(0.0);
                                        let y2 = mediabox[3].as_float().unwrap_or(0.0);
                                        
                                        let width = (x2 - x1).abs();
                                        let height = (y2 - y1).abs();
                                        let content = page_texts[(page_num - 1) as usize].clone();
                                        let segment = Segment {
                                            segment_id: Uuid::new_v4().to_string(),
                                            content: content,
                                            bbox: BoundingBox {
                                                top: 0.0,
                                                left: 0.0,
                                                width,
                                                height,
                                            },
                                            page_number: (page_num) as u32,
                                            page_width: width,
                                            page_height: height,
                                            segment_type: SegmentType::Page,
                                            image: None,
                                            html: None,
                                            markdown: None,
                                            ocr: None,
                                        };
                                        segments.push(segment);
                                    }
                                }
                            }
                        }
                        
                        let mut output_temp_file = NamedTempFile::new()?;
                        output_temp_file.write_all(serde_json::to_string(&segments)?.as_bytes())?;
                        
                        upload_to_s3(&s3_client, &extraction_payload.output_location, &output_temp_file.path()).await?;
                        
                        produce_extraction_payloads(config.queue_postprocess, extraction_payload).await?;
                        log_task(
                            task_id.clone(),
                            Status::Processing,
                            Some("Chunking queued".to_string()),
                            None,
                            &pg_pool
                        ).await?;
                        
                    }
                }
            }
            Ok(())
        }
        Err(e) => {
            eprintln!("Error processing task: {:?}", e);
            let error_message = 
            if e.to_string().to_lowercase().contains("usage limit exceeded") {
                "Usage limit exceeded".to_string()
            } else {
                "Preprocessing failed".to_string()
            };

            if payload.attempt >= payload.max_attempts {
                eprintln!("Task failed after {} attempts", payload.max_attempts);
                log_task(
                    task_id.clone(),
                    Status::Failed,
                    Some(error_message),
                    Some(Utc::now()),
                    &pg_pool
                ).await?;
            }
            Err(e)
        }
    }
}

