use crate::models::rrq::queue::QueuePayload;
use crate::models::server::extract::{ ExtractionPayload, SegmentationModel };
use crate::models::server::task::Status;
use crate::utils::configs::extraction_config::Config;
use crate::utils::db::deadpool_postgres::{ Client, create_pool };
use crate::utils::services::pdf::pdf_2_images;
use crate::utils::storage::config_s3::create_client;
use crate::utils::storage::services::{ download_to_tempfile, upload_to_s3 };
use crate::utils::workers::{ log::log_task, payload::produce_extraction_payloads };
use chrono::Utc;
use std::{ error::Error, path::{ Path, PathBuf }, process::Command };
use tempfile::TempDir;

fn is_valid_file_type(file_path: &str) -> Result<(bool, String), Box<dyn Error>> {
    let output = Command::new("file").arg("--mime-type").arg("-b").arg(file_path).output()?;

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

    let result: Result<(), Box<dyn std::error::Error>> = (async {
        log_task(
            task_id.clone(),
            Status::Processing,
            Some("Task started".to_string()),
            None,
            &pg_pool
        ).await?;

        let input_file = download_to_tempfile(
            &s3_client,
            &reqwest_client,
            &extraction_payload.input_location,
            None
        ).await?;

        let (is_valid, detected_mime_type) = is_valid_file_type(
            &extraction_payload.input_location
        )?;
        if !is_valid {
            log_task(
                task_id.clone(),
                Status::Failed,
                Some(format!("Not a valid file type: {}", detected_mime_type)),
                Some(Utc::now()),
                &pg_pool
            ).await?;
            return Ok(());
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

        let image_paths = pdf_2_images(&pdf_path, &temp_dir.path())?;

        let page_count = image_paths.len() as i32;
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
                    return Ok(());
                } else {
                    return Err(e.into());
                }
            }
        })?;

        for image_path in image_paths {
            let image_name = image_path.file_name().unwrap().to_str().unwrap().to_string();
            let image_location = format!("{}/{}", extraction_payload.input_location, image_name.to_string());
            upload_to_s3(&s3_client, &image_location, &image_path).await?;
        }

        Ok(())
    }).await;

    match result {
        Ok(_) => {
            println!("Task succeeded");
            let config = Config::from_env()?;
            let queue_name = match extraction_payload.model {
                SegmentationModel::PdlaFast => config.queue_fast,
                SegmentationModel::Pdla => config.queue_high_quality,
            };
            produce_extraction_payloads(queue_name, extraction_payload).await?;
            log_task(
                task_id.clone(),
                Status::Processing,
                Some("Segmentation queued".to_string()),
                None,
                &pg_pool
            ).await?;
            Ok(())
        }
        Err(e) => {
            eprintln!("Error processing task: {:?}", e);
            let error_message = if e.to_string().to_lowercase().contains("usage limit exceeded") {
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
