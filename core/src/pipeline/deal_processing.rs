use crate::events::document_events::{DocumentEvent, publish_event};
use crate::services::ocr_service::{create_ocr_service, OCRResponse};
use crate::utils::clients::get_pg_client;
use crate::utils::storage::services::download_to_tempfile;
use chrono;
use serde_json::json;
use std::error::Error;
use std::time::{Duration, Instant};
use tokio::time::sleep;

/// Simplified document processing pipeline for deal documents
/// Bypasses heavy Chunkr pipeline and uses cloud OCR directly
pub struct DealDocumentProcessor {
    document_id: String,
    deal_id: String,
    user_id: String,
    s3_location: String,
}

impl DealDocumentProcessor {
    pub fn new(document_id: String, deal_id: String, user_id: String, s3_location: String) -> Self {
        Self {
            document_id,
            deal_id,
            user_id,
            s3_location,
        }
    }

    /// Main processing pipeline with retry logic and metrics
    pub async fn process(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        let start_time = Instant::now();
        self.log_metric("pipeline_start", json!({
            "document_id": &self.document_id,
            "deal_id": &self.deal_id,
        }));

        // Publish OCR started event
        publish_event(DocumentEvent::OCRStarted {
            document_id: self.document_id.clone(),
            deal_id: self.deal_id.clone(),
            user_id: self.user_id.clone(),
        }).await?;

        // Update status to processing
        self.update_status("processing").await?;

        // Step 1: Download document from S3 with retry
        let download_start = Instant::now();
        println!("🔽 Downloading from S3: {}", self.s3_location);
        // #region agent log
        let _ = std::fs::OpenOptions::new().create(true).append(true).open("/Users/harishmaiya/Documents/GitHub/data-extract/.cursor/debug.log").and_then(|mut f| std::io::Write::write_all(&mut f, format!("{{\"location\":\"deal_processing.rs:51\",\"message\":\"Starting S3 download\",\"data\":{{\"s3_location\":\"{}\"}},\"timestamp\":{},\"sessionId\":\"debug-session\",\"runId\":\"doc-processing\",\"hypothesisId\":\"S3\"}}\n", self.s3_location, std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis()).as_bytes()));
        // #endregion
        let temp_file = self.retry_operation(|| async {
            download_to_tempfile(&self.s3_location, None, "application/octet-stream")
                .await
                .map_err(|e| -> Box<dyn Error + Send + Sync> { 
                    let error_msg = format!("Failed to download from S3: {}", e);
                    eprintln!("❌ S3 Download Error: {} | Location: {}", e, &self.s3_location);
                    // #region agent log
                    let error_str = error_msg.replace("\"", "\\\"");
                    let _ = std::fs::OpenOptions::new().create(true).append(true).open("/Users/harishmaiya/Documents/GitHub/data-extract/.cursor/debug.log").and_then(|mut f| std::io::Write::write_all(&mut f, format!("{{\"location\":\"deal_processing.rs:58\",\"message\":\"S3 download failed\",\"data\":{{\"error\":\"{}\",\"s3_location\":\"{}\"}},\"timestamp\":{},\"sessionId\":\"debug-session\",\"runId\":\"doc-processing\",\"hypothesisId\":\"S3\"}}\n", error_str, &self.s3_location, std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis()).as_bytes()));
                    // #endregion
                    error_msg.into()
                })
        }, 3).await?;
        println!("✅ S3 download successful");
        self.log_metric("s3_download_duration_ms", json!(download_start.elapsed().as_millis()));

        // Step 2: Send ORIGINAL file directly to Docling (no PDF conversion!)
        // Docling natively supports PDF, DOCX, XLSX, PPTX, images, and more
        let file_path = temp_file.path();
        let file_ext = file_path.extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();
        
        self.log_metric("file_type", json!(file_ext));
        self.log_metric("optimization", json!({
            "skip_pdf_conversion": true,
            "send_original_to_docling": true
        }));
        
        let ocr_start = Instant::now();
        let ocr_results = self.retry_operation(|| async {
            // Use Docling for ALL file types - it handles them natively
                let ocr_service = create_ocr_service()?;
            println!("📄 Sending {} file directly to Docling (optimized path)", file_ext.to_uppercase());
            println!("✨ Skip PDF conversion - Docling handles original format natively");
            ocr_service.process_document(file_path).await
        }, 3).await?;
        
        self.log_metric("ocr_duration_ms", json!(ocr_start.elapsed().as_millis()));
        self.log_metric("page_count", json!(ocr_results.page_count));

        // Publish OCR completed event
        publish_event(DocumentEvent::OCRCompleted {
            document_id: self.document_id.clone(),
            deal_id: self.deal_id.clone(),
            user_id: self.user_id.clone(),
            page_count: ocr_results.page_count,
        }).await?;

        // Step 3: Store OCR results
        self.store_ocr_results(&ocr_results).await?;

        // Step 4: Generate PDF for viewing (async from OCR - doesn't block fact extraction)
        // This happens in parallel with fact extraction for faster overall processing
        // #region agent log
        let _ = std::fs::OpenOptions::new().create(true).append(true).open("/Users/harishmaiya/Documents/GitHub/data-extract/.cursor/debug.log").and_then(|mut f| std::io::Write::write_all(&mut f, format!("{{\"location\":\"deal_processing.rs:110\",\"message\":\"Spawning async PDF generation\",\"data\":{{\"document_id\":\"{}\",\"s3_location\":\"{}\"}},\"timestamp\":{},\"sessionId\":\"debug-session\",\"runId\":\"doc-processing\",\"hypothesisId\":\"H1\"}}\n", self.document_id, self.s3_location, std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis()).as_bytes()));
        // #endregion
        tokio::spawn({
            let s3_location = self.s3_location.clone();
            let document_id = self.document_id.clone();
            let deal_id = self.deal_id.clone();
            async move {
                match Self::generate_viewer_pdf_async(&s3_location, &document_id).await {
                    Ok(pdf_s3_location) => {
                        println!("✅ PDF generated for viewing: {}", document_id);
                        // Update the document with the PDF S3 location
                        if let Err(e) = Self::update_pdf_url(&document_id, &deal_id, &pdf_s3_location).await {
                            eprintln!("⚠️  Failed to update PDF S3 location in database: {}", e);
                        }
                    },
                    Err(e) => {
                        eprintln!("⚠️  PDF generation failed (non-critical): {}", e);
                    },
                }
            }
        });

        // Step 5: Trigger AI agent for fact extraction (doesn't wait for PDF)
        self.trigger_fact_extraction(&ocr_results).await?;

        // Step 6: Update document status
        self.update_status("completed").await?;

        let total_duration = start_time.elapsed();
        self.log_metric("pipeline_complete", json!({
            "total_duration_ms": total_duration.as_millis(),
            "document_id": &self.document_id,
            "status": "completed"
        }));

        Ok(())
    }

    async fn store_ocr_results(&self, ocr_results: &OCRResponse) -> Result<(), Box<dyn Error + Send + Sync>> {
        let client = get_pg_client().await?;
        
        let ocr_json = json!({
            "pages": ocr_results.pages,
            "page_count": ocr_results.page_count,
            "document_type": ocr_results.document_type,
        });

        // Update document with OCR results using raw SQL
        // Note: ocr_json is serde_json::Value which implements ToSql for JSONB
        client.execute(
            "UPDATE documents SET status = $1, page_count = $2, ocr_output = $3 WHERE document_id = $4",
            &[
                &"processing",
                &ocr_results.page_count,
                &ocr_json as &serde_json::Value,
                &self.document_id,
            ]
        ).await?;

        Ok(())
    }

    async fn trigger_fact_extraction(&self, ocr_results: &OCRResponse) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Queue fact extraction task
        let extraction_message = json!({
            "document_id": self.document_id,
            "deal_id": self.deal_id,
            "user_id": self.user_id,
            "ocr_results": ocr_results,
        });

        let pool = crate::utils::clients::get_redis_pool();
        let mut conn = pool.get().await?;

        deadpool_redis::redis::cmd("RPUSH")
            .arg("fact_extraction")
            .arg(serde_json::to_string(&extraction_message)?)
            .query_async::<i64>(&mut conn)
            .await?;

        Ok(())
    }

    async fn update_status(&self, new_status: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
        let client = get_pg_client().await?;
        
        // Update document status in documents table (if it exists, for real deals)
        let _ = client.execute(
            "UPDATE documents SET status = $1 WHERE document_id = $2",
            &[&new_status, &self.document_id]
        ).await; // Ignore errors if table doesn't exist
        
        // CRITICAL: Also update the tasks table status so the frontend can see it
        // Map our internal status to Task status enum
        let task_status = match new_status {
            "completed" => "Succeeded",
            "failed" => "Failed",
            "processing" => "Processing",
            _ => "Starting"
        };
        
        client.execute(
            "UPDATE tasks SET status = $1 WHERE task_id = $2",
            &[&task_status, &self.document_id]
        ).await?;

        Ok(())
    }

    /// Retry operation with exponential backoff
    async fn retry_operation<F, Fut, T>(&self, operation: F, max_retries: u32) -> Result<T, Box<dyn Error + Send + Sync>>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T, Box<dyn Error + Send + Sync>>>,
    {
        let mut retries = 0;
        loop {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(e) if retries < max_retries => {
                    retries += 1;
                    let delay = Duration::from_secs(2u64.pow(retries));
                    self.log_metric("retry_attempt", json!({
                        "attempt": retries,
                        "max_retries": max_retries,
                        "error": e.to_string(),
                        "delay_secs": delay.as_secs()
                    }));
                    sleep(delay).await;
                }
                Err(e) => {
                    // Update document status to failed
                    let _ = self.update_status("failed").await;
                    self.log_metric("pipeline_failed", json!({
                        "document_id": &self.document_id,
                        "error": e.to_string(),
                        "retries": max_retries
                    }));
                    return Err(format!("Operation failed after {} retries: {}", max_retries, e).into());
                }
            }
        }
    }
    
    /// Log structured metrics for monitoring
    fn log_metric(&self, metric_name: &str, data: serde_json::Value) {
        let log_entry = json!({
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "metric": metric_name,
            "document_id": &self.document_id,
            "deal_id": &self.deal_id,
            "data": data
        });
        println!("[METRIC] {}", serde_json::to_string(&log_entry).unwrap_or_default());
    }

    /// Generate PDF for viewing purposes (runs async, doesn't block OCR/fact extraction)
    /// Returns the presigned URL for the viewer
    async fn generate_viewer_pdf_async(
        s3_location: &str,
        document_id: &str,
    ) -> Result<String, Box<dyn Error + Send + Sync>> {
        use crate::utils::services::file_operations::check_file_type;
        use crate::utils::storage::services::{download_to_tempfile, upload_to_s3};
        
        println!("🔄 Generating PDF for viewer: document_id={}", document_id);
        
        // Download original file
        let temp_file = download_to_tempfile(s3_location, None, "application/octet-stream")
            .await
            .map_err(|e| -> Box<dyn Error + Send + Sync> { format!("Download failed: {}", e).into() })?;
        
        // Check file type
        let (mime_type, _) = check_file_type(&temp_file, None)
            .map_err(|e| -> Box<dyn Error + Send + Sync> { format!("File type check failed: {}", e).into() })?;
        
        // Generate PDF location - for PDFs, use the same location; for others, replace extension with .pdf
        let pdf_location = if s3_location.ends_with(".pdf") {
            s3_location.to_string()
        } else {
            // Replace the file extension with .pdf
            // s3://bucket/path/file.docx -> s3://bucket/path/file.pdf
            let path_without_ext = if let Some(dot_pos) = s3_location.rfind('.') {
                &s3_location[..dot_pos]
            } else {
                s3_location
            };
            format!("{}.pdf", path_without_ext)
        };
        // #region agent log
        let _ = std::fs::OpenOptions::new().create(true).append(true).open("/Users/harishmaiya/Documents/GitHub/data-extract/.cursor/debug.log").and_then(|mut f| std::io::Write::write_all(&mut f, format!("{{\"location\":\"deal_processing.rs:260\",\"message\":\"PDF location determined\",\"data\":{{\"pdf_location\":\"{}\",\"mime_type\":\"{}\"}},\"timestamp\":{},\"sessionId\":\"debug-session\",\"runId\":\"doc-processing\",\"hypothesisId\":\"H3\"}}\n", pdf_location, mime_type, std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis()).as_bytes()));
        // #endregion
        
        // Try to generate/upload PDF for viewing
        let final_location = if mime_type == "application/pdf" {
            // Already PDF, use original location (no copy needed)
            println!("✅ File is already PDF");
            s3_location.to_string()
        } else {
            // Try to convert to PDF for viewing (LibreOffice/ImageMagick required)
            println!("🔄 Converting {} to PDF", mime_type);
            
            // Convert synchronously first (outside of async context)
            use crate::utils::services::file_operations::convert_to_pdf;
            let conversion_result = convert_to_pdf(&temp_file, None)
                .map_err(|e| format!("{}", e));
            
            match conversion_result {
                Ok(pdf_file) => {
                    // Now upload the converted PDF
                    match upload_to_s3(&pdf_location, pdf_file.path()).await {
                        Ok(_) => {
                            println!("✅ Converted and uploaded PDF to: {}", pdf_location);
                            pdf_location.clone()
                        },
                        Err(e) => {
                            eprintln!("⚠️  PDF upload failed, using original: {}", e);
                            s3_location.to_string()
                        }
                    }
                },
                Err(e) => {
                    eprintln!("⚠️  PDF conversion failed (tools not installed?), using original file: {}", e);
                    println!("💡 Tip: For local dev, install LibreOffice for DOCX→PDF or just use original files");
                    s3_location.to_string()
                }
            }
        };
        
        // Return the S3 location (not presigned URL!)
        // The API will generate fresh presigned URLs on-demand via create_output()
        println!("✅ Viewer PDF location determined: {} (S3: {})", document_id, final_location);
        
        Ok(final_location)
    }

    /// Update the document's PDF S3 location in the database after async PDF generation
    /// The API will generate fresh presigned URLs from this location on-demand
    async fn update_pdf_url(
        document_id: &str,
        _deal_id: &str,
        s3_location: &str,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let client = get_pg_client().await?;
        
        // Store the S3 location (not presigned URL!)
        // The API task.create_output() will generate fresh presigned URLs on each request
        client.execute(
            "UPDATE tasks 
             SET pdf_location = $1
             WHERE task_id = $2",
            &[&s3_location, &document_id],
        ).await?;
        
        println!("✅ Updated PDF S3 location for task {} in database", document_id);
        Ok(())
    }

    // ============================================================================
    // REMOVED: process_csv, process_excel, process_pdf, extract_pdf_text
    // 
    // These functions are no longer needed! Docling handles all file types
    // natively without requiring custom processing per format.
    //
    // Benefits of removal:
    // - Simpler codebase (150+ lines removed)
    // - Unified processing pipeline
    // - Better quality (Docling sees original formatting)
    // - Faster (no intermediate conversions)
    // ============================================================================

    pub async fn handle_error(&self, error: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
        println!("Error processing document {}: {}", self.document_id, error);
        
        // Publish processing failed event
        publish_event(DocumentEvent::ProcessingFailed {
            document_id: self.document_id.clone(),
            deal_id: self.deal_id.clone(),
            user_id: self.user_id.clone(),
            error: error.to_string(),
        }).await?;
        
        self.update_status("failed").await?;
        Ok(())
    }
}

/// Process document from Redis queue message
pub async fn process_document_from_queue(message: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
    let data: serde_json::Value = serde_json::from_str(message)?;
    
    let document_id = data["document_id"].as_str().ok_or("Missing document_id")?;
    let deal_id = data["deal_id"].as_str().ok_or("Missing deal_id")?;
    let user_id = data["user_id"].as_str().ok_or("Missing user_id")?;
    let s3_location = data["s3_location"].as_str().ok_or("Missing s3_location")?;

    let processor = DealDocumentProcessor::new(
        document_id.to_string(),
        deal_id.to_string(),
        user_id.to_string(),
        s3_location.to_string(),
    );

    match processor.process().await {
        Ok(_) => Ok(()),
        Err(e) => {
            processor.handle_error(&e.to_string()).await?;
            Err(e)
        }
    }
}

