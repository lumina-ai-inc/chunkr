use crate::events::document_events::{DocumentEvent, publish_event};
use crate::services::ocr_service::{create_ocr_service, OCRResponse, OCRResult, BoundingBox};
use crate::utils::clients::get_pg_client;
use crate::utils::storage::services::download_to_tempfile;
use chrono;
use serde_json::json;
use std::error::Error;
use std::fs;
use std::path::Path;
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
        let temp_file = self.retry_operation(|| async {
            download_to_tempfile(&self.s3_location, None, "application/octet-stream")
                .await
                .map_err(|e| -> Box<dyn Error + Send + Sync> { 
                    format!("Failed to download from S3: {}", e).into()
                })
        }, 3).await?;
        self.log_metric("s3_download_duration_ms", json!(download_start.elapsed().as_millis()));

        // Step 2: Determine file type and process accordingly with retry
        let file_path = temp_file.path();
        let file_ext = file_path.extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();
        
        self.log_metric("file_type", json!(file_ext));
        let ocr_start = Instant::now();
        let ocr_results = self.retry_operation(|| async {
            match file_ext.as_str() {
                "csv" => {
                    // CSV files don't need OCR, just parse directly
                    println!("Processing CSV file: {}", self.document_id);
                    self.process_csv(file_path).await
                }
                "xls" | "xlsx" => {
                    // Excel files - extract to text
                    println!("Processing Excel file: {}", self.document_id);
                    self.process_excel(file_path).await
                }
                "pdf" => {
                    // PDF - try free OCR first, fallback to Azure if configured
                    println!("Processing PDF file: {}", self.document_id);
                    self.process_pdf(file_path).await
                }
                _ => {
                    // Other files - try OCR service
                    let ocr_service = create_ocr_service()?;
                    println!("Using OCR service: {}", ocr_service.name());
                    ocr_service.process_document(file_path).await
                }
            }
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

        // Step 4: Trigger AI agent for fact extraction
        self.trigger_fact_extraction(&ocr_results).await?;

        // Step 5: Update document status
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
        
        // Update document status using raw SQL
        client.execute(
            "UPDATE documents SET status = $1 WHERE document_id = $2",
            &[&new_status, &self.document_id]
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

    /// Process CSV files (no OCR needed, just parse the text)
    async fn process_csv(&self, file_path: &Path) -> Result<OCRResponse, Box<dyn Error + Send + Sync>> {
        let content = fs::read_to_string(file_path)?;
        let _lines: Vec<&str> = content.lines().collect();
        
        // Create a single page result with all CSV content
        let result = OCRResult {
            page_number: 1,
            text: content.clone(),
            confidence: Some(1.0), // CSV is always 100% confident
            bounding_boxes: vec![BoundingBox {
                text: content,
                x: 0.0,
                y: 0.0,
                width: 100.0,
                height: 100.0,
                confidence: Some(1.0),
            }],
        };

        Ok(OCRResponse {
            page_count: 1,
            pages: vec![result],
            document_type: Some("CSV".to_string()),
        })
    }

    /// Process Excel files (XLS/XLSX) - extract to text
    async fn process_excel(&self, file_path: &Path) -> Result<OCRResponse, Box<dyn Error + Send + Sync>> {
        use calamine::{Reader, open_workbook_auto};
        
        let mut workbook = open_workbook_auto(file_path)
            .map_err(|e| format!("Failed to open Excel file: {}", e))?;
        
        let mut all_text = String::new();
        let sheet_names: Vec<String> = workbook.sheet_names().to_vec();
        
        for sheet_name in sheet_names {
            if let Ok(range) = workbook.worksheet_range(&sheet_name) {
                all_text.push_str(&format!("=== Sheet: {} ===\n", sheet_name));
                
                for row in range.rows() {
                    let row_text: Vec<String> = row.iter()
                        .map(|cell| cell.to_string())
                        .collect();
                    all_text.push_str(&row_text.join("\t"));
                    all_text.push('\n');
                }
                all_text.push('\n');
            }
        }
        
        let result = OCRResult {
            page_number: 1,
            text: all_text.clone(),
            confidence: Some(1.0),
            bounding_boxes: vec![BoundingBox {
                text: all_text,
                x: 0.0,
                y: 0.0,
                width: 100.0,
                height: 100.0,
                confidence: Some(1.0),
            }],
        };

        Ok(OCRResponse {
            page_count: 1,
            pages: vec![result],
            document_type: Some("EXCEL".to_string()),
        })
    }

    /// Process PDF files - try free PDFium extraction first, fallback to Azure if needed
    async fn process_pdf(&self, file_path: &Path) -> Result<OCRResponse, Box<dyn Error + Send + Sync>> {
        // Try PDFium text extraction first (free, built-in)
        match self.extract_pdf_text(file_path).await {
            Ok(ocr_response) if !ocr_response.pages.is_empty() => {
                println!("Successfully extracted text from PDF using PDFium (free)");
                return Ok(ocr_response);
            }
            Ok(_) | Err(_) => {
                // PDFium failed or no text found, try Azure if configured
                println!("PDFium extraction failed or found no text, trying Azure OCR...");
                
                // Check if Azure is configured
                if std::env::var("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT").is_ok() 
                    && !std::env::var("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT").unwrap_or_default().is_empty() {
                    let ocr_service = create_ocr_service()?;
                    println!("Using Azure OCR (paid) as fallback");
                    ocr_service.process_document(file_path).await
                } else {
                    // No Azure configured, return error with helpful message
                    Err("PDF has no extractable text and Azure OCR is not configured. Please add AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and AZURE_DOCUMENT_INTELLIGENCE_KEY to .env file.".into())
                }
            }
        }
    }

    /// Extract text from PDF using PDFium (free, built-in)
    async fn extract_pdf_text(&self, file_path: &Path) -> Result<OCRResponse, Box<dyn Error + Send + Sync>> {
        use pdfium_render::prelude::*;
        
        let pdfium = Pdfium::new(
            Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path("./pdfium-binaries/"))
                .or_else(|_| Pdfium::bind_to_system_library())
                .map_err(|e| format!("Failed to load PDFium: {}", e))?
        );
        
        let document = pdfium.load_pdf_from_file(file_path, None)
            .map_err(|e| format!("Failed to open PDF: {}", e))?;
        
        let mut pages = Vec::new();
        
        for (page_index, page) in document.pages().iter().enumerate() {
            let text = page.text()
                .map_err(|e| format!("Failed to extract text from page {}: {}", page_index + 1, e))?
                .all();
            
            if !text.trim().is_empty() {
                pages.push(OCRResult {
                    page_number: (page_index + 1) as i32,
                    text: text.clone(),
                    confidence: Some(1.0), // Direct text extraction is 100% confident
                    bounding_boxes: vec![BoundingBox {
                        text,
                        x: 0.0,
                        y: 0.0,
                        width: 100.0,
                        height: 100.0,
                        confidence: Some(1.0),
                    }],
                });
            }
        }
        
        if pages.is_empty() {
            return Err("PDF contains no extractable text (might be scanned images)".into());
        }
        
        Ok(OCRResponse {
            page_count: pages.len() as i32,
            pages,
            document_type: Some("PDF".to_string()),
        })
    }

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

