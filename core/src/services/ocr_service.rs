use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OCRResult {
    pub page_number: i32,
    pub text: String,
    pub confidence: Option<f32>,
    pub bounding_boxes: Vec<BoundingBox>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub text: String,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OCRResponse {
    pub pages: Vec<OCRResult>,
    pub document_type: Option<String>,
    pub page_count: i32,
}

#[async_trait]
pub trait OCRService: Send + Sync {
    async fn process_document(&self, file_path: &Path) -> Result<OCRResponse, Box<dyn Error + Send + Sync>>;
    fn name(&self) -> &str;
}

/// Azure Document Intelligence OCR Service
pub struct AzureOCRService {
    endpoint: String,
    key: String,
}

impl AzureOCRService {
    pub fn new(endpoint: String, key: String) -> Self {
        Self { endpoint, key }
    }
    
    pub fn from_env() -> Result<Self, Box<dyn Error + Send + Sync>> {
        let endpoint = std::env::var("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
            .or_else(|_| std::env::var("AZURE__ENDPOINT"))
            .map_err(|_| "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT not set")?;
        let key = std::env::var("AZURE_DOCUMENT_INTELLIGENCE_KEY")
            .or_else(|_| std::env::var("AZURE__KEY"))
            .map_err(|_| "AZURE_DOCUMENT_INTELLIGENCE_KEY not set")?;
        Ok(Self::new(endpoint, key))
    }
}

#[async_trait]
impl OCRService for AzureOCRService {
    async fn process_document(&self, file_path: &Path) -> Result<OCRResponse, Box<dyn Error + Send + Sync>> {
        use crate::utils::services::azure::perform_azure_analysis;
        use crate::models::upload::SegmentationStrategy;
        use base64::{engine::general_purpose, Engine as _};
        
        // Create temp file from path
        let temp_file = tempfile::NamedTempFile::new()?;
        std::fs::copy(file_path, temp_file.path())?;
        
        // Use existing Azure analysis
        let chunks = match perform_azure_analysis(&temp_file, None, SegmentationStrategy::LayoutAnalysis).await {
            Ok(c) => c,
            Err(e) => {
                let err_msg = format!("Azure analysis failed: {}", e);
                return Err(err_msg.into());
            }
        };
        
        // Convert chunks to OCR results
        let mut pages = Vec::new();
        let mut page_map: std::collections::HashMap<usize, Vec<BoundingBox>> = std::collections::HashMap::new();
        
        for chunk in chunks {
            // segments is a Vec, not Option<Vec>
            for segment in chunk.segments {
                let page_num = segment.page_number as usize;
                let bbox = BoundingBox {
                    text: segment.content.clone(),
                    x: segment.bbox.left,
                    y: segment.bbox.top,
                    width: segment.bbox.width,
                    height: segment.bbox.height,
                    confidence: None,
                };
                page_map.entry(page_num).or_insert_with(Vec::new).push(bbox);
            }
        }
        
        // Create page results
        for (page_num, bboxes) in page_map.iter() {
            let text = bboxes.iter().map(|b| b.text.as_str()).collect::<Vec<_>>().join(" ");
            pages.push(OCRResult {
                page_number: *page_num as i32,
                text,
                confidence: None,
                bounding_boxes: bboxes.clone(),
            });
        }
        
        pages.sort_by_key(|p| p.page_number);
        
        Ok(OCRResponse {
            page_count: pages.len() as i32,
            pages,
            document_type: None,
        })
    }
    
    fn name(&self) -> &str {
        "Azure Document Intelligence"
    }
}

/// Docling OCR Service - modern document understanding with table extraction
pub struct DoclingOCRService {
    client: Client,
    service_url: String,
}

impl DoclingOCRService {
    pub fn new() -> Self {
        let service_url = std::env::var("DOCLING_SERVICE_URL")
            .unwrap_or_else(|_| "http://docling-service:8000".to_string());
        
        Self {
            client: Client::new(),
            service_url,
        }
    }
}

#[async_trait]
impl OCRService for DoclingOCRService {
    async fn process_document(&self, file_path: &Path) -> Result<OCRResponse, Box<dyn Error + Send + Sync>> {
        println!("Using Docling service at: {}", self.service_url);
        
        // Read file
        let file_bytes = tokio::fs::read(file_path).await?;
        let file_name = file_path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("document.pdf")
            .to_string();
        
        // Create multipart form
        let part = reqwest::multipart::Part::bytes(file_bytes)
            .file_name(file_name);
        
        let form = reqwest::multipart::Form::new()
            .part("file", part);
        
        // Call Docling service with timeout
        let response = self.client
            .post(&format!("{}/convert", self.service_url))
            .multipart(form)
            .timeout(std::time::Duration::from_secs(120))
            .send()
            .await
            .map_err(|e| format!("Docling request failed: {}", e))?;
        
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(format!("Docling failed with status {}: {}", status, error_text).into());
        }
        
        // Parse response
        let result: serde_json::Value = response.json().await
            .map_err(|e| format!("Failed to parse Docling response: {}", e))?;
        
        // Convert to OCRResponse format
        let pages = result["pages"]
            .as_array()
            .ok_or("No pages in response")?;
        
        let ocr_pages: Vec<OCRResult> = pages.iter().map(|page| {
            let text = page["text"].as_str().unwrap_or("").to_string();
            let page_num = page["page_number"].as_i64().unwrap_or(1) as i32;
            
            OCRResult {
                page_number: page_num,
                text: text.clone(),
                confidence: Some(0.95), // Docling has high accuracy
                bounding_boxes: vec![BoundingBox {
                    text,
                    x: 0.0,
                    y: 0.0,
                    width: 100.0,
                    height: 100.0,
                    confidence: Some(0.95),
                }],
            }
        }).collect();
        
        Ok(OCRResponse {
            page_count: ocr_pages.len() as i32,
            pages: ocr_pages,
            document_type: Some("Docling".to_string()),
        })
    }
    
    fn name(&self) -> &str {
        "Docling Document Understanding"
    }
}

/// Cached OCR Service wrapper
pub struct CachedOCRService {
    inner: Box<dyn OCRService>,
    cache_enabled: bool,
}

impl CachedOCRService {
    pub fn new(inner: Box<dyn OCRService>) -> Self {
        let cache_enabled = std::env::var("OCR_CACHE_ENABLED")
            .unwrap_or_else(|_| "true".to_string())
            .parse()
            .unwrap_or(true);
        
        Self {
            inner,
            cache_enabled,
        }
    }
    
    fn get_cache_key(&self, file_path: &Path) -> Result<String, Box<dyn Error + Send + Sync>> {
        use sha2::{Sha256, Digest};
        
        let content = std::fs::read(file_path)?;
        let mut hasher = Sha256::new();
        hasher.update(&content);
        let hash = format!("{:x}", hasher.finalize());
        
        Ok(format!("ocr_cache:{}", hash))
    }
    
    async fn get_cached(&self, cache_key: &str) -> Result<Option<OCRResponse>, Box<dyn Error + Send + Sync>> {
        if !self.cache_enabled {
            return Ok(None);
        }
        
        let pool = crate::utils::clients::get_redis_pool();
        let mut conn = pool.get().await?;
        
        let cached: Option<String> = deadpool_redis::redis::cmd("GET")
            .arg(cache_key)
            .query_async(&mut conn)
            .await?;
        
        if let Some(json_str) = cached {
            let response: OCRResponse = serde_json::from_str(&json_str)?;
            println!("OCR cache hit for key: {}", cache_key);
            return Ok(Some(response));
        }
        
        Ok(None)
    }
    
    async fn set_cached(&self, cache_key: &str, response: &OCRResponse) -> Result<(), Box<dyn Error + Send + Sync>> {
        if !self.cache_enabled {
            return Ok(());
        }
        
        let pool = crate::utils::clients::get_redis_pool();
        let mut conn = pool.get().await?;
        
        let json_str = serde_json::to_string(response)?;
        
        // Cache for 7 days
        deadpool_redis::redis::cmd("SETEX")
            .arg(cache_key)
            .arg(7 * 24 * 60 * 60) // 7 days in seconds
            .arg(json_str)
            .query_async::<()>(&mut conn)
            .await?;
        
        println!("OCR results cached for key: {}", cache_key);
        Ok(())
    }
}

#[async_trait]
impl OCRService for CachedOCRService {
    async fn process_document(&self, file_path: &Path) -> Result<OCRResponse, Box<dyn Error + Send + Sync>> {
        let cache_key = self.get_cache_key(file_path)?;
        
        // Try to get from cache
        if let Some(cached_response) = self.get_cached(&cache_key).await? {
            return Ok(cached_response);
        }
        
        // Process with underlying service
        let response = self.inner.process_document(file_path).await?;
        
        // Cache the result
        self.set_cached(&cache_key, &response).await?;
        
        Ok(response)
    }
    
    fn name(&self) -> &str {
        self.inner.name()
    }
}

/// Factory to create OCR service based on configuration
pub fn create_ocr_service() -> Result<Box<dyn OCRService>, Box<dyn Error + Send + Sync>> {
    let provider = std::env::var("OCR_PROVIDER").unwrap_or_else(|_| "docling".to_string());
    
    match provider.as_str() {
        "docling" => {
            // Use Docling for advanced document understanding (tables, layout, OCR)
            let service = DoclingOCRService::new();
            Ok(Box::new(CachedOCRService::new(Box::new(service))))
        }
        "azure" => {
            let service = AzureOCRService::from_env()?;
            Ok(Box::new(CachedOCRService::new(Box::new(service))))
        }
        _ => Err(format!("Unsupported OCR provider: {}", provider).into()),
    }
}

