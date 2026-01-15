use crate::agents::common::{AgentContext, AgentError, AgentResult, log_execution};
use crate::models::document::{Document, DocumentType};
use crate::models::fact::{FactType, NewFact, SourceCitation};
use crate::models::output::OCRResult;
use crate::services::ai::{OpenAIClient, VectorDB};
use crate::services::ai::prompts::{get_document_extraction_prompt, DOCUMENT_CLASSIFIER_SYSTEM};
use crate::utils::clients::get_pg_client;
use chrono::Utc;
use diesel::prelude::*;
use serde_json::{json, Value as JsonValue};
use std::collections::HashMap;
use std::error::Error;
use uuid::Uuid;

pub struct DocumentParserAgent {
    openai_client: OpenAIClient,
    vector_db: VectorDB,
}

impl DocumentParserAgent {
    pub async fn new() -> Result<Self, Box<dyn Error + Send + Sync>> {
        let openai_client = OpenAIClient::new().await?;
        let vector_db = VectorDB::new().await?;
        Ok(Self {
            openai_client,
            vector_db,
        })
    }

    /// Main entry point: Parse document and extract facts
    pub async fn parse_document(
        &self,
        document_id: &str,
        deal_id: &str,
        ocr_results: Vec<Vec<OCRResult>>,
        context: AgentContext,
    ) -> Result<AgentResult, Box<dyn Error + Send + Sync>> {
        let start_time = std::time::Instant::now();
        let input = json!({
            "document_id": document_id,
            "deal_id": deal_id,
            "ocr_pages": ocr_results.len(),
        });

        // Get document from database
        let mut client = get_pg_client().await?;
        let document = self.get_document(&mut client, document_id).await?;

        // Step 1: Classify document type
        let document_type = self.classify_document(&document, &ocr_results).await?;
        
        // Step 2: Extract facts using AI
        let extracted_facts = self.extract_facts_ai(&document, &ocr_results, &document_type).await?;
        
        // Step 3: Validate and cite facts
        let validated_facts = self.validate_and_cite_facts(extracted_facts, &ocr_results)?;
        
        // Step 4: Store facts in database
        let fact_ids = self.store_facts(&mut client, deal_id, document_id, validated_facts.clone()).await?;
        
        // Step 5: Create embeddings and store in vector DB
        self.create_and_store_embeddings(document_id, deal_id, &validated_facts, &ocr_results).await?;
        
        // Step 6: Update document status
        self.update_document_status(&mut client, document_id, "completed").await?;

        let execution_time = start_time.elapsed().as_millis() as i32;
        
        // Log execution
        let output = json!({
            "document_type": document_type,
            "facts_extracted": fact_ids.len(),
            "fact_ids": fact_ids,
        });

        let _execution_id = log_execution(
            "document_parser",
            Some(document_id.to_string()),
            Some("document".to_string()),
            input,
            Some(output.clone()),
            "completed",
            None,
            Some("openai".to_string()),
            Some("gpt-4-turbo".to_string()),
            None, // Tokens tracked internally
            Some(execution_time),
        )
        .await?;

        Ok(AgentResult::success(output, None, execution_time))
    }

    /// Classify document type using AI
    async fn classify_document(
        &self,
        document: &Document,
        ocr_results: &[Vec<OCRResult>],
    ) -> Result<String, Box<dyn Error + Send + Sync>> {
        // Combine first few pages of OCR text for classification
        let ocr_text: String = ocr_results
            .iter()
            .take(3) // Only use first 3 pages for classification
            .flatten()
            .map(|r| r.text.clone())
            .collect::<Vec<_>>()
            .join(" ");

        let truncated_text = if ocr_text.len() > 2000 {
            &ocr_text[..2000]
        } else {
            &ocr_text
        };

        // Simple classification based on filename and keywords first
        let file_name = document.file_name.to_lowercase();
        if file_name.contains("rent") && file_name.contains("roll") {
            return Ok("RENT_ROLL".to_string());
        }
        if file_name.contains("p&l") || file_name.contains("profit") || file_name.contains("loss") {
            return Ok("PROFIT_AND_LOSS".to_string());
        }
        
        // Use AI for ambiguous cases
        let prompt = format!(
            "File name: {}\n\nContent preview:\n{}",
            document.file_name, truncated_text
        );

        let response = self.openai_client
            .chat_completion(
                vec![
                    async_openai::types::ChatCompletionRequestMessage::System(
                        async_openai::types::ChatCompletionRequestSystemMessageArgs::default()
                            .content(DOCUMENT_CLASSIFIER_SYSTEM)
                            .build()?
                    ),
                    async_openai::types::ChatCompletionRequestMessage::User(
                        async_openai::types::ChatCompletionRequestUserMessageArgs::default()
                            .content(prompt)
                            .build()?
                    ),
                ],
                Some(0.1),
            )
            .await?;

        // Extract document type from response
        let doc_type = if response.content.contains("RENT_ROLL") {
            "RENT_ROLL"
        } else if response.content.contains("PROFIT_AND_LOSS") {
            "PROFIT_AND_LOSS"
        } else if response.content.contains("MORTGAGE") {
            "MORTGAGE_STATEMENT"
        } else if response.content.contains("TAX") {
            "TAX_DOCUMENT"
        } else {
            "OTHER"
        };

        Ok(doc_type.to_string())
    }

    /// Extract facts from document using AI
    async fn extract_facts_ai(
        &self,
        document: &Document,
        ocr_results: &[Vec<OCRResult>],
        document_type: &str,
    ) -> Result<Vec<ExtractedFact>, Box<dyn Error + Send + Sync>> {
        // Combine all OCR text
        let ocr_text: String = ocr_results
            .iter()
            .flatten()
            .map(|r| r.text.clone())
            .collect::<Vec<_>>()
            .join("\n");

        // Get appropriate prompt for document type
        let system_prompt = get_document_extraction_prompt(document_type);

        // Extract facts using GPT-4 with structured output
        let facts = self.openai_client
            .extract_facts_from_ocr(
                document_type,
                &ocr_text,
                system_prompt,
            )
            .await?;

        Ok(facts.facts.into_iter().map(|f| ExtractedFact {
            fact_type: f.fact_type,
            label: f.label,
            value: f.value,
            unit: f.unit,
            confidence: f.confidence,
            source_page: f.source_page,
            source_text: f.source_text,
        }).collect())
    }

    /// Validate facts and add source citations
    fn validate_and_cite_facts(
        &self,
        facts: Vec<ExtractedFact>,
        ocr_results: &[Vec<OCRResult>],
    ) -> Result<Vec<ValidatedFact>, Box<dyn Error + Send + Sync>> {
        let mut validated = Vec::new();

        for fact in facts {
            // Find source in OCR results if page is specified
            let source_citation = if let Some(page_num) = fact.source_page {
                let page_idx = (page_num - 1) as usize;
                if page_idx < ocr_results.len() {
                    // Find the exact text in the page
                    if let Some(source_text) = &fact.source_text {
                        if let Some(ocr_match) = ocr_results[page_idx]
                            .iter()
                            .find(|r| r.text.contains(source_text))
                        {
                            Some(SourceCitation {
                                document_page: page_num,
                                text: source_text.clone(),
                                bbox: Some(crate::models::fact::BoundingBox {
                                    left: ocr_match.bbox.x,
                                    top: ocr_match.bbox.y,
                                    width: ocr_match.bbox.width,
                                    height: ocr_match.bbox.height,
                                    page: page_num,
                                }),
                            })
                        } else {
                            Some(SourceCitation {
                                document_page: page_num,
                                text: source_text.clone(),
                                bbox: None,
                            })
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            validated.push(ValidatedFact {
                fact_type: fact.fact_type,
                label: fact.label,
                value: fact.value,
                unit: fact.unit,
                confidence: fact.confidence,
                source_citation,
            });
        }

        Ok(validated)
    }

    /// Store facts in database
    async fn store_facts(
        &self,
        client: &mut deadpool_postgres::Client,
        deal_id: &str,
        document_id: &str,
        facts: Vec<ValidatedFact>,
    ) -> Result<Vec<String>, Box<dyn Error + Send + Sync>> {
        use crate::data::schema::facts;

        let mut fact_ids = Vec::new();

        for fact in facts {
            let fact_id = Uuid::new_v4().to_string();
            let now = Utc::now().naive_utc();

            let source_json = if let Some(citation) = fact.source_citation {
                json!(citation)
            } else {
                json!({})
            };

            let new_fact = NewFact {
                fact_id: fact_id.clone(),
                document_id: document_id.to_string(),
                deal_id: deal_id.to_string(),
                fact_type: fact.fact_type,
                label: fact.label,
                value: fact.value,
                unit: fact.unit,
                source_citation: source_json,
                status: "needs_review".to_string(),
                confidence_score: Some(fact.confidence),
                extraction_method: Some("ai".to_string()),
                reviewed_by_user: Some(false),
                embedding_id: None,
                locked: false,
                created_at: now,
            };

            diesel::insert_into(facts::table)
                .values(&new_fact)
                .execute(client)
                .await?;

            fact_ids.push(fact_id);
        }

        Ok(fact_ids)
    }

    /// Create embeddings and store in vector DB
    async fn create_and_store_embeddings(
        &self,
        document_id: &str,
        deal_id: &str,
        facts: &[ValidatedFact],
        ocr_results: &[Vec<OCRResult>],
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Create document chunks for embedding
        let mut embedding_tasks = Vec::new();

        // Chunk OCR results by page
        for (page_num, page_results) in ocr_results.iter().enumerate() {
            let page_text: String = page_results
                .iter()
                .map(|r| r.text.clone())
                .collect::<Vec<_>>()
                .join(" ");

            if !page_text.is_empty() {
                embedding_tasks.push((
                    format!("{}-page-{}", document_id, page_num + 1),
                    page_text,
                    page_num + 1,
                ));
            }
        }

        // Create embeddings in batch
        let texts: Vec<String> = embedding_tasks.iter().map(|(_, text, _)| text.clone()).collect();
        let embeddings = self.openai_client.create_embeddings_batch(texts).await?;

        // Store in vector DB
        let points: Vec<(String, Vec<f32>, HashMap<String, JsonValue>)> = embedding_tasks
            .into_iter()
            .zip(embeddings.into_iter())
            .map(|((id, text, page_num), embedding)| {
                let mut payload = HashMap::new();
                payload.insert("document_id".to_string(), json!(document_id));
                payload.insert("deal_id".to_string(), json!(deal_id));
                payload.insert("page_number".to_string(), json!(page_num));
                payload.insert("text".to_string(), json!(text));
                payload.insert("chunk_type".to_string(), json!("page"));
                
                (id, embedding, payload)
            })
            .collect();

        self.vector_db
            .upsert_embeddings_batch("documents", points)
            .await?;

        // Create embeddings for individual facts
        let fact_texts: Vec<String> = facts
            .iter()
            .map(|f| format!("{}: {} {}", f.label, f.value, f.unit.as_ref().unwrap_or(&"".to_string())))
            .collect();

        if !fact_texts.is_empty() {
            let fact_embeddings = self.openai_client.create_embeddings_batch(fact_texts.clone()).await?;
            
            let fact_points: Vec<(String, Vec<f32>, HashMap<String, JsonValue>)> = facts
                .iter()
                .zip(fact_embeddings.into_iter())
                .enumerate()
                .map(|(idx, (fact, embedding))| {
                    let id = format!("{}-fact-{}", document_id, idx);
                    let mut payload = HashMap::new();
                    payload.insert("document_id".to_string(), json!(document_id));
                    payload.insert("deal_id".to_string(), json!(deal_id));
                    payload.insert("fact_type".to_string(), json!(fact.fact_type));
                    payload.insert("label".to_string(), json!(fact.label));
                    payload.insert("value".to_string(), json!(fact.value));
                    payload.insert("confidence".to_string(), json!(fact.confidence));
                    
                    (id, embedding, payload)
                })
                .collect();

            self.vector_db
                .upsert_embeddings_batch("facts", fact_points)
                .await?;
        }

        Ok(())
    }

    /// Update document processing status
    async fn update_document_status(
        &self,
        client: &mut deadpool_postgres::Client,
        document_id: &str,
        status: &str,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        use crate::data::schema::documents;

        let now = Utc::now().naive_utc();

        diesel::update(documents::table.filter(documents::document_id.eq(document_id)))
            .set((
                documents::fact_extraction_status.eq(status),
                documents::fact_extraction_completed_at.eq(Some(now)),
            ))
            .execute(client)
            .await?;

        Ok(())
    }

    /// Helper: Get document from database
    async fn get_document(
        &self,
        client: &mut deadpool_postgres::Client,
        document_id: &str,
    ) -> Result<Document, Box<dyn Error + Send + Sync>> {
        use crate::data::schema::documents;

        let document = documents::table
            .filter(documents::document_id.eq(document_id))
            .first::<Document>(client)
            .await?;

        Ok(document)
    }
}

#[derive(Debug, Clone)]
struct ExtractedFact {
    pub fact_type: String,
    pub label: String,
    pub value: String,
    pub unit: Option<String>,
    pub confidence: f64,
    pub source_page: Option<i32>,
    pub source_text: Option<String>,
}

#[derive(Debug, Clone)]
struct ValidatedFact {
    pub fact_type: String,
    pub label: String,
    pub value: String,
    pub unit: Option<String>,
    pub confidence: f64,
    pub source_citation: Option<SourceCitation>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires database and API keys
    async fn test_document_parser() {
        let agent = DocumentParserAgent::new().await.unwrap();
        // Add test implementation
    }
}

