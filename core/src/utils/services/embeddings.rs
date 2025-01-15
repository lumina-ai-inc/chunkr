use reqwest::Client;
use serde::Serialize;
use std::collections::HashMap;
use std::error::Error;

#[derive(Clone, Serialize)]
pub struct EmbeddingRequest {
    inputs: Vec<String>,
}

#[derive(Clone)]
pub struct EmbeddingCache {
    pub embeddings: HashMap<String, Vec<f32>>,
}

impl EmbeddingCache {
    pub fn new() -> Self {
        EmbeddingCache {
            embeddings: HashMap::new(),
        }
    }

    pub fn get_embedding(&self, text: &str) -> Option<&Vec<f32>> {
        self.embeddings.get(text)
    }

    async fn generate_embeddings(
        &self,
        client: &Client,
        embedding_url: &str,
        texts: Vec<String>,
        batch_size: usize,
    ) -> Result<Vec<Vec<f32>>, Box<dyn Error + Send + Sync>> {
        let mut all_embeddings = Vec::new();
        let mut futures = Vec::new();
        for chunk in texts.chunks(batch_size) {
            let request = EmbeddingRequest {
                inputs: chunk.to_vec(),
            };
            let future = client.post(embedding_url).json(&request).send();
            futures.push(future);
        }

        let responses = futures::future::join_all(futures).await;

        for response in responses {
            let response = response?;
            let status = response.status();
            let response_text = response.text().await?;

            if !status.is_success() {
                println!("Error: Embedding server returned status {}", status);
                return Err(format!("Embedding server error: {}", status).into());
            }

            if response_text.trim().is_empty() {
                println!("Error: Empty response from embedding server");
                return Err("Empty response from embedding server".into());
            }

            match serde_json::from_str::<Vec<Vec<f32>>>(&response_text) {
                Ok(embeddings) => {
                    all_embeddings.extend(embeddings);
                }
                Err(e) => {
                    println!("Error parsing embeddings: {}", e);
                    return Err(Box::new(e));
                }
            }
        }

        Ok(all_embeddings)
    }

    pub async fn get_or_generate_embeddings(
        &mut self,
        client: &Client,
        embedding_url: &str,
        texts: Vec<String>,
        batch_size: usize,
    ) -> Result<Vec<Vec<f32>>, Box<dyn Error + Send + Sync>> {
        let mut result = Vec::new();
        let mut texts_to_generate = Vec::new();

        for text in texts {
            if let Some(embedding) = self.embeddings.get(&text) {
                result.push(embedding.clone());
            } else {
                texts_to_generate.push(text);
            }
        }

        if !texts_to_generate.is_empty() {
            let new_embeddings = self
                .generate_embeddings(client, embedding_url, texts_to_generate, batch_size)
                .await?;

            result.extend(new_embeddings);
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::configs::search_config::Config as SearchConfig;
    use crate::models::chunkr::output::{BoundingBox, Segment, SegmentType};
    use tokio;
    #[tokio::test]
    async fn embeddings() {
        // Mock client and response
        let client = reqwest::Client::new();
        let search_config = SearchConfig::from_env().unwrap();
        let embedding_url = format!("{}/embed", search_config.dense_vector_url);
        let segments = vec![
            Segment {
                segment_id: "1".to_string(),
                confidence: Some(1.0),
                content: "Today is a nice day".to_string(),
                llm: None,
                bbox: BoundingBox {
                    left: 0.0,
                    top: 0.0,
                    width: 100.0,
                    height: 100.0,
                },
                page_number: 1,
                page_width: 1000.0,
                page_height: 1000.0,
                segment_type: SegmentType::Text,
                ocr: vec![],
                image: None,
                html: None,
                markdown: Some("**Today is a nice day**".to_string()),
            },
            Segment {
                segment_id: "2".to_string(),
                confidence: Some(1.0),
                content: "I like you".to_string(),
                llm: None,
                bbox: BoundingBox {
                    left: 0.0,
                    top: 0.0,
                    width: 100.0,
                    height: 100.0,
                },
                page_number: 1,
                page_width: 1000.0,
                page_height: 1000.0,
                segment_type: SegmentType::Text,
                ocr: vec![],
                image: None,
                html: None,
                markdown: Some("_I like you_".to_string()),
            },
        ];
        let batch_size = 2;

        let markdown_texts: Vec<String> = segments
            .iter()
            .filter_map(|segment| segment.markdown.clone())
            .collect();

        let mut cache = EmbeddingCache {
            embeddings: HashMap::new(),
        };

        let result = cache
            .generate_embeddings(&client, &embedding_url, markdown_texts.clone(), batch_size)
            .await;
        assert!(result.is_ok());

        let get_or_generate_result = cache
            .get_or_generate_embeddings(&client, &embedding_url, markdown_texts, batch_size)
            .await;
        assert!(get_or_generate_result.is_ok());
        // println!("{:?}", result);
    }
}
