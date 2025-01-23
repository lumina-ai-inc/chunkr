use crate::configs::search_config;
use crate::models::chunkr::output::Chunk;
use crate::utils::clients::get_reqwest_client;
use serde::{Deserialize, Serialize};
use std::error::Error;
use utoipa::ToSchema;

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema)]
#[serde(untagged)]
pub enum ChunkContent {
    Full(Chunk),
    Simple(SimpleChunk),
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema)]
pub struct SimpleChunk {
    pub id: String,
    pub content: String,
}

#[derive(Clone, Serialize)]
struct EmbeddingRequest {
    inputs: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub chunk: SimpleChunk,
    pub score: f32,
}

pub struct Search {
    chunks: Vec<SimpleChunk>,
    embeddings: Vec<Vec<f32>>,
}

impl Search {
    pub async fn new(content: Vec<ChunkContent>) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let chunks: Vec<SimpleChunk> = content
            .into_iter()
            .map(|chunk| match chunk {
                ChunkContent::Simple(simple) => simple,
                ChunkContent::Full(full) => full.to_simple(),
            })
            .collect();

        let mut search = Search {
            chunks: Vec::new(),
            embeddings: Vec::new(),
        };
        let texts: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();
        search.embeddings = search.generate_embeddings(texts).await?;
        search.chunks = chunks;
        Ok(search)
    }

    pub async fn search(
        &self,
        query: &str,
    ) -> Result<Vec<SearchResult>, Box<dyn Error + Send + Sync>> {
        let query_embedding = self.generate_embeddings(vec![query.to_string()]).await?;
        let similar_chunks = self.find_similar_chunks(&query_embedding[0]);
        Ok(similar_chunks)
    }

    async fn generate_embeddings(
        &self,
        texts: Vec<String>,
    ) -> Result<Vec<Vec<f32>>, Box<dyn Error + Send + Sync>> {
        let search_config = search_config::Config::from_env()?;
        let client = get_reqwest_client();
        let responses = futures::future::try_join_all(
            texts
                .into_iter()
                .filter(|text| !text.trim().is_empty())
                .collect::<Vec<_>>()
                .chunks(search_config.batch_size)
                .map(|chunk| {
                    client
                        .post(&search_config.dense_vector_url)
                        .json(&EmbeddingRequest {
                            inputs: chunk.to_vec(),
                        })
                        .send()
                }),
        )
        .await?;
        let response_texts: Vec<String> =
            futures::future::try_join_all(responses.into_iter().map(|r| r.text())).await?;
        let embeddings: Vec<Vec<f32>> = serde_json::from_str(&response_texts.join(""))?;
        Ok(embeddings)
    }

    fn find_similar_chunks(&self, query_embedding: &[f32]) -> Vec<SearchResult> {
        let search_config = search_config::Config::from_env().unwrap();
        let mut similarities: Vec<(usize, f32)> = self
            .embeddings
            .iter()
            .enumerate()
            .map(|(idx, embedding)| (idx, self.cosine_similarity(query_embedding, embedding)))
            .collect();
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(search_config.top_k);
        similarities
            .into_iter()
            .map(|(idx, score)| SearchResult {
                chunk: self.chunks[idx].clone(),
                score,
            })
            .collect()
    }

    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot_product / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::clients::initialize;
    #[tokio::test]
    async fn test_search() -> Result<(), Box<dyn Error + Send + Sync>> {
        initialize().await;
        let chunks = vec![
            ChunkContent::Simple(SimpleChunk {
                id: "1".to_string(),
                content: "Apple is a sweet fruit".to_string(),
            }),
            ChunkContent::Simple(SimpleChunk {
                id: "2".to_string(),
                content: "Orange is a citrus fruit".to_string(),
            }),
            ChunkContent::Simple(SimpleChunk {
                id: "3".to_string(),
                content: "Carrot is a root vegetable".to_string(),
            }),
        ];

        let search = Search::new(chunks).await?;

        let results = search.search("citrus fruits").await?;
        println!("Results: {:?}", results);
        assert!(!results.is_empty(), "Should return at least one result");
        assert!(
            results[0].chunk.content.contains("citrus"),
            "First result should be about citrus"
        );

        Ok(())
    }
}
