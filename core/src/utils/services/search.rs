use crate::models::chunkr::output::Segment;

pub struct SearchResult {
    pub segment: Segment,
    pub score: f32,
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

pub fn search_embeddings(
    query_embedding: &[f32],
    texts: &[String],
    text_embeddings: &Vec<Vec<f32>>,
    top_k: usize,
) -> Vec<String> {
    let mut results: Vec<(String, f32)> = texts
        .iter()
        .zip(text_embeddings.iter())
        .map(|(text, embedding)| (text.clone(), cosine_similarity(query_embedding, embedding)))
        .collect();
    results.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(top_k);
    results.into_iter().map(|(text, _)| text).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::configs::search_config::Config as SearchConfig;
    use crate::utils::services::embeddings::EmbeddingCache;
    use std::error::Error;
    use tokio;

    #[tokio::test]
    async fn search() -> Result<(), Box<dyn Error + Send + Sync>> {
        let client = reqwest::Client::new();
        let search_config = SearchConfig::from_env()?;
        let embedding_url = search_config.dense_vector_url;

        let texts = vec![
            "**Apple**: A sweet fruit".to_string(),
            "**Orange**: A citrus fruit".to_string(), 
            "**Carrot**: A root vegetable".to_string(),
            "**Lemon**: A citrus fruit".to_string()
        ];

        let mut cache = EmbeddingCache::new();
        let text_embeddings = cache
            .get_or_generate_embeddings(&client, &embedding_url, texts.clone(), 2)
            .await?;

        let query = "citrus fruits".to_string();
        let query_embedding = cache
            .get_or_generate_embeddings(&client, &embedding_url, vec![query], 1)
            .await?;

        let results = search_embeddings(&query_embedding[0], &texts, &text_embeddings, 4);

        assert!(!results.is_empty(), "Should return at least one result");

        assert!(
            results[0].contains("Orange") || results[0].contains("Lemon"),
            "First result should be the most relevant"
        );

        for result in results {
            println!("Content: {}", result);
        }

        Ok(())
    }
}
