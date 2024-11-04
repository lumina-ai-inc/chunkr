use crate::models::server::segment::Segment;

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
    segments: &[Segment],
    segment_embeddings: &Vec<Vec<f32>>,
    top_k: usize,
) -> Vec<SearchResult> {
    let mut results: Vec<SearchResult> = segments
        .iter()
        .zip(segment_embeddings.iter())
        .map(|(segment, embedding)| SearchResult {
            segment: segment.clone(),
            score: cosine_similarity(query_embedding, embedding),
        })
        .collect();
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(top_k);
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::server::segment::{BoundingBox, SegmentType};
    use crate::utils::services::embeddings::EmbeddingCache;
    use std::error::Error;
    use tokio;

    #[tokio::test]
    async fn search() -> Result<(), Box<dyn Error + Send + Sync>> {
        let client = reqwest::Client::new();
        let embedding_url = "http://127.0.0.1:8085/embed";
        let segments = vec![
            Segment {
                segment_id: "1".to_string(),
                content: "Apple".to_string(),
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
                ocr: None,
                image: None,
                html: None,
                markdown: Some("**Apple**".to_string()),
            },
            Segment {
                segment_id: "2".to_string(),
                content: "Orange".to_string(),
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
                ocr: None,
                image: None,
                html: None,
                markdown: Some("_Orange_".to_string()),
            },
            Segment {
                segment_id: "3".to_string(),
                content: "Carrot".to_string(),
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
                ocr: None,
                image: None,
                html: None,
                markdown: Some("_Carrot_".to_string()),
            },
            Segment {
                segment_id: "4".to_string(),
                content: "Lemon".to_string(),
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
                ocr: None,
                image: None,
                html: None,
                markdown: Some("_Lemon_".to_string()),
            },
        ];

        let mut cache = EmbeddingCache::new();
        let markdown_texts: Vec<String> = segments
            .iter()
            .filter_map(|segment| segment.markdown.clone())
            .collect();

        let segment_embeddings = cache
            .get_or_generate_embeddings(&client, embedding_url, markdown_texts.clone(), 2)
            .await?;

        let query = "citrus fruits".to_string();
        let query_embedding = cache
            .get_or_generate_embeddings(&client, embedding_url, vec![query], 1)
            .await?;

        let results = search_embeddings(&query_embedding[0], &segments, &segment_embeddings, 4);

        assert!(!results.is_empty(), "Should return at least one result");

        assert!(
            results[0].segment.content == "Orange" || results[0].segment.content == "Lemon",
            "First result should be the most relevant"
        );
        for result in results {
            println!(
                "Content: {}, Score: {:.4}",
                result.segment.content, result.score
            );
        }

        Ok(())
    }
}
