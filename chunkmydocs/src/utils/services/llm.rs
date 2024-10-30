use crate::models::server::segment::{Chunk, Segment};
use base64::{engine::general_purpose, Engine as _};
use serde_json::to_string_pretty;
use serde_json::{json, Value};
use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct Field {
    pub name: String,
    pub field_type: String,
    pub description: String,
}

#[derive(Serialize, Deserialize)]
pub struct Context {
    pub ranked_segments: Vec<Segment>,
    pub field: Field,
}

#[derive(Serialize, Deserialize)]
pub struct RerankRequest {
    pub query: String,
    pub segments: Vec<Segment>,
}

#[derive(Serialize, Deserialize)]
pub struct BatchRerankRequest {
    pub queries: Vec<String>,
    pub segments: Vec<Segment>,
}

#[derive(Serialize, Deserialize)]
pub struct RerankerRequest {
    pub query: String,
    pub texts: Vec<String>,
    pub raw_scores: bool,
}

async fn rerank(
    client: &Client,
    reranker_url: &str,
    chunks: Vec<Chunk>,
    fields: Vec<Field>,
) -> Result<Vec<Context>, Box<dyn Error + Send + Sync>> {
    let mut contexts = Vec::new();

    // Process each field/query separately
    for field in &fields {
        let query = format!("{}: {}", field.name, field.description);

        // Extract all texts from segments
        let texts: Vec<String> = chunks
            .iter()
            .flat_map(|chunk| &chunk.segments)
            .map(|segment| segment.content.clone())
            .collect();

        let segments: Vec<Segment> = chunks
            .iter()
            .flat_map(|chunk| chunk.segments.clone())
            .collect();

        let reranker_request = RerankerRequest {
            query,
            texts,
            raw_scores: false,
        };

        println!("Sending request to {}", reranker_url);
        println!("Request body: {}", to_string_pretty(&reranker_request)?);

        let response = client
            .post(reranker_url)
            .json(&reranker_request)
            .send()
            .await?;

        let status = response.status();
        let text = response.text().await?;
        println!("Response status: {}", status);
        println!("Response body: {}", text);

        let response_body: Vec<serde_json::Value> = serde_json::from_str(&text)?;

        // Create ranked segments based on the response
        let mut ranked_segments = Vec::new();
        for result in &response_body {
            let index = result["index"]
                .as_u64()
                .ok_or("Missing or invalid 'index' field")? as usize;
            ranked_segments.push(segments[index].clone());
        }

        contexts.push(Context {
            ranked_segments,
            field: field.clone(),
        });
    }

    Ok(contexts)
}

pub async fn llm_call(
    url: String,
    key: String,
    prompt: String,
    model: String,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
) -> Result<String, Box<dyn Error>> {
    let client = reqwest::Client::new();

    let payload = json!({
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": max_tokens.unwrap_or(8000),
        "temperature": temperature.unwrap_or(0.0)
    });

    let response = client
        .post(format!("{}/v1/chat/completions", url))
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {}", key))
        .json(&payload)
        .send()
        .await?;
    let response_body: Value = response.json().await?;
    let completion = response_body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("No response")
        .to_string();
    Ok(completion)
}

pub async fn vision_llm_call(
    file_path: &Path,
    url: String,
    key: String,
    prompt: String,
    model: String,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
) -> Result<String, Box<dyn Error>> {
    let client = reqwest::Client::new();

    let mut file = File::open(file_path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    let base64_image = general_purpose::STANDARD.encode(&buffer);

    let payload = json!({
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": format!("data:image/jpeg;base64,{}", base64_image)
                        }
                    }
                ]
            }
        ],
        "max_tokens": max_tokens.unwrap_or(8000),
        "temperature": temperature.unwrap_or(0.0)
    });

    let response = client
        .post(format!("{}/v1/chat/completions", url))
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {}", key))
        .json(&payload)
        .send()
        .await?;

    let response_body: Value = response.json().await?;
    println!("Response body: {}", response_body);
    let completion = response_body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("No response")
        .to_string();

    Ok(completion)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::server::segment::BoundingBox;
    use crate::models::server::segment::SegmentType;

    use crate::utils::configs::extraction_config::Config;
    use crate::utils::configs::extraction_config::Config;
    use std::fs;
    use std::path::Path;
    use std::time::Instant;
    use tokio;

    #[tokio::test]
    async fn test_ocr_llm() -> Result<(), Box<dyn Error + Send + Sync>> {
        println!("Starting test_ocr_llm_with_image");

        let input_dir = Path::new("input");
        let output_dir = Path::new("output");
        fs::create_dir_all(output_dir)?;

        let prompt = "Task: Convert the provided image into a complete and precise HTML representation using advanced HTML techniques. DO NOT SIMPLIFY THE TABLE STRUCTURE AND ENSURE YOU COMPLETE THE ENTIRE TABLE. Get all content including titles, headers and footers.
        Use basic CSS styling where needed.
        Instructions:
        1. Analyze the table thoroughly, ensuring no detail is overlooked.
        2. Create an HTML structure that exactly mirrors the layout and content of the table, using advanced HTML5 elements where appropriate (e.g., <thead>, <tbody>, <tfoot>, <colgroup>, <caption>).
        3. Include ALL text, numbers, and any other content present in the table.
        4. Preserve the table structure exactly as it appears, using appropriate HTML tags (<table>, <tr>, <td>, <th>, etc.).
        5. Maintain proper formatting, including any bold, italic, or other text styles visible. 
        6. Transcribe the content exactly as it appears, using appropriate semantic HTML elements (e.g., <strong>, <em>, <sup>, <sub>) where necessary.
        7. If applicable, use the <colgroup> and <col> elements to define column properties.
        8. Implement proper accessibility features, such as using 'scope' attributes for header cells and including a <caption> for the table if a title is present.
        9. PRESERVE ALL COMPLEXITY OF THE ORIGINAL TABLE IN YOUR HTML, including any nested tables, rowspans, or colspans.
        10. If the table contains any interactive elements or complex layouts, consider using appropriate ARIA attributes to enhance accessibility.
        11. ENSURE YOU COMPLETE THE ENTIRE TABLE. Do not stop until you have transcribed every single row and column. Use colspan and rowspan attributes to handle merged cells. If the table is extensive, prioritize completeness over perfect formatting.
        12. For large tables that may exceed token limits, focus on capturing all data even if it means simplifying some formatting. It's crucial to include every piece of information from the original table.
        13. If you encounter any issues or limitations while transcribing, indicate this clearly within HTML comments at the end of your output.
        14. Double-check that you have included all rows and columns before finishing.
        15. Pay special attention to merged cells, split cells, and complex table structures. Use appropriate colspan and rowspan attributes to accurately represent the table layout. For cells split diagonally or containing multiple pieces of information, consider using nested tables or CSS positioning to faithfully reproduce the layout.

        
        Output: Provide the complete HTML code, without any explanations or comments. 
        Your response will be evaluated based on its completeness, accuracy, and use of advanced HTML techniques in representing every aspect of the table.
        It is crucial to maintain the exact structure and visual appearance of the input table and all its complexity - do not simplify or alter the table structure in any way.
        YOU MUST COMPLETE THE ENTIRE TABLE, INCLUDING ALL ROWS AND COLUMNS. Failure to do so will result in an incorrect response.
        
        IMPORTANT: If you find yourself running out of space, prioritize completing all the data in the table over maintaining perfect formatting. It's better to have all the data in a slightly less formatted table than to have an incomplete table.
        Wrap your output in <output> </output> tags.";

        let models = vec![
            "qwen/qwen-2-vl-7b-instruct",
            "google/gemini-flash-1.5-8b",
            // "meta-llama/llama-3.2-11b-vision-instruct",
            // "anthropic/claude-3-haiku",
            // "openai/gpt-4o-mini",
        ];

        let input_files: Vec<_> = fs::read_dir(input_dir)?
            .filter_map(|entry| {
                entry.ok().and_then(|e| {
                    let path = e.path();
                    if path.extension().and_then(|ext| ext.to_str()) == Some("jpg") {
                        Some(path)
                    } else {
                        None
                    }
                })
            })
            .collect();

        let mut tasks = Vec::new();
        let config = Config::from_env().unwrap();
        let url = config.ocr_llm_url;
        let key = config.ocr_llm_key;
        for input_file in input_files {
            let table_name = input_file
                .file_stem()
                .and_then(|s| s.to_str())
                .ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid file name")
                })?
                .to_string();
            let table_dir = output_dir.join(&table_name);
            fs::create_dir_all(&table_dir)?;

            let original_image_path = table_dir.join("original.jpg");
            fs::copy(&input_file, &original_image_path)?;

            println!("Processing table: {:?}", table_name);

            for model in &models {
                let input_file = input_file.clone();
                let prompt = prompt.to_string();
                let model = model.to_string();
                let model_clone = model.clone(); // Clone the model string

                let table_dir = table_dir.clone();
                let table_name = table_name.clone();
                let url = url.clone();
                let key = key.clone();

                let task = tokio::spawn(async move {
                    let start_time = Instant::now();

                    match vision_llm_call(&input_file, url, key, prompt, model, None, None).await {
                        Ok(response) => {
                            let duration = start_time.elapsed();
                            println!(
                                "Model: {}, Table: {}, Time: {:?}",
                                &model_clone, &table_name, duration
                            );

                            let html_content = response
                                .split("<output>")
                                .nth(1)
                                .and_then(|s| s.split("</output>").next())
                                .unwrap_or(&response);

                            let html_file =
                                table_dir.join(format!("{}.html", model_clone.replace("/", "_")));
                            fs::write(&html_file, html_content)?;
                            println!("HTML for {} saved to {:?}", model_clone, html_file);

                            let csv_file =
                                table_dir.join(format!("{}.csv", model_clone.replace("/", "_")));
                            let csv_content = format!(
                                "Model,Table,Duration\n{},{},{:?}\n",
                                model_clone, table_name, duration
                            );
                            fs::write(&csv_file, csv_content)?;
                            println!("CSV for {} saved to {:?}", model_clone, csv_file);
                        }
                        Err(e) => {
                            println!(
                                "Error processing {} with model {}: {:?}",
                                table_name, model_clone, e
                            );
                        }
                    }
                    Ok::<_, Box<dyn Error + Send + Sync>>(())
                });

                tasks.push(task);
            }
        }

        for task in tasks {
            task.await??;
        }

        println!("test_ocr_llm_with_image completed successfully");
        Ok(())
    }
    #[tokio::test]
    async fn test_reranker() -> Result<(), Box<dyn Error + Send + Sync>> {
        // Test data setup
        let texts = vec!["apple", "orange", "carrot", "lettuce"];
        let queries = vec!["fruits", "vegetables"];

        // Create segments from texts
        let segments: Vec<Segment> = texts
            .iter()
            .enumerate()
            .map(|(i, &text)| Segment {
                segment_id: format!("id-{}", i),
                bbox: BoundingBox {
                    left: 0.0,
                    top: 0.0,
                    width: 100.0,
                    height: 100.0,
                },
                page_number: 1,
                page_width: 1000.0,
                page_height: 1000.0,
                content: text.to_string(),
                segment_type: SegmentType::Text,
                ocr: None,
                image: None,
                html: None,
                markdown: None,
            })
            .collect();

        // Create chunks containing the segments
        let chunks = vec![Chunk {
            segments: segments.clone(),
            chunk_length: segments.len() as i32,
        }];

        // Create fields for each query
        let fields: Vec<Field> = queries
            .into_iter()
            .map(|query| Field {
                name: query.to_string(),
                field_type: "extraction_set".to_string(),
                description: query.to_string(),
            })
            .collect();

        // Initialize HTTP client
        let client = Client::new();
        let reranker_url = "http://127.0.0.1:8086/rerank";

        // Call rerank function
        let contexts = rerank(&client, reranker_url, chunks, fields).await?;

        // Validate results
        for context in &contexts {
            // Add & here to borrow instead of move
            println!("Query: {}", context.field.name);
            println!("Ranked segments:");
            for segment in &context.ranked_segments {
                // Add & here as well
                println!("  - {} ({})", segment.content, segment.segment_id);
            }
            println!();

            match context.field.name.as_str() {
                "fruits" => {
                    assert!(context.ranked_segments.iter().any(|s| s.content == "apple"));
                    assert!(context
                        .ranked_segments
                        .iter()
                        .any(|s| s.content == "orange"));
                }
                "vegetables" => {
                    assert!(context
                        .ranked_segments
                        .iter()
                        .any(|s| s.content == "carrot"));
                    assert!(context
                        .ranked_segments
                        .iter()
                        .any(|s| s.content == "lettuce"));
                }
                _ => panic!("Unexpected field name"),
            }
        }

        Ok(())
    }
}
