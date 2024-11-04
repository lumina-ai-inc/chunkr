use crate::models::server::segment::Segment;
use base64::{engine::general_purpose, Engine as _};
use serde_json::{json, Value};
use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::Path;

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
        .post(url)
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

    use crate::utils::configs::structured_extract::Config;
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
        let url = config.llm_url;
        let key = config.llm_key;
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
}
