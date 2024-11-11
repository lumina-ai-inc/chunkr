use base64::{engine::general_purpose, Engine as _};
use serde_json::{json, Value};
use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub async fn llm_call(
    url: String,
    key: String,
    model: String,
    prompt: String,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
) -> Result<String, Box<dyn Error>> {
    let client = reqwest::Client::new();
    println!("LLM call: {:?}", prompt);
    let payload = json!({
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": max_tokens.unwrap_or(16000),
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

pub async fn vlm_call(
    file_path: &Path,
    url: String,
    key: String,
    model: String,
    prompt: String,
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
        .post(url)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {}", key))
        .json(&payload)
        .send()
        .await?
        .error_for_status()?;

    let response_body: Value = response.json().await?;

    let completion = response_body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("No response")
        .to_string();

    Ok(completion)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::utils::configs::llm_config::{get_prompt, Config as LlmConfig};
    use crate::utils::services::ocr::get_html_from_vllm_table_ocr;
    use std::collections::HashMap;
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

        let prompt = get_prompt("table", &HashMap::new())?;

        let models = vec![
            // "qwen/qwen-2-vl-7b-instruct",
            // "google/gemini-flash-1.5-8b",
            "google/gemini-pro-1.5",
            // "meta-llama/llama-3.2-11b-vision-instruct",
            // "anthropic/claude-3-haiku",
            // "openai/chatgpt-4o-latest",
            // "gpt-4o",
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
        let llm_config = LlmConfig::from_env().unwrap();
        let url = llm_config.url;
        let key = llm_config.key;
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

                    match vlm_call(&input_file, url, key, model, prompt, None, None).await {
                        Ok(response) => {
                            let duration = start_time.elapsed();
                            let html_content = get_html_from_vllm_table_ocr(response)?;

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
                        }
                        Err(e) => {
                            println!(
                                "Error processing {} with model {}: {:?}",
                                table_name, model_clone, e
                            );
                            assert!(false);
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
