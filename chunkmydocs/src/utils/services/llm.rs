use crate::models::workers::open_ai::{
    ContentPart, ImageUrl, Message, MessageContent, OpenAiRequest, OpenAiResponse,
};
use base64::{engine::general_purpose, Engine as _};
use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub async fn open_ai_call(
    url: String,
    key: String,
    model: String,
    messages: Vec<Message>,
    max_completion_tokens: Option<u32>,
    temperature: Option<f32>,
) -> Result<OpenAiResponse, Box<dyn Error>> {
    let request = OpenAiRequest {
        model,
        messages,
        max_completion_tokens,
        temperature,
    };

    let client = reqwest::Client::new();
    let response = client
        .post(url)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {}", key))
        .json(&request)
        .send()
        .await?
        .error_for_status()?;

    let response: OpenAiResponse = response.json().await?;
    Ok(response)
}

pub fn get_basic_message(prompt: String) -> Result<Vec<Message>, Box<dyn Error>> {
    Ok(vec![Message {
        role: "user".to_string(),
        content: MessageContent::String { content: prompt },
    }])
}

pub fn get_basic_image_message(
    file_path: &Path,
    prompt: String,
) -> Result<Vec<Message>, Box<dyn Error>> {
    let mut file = File::open(file_path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    let base64_image = general_purpose::STANDARD.encode(&buffer);
    Ok(vec![Message {
        role: "user".to_string(),
        content: MessageContent::Array {
            content: vec![
                ContentPart {
                    content_type: "text".to_string(),
                    text: Some(prompt),
                    image_url: None,
                },
                ContentPart {
                    content_type: "image_url".to_string(),
                    text: None,
                    image_url: Some(ImageUrl {
                        url: format!("data:image/jpeg;base64,{}", base64_image),
                    }),
                },
            ],
        },
    }])
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::utils::configs::llm_config::{get_prompt, Config as LlmConfig};
    use crate::utils::services::segment_ocr::get_html_from_llm_table_ocr;
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
        let llm_config = LlmConfig::from_env().unwrap();
        let url = llm_config.url;
        let key = llm_config.key;

        let models = vec![
            // "qwen/qwen-2-vl-7b-instruct",
            // "google/gemini-flash-1.5-8b",
            // "google/gemini-pro-1.5",
            // "meta-llama/llama-3.2-11b-vision-instruct",
            // "anthropic/claude-3-haiku",
            // "openai/chatgpt-4o-latest",
            // "gpt-4o",
            llm_config.model,
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
                let model_clone = model.clone();

                let table_dir = table_dir.clone();
                let table_name = table_name.clone();
                let url = url.clone();
                let key = key.clone();

                let task = tokio::spawn(async move {
                    let start_time = Instant::now();
                    let messages = match get_basic_image_message(&input_file, prompt) {
                        Ok(messages) => messages,
                        Err(e) => {
                            println!("Error getting basic image message: {:?}", e);
                            return Err(e.to_string().into());
                        }
                    };

                    match open_ai_call(url, key, model, messages, None, None).await {
                        Ok(response) => {
                            let duration = start_time.elapsed();
                            let content = response.choices[0].message.content.clone();
                            if let MessageContent::String { content } = content {
                                let html_content = get_html_from_llm_table_ocr(content)?;
                                let html_file = table_dir
                                    .join(format!("{}.html", model_clone.replace("/", "_")));
                                fs::write(&html_file, html_content)?;
                                println!("HTML for {} saved to {:?}", model_clone, html_file);
                            } else {
                                return Err("Invalid content type".into());
                            }

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
