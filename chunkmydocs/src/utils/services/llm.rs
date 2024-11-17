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
    let response = match client
        .post(url)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {}", key))
        .json(&request)
        .send()
        .await?
        .error_for_status() {
            Ok(response) => response,
            Err(e) => {
                println!("Error status: {}", e.status().unwrap_or_default());
                return Err(Box::new(e));
            }
        };

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
    use crate::utils::services::ocr::get_html_from_llm_table_ocr;
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

        let llm_config = LlmConfig::from_env().unwrap();
        let url = llm_config.url;
        let key = llm_config.key;

        let models = HashMap::from([
            ("geminiflash8b", "google/gemini-flash-1.5-8b"),
            ("geminiflash", "google/gemini-flash-1.5"),
            ("geminipro", "google/gemini-pro-1.5"),
        ]);

        let prompts = HashMap::from([
            ("prompt1", "Analyze this image and convert the table to HTML format maintaining the original structure. Output the table directly in ```html``` tags."),
            ("prompt2", "Task: Convert the provided image into a complete and precise HTML representation using advanced HTML techniques. 
            DO NOT SIMPLIFY THE TABLE STRUCTURE AND ENSURE YOU COMPLETE THE ENTIRE TABLE. Get all content including titles, headers and footers.
                HTML only.
                Instructions:
                1. Analyze the table thoroughly, ensuring no detail is overlooked.
                2. Create an HTML structure that exactly mirrors the layout and content of the table, using advanced HTML5 elements where appropriate (e.g., <thead>, <tbody>, <tfoot>, <colgroup>, <caption>).
                3. Include ALL text, numbers, and any other content present in the table.
                4. Preserve the table structure exactly as it appears, using appropriate HTML tags (<table>, <tr>, <td>, <th>, etc.).
                5. Maintain proper formatting, including any bold, italic, or other text styles visible. 
                6. Transcribe the content exactly as it appears, using appropriate semantic HTML elements (e.g., <strong>, <em>, <sup>, <sub>) where necessary.
                7. Implement proper accessibility features, such as including a <caption> for the table if a title is present.
                8. PRESERVE ALL COMPLEXITY OF THE ORIGINAL TABLE IN YOUR HTML, including any nested tables, rowspans, or colspans.
                9. If the table contains any interactive elements or complex layouts, consider using appropriate ARIA attributes to enhance accessibility.
                10. ENSURE YOU COMPLETE THE ENTIRE TABLE. Do not stop until you have transcribed every single row and column. Use colspan and rowspan attributes to handle merged cells. If the table is extensive, prioritize completeness over perfect formatting.
                11. Double-check that you have included all rows and columns before finishing.
                12. Pay special attention to merged cells, split cells, and complex table structures. 
                Use appropriate colspan and rowspan attributes with the right number to accurately represent the table layout. For cells split diagonally or containing multiple pieces of information, consider using nested tables or CSS positioning to faithfully reproduce the layout.
                Ensure the structure of information is accurately represented in your html.
                Output: Provide the complete HTML code, without any explanations or comments. 
                Your response will be evaluated based on its completeness, accuracy, and use of HTML techniques in representing every aspect of the table.
                Intelligently include any text around the table if present. Make intelligent decisions about how to represent the table. It must still be contained inside the ```html ... ``` tags.
                Wrap your output in a code block like ```html ... ```."
            ),
            ("prompt3", "Task: Convert the provided image into a complete and precise HTML representation using advanced HTML techniques. 
            DO NOT SIMPLIFY THE TABLE STRUCTURE AND ENSURE YOU COMPLETE THE ENTIRE TABLE. Get all content including titles, headers and footers.
                HTML only.
                Instructions:
                1. Analyze the table thoroughly, ensuring no detail is overlooked.
                2. Create an HTML structure that exactly mirrors the layout and content of the table, using advanced HTML5 elements where appropriate (e.g., <thead>, <tbody>, <tfoot>, <colgroup>, <caption>).
                3. Include ALL text, numbers, and any other content present in the table.
                4. Preserve the table structure exactly as it appears, using appropriate HTML tags (<table>, <tr>, <td>, <th>, etc.).
                5. Maintain proper formatting, including any bold, italic, or other text styles visible. 
                6. Transcribe the content exactly as it appears, using appropriate semantic HTML elements (e.g., <strong>, <em>, <sup>, <sub>) where necessary.
                7. Implement proper accessibility features, such as including a <caption> for the table if a title is present.
                8. PRESERVE ALL COMPLEXITY OF THE ORIGINAL TABLE IN YOUR HTML, including any nested tables, rowspans, or colspans.
                9. If the table contains any interactive elements or complex layouts, consider using appropriate ARIA attributes to enhance accessibility.
                10. ENSURE YOU COMPLETE THE ENTIRE TABLE. Do not stop until you have transcribed every single row and column. Use colspan and rowspan attributes to handle merged cells. If the table is extensive, prioritize completeness over perfect formatting.
                11. Double-check that you have included all rows and columns before finishing.
                12. Pay special attention to merged cells, split cells, and complex table structures. 
                Use appropriate colspan and rowspan attributes with the right number to accurately represent the table layout. For cells split diagonally or containing multiple pieces of information, consider using nested tables or CSS positioning to faithfully reproduce the layout.
                Ensure the structure of information is accurately represented in your html.
                Output: Provide the complete HTML code, without any explanations or comments. 
                Your response will be evaluated based on its completeness, accuracy, and use of HTML techniques in representing every aspect of the table.
                Intelligently include any text around the table if present. Make intelligent decisions about how to represent the table. It must still be contained inside the ```html ... ``` tags.
                
                
                Output instructions:
                1. before you start, output a plan with special considerations for how you will write html for the table, take into account quirks, edge cases, complex structure, and write in plain english how in HTML you will solve it. Output this in <thinking> </thinking> tags.  
                2. taking your thinking into consideration, output your final HTML in ```html ... ``` tags.
                
                Wrap your output in a code block like ```html ... ```."
            )
        ]);

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
        let mut stats = HashMap::new();
        for (model_shorthand, _) in &models {
            stats.insert(model_shorthand.to_string(), (0, 0)); // (total, broken)
        }
        
        // Process specified number of files or all if n_files is 0
        let n_files = 100; // Change this to limit number of files processed
        let files_to_process = if n_files > 0 {
            input_files.iter().take(n_files).collect::<Vec<_>>()
        } else {
            input_files.iter().collect::<Vec<_>>()
        };
        
        for input_file in files_to_process {
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

            for (model_shorthand, model) in &models {
                for (prompt_shorthand, prompt) in &prompts {
                    let input_file = input_file.clone();
                    let prompt = prompt.to_string();
                    let model = model.to_string();
                    let model_shorthand = model_shorthand.to_string();
                    let prompt_shorthand = prompt_shorthand.to_string();
                    let model_for_write = model.clone();
                    let prompt_for_write = prompt.clone();

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
                        // Add small random delay to prevent rate limit issues
                        let jitter = rand::random::<f32>() * 0.3 + 0.2; // Random delay between 0.2-0.5 seconds
                        tokio::time::sleep(std::time::Duration::from_secs_f32(jitter)).await;
                        match open_ai_call(url, key, model, messages, None, None).await {
                            Ok(response) => {
                                let duration = start_time.elapsed();
                                let mut content: MessageContent = response.choices[0].message.content.clone();
                                content = match content {
                                    MessageContent::String { content: text } => {
                                        let re = regex::Regex::new(r"<thinking>.*?</thinking>").unwrap();
                                        MessageContent::String { content: re.replace_all(&text, "").to_string() }
                                    },
                                    _ => content
                                };
                                if let MessageContent::String { content } = content {
                                    let html_result = get_html_from_llm_table_ocr(content.clone());
                                    let html_content = html_result.unwrap_or(content);
                                    

                                    let html_file = table_dir
                                        .join(format!("{}_{}.html", model_shorthand, prompt_shorthand));
                                    fs::write(&html_file, html_content)?;
                                    println!("HTML for {} saved to {:?}", model_shorthand, html_file);

                                    let csv_file = table_dir
                                        .join(format!("{}_{}.csv", model_shorthand, prompt_shorthand));
                                    let csv_content = format!(
                                        "Model,Table,Prompt,Duration\n{},{},{},{:?}\n",
                                        model_for_write, table_name, prompt_for_write, duration
                                    );
                                    fs::write(&csv_file, csv_content)?;
                                } else {
                                    let html_file = table_dir
                                        .join(format!("{}_{}.html", model_shorthand, prompt_shorthand));
                                    fs::write(&html_file, "No choices in response")?;

                                    let csv_file = table_dir
                                        .join(format!("{}_{}.csv", model_shorthand, prompt_shorthand));
                                    let csv_content = format!(
                                        "Model,Table,Prompt,Duration\n{},{},{},{:?}\n",
                                        model_for_write, table_name, prompt_for_write, duration
                                    );
                                    fs::write(&csv_file, csv_content)?;
                                }
                            }
                            Err(e) => {
                                println!(
                                    "Error processing {} with model {} and prompt '{}': {:?}",
                                    table_name, model_shorthand, prompt_for_write, e
                                );

                                return Err(e.to_string().into());
                            }
                        }
                        Ok::<_, Box<dyn Error + Send + Sync>>(())
                    });

                    tasks.push(task);
                }
            }
        }

        for task in tasks {
            task.await??;
        }

        // After all tasks complete, print statistics
        println!("\nFinal Statistics:");
        for (model, (total, broken)) in stats {
            println!(
                "{}: {}/{} successful ({:.1}% broken)", 
                model, 
                total - broken, 
                total, 
                (broken as f64 / total as f64) * 100.0
            );
        }

        println!("test_ocr_llm_with_image completed successfully");
        Ok(())
    }
}
