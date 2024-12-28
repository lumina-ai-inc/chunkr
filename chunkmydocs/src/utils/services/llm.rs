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
        .error_for_status()?
        .json::<OpenAiResponse>()
        .await?;

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

    use crate::utils::configs::llm_config::Config as LlmConfig;
    use crate::utils::services::ocr::{
        get_html_from_llm_table_ocr, get_markdown_from_llm_table_ocr,
    };
    use std::collections::HashMap;
    use std::fs;
    use std::path::Path;
    use std::time::Instant;
    use tokio;

    #[tokio::test]
    async fn test_html_ocr_llm() -> Result<(), Box<dyn Error + Send + Sync>> {
        println!("Starting test_ocr_llm_with_image");

        let input_dir = Path::new("input");
        let output_dir = Path::new("output");
        fs::create_dir_all(output_dir)?;

        let llm_config = LlmConfig::from_env().unwrap();
        let url = llm_config.url;
        let key = llm_config.key;

        let models = HashMap::from([("geminipro", "gemini-1.5-flash")]);

        let prompts = HashMap::from([
            ("prompt1", "Analyze this image and convert the table to HTML format maintaining the original structure. If the image provided is not a table (for example if it is a image, formula, chart, etc), then represent the information (maintain all the text exactly as it is) but structure it gracefully into html.
. Output the table directly in ```html``` tags."),
            ("prompt2", "Analyze this image and convert the table to HTML format maintaining the original structure. If the image provided is not a table (for example if it is a image, formula, chart, etc), then represent the information (maintain all the text exactly as it is) but structure it gracefully into html.
            . Include all text shown and any empty cells. Output the table directly in ```html``` tags."),
            // ("prompt3", "Task: Convert the provided image into a complete and precise HTML representation using advanced HTML techniques. 
            // DO NOT SIMPLIFY THE TABLE STRUCTURE AND ENSURE YOU COMPLETE THE ENTIRE TABLE. Get all content including titles, headers and footers.
            //     HTML only.
            //     Instructions:
            //     1. Analyze the table thoroughly, ensuring no detail is overlooked.
            //     2. Create an HTML structure that exactly mirrors the layout and content of the table, using advanced HTML5 elements where appropriate (e.g., <thead>, <tbody>, <tfoot>, <colgroup>, <caption>).
            //     3. Include ALL text, numbers, and any other content present in the table.
            //     4. Preserve the table structure exactly as it appears, using appropriate HTML tags (<table>, <tr>, <td>, <th>, etc.).
            //     5. Maintain proper formatting, including any bold, italic, or other text styles visible. 
            //     6. Transcribe the content exactly as it appears, using appropriate semantic HTML elements (e.g., <strong>, <em>, <sup>, <sub>) where necessary.
            //     7. Implement proper accessibility features, such as including a <caption> for the table if a title is present.
            //     8. PRESERVE ALL COMPLEXITY OF THE ORIGINAL TABLE IN YOUR HTML, including any nested tables, rowspans, or colspans.
            //     9. If the table contains any interactive elements or complex layouts, consider using appropriate ARIA attributes to enhance accessibility.
            //     10. ENSURE YOU COMPLETE THE ENTIRE TABLE. Do not stop until you have transcribed every single row and column. Use colspan and rowspan attributes to handle merged cells. If the table is extensive, prioritize completeness over perfect formatting.
            //     11. Double-check that you have included all rows and columns before finishing.
            //     12. Pay special attention to merged cells, split cells, and complex table structures. 
            //     Use appropriate colspan and rowspan attributes with the right number to accurately represent the table layout. For cells split diagonally or containing multiple pieces of information, consider using nested tables or CSS positioning to faithfully reproduce the layout.
            //     Ensure the structure of information is accurately represented in your html.
            //     Output: Provide the complete HTML code, without any explanations or comments. 
            //     Your response will be evaluated based on its completeness, accuracy, and use of HTML techniques in representing every aspect of the table.
            //     Intelligently include any text around the table if present. Make intelligent decisions about how to represent the table. It must still be contained inside the ```html ... ``` tags.
            //     if the image provided is not a table (for example if it is a image, formula, chart, etc), then represent the information (maintain all the text exactly as it is) but structure it gracefully into html.
            //     output all your html in a <body> </body> tag enclosed in ```html ... ``` tags. For example:
            //     ```html
            //     <body>
            //     ...
            //     </body>
            //     ```
            //     you are always outputting HTML code. Wrap your output in a code block like ```html ... ```."
            // )
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

        let mut tasks: Vec<tokio::task::JoinHandle<Result<(), Box<dyn Error + Send + Sync>>>> =
            Vec::new();
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
                                return Ok(()); // Continue with next task
                            }
                        };
                        // Add small random delay to prevent rate limit issues
                        let jitter = rand::random::<f32>() * 0.3 + 0.2; // Random delay between 0.2-0.5 seconds
                        tokio::time::sleep(std::time::Duration::from_secs_f32(jitter)).await;
                        match open_ai_call(url, key, model, messages, None, None).await {
                            Ok(response) => {
                                let duration = start_time.elapsed();
                                let mut content: MessageContent =
                                    response.choices[0].message.content.clone();
                                content = match content {
                                    MessageContent::String { content: text } => {
                                        let re =
                                            regex::Regex::new(r"<thinking>.*?</thinking>").unwrap();
                                        MessageContent::String {
                                            content: re.replace_all(&text, "").to_string(),
                                        }
                                    }
                                    _ => content,
                                };
                                if let MessageContent::String { content } = content {
                                    let html_result = get_html_from_llm_table_ocr(content.clone());
                                    let html_content = html_result.unwrap_or(content);

                                    let html_file = table_dir.join(format!(
                                        "{}_{}.html",
                                        model_shorthand, prompt_shorthand
                                    ));
                                    if let Err(e) = fs::write(&html_file, html_content) {
                                        println!("Error writing HTML file: {:?}", e);
                                    }
                                    println!(
                                        "HTML for {} saved to {:?}",
                                        model_shorthand, html_file
                                    );

                                    let csv_file = table_dir.join(format!(
                                        "{}_{}.csv",
                                        model_shorthand, prompt_shorthand
                                    ));
                                    let csv_content = format!(
                                        "Model,Table,Prompt,Duration\n{},{},{},{:?}\n",
                                        model_for_write, table_name, prompt_for_write, duration
                                    );
                                    if let Err(e) = fs::write(&csv_file, csv_content) {
                                        println!("Error writing CSV file: {:?}", e);
                                    }
                                } else {
                                    let html_file = table_dir.join(format!(
                                        "{}_{}.html",
                                        model_shorthand, prompt_shorthand
                                    ));
                                    if let Err(e) = fs::write(&html_file, "No choices in response")
                                    {
                                        println!("Error writing HTML file: {:?}", e);
                                    }

                                    let csv_file = table_dir.join(format!(
                                        "{}_{}.csv",
                                        model_shorthand, prompt_shorthand
                                    ));
                                    let csv_content = format!(
                                        "Model,Table,Prompt,Duration\n{},{},{},{:?}\n",
                                        model_for_write, table_name, prompt_for_write, duration
                                    );
                                    if let Err(e) = fs::write(&csv_file, csv_content) {
                                        println!("Error writing CSV file: {:?}", e);
                                    }
                                }
                            }
                            Err(e) => {
                                println!(
                                    "Error processing {} with model {} and prompt '{}': {:?}",
                                    table_name, model_shorthand, prompt_for_write, e
                                );
                            }
                        }
                        Ok(())
                    });

                    tasks.push(task);
                }
            }
        }

        for task in tasks {
            // Ignore errors from individual tasks
            let _ = task.await;
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

    #[tokio::test]
    async fn test_mkd_ocr_llm() -> Result<(), Box<dyn Error + Send + Sync>> {
        println!("Starting test_ocr_llm_with_image");

        let input_dir = Path::new("input");
        let output_dir = Path::new("output");
        fs::create_dir_all(output_dir)?;

        let llm_config = LlmConfig::from_env().unwrap();
        let url = llm_config.url;
        let key = llm_config.key;

        let models = HashMap::from([("geminipro", "google/gemini-pro-1.5")]);

        let prompts = HashMap::from([
            ("prompt1", "you are always outputting HTML code.Analyze this image and convert the table to markdown format maintaining the original structure.                 3. if the image provided is not a table, then represent the information (maintain all the text exactly as it is) but structure it gracefully into markdown. Output the table directly in ```markdown``` tags."),
            // ("prompt2", "you are always outputting HTML code. Analyze this image and convert the table to markdown format maintaining the original structure.                 3. if the image provided is not a table, then represent the information (maintain all the text exactly as it is) but structure it gracefully into markdown. Output the table directly in ```markdown``` tags."),
            // ("prompt2", "Task: Convert the provided image into a complete and precise markdown representation using advanced markdown techniques. 
            // DO NOT SIMPLIFY THE TABLE STRUCTURE AND ENSURE YOU COMPLETE THE ENTIRE TABLE. Get all content including titles, headers and footers.
            //     Markdown only.
            //     Instructions:
            //     1. Analyze the table thoroughly, ensuring no detail is overlooked.
            //     2. Create a markdown structure that exactly mirrors the layout and content of the table.
            //     3. Include ALL text, numbers, and any other content present in the table.
            //     4. Preserve the table structure exactly as it appears, using appropriate markdown table syntax.
            //     5. Maintain proper formatting, including any bold, italic, or other text styles visible.
            //     6. Transcribe the content exactly as it appears, using appropriate markdown formatting (e.g., **bold**, *italic*).
            //     7. Include table headers and alignment if present.
            //     8. PRESERVE ALL COMPLEXITY OF THE ORIGINAL TABLE IN YOUR MARKDOWN.
            //     9. ENSURE YOU COMPLETE THE ENTIRE TABLE. Do not stop until you have transcribed every single row and column.
            //     10. Double-check that you have included all rows and columns before finishing.
            //     11. Pay special attention to complex table structures.
            //     Ensure the structure of information is accurately represented in your markdown.
            //     Output: Provide the complete markdown code, without any explanations or comments.
            //     Your response will be evaluated based on its completeness, accuracy, and use of markdown techniques in representing every aspect of the table.
            //     Intelligently include any text around the table if present. Make intelligent decisions about how to represent the table. It must still be contained inside the ```markdown ... ``` tags.
            //     if the image provided is not a table, then represent the information (maintain all the text exactly as it is) but structure it gracefully into markdown.

            //     you are always outputting HTML code. Wrap your output in a code block like ```markdown ... ```."
            // )
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

        let mut tasks: Vec<tokio::task::JoinHandle<Result<(), Box<dyn Error + Send + Sync>>>> =
            Vec::new();
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
                                return Ok(()); // Continue with next task
                            }
                        };
                        // Add small random delay to prevent rate limit issues
                        let jitter = rand::random::<f32>() * 0.3 + 0.2; // Random delay between 0.2-0.5 seconds
                        tokio::time::sleep(std::time::Duration::from_secs_f32(jitter)).await;
                        match open_ai_call(url, key, model, messages, None, None).await {
                            Ok(response) => {
                                let duration = start_time.elapsed();
                                let mut content: MessageContent =
                                    response.choices[0].message.content.clone();
                                content = match content {
                                    MessageContent::String { content: text } => {
                                        let re =
                                            regex::Regex::new(r"<thinking>.*?</thinking>").unwrap();
                                        MessageContent::String {
                                            content: re.replace_all(&text, "").to_string(),
                                        }
                                    }
                                    _ => content,
                                };
                                if let MessageContent::String { content } = content {
                                    let markdown_result =
                                        get_markdown_from_llm_table_ocr(content.clone());
                                    let markdown_content = markdown_result.unwrap_or(content);

                                    let markdown_file = table_dir.join(format!(
                                        "{}_{}.md",
                                        model_shorthand, prompt_shorthand
                                    ));
                                    if let Err(e) = fs::write(&markdown_file, markdown_content) {
                                        println!("Error writing markdown file: {:?}", e);
                                    }
                                    println!(
                                        "Markdown for {} saved to {:?}",
                                        model_shorthand, markdown_file
                                    );

                                    let csv_file = table_dir.join(format!(
                                        "{}_{}.csv",
                                        model_shorthand, prompt_shorthand
                                    ));
                                    let csv_content = format!(
                                        "Model,Table,Prompt,Duration\n{},{},{},{:?}\n",
                                        model_for_write, table_name, prompt_for_write, duration
                                    );
                                    if let Err(e) = fs::write(&csv_file, csv_content) {
                                        println!("Error writing CSV file: {:?}", e);
                                    }
                                } else {
                                    let markdown_file = table_dir.join(format!(
                                        "{}_{}.md",
                                        model_shorthand, prompt_shorthand
                                    ));
                                    if let Err(e) =
                                        fs::write(&markdown_file, "No choices in response")
                                    {
                                        println!("Error writing markdown file: {:?}", e);
                                    }

                                    let csv_file = table_dir.join(format!(
                                        "{}_{}.csv",
                                        model_shorthand, prompt_shorthand
                                    ));
                                    let csv_content = format!(
                                        "Model,Table,Prompt,Duration\n{},{},{},{:?}\n",
                                        model_for_write, table_name, prompt_for_write, duration
                                    );
                                    if let Err(e) = fs::write(&csv_file, csv_content) {
                                        println!("Error writing CSV file: {:?}", e);
                                    }
                                }
                            }
                            Err(e) => {
                                println!(
                                    "Error processing {} with model {} and prompt '{}': {:?}",
                                    table_name, model_shorthand, prompt_for_write, e
                                );
                            }
                        }
                        Ok(())
                    });

                    tasks.push(task);
                }
            }
        }

        for task in tasks {
            // Ignore any errors from individual tasks
            let _ = task.await;
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
