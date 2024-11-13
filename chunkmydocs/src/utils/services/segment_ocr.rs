use crate::models::server::segment::OCRResult;
use crate::models::workers::table_ocr::PaddleTableRecognitionResult;
use crate::utils::configs::worker_config::{Config as WorkerConfig, TableOcrModel};
use crate::utils::services::html::{
    clean_img_tags, convert_table_to_markdown, extract_table_html, validate_html,
};
use crate::utils::services::ocr::{llm_formula_ocr, llm_table_ocr, paddle_ocr, paddle_table_ocr};
use crate::utils::storage::services::download_to_tempfile;
use aws_sdk_s3::Client as S3Client;
use reqwest::Client as ReqwestClient;

pub async fn download_and_ocr(
    s3_client: &S3Client,
    reqwest_client: &ReqwestClient,
    image_location: &str,
) -> Result<(Vec<OCRResult>, String, String), Box<dyn std::error::Error>> {
    let original_file =
        download_to_tempfile(s3_client, reqwest_client, image_location, None).await?;
    let ocr_results = match paddle_ocr(original_file.path()).await {
        Ok(ocr_results) => ocr_results,
        Err(e) => {
            return Err(e.to_string().into());
        }
    };
    Ok((ocr_results, "".to_string(), "".to_string()))
}

pub async fn download_and_formula_ocr(
    s3_client: &S3Client,
    reqwest_client: &ReqwestClient,
    image_location: &str,
) -> Result<(Vec<OCRResult>, String, String), Box<dyn std::error::Error>> {
    let original_file =
        download_to_tempfile(s3_client, reqwest_client, image_location, None).await?;
    let latex_formula = match llm_formula_ocr(original_file.path()).await {
        Ok(latex_formula) => {
            get_latex_from_vllm_formula_ocr(latex_formula).unwrap_or("".to_string())
        }
        Err(e) => {
            return Err(e.to_string().into());
        }
    };

    Ok((
        vec![],
        format!("<span class=\"formula\">{}</span>", latex_formula.clone()),
        format!("${}$", latex_formula),
    ))
}

pub async fn download_and_table_ocr(
    s3_client: &S3Client,
    reqwest_client: &ReqwestClient,
    image_location: &str,
) -> Result<(Vec<OCRResult>, String, String), Box<dyn std::error::Error>> {
    let worker_config = WorkerConfig::from_env()?;
    let original_file =
        download_to_tempfile(s3_client, reqwest_client, image_location, None).await?;
    let original_file_path = original_file.path().to_owned();
    let original_file_path_clone = original_file_path.clone();
    let table_ocr_task = tokio::task::spawn(async move {
        match worker_config.table_ocr_model {
            TableOcrModel::Paddle => {
                let result = paddle_table_ocr(&original_file_path).await?;
                get_html_from_paddle_table_ocr(result)
            }
            TableOcrModel::LLM => {
                let result = llm_table_ocr(&original_file_path).await?;
                get_html_from_llm_table_ocr(result)
            }
        }
    });
    let paddle_ocr_task =
        tokio::task::spawn(async move { paddle_ocr(&original_file_path_clone).await });
    let ocr_results = match paddle_ocr_task.await {
        Ok(ocr_results) => ocr_results.unwrap_or_default(),
        Err(e) => {
            println!("Error getting OCR results: {}", e);
            vec![]
        }
    };

    let table_ocr_result: Result<(String, String), Box<dyn std::error::Error>> =
        match table_ocr_task.await {
            Ok(html) => match html {
                Ok(html) => {
                    let html = extract_table_html(html);
                    let markdown = convert_table_to_markdown(html.clone());
                    Ok((html, markdown))
                }
                Err(e) => {
                    println!("Error getting table OCR results: {}", e);
                    Ok(("".to_string(), "".to_string()))
                }
            },
            Err(e) => Err(e.to_string().into()),
        };

    match table_ocr_result {
        Ok(result) => Ok((ocr_results, result.0, result.1)),
        Err(e) => {
            println!("Error getting table OCR results: {}", e);
            Ok((ocr_results, "".to_string(), "".to_string()))
        }
    }
}

fn get_html_from_paddle_table_ocr(
    table_ocr_result: PaddleTableRecognitionResult,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let first_table = table_ocr_result.tables.first().cloned();
    match first_table {
        Some(table) => Ok(table.html),
        None => Err("No table structure found".to_string().into()),
    }
}

pub fn get_html_from_llm_table_ocr(
    table_ocr_result: String,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    if let Some(html_content) = table_ocr_result.split("```html").nth(1) {
        if let Some(html) = html_content.split("```").next() {
            let html = html.trim().to_string();
            let cleaned_html = clean_img_tags(&html);
            return match validate_html(&cleaned_html) {
                Ok(_) => Ok(cleaned_html),
                Err(e) => Err(e.to_string().into()),
            };
        }
    }
    Err("No HTML content found in table OCR result".into())
}

pub fn get_latex_from_vllm_formula_ocr(
    formula_ocr_result: String,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    if let Some(latex_content) = formula_ocr_result.split("```latex").nth(1) {
        if let Some(latex) = latex_content.split("```").next() {
            return Ok(latex.trim().to_string());
        }
    }
    Err("No LaTeX content found in formula OCR result".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_html() {
        let input = r#"
        ```html
        <table>
            <tr>
                <td>1</td>
                <td>2</td>
            </tr>
        </table>
        ```
        "#;
        let html = get_html_from_llm_table_ocr(input.to_string()).unwrap();
        assert_eq!(
            html,
            r#"<table>
            <tr>
                <td>1</td>
                <td>2</td>
            </tr>
        </table>"#
        );
    }

    #[test]
    fn test_invalid_html() {
        let input = r#"
        ```html
        <table>
            <tr>
                <td>1
                <td>2</td>
            </tr>
        </table>
        ```
        "#;
        let result = get_html_from_llm_table_ocr(input.to_string());
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "Mismatched HTML tags: expected </td>, found </tr>"
        );
    }

    #[test]
    fn test_image_alt_text() {
        let input = r#"
        ```html
        <table>
            <tr>
                <td>1</td>
                <td>2</td>
                <td><img src="pizza.png" alt="pizza"></td>
            </tr>
        </table>
        "#;
        let html = get_html_from_llm_table_ocr(input.to_string()).unwrap();
        assert_eq!(
            html,
            r#"<table>
            <tr>
                <td>1</td>
                <td>2</td>
                <td>pizza</td>
            </tr>
        </table>"#
        );
    }

    #[test]
    fn test_image_no_alt_text() {
        let input = r#"
        ```html
        <table>
            <tr>
                <td>1</td>
                <td>2</td>
                <td><img src="pizza.png"></td>
            </tr>
        </table>
        "#;
        let html = get_html_from_llm_table_ocr(input.to_string()).unwrap();
        assert_eq!(
            html,
            r#"<table>
            <tr>
                <td>1</td>
                <td>2</td>
                <td></td>
            </tr>
        </table>"#
        );
    }

    #[test]
    fn test_void_elements() {
        let input = r#"
        ```html
        <table>
            <tr>
                <td><img src="pizza.png" /></td>
                <td>2<br>3</td>
            </tr>
        </table>
        ```
        "#;
        let html = get_html_from_llm_table_ocr(input.to_string()).unwrap();
        assert_eq!(
            html,
            r#"<table>
            <tr>
                <td></td>
                <td>2<br>3</td>
            </tr>
        </table>"#
        );
    }
}
