
use crate::models::chunkr::structured_extraction::ExtractionRequest;
use crate::models::chunkr::structured_extraction::ExtractionResponse;

use crate::utils::services::structured_extraction::perform_structured_extraction;



pub async fn handle_structured_extraction(
    req: ExtractionRequest,
) -> Result<ExtractionResponse, Box<dyn std::error::Error>> {
    match perform_structured_extraction(req.json_schema, req.contents, req.content_type).await {
        Ok(extracted_json) => Ok(ExtractionResponse { extracted_json }),
        Err(e) => Err(format!("Extraction failed: {}", e).into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::chunkr::structured_extraction::Property;
    use crate::models::chunkr::structured_extraction::JsonSchema;
    use crate::utils::clients;
    use tokio;

    #[tokio::test]
    async fn test_structured_extraction() {
        clients::initialize().await;
        let json_schema = JsonSchema {
            title: "Fruit Facts".to_string(),
            properties: vec![
                Property {
                    name: "fruit_name".to_string(),
                    title: Some("Fruit Name".to_string()),
                    prop_type: "string".to_string(),
                    description: Some("The name of the fruit".to_string()),
                    default: None,
                },
                Property {
                    name: "color".to_string(),
                    title: Some("Color".to_string()),
                    prop_type: "string".to_string(),
                    description: Some("The color of the fruit when ripe".to_string()),
                    default: None,
                },
                Property {
                    name: "calories_per_100g".to_string(),
                    title: Some("Calories".to_string()),
                    prop_type: "number".to_string(),
                    description: Some("Number of calories per 100g serving".to_string()),
                    default: None,
                },
                Property {
                    name: "is_citrus".to_string(),
                    title: Some("Is Citrus".to_string()),
                    prop_type: "boolean".to_string(),
                    description: Some("Whether the fruit is a citrus fruit".to_string()),
                    default: None,
                },
            ],
            schema_type: None,
        };

        let contents = vec![
            "Oranges are bright orange citrus fruits that contain about 47 calories per 100g serving.".to_string(),
            "Bananas have a yellow peel and are not citrus fruits. They contain approximately 89 calories per 100g.".to_string(),
            "Lemons are yellow citrus fruits with around 29 calories per 100g.".to_string()
        ];

        let request = ExtractionRequest {
            json_schema,
            contents,
            content_type: "text".to_string(),
        };

        let response = handle_structured_extraction(request).await;
        println!("Response: {:?}", response);
        assert!(response.is_ok());
    }
}
