use serde::{ Deserialize, Serialize };

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct ApiInfo {
    pub api_key: String,
    pub user_id: String,
}