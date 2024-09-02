use serde::{ Deserialize, Serialize };

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct UserInfo {
    pub user_id: String,
    pub api_key: Option<String>,
}
