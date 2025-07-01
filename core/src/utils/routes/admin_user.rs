use super::{create_user::create_user, get_user::get_user};
use crate::models::auth::UserInfo;
use crate::models::user::User;

pub async fn get_or_create_admin_user() -> Result<User, Box<dyn std::error::Error>> {
    let user_info = UserInfo {
        user_id: "admin".to_string(),
        email: Some("admin@chunkr.ai".to_string()),
        first_name: Some("admin".to_string()),
        last_name: Some("admin".to_string()),
        api_key: None,
    };

    let user = match get_user(user_info.user_id.clone()).await {
        Ok(user) => user,
        Err(e) => {
            if e.to_string().contains("not found") {
                let user = create_user(user_info).await?;
                println!("Admin user created: {user:#?}");
                println!("IMPORTANT: The admin user details will only be displayed once upon creation. Please save the API key securely as it won't be shown again.");
                return Ok(user);
            } else {
                return Err(e);
            }
        }
    };

    Ok(user)
}
