use crate::models::server::user::User;
use crate::models::auth::auth::UserInfo;
use crate::utils::db::deadpool_postgres::Pool;
use super::{ get_user::get_user, create_user::create_user };

pub async fn get_or_create_admin_user(pool: &Pool) -> Result<User, Box<dyn std::error::Error>> {
    let user_info = UserInfo {
        user_id: "admin".to_string(),
        email: Some("admin@chunkmydocs.com".to_string()),
        first_name: Some("admin".to_string()),
        last_name: Some("admin".to_string()),
        api_key: None,
    };

    let user = match get_user(user_info.user_id.clone(), &pool).await {
        Ok(user) => user,
        Err(e) => {
            if e.to_string().contains("not found") {
                let user = create_user(user_info, &pool).await?;
                println!("Admin user created: {:#?}", user);
                println!("IMPORTANT: The admin user details will only be displayed once upon creation. Please save the API key securely as it won't be shown again.");
                return Ok(user);
            } else {
                return Err(e.into());
            }
        }
    };

    Ok(user)
}